#include "cuda/orb_adaptive_kernels.cuh"
#include "cuda/orb_gpu.hpp" // 為了 BORDER_SKIP 等常數

namespace orb_cuda {

// 這是一個 __device__ 函式，用於在 GPU 上模擬 OpenCV 的 cornerScore
// 它的邏輯直接從 opencv/modules/features2d/src/fast_score.cpp 移植而來
__device__ inline int computeCornerScore_OpenCV(const int v, const unsigned char* ptr, const int p[16], int initial_threshold)
{
    // 為了環形存取，預先計算中心點與周圍點的灰度差
    short d[25];
    for (int k = 0; k < 16; k++)
        d[k] = (short)(v - ptr[p[k]]);
    for (int k = 16; k < 25; k++)
        d[k] = d[k - 16];
    // 一組為 9 個連續亮或暗點
    // --- 計算「暗」方向的最大安全閾值 ---
    int a0 = initial_threshold;
    for (int k = 0; k < 16; k += 2)
    {
        int a = min((int)d[k+1], (int)d[k+2]);
        if (a <= a0) continue;
        a = min(a, (int)d[k+3]);
        a = min(a, (int)d[k+4]);
        a = min(a, (int)d[k+5]);
        a = min(a, (int)d[k+6]);
        a = min(a, (int)d[k+7]);
        a = min(a, (int)d[k+8]);
        
        a0 = max(a0, min(a, (int)d[k]));
        a0 = max(a0, min(a, (int)d[k+9]));
    }

    // --- 計算「亮」方向的最大安全閾值 ---
    int b0 = -initial_threshold;
    for (int k = 0; k < 16; k += 2)
    {
        int b = max((int)d[k+1], (int)d[k+2]);
        if (b >= b0) continue;
        b = max(b, (int)d[k+3]);
        b = max(b, (int)d[k+4]);
        b = max(b, (int)d[k+5]);
        b = max(b, (int)d[k+6]);
        b = max(b, (int)d[k+7]);
        b = max(b, (int)d[k+8]);

        b0 = min(b0, max(b, (int)d[k]));
        b0 = min(b0, max(b, (int)d[k+9]));
    }

    // 最終分數是兩個方向中的較大值，再減 1
    return max(a0, -b0) - 1;
}

// 核心 A: 檢查 Block 狀態
__global__ void CheckAndSetBlockStatusKernel(
    const int* score_high_nms, 
    int* block_status, 
    int score_pitch,
    int height,
    int width)
{
    // 建立一個 block 範圍的共享旗標
    __shared__ int s_found_in_block;

    // 由 Block 中的第一個 thread 初始化旗標
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_found_in_block = 0;
    }
    __syncthreads(); // 確保旗標已被初始化

    // 每個 thread 檢查自己的像素
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w < width && h < height) {
        // 如果還沒有其他 thread 在這個 block 中找到特徵點
        if (s_found_in_block == 0) {
            if (score_high_nms[h * score_pitch + w] > 0) {
                // 就用原子操作設定共享旗標，通知 block 內的其他 thread
                atomicExch(&s_found_in_block, 1);
            }
        }
    }
    __syncthreads(); // 確保所有 thread 都完成了檢查

    // 最後，只讓 Block 中的第一個 thread 將最終結果寫回 Global Memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (s_found_in_block == 1) {
            block_status[blockIdx.y * gridDim.x + blockIdx.x] = 1;
        }
    }
}

// 核心 B: 條件式 FAST 分數計算
__global__ void ConditionalFASTScoreKernel_Impl(
    int imheight, int imwidth,
    const int* __restrict__ lookup_table,
    const unsigned char* __restrict__ image_data, int image_pitch,
    const unsigned char* __restrict__ mask_data, int mask_pitch,
    int* __restrict__ score_data, int score_pitch,
    const int* __restrict__ block_status,
    int threshold_low)
{
    // 使用 extern 關鍵字宣告動態共享記憶體
    extern __shared__ unsigned char s_tile[];

    if (block_status[blockIdx.y * gridDim.x + blockIdx.x] == 1) {
        return;
    }

    const int BORDER = 3;
    const int tile_start_x = blockIdx.x * blockDim.x;
    const int tile_start_y = blockIdx.y * blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int w = tile_start_x + tx;
    const int h = tile_start_y + ty;
    
    for (int i = ty; i < blockDim.y + 2 * BORDER; i += blockDim.y) {
        for (int j = tx; j < blockDim.x + 2 * BORDER; j += blockDim.x) {
            int load_x = tile_start_x + j - BORDER;
            int load_y = tile_start_y + i - BORDER;
            if (load_x >= 0 && load_x < imwidth && load_y >= 0 && load_y < imheight) {
                s_tile[i * (blockDim.x + 2 * BORDER) + j] = image_data[load_y * image_pitch + load_x];
            }
        }
    }
    __syncthreads();

    int local_score = 0;
    if (w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && w >= BORDER_SKIP && h >= BORDER_SKIP)
    {
        if (mask_data[h * mask_pitch + w])
        {
            const unsigned char* ptr = &s_tile[(ty + BORDER) * (blockDim.x + 2 * BORDER) + (tx + BORDER)];
            const int v = ptr[0];
            const int s_pitch = blockDim.x + 2 * BORDER;
            const int p[16] = { 3*s_pitch+0, 3*s_pitch+1, 2*s_pitch+2, 1*s_pitch+3, 0*s_pitch+3, -1*s_pitch+3, -2*s_pitch+2, -3*s_pitch+1, -3*s_pitch+0, -3*s_pitch-1, -2*s_pitch-2, -1*s_pitch-3, 0*s_pitch-3, 1*s_pitch-3, 2*s_pitch-2, 3*s_pitch-1};

            
            int vt_low = v + threshold_low;
            int v_t_low = v - threshold_low;
            int bright_idx = 0, dark_idx = 0;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int val = ptr[p[i]];
                if (val > vt_low) bright_idx |= (1 << i);
                if (val < v_t_low) dark_idx |= (1 << i);
            }

            if (lookup_table[bright_idx] || lookup_table[dark_idx]) {
                /* SAD評分
                int sum_abs_diff = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) { sum_abs_diff += abs((int)ptr[p[i]] - v); }
                local_score = sum_abs_diff;
                */
                local_score = computeCornerScore_OpenCV(v, ptr, p, threshold_low);
            }
        }
    }

    if (w < imwidth && h < imheight) {
        score_data[h * score_pitch + w] = local_score;
    }
}

// 核心 C: 合併結果
__global__ void MergeResultsKernel(
    const int* score_high_nms, 
    const int* score_low_nms, 
    int* final_score_nms,
    int pitch,
    int height,
    int width)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w < width && h < height) {
        int idx = h * pitch + w;
        int high_val = score_high_nms[idx];

        if (high_val > 0) {
            final_score_nms[idx] = high_val;
        } else {
            final_score_nms[idx] = score_low_nms[idx];
        }
    }
}



//#################### Host 端啟動函式 ####################

void ORB_GPU::LaunchCheckAndSetBlockStatus(const int* score_high_nms, int* block_status, int score_pitch, int height, int width, cudaStream_t stream)
{
    const int block_dim_x = this->fast_score_block_dim_x_;
    const int block_dim_y = this->fast_score_block_dim_y_;
    dim3 block_dim(block_dim_x, block_dim_y);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    CheckAndSetBlockStatusKernel<<<grid_dim, block_dim, 0, stream>>>(score_high_nms, block_status, score_pitch, height, width);
}

void ORB_GPU::LaunchConditionalFASTScore(int imheight, int imwidth, const unsigned char* image_data, int image_pitch, const unsigned char* mask_data, int mask_pitch, const int* lookup_table, int* score_data, int score_pitch, const int* block_status, int threshold_low, cudaStream_t stream)
{
    const int block_dim_x = this->fast_score_block_dim_x_;
    const int block_dim_y = this->fast_score_block_dim_y_;

    dim3 grid_dim((imwidth + block_dim_x - 1) / block_dim_x, (imheight + block_dim_y - 1) / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);

    const int BORDER = 3;
    size_t shared_mem_size = (block_dim_y + 2 * BORDER) * (block_dim_x + 2 * BORDER) * sizeof(unsigned char);
    
    ConditionalFASTScoreKernel_Impl<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        imheight, imwidth, lookup_table, 
        image_data, image_pitch, 
        mask_data, mask_pitch, 
        score_data, score_pitch, 
        block_status, threshold_low);
        
}

void ORB_GPU::LaunchMergeResults(const int* score_high_nms, const int* score_low_nms, int* final_score_nms, int pitch, int height, int width, cudaStream_t stream)
{
    dim3 block_dim(32, 8);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
    MergeResultsKernel<<<grid_dim, block_dim, 0, stream>>>(score_high_nms, score_low_nms, final_score_nms, pitch, height, width);
}

} // namespace orb_cuda