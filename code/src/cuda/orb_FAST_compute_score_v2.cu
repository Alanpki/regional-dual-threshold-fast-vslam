#include "cuda/orb_gpu.hpp"
#include <cooperative_groups.h> 
#include "cuda/orb_adaptive_kernels.cuh" 

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


__global__ void TileBased_FAST_Score_Two_Thresh_Kernel(
    int imheight, int imwidth,
    int threshold_high, int threshold_low,
    const int* __restrict__ lookup_table,
    const unsigned char* __restrict__ image_data, int image_pitch,
    const unsigned char* __restrict__ mask_data, int mask_pitch,
    int* __restrict__ score_data, int score_pitch)
{
    // 1. 變數與座標計算
    const int BORDER = 3; 

    const int tile_start_x = blockIdx.x * blockDim.x;
    const int tile_start_y = blockIdx.y * blockDim.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int w = tile_start_x + tx;
    const int h = tile_start_y + ty;

    extern __shared__ unsigned char s_tile[];
    // 用於區塊內歸約的共享旗標
    __shared__ int s_keypoint_found_flag;

    // 2. 載入 Tile 資料到 Shared Memory
    for (int i = ty; i < blockDim.y + 2 * BORDER; i += blockDim.y) {
        for (int j = tx; j < blockDim.x + 2 * BORDER; j += blockDim.x) {
            int load_x = tile_start_x + j - BORDER;
            int load_y = tile_start_y + i - BORDER;
            if (load_x >= 0 && load_x < imwidth && load_y >= 0 && load_y < imheight) {
                s_tile[i * (blockDim.x + 2 * BORDER) + j] = image_data[load_y * image_pitch + load_x];
            }
        }
    }

    // 由 Block 中的第一個 thread 初始化旗標
    if (tx == 0 && ty == 0) {
        s_keypoint_found_flag = 0;
    }
    
    // 同步，確保所有 thread 都完成了載入和旗標初始化
    __syncthreads();

    // 3. 雙門檻值角點檢測
    int local_score = 0;
    if (w < imwidth && h < imheight && w >= BORDER_SKIP && w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && h >= BORDER_SKIP)
    {
        if (mask_data[h * mask_pitch + w])
        {
            const unsigned char* ptr = &s_tile[(ty + BORDER) * (blockDim.x + 2 * BORDER) + (tx + BORDER)];
            const int v = ptr[0];
            const int s_pitch = blockDim.x + 2 * BORDER;
            const int p[16] = {
                3*s_pitch+0,  3*s_pitch+1,  2*s_pitch+2,  1*s_pitch+3, 0*s_pitch+3, -1*s_pitch+3, -2*s_pitch+2, -3*s_pitch+1,
               -3*s_pitch+0, -3*s_pitch-1, -2*s_pitch-2, -1*s_pitch-3, 0*s_pitch-3,  1*s_pitch-3,  2*s_pitch-2,  3*s_pitch-1
            };

            // Pass 1: 高門檻值檢測
            int vt_high = v + threshold_high;
            int v_t_high = v - threshold_high;
            int bright_idx = 0, dark_idx = 0;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int val = ptr[p[i]];
                if (val > vt_high) bright_idx |= (1 << i);
                if (val < v_t_high) dark_idx |= (1 << i);
            }
            if (lookup_table[bright_idx] || lookup_table[dark_idx]) {
                int sum_abs_diff = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) { sum_abs_diff += abs((int)ptr[p[i]] - v); }
                local_score = sum_abs_diff; // 正規化
            }

            if (local_score > 0) {
                atomicExch(&s_keypoint_found_flag, 1);
            }
            __syncthreads();

            // Pass 2: 低門檻值檢測 (僅在 Pass 1 失敗時)
            if (s_keypoint_found_flag == 0) {
                local_score = 0;
                int vt_low = v + threshold_low;
                int v_t_low = v - threshold_low;
                bright_idx = 0; dark_idx = 0;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int val = ptr[p[i]];
                    if (val > vt_low) bright_idx |= (1 << i);
                    if (val < v_t_low) dark_idx |= (1 << i);
                }
                if (lookup_table[bright_idx] || lookup_table[dark_idx]) {
                    int sum_abs_diff = 0;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) { sum_abs_diff += abs((int)ptr[p[i]] - v); }
                    local_score = sum_abs_diff; // 正規化
                }
            }
        }
    }

    // 4. 寫回 Global Memory
    if (w < imwidth && h < imheight) {
        score_data[h * score_pitch + w] = local_score;
    }
}

__global__ void SingleThreshFASTScoreKernel_Impl(
    int imheight, int imwidth,
    int threshold, // 只接收一個門檻值
    const int* __restrict__ lookup_table,
    const unsigned char* __restrict__ image_data, int image_pitch,
    const unsigned char* __restrict__ mask_data, int mask_pitch,
    int* __restrict__ score_data, int score_pitch)
{
    // --- 變數與 Tile 載入邏輯 ---
    const int BORDER = 3; 
    const int tile_start_x = blockIdx.x * blockDim.x;
    const int tile_start_y = blockIdx.y * blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int w = tile_start_x + tx;
    const int h = tile_start_y + ty;

    extern __shared__ unsigned char s_tile[];
    
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

    // --- 單門檻檢測邏輯 ---
    int local_score = 0;
    if (w < imwidth - BORDER_SKIP && h < imheight - BORDER_SKIP && w >= BORDER_SKIP && h >= BORDER_SKIP)
    {
        if (mask_data[h * mask_pitch + w])
        {
            const unsigned char* ptr = &s_tile[(ty + BORDER) * (blockDim.x + 2 * BORDER) + (tx + BORDER)];
            const int v = ptr[0];
            const int s_pitch = blockDim.x + 2 * BORDER;
            const int p[16] = { 3*s_pitch+0, 3*s_pitch+1, 2*s_pitch+2, 1*s_pitch+3, 0*s_pitch+3, -1*s_pitch+3, -2*s_pitch+2, -3*s_pitch+1, -3*s_pitch+0, -3*s_pitch-1, -2*s_pitch-2, -1*s_pitch-3, 0*s_pitch-3, 1*s_pitch-3, 2*s_pitch-2, 3*s_pitch-1};

            int vt = v + threshold;
            int v_t = v - threshold;
            int bright_idx = 0, dark_idx = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int val = ptr[p[i]];
                if (val > vt) bright_idx |= (1 << i);
                if (val < v_t) dark_idx |= (1 << i);
            }

            if (lookup_table[bright_idx] || lookup_table[dark_idx]) {
                local_score = computeCornerScore_OpenCV(v, ptr, p, threshold);
            }
        }
    }

    // --- 寫回 Global Memory ---
    if (w < imwidth && h < imheight) {
        score_data[h * score_pitch + w] = local_score;
    }
}

// #################### Host 端啟動函式 ####################
// 雙閾值
void ORB_GPU::FAST_compute_score_tile_based_templated(
    int height, int width,
    unsigned char* image_data_gpu, int image_pitch,
    unsigned char* mask_data_gpu, int mask_pitch,
    int threshold_high, int threshold_low,
    int* lookup_table_gpu,
    int* score_data_gpu, int score_pitch,
    cudaStream_t& cuda_stream)
{
    const int block_dim_x = this->fast_score_block_dim_x_;
    const int block_dim_y = this->fast_score_block_dim_y_;

    dim3 grid_dim( (width + block_dim_x - 1) / block_dim_x, (height + block_dim_y - 1) / block_dim_y );
    dim3 block_dim(block_dim_x, block_dim_y);

    const int BORDER = 3;
    size_t shared_mem_size = (block_dim_y + 2 * BORDER) * (block_dim_x + 2 * BORDER) * sizeof(unsigned char);

    TileBased_FAST_Score_Two_Thresh_Kernel<<<grid_dim, block_dim, shared_mem_size, cuda_stream>>>(
        height, width,
        threshold_high, threshold_low,
        lookup_table_gpu,
        image_data_gpu, image_pitch,
        mask_data_gpu, mask_pitch,
        score_data_gpu, score_pitch
    );
}

void ORB_GPU::FAST_compute_score_single_thresh(
    int height, int width,
    const unsigned char* image_data_gpu, int image_pitch,
    const unsigned char* mask_data_gpu, int mask_pitch,
    const int* lookup_table_gpu,
    int* score_data_gpu, int score_pitch,
    int threshold, // 只接收一個門檻值
    cudaStream_t& cuda_stream)
{
    const int block_dim_x = this->fast_score_block_dim_x_;
    const int block_dim_y = this->fast_score_block_dim_y_;

    dim3 grid_dim( (width + block_dim_x - 1) / block_dim_x, (height + block_dim_y - 1) / block_dim_y );
    dim3 block_dim(block_dim_x, block_dim_y);

    const int BORDER = 3;
    size_t shared_mem_size = (block_dim_y + 2 * BORDER) * (block_dim_x + 2 * BORDER) * sizeof(unsigned char);

    // 啟動單門檻 Kernel
    SingleThreshFASTScoreKernel_Impl<<<grid_dim, block_dim, shared_mem_size, cuda_stream>>>(
        height, width,
        threshold, // 傳入單一門檻值
        lookup_table_gpu,
        image_data_gpu, image_pitch,
        mask_data_gpu, mask_pitch,
        score_data_gpu, score_pitch
    );
}
// #################### Host 端啟動函式 ####################

// 重新定義 ORB_GPU 的成員函式，使其內部呼叫我們的優化版本
void ORB_GPU::FAST_compute_score_lookpup_mask_v2(int height, int width,
                                         unsigned char* image_data_gpu,
                                         int image_pitch,
                                         unsigned char* mask_data_gpu,
                                         int mask_pitch,
                                         int *lookup_table_gpu,
                                         int* score_data_gpu,
                                         int score_pitch,
                                         cudaStream_t& cuda_stream)
{
    FAST_compute_score_tile_based_templated(height, width,
                                  image_data_gpu, image_pitch,
                                  mask_data_gpu, mask_pitch,
                                  this->th_FAST_MAX_, // 高門檻值
                                  this->th_FAST_MIN_, // 低門檻值
                                  lookup_table_gpu,
                                  score_data_gpu, score_pitch,
                                  cuda_stream);
}

} // namespace orb_cuda