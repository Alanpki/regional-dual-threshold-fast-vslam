#include "cuda/orb_matcher.hpp"

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<stdio.h>

#include<iostream>

#include<chrono>

#include <device_functions.h> // sqrtf

namespace orb_cuda{

#define CUDA_NUM_THREADS_PER_BLOCK 512

// --- 可配置參數 ---
constexpr int PATCH_SIZE = 9;            // Patch大小，例如 9x9
constexpr int PATCH_HALF_SIZE = PATCH_SIZE / 2;
constexpr int STRICT_HAMMING_THRESHOLD = 100;    // 高可信度匹配的漢明距離上限
constexpr float BRIGHTNESS_DIFF_THRESHOLD = 10.0f; // Patch平均亮度差異的閾值 (0-255之間，可調)
constexpr float BRIGHTNESS_STD_DEV_THRESHOLD = 10.0f; // Patch亮度標準差的閾值 (0-255之間，可調)

__global__ void ORB_Search_by_projection_project_on_GPU(int n_threads,
                                                        float* Px, float* Py, float* Pz,
                                                        float* Rcw, float* tcw,
                                                        float fx, float fy, float cx, float cy,
                                                        float minX, float maxX, float minY, float maxY,
                                                        float* u, float* v, float* invz,
                                                        unsigned char* is_valid)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        unsigned char isvalid = 0;

        const float Pwx = Px[index];
        const float Pwy = Py[index];
        const float Pwz = Pz[index];

        const float Pcx = Pwx * Rcw[0] + Pwy * Rcw[1] + Pwz * Rcw[2] + tcw[0];
        const float Pcy = Pwx * Rcw[3] + Pwy * Rcw[4] + Pwz * Rcw[5] + tcw[1];
        const float Pcz = Pwx * Rcw[6] + Pwy * Rcw[7] + Pwz * Rcw[8] + tcw[2];

        float im_invz = -1;
        float im_u = -1;
        float im_v = -1;

        // Check positive depth
        if(Pcz > 0.0f)
        {
            // Project in image and check it is not outside
            im_invz = 1.0f / Pcz;
            im_u = fx * Pcx * im_invz + cx;
            im_v = fy * Pcy * im_invz + cy;

            if(!(im_u < minX || im_u > maxX || im_v < minY || im_v > maxY))
                isvalid = 1;

        }

        u[index] = im_u;
        v[index] = im_v;
        invz[index] = im_invz;
        is_valid[index] = isvalid;
    }
}


void ORB_Search_by_projection_project_on_frame(int n_points,
                                               float* Px_gpu, float* Py_gpu, float* Pz_gpu,
                                               float* Rcw_gpu, float* tcw_gpu,
                                               float& fx, float& fy, float& cx, float& cy,
                                               float& minX, float& maxX, float& minY, float& maxY,
                                               float* u_gpu, float* v_gpu, float* invz_gpu,
                                               unsigned char* is_valid_gpu)
{

    int n_threads = n_points;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

    ORB_Search_by_projection_project_on_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                                               n_threads,
                                                                                               Px_gpu, Py_gpu, Pz_gpu,
                                                                                               Rcw_gpu, tcw_gpu,
                                                                                               fx, fy, cx, cy,
                                                                                               minX, maxX, minY, maxY,
                                                                                               u_gpu,  v_gpu, invz_gpu,
                                                                                               is_valid_gpu
                                                                                               );

    cudaStreamSynchronize(0);
}





__global__ void ORB_compute_descriptor_Distance_GPU(int n_threads,
                                                    int* idx_left, int* idx_right,
                                                    unsigned char*  descriptor_left,
                                                    unsigned char*  descriptor_right,
                                                    int* distance)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int* d_left  = (int*)(descriptor_left  + idx_left[index] * 32);
        int* d_right = (int*)(descriptor_right + idx_right[index] * 32);

        int dist=0;

        for(int i=0; i<8; i++)
        {
            unsigned  int v = d_left[i] ^ d_right[i];
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        distance[index] = dist;
    }
}


void ORB_compute_distances(int n_points,
                           int* idx_left, int* idx_right,
                           unsigned char*  descriptor_left,
                           unsigned char*  descriptor_right,
                           int* distance)
{
    int n_threads =  n_points;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

    ORB_compute_descriptor_Distance_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK>>>(
                                                                                           n_threads,
                                                                                           idx_left, idx_right,
                                                                                           descriptor_left,
                                                                                           descriptor_right,
                                                                                           distance
                                                                                           );


    cudaStreamSynchronize(0);

//    std::cout << "line " << __LINE__ << " curror =  " << cudaGetLastError() << "\n";

}


// =================================================================================
// ========================== 新增的二階段匹配GPU程式碼 ==========================
// =================================================================================

// --- 裝置端輔助函式 ---

// 計算漢明距離的裝置函式
__device__ int compute_hamming_distance(const int* pa, const int* pb)
{
    int dist = 0;
    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

// 刪除舊的 get_patch_mean 函式，替換為下面這個新的函式
__device__ void get_patch_stats(const unsigned char* image, int img_width, int img_height,
                                float u, float v, float& mean, float& std_dev)
{

    // 在原有的基礎上，對右側和底側邊界再減去 1，以處理 roundf() 函式在 .5 處的進位問題。
    // 這確保了即使 u = 750.5，也會被正確地攔截。
    if (u < PATCH_HALF_SIZE || u >= (img_width - 1) - PATCH_HALF_SIZE ||
        v < PATCH_HALF_SIZE || v >= (img_height - 1) - PATCH_HALF_SIZE)
    {
        mean = 0.0f;
        std_dev = 0.0f;
        return;
    }

    float sum = 0.0f;
    float sq_sum = 0.0f;
    const int n_pixels = PATCH_SIZE * PATCH_SIZE;

    int base_u = roundf(u) - PATCH_HALF_SIZE;
    int base_v = roundf(v) - PATCH_HALF_SIZE;

    for (int r = 0; r < PATCH_SIZE; ++r) {
        for (int c = 0; c < PATCH_SIZE; ++c) {
            int y = base_v + r;
            int x = base_u + c;
            if (x >= 0 && x < img_width && y >= 0 && y < img_height) {
                float pixel_val = (float)image[y * img_width + x];
                sum += pixel_val;
                sq_sum += pixel_val * pixel_val;
            }
        }
    }
    
    mean = sum / n_pixels;
    // 計算標準差: sqrt( E[X^2] - (E[X])^2 )
    float variance = (sq_sum / n_pixels) - (mean * mean);
    // 避免負數開根號
    std_dev = (variance > 0.0f) ? sqrtf(variance) : 0.0f;
}

// 計算兩個Patch的L2距離
__device__ float compute_l2_distance(const float* patch1, const float* patch2, 
                                     float mean1, float std1, float mean2, float std2)
{
    float l2_dist = 0.0f;
    float s1 = (std1 > 1e-6f) ? (1.0f / std1) : 1.0f;
    float s2 = (std2 > 1e-6f) ? (1.0f / std2) : 1.0f;

    for (int i = 0; i < PATCH_SIZE * PATCH_SIZE; ++i) {
        float p1_norm = (patch1[i] - mean1) * s1;
        float p2_norm = (patch2[i] - mean2) * s2;
        float diff = p1_norm - p2_norm;
        l2_dist += diff * diff;
    }
    return sqrtf(l2_dist);
}

// v6 新增: 一個只計算Patch平均亮度的輔助函式，更高效
__device__ float get_patch_mean(const unsigned char* image, int img_width, int img_height, float u, float v)
{
    float mean = 0.0f;
    int base_u = roundf(u) - PATCH_HALF_SIZE;
    int base_v = roundf(v) - PATCH_HALF_SIZE;

    for (int r = 0; r < PATCH_SIZE; ++r) {
        for (int c = 0; c < PATCH_SIZE; ++c) {
            int y = base_v + r;
            int x = base_u + c;
            if (x >= 0 && x < img_width && y >= 0 && y < img_height) {
                mean += (float)image[y * img_width + x];
            }
        }
    }
    return mean / (float)(PATCH_SIZE * PATCH_SIZE);
}

__global__ void advanced_match_kernel(int n_query_points,
                                             const unsigned char* d_query_descriptors,
                                             const float* d_query_keypoints,
                                             const unsigned char* d_query_image,
                                             int query_img_width, int query_img_height,
                                             const unsigned char* d_candidate_descriptors,
                                             const float* d_candidate_keypoints,
                                             const unsigned char* d_candidate_image,
                                             int cand_img_width, int cand_img_height,
                                             const int2* d_candidate_indices,
                                             MatchResult* d_results,
                                             MatchStats* d_stats)
{
    const int query_idx = blockIdx.x;
    if (query_idx >= n_query_points) return;

    const int2 candidate_range = d_candidate_indices[query_idx];
    const int start_idx = candidate_range.x;
    const int num_candidates = candidate_range.y;

    if (num_candidates == 0) {
        if (threadIdx.x == 0) d_results[query_idx] = {-1, NO_MATCH};
        return;
    }
    
    // --- 共享記憶體，用於在區塊內進行歸約 (Reduction) ---
    __shared__ int best_dist_shared;
    __shared__ int best_idx_shared;

    if (threadIdx.x == 0) {
        best_dist_shared = 257; // 初始化為最大不可能值
        best_idx_shared = -1;
    }
    __syncthreads();

    // ==================================================================
    // 階段一：尋找最佳漢明距離匹配點
    // ==================================================================
    const int* query_desc = (const int*)(d_query_descriptors + query_idx * 32);
    
    for (int i = threadIdx.x; i < num_candidates; i += blockDim.x) {
        const int cand_idx = start_idx + i;
        const int* cand_desc = (const int*)(d_candidate_descriptors + cand_idx * 32);
        atomicMin(&best_dist_shared, compute_hamming_distance(query_desc, cand_desc));
    }
    __syncthreads();

    // 只有當找到了可能的匹配時才繼續
    if (best_dist_shared > STRICT_HAMMING_THRESHOLD) {
         if (threadIdx.x == 0) {
            d_results[query_idx] = {-1, NO_MATCH};
            atomicAdd(&d_stats->low_confidence_failure, 1);
        }
        return;
    }
    
    // 找到擁有最小距離的那個候選點的索引
    if (best_dist_shared < 257) {
        for (int i = threadIdx.x; i < num_candidates; i += blockDim.x) {
            if (best_idx_shared != -1) // 一旦找到，其他執行緒就不用再找了
                break;

            const int cand_idx = start_idx + i;
            const int* cand_desc = (const int*)(d_candidate_descriptors + cand_idx * 32);
            if (compute_hamming_distance(query_desc, cand_desc) == best_dist_shared) {
                atomicExch(&best_idx_shared, cand_idx);
            }
        }
    }
    __syncthreads();

    // ==================================================================
    // 階段二：決策
    // ==================================================================
    
    // 只有一個執行緒負責做最終的決策和寫入
    if (threadIdx.x == 0) {
        // --- 高可信度分支 ---
        if (best_dist_shared < STRICT_HAMMING_THRESHOLD) {
            d_results[query_idx] = {best_idx_shared, HIGH_CONFIDENCE};
            atomicAdd(&d_stats->high_confidence_success, 1);
            return;
        }

        // --- 低可信度分支：執行二次驗證 ---
        
        // 1. 計算查詢點的 Patch 統計數據
        float query_u = d_query_keypoints[query_idx * 2];
        float query_v = d_query_keypoints[query_idx * 2 + 1];
        float query_mean = 0.0f, query_std_dev = 0.0f;
        bool is_query_patch_valid = false;

        if (!(query_u < PATCH_HALF_SIZE || query_u >= (float)(query_img_width - 1) - PATCH_HALF_SIZE ||
              query_v < PATCH_HALF_SIZE || query_v >= (float)(query_img_height - 1) - PATCH_HALF_SIZE))
        {
            is_query_patch_valid = true;
            float q_sum = 0.0f, q_sq_sum = 0.0f;
            int q_base_u = roundf(query_u) - PATCH_HALF_SIZE;
            int q_base_v = roundf(query_v) - PATCH_HALF_SIZE;
            for (int r = 0; r < PATCH_SIZE; ++r) {
                for (int c = 0; c < PATCH_SIZE; ++c) {
                    float pixel_val = (float)d_query_image[(q_base_v + r) * query_img_width + (q_base_u + c)];
                    q_sum += pixel_val;
                    q_sq_sum += pixel_val * pixel_val;
                }
            }
            query_mean = q_sum / (PATCH_SIZE * PATCH_SIZE);
            float q_variance = q_sq_sum / (PATCH_SIZE * PATCH_SIZE) - query_mean * query_mean;
            query_std_dev = (q_variance > 0.0f) ? sqrtf(q_variance) : 0.0f;
        }

        // 2. 檢查最佳候選點的 Patch
        if (is_query_patch_valid && best_idx_shared != -1) {
            float cand_u = d_candidate_keypoints[best_idx_shared * 2];
            float cand_v = d_candidate_keypoints[best_idx_shared * 2 + 1];
            
            if (!(cand_u < PATCH_HALF_SIZE || cand_u >= (float)(cand_img_width - 1) - PATCH_HALF_SIZE ||
                  cand_v < PATCH_HALF_SIZE || cand_v >= (float)(cand_img_height - 1) - PATCH_HALF_SIZE))
            {
                float cand_mean = 0.0f, cand_std_dev = 0.0f;
                float c_sum = 0.0f, c_sq_sum = 0.0f;
                int c_base_u = roundf(cand_u) - PATCH_HALF_SIZE;
                int c_base_v = roundf(cand_v) - PATCH_HALF_SIZE;
                for (int r = 0; r < PATCH_SIZE; ++r) {
                    for (int c = 0; c < PATCH_SIZE; ++c) {
                        float pixel_val = (float)d_candidate_image[(c_base_v + r) * cand_img_width + (c_base_u + c)];
                        c_sum += pixel_val;
                        c_sq_sum += pixel_val * pixel_val;
                    }
                }
                cand_mean = c_sum / (PATCH_SIZE * PATCH_SIZE);
                float c_variance = c_sq_sum / (PATCH_SIZE * PATCH_SIZE) - cand_mean * cand_mean;
                cand_std_dev = (c_variance > 0.0f) ? sqrtf(c_variance) : 0.0f;

                // 3. 最終比較
                if (fabsf(cand_mean - query_mean) < BRIGHTNESS_DIFF_THRESHOLD &&
                    fabsf(cand_std_dev - query_std_dev) < BRIGHTNESS_STD_DEV_THRESHOLD)
                {
                    d_results[query_idx] = {best_idx_shared, LOW_CONFIDENCE_SUCCESS};
                    atomicAdd(&d_stats->low_confidence_success, 1);
                    return;
                }
            }
        }
        
        // 如果二次驗證失敗或無法進行
        d_results[query_idx] = {-1, NO_MATCH};
        atomicAdd(&d_stats->low_confidence_failure, 1);
    }
}



// --- 新增的C++封裝函式 ---
void find_best_match_with_patch_verification(int n_query_points,
                                             const unsigned char* d_query_descriptors,
                                             const float* d_query_keypoints,
                                             const unsigned char* d_query_image,
                                             const cv::Size& query_image_dims,
                                             int n_candidate_points,
                                             const unsigned char* d_candidate_descriptors,
                                             const float* d_candidate_keypoints,
                                             const unsigned char* d_candidate_image,
                                             const cv::Size& candidate_image_dims,
                                             const int2* d_candidate_indices,
                                             MatchResult* d_best_matches, // <--- 修改點 1: 參數類型
                                             MatchStats* d_stats) // <-- v12 修改參數
{
    // 每個查詢點使用一個執行緒塊
    dim3 blocks(n_query_points);
    // 執行緒塊的大小可以根據硬體調整，但通常建議是Warp大小的倍數
    // 此處設為128，可以根據候選點數量動態調整，但為簡化，先固定
    dim3 threads(128); 

    advanced_match_kernel<<<blocks, threads>>>(
        n_query_points,
        d_query_descriptors,
        d_query_keypoints,
        d_query_image,
        query_image_dims.width,
        query_image_dims.height,
        d_candidate_descriptors,
        d_candidate_keypoints,
        d_candidate_image,
        candidate_image_dims.width,
        candidate_image_dims.height,
        d_candidate_indices,
        d_best_matches,
        d_stats // <-- v12 修改參數
    );
    

    // 等待核心函式執行完成
    cudaDeviceSynchronize(); 
}

}

