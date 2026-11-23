#include "cuda/orb_gpu.hpp"
#include <cmath> // For roundf

namespace orb_cuda {

/**
 * @brief 將所有層級的 keypoints 座標和大小縮放回 Level 0 的座標系
 * @param kp_data               包含所有 keypoints 資訊的線性記憶體
 * @param total_kps             總 keypoints 數量
 * @param scale_factors         包含各層級縮放因子的陣列
 */
__global__ void ScaleKeypointsToLevel0(
    int* kp_data,
    float* kp_data_float, // 將浮點數部分分開傳遞，避免類型轉換的複雜性
    int total_kps,
    const float* scale_factors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_kps) return;

    // 從 keypoint data 中讀取當前點所在的層級 (octave)
    // 根據 Jetson-SLAM 的格式, octave 位於第 4 個區塊
    const int out_l_offset = 4 * total_kps;
    int level = kp_data[out_l_offset + idx];

    // Level 0 的縮放因子是 1.0，不需要處理
    if (level == 0) {
        return;
    }

    float scale = scale_factors[level];

    // 獲取 x, y 座標的指標 (位於第 0, 1 區塊)
    const int out_x_offset = 0 * total_kps;
    const int out_y_offset = 1 * total_kps;

    // 縮放 x, y 座標
    kp_data[out_x_offset + idx] = roundf(kp_data[out_x_offset + idx] * scale);
    kp_data[out_y_offset + idx] = roundf(kp_data[out_y_offset + idx] * scale);

    // 縮放特徵點大小 (size)
    // 根據 Jetson-SLAM 格式, size 是 float, 位於第 5 個區塊
    const int out_sz_offset = 5 * total_kps;
    float* size_ptr = (float*)(&kp_data[out_sz_offset]);
    size_ptr[idx] *= scale;
}

void ORB_GPU::ScaleKeypoints(int total_kps, SyncedMem<int>& keypoints, cudaStream_t& cuda_stream)
{
    if (total_kps == 0) return;

    SyncedMem<float> scales_gpu;
    scales_gpu.resize(n_levels_);
    memcpy(scales_gpu.cpu_data(), scale_.data(), n_levels_ * sizeof(float));
    scales_gpu.to_gpu();

    dim3 block_dim(256);
    dim3 grid_dim((total_kps + block_dim.x - 1) / block_dim.x);

    ScaleKeypointsToLevel0<<<grid_dim, block_dim, 0, cuda_stream>>>(
        keypoints.gpu_data(),
        (float*)keypoints.gpu_data(),
        total_kps,
        scales_gpu.gpu_data()
    );
}

} // namespace orb_cuda