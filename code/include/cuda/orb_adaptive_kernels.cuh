#pragma once

#include <cuda_runtime.h>

namespace orb_cuda {

// 輔助函式：在指定的閾值 t 下，判斷一個點是否為 FAST 角點
__device__ inline bool is_corner_at_threshold(
    const unsigned char* center_ptr,
    int center_value,
    const int* p, // 16個鄰域點的偏移量
    const int* lookup_table,
    int t)
{
    int vt = center_value + t;
    int v_t = center_value - t;
    int bright_idx = 0, dark_idx = 0;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int val = center_ptr[p[i]];
        if (val > vt) bright_idx |= (1 << i);
        if (val < v_t) dark_idx |= (1 << i);
    }

    return (lookup_table[bright_idx] || lookup_table[dark_idx]);
}

// 啟動核心來檢查 Block 狀態
void LaunchCheckAndSetBlockStatus(
    const int* score_high_nms, 
    int* block_status, 
    int score_pitch,
    int height,
    int width,
    cudaStream_t stream);

// 啟動條件式 FAST 分數計算核心
void LaunchConditionalFASTScore(
    int imheight, int imwidth,
    const unsigned char* image_data, int image_pitch,
    const unsigned char* mask_data, int mask_pitch,
    const int* lookup_table,
    int* score_data, int score_pitch,
    const int* block_status,
    int threshold_low,
    cudaStream_t stream);

// 啟動合併結果的核心
void LaunchMergeResults(
    const int* score_high_nms, 
    const int* score_low_nms, 
    int* final_score_nms,
    int pitch,
    int height,
    int width,
    cudaStream_t stream);

} // namespace orb_cuda