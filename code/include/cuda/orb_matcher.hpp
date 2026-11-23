#ifndef __ORB_MATCHER_GPU_HPP__
#define __ORB_MATCHER_GPU_HPP__


#include<vector>
// 引入OpenCV類型，以便傳遞影像尺寸
#include <opencv2/core/core.hpp>

namespace orb_cuda{

// v_final: 新增匹配結果的類型定義
enum MatchType {
    NO_MATCH = -1,
    HIGH_CONFIDENCE = 1,        // 漢明距離 < 閾值
    LOW_CONFIDENCE_SUCCESS = 2  // 漢明距離 >= 閾值，但通過亮度驗證
};

// v_final: 新增的匹配結果結構體
struct MatchResult {
    int match_index; // 在 packed candidate map 中的索引
    MatchType type;
};

// v12: 新增用於統計的結構體
struct MatchStats {
    unsigned int high_confidence_success;
    unsigned int low_confidence_success;
    unsigned int low_confidence_failure;
};

void ORB_Search_by_projection_project_on_frame(int n_points,
                                             float* Px_gpu, float* Py_gpu, float* Pz_gpu,
                                             float* Rcw_gpu, float* tcw_gpu,
                                             float& fx, float& fy, float& cx, float& cy,
                                             float &minX, float &maxX, float &minY, float &maxY,
                                             float* u_gpu, float* v_gpu, float* invz_gpu,
                                             unsigned char* is_valid_gpu);

void ORB_compute_distances(int n_points,
                           int* idx_left, int* idx_right,
                           unsigned char*  descriptor_left,
                           unsigned char*  descriptor_right,
                           int* distance);

/**
 * @brief 使用二階段方法在GPU上搜尋最佳匹配
 * * 階段一：計算漢明距離，篩選出Top-N候選點。
 * 階段二：對Top-N候選點提取Patch，進行光照標準化後計算L2距離。
 * 最終根據漢明距離和Patch L2距離的加權和，選出最佳匹配。
 *
 * @param n_query_points            查詢特徵點的數量
 * @param d_query_descriptors       查詢特徵點的描述子 (在GPU上)
 * @param d_query_keypoints         查詢特徵點的KeyPoint資訊 (u,v) (在GPU上)
 * @param d_query_image             查詢幀的灰階影像 (在GPU上)
 * @param query_image_dims          查詢幀的影像尺寸 (width, height)
 * @param n_candidate_points        候選特徵點的總數
 * @param d_candidate_descriptors   候選特徵點的描述子 (在GPU上)
 * @param d_candidate_keypoints     候選特徵點的KeyPoint資訊 (u,v) (在GPU上)
 * @param d_candidate_image         候選幀的灰階影像 (在GPU上)
 * @param candidate_image_dims      候選幀的影像尺寸 (width, height)
 * @param d_candidate_indices       記錄每個查詢點對應的候選點起始索引和數量 (在GPU上)
 * @param d_best_matches            輸出的最佳匹配結果 (每個查詢點對應的最佳候選點索引) (在GPU上)
 */
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
                                             MatchResult* d_best_matches, // <--- 修改點
                                             MatchStats* d_stats); // <-- v12 修改參數

}

#endif
