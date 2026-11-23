#ifndef __CUDA_ORB_GPU_HPP__
#define __CUDA_ORB_GPU_HPP__

#include<vector>
#include<list> // 四叉樹
#include<algorithm> // 四叉樹

#include<opencv2/opencv.hpp>

#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<cuda/synced_mem_holder.hpp>


namespace orb_cuda
{

// [20261116] 延遲優化：GPU 端壓縮 NMS 後的候選點（定義在 src/cuda/orb_compact.cu）
void compact_score_map(const int* score, int pitch_elems,
                       int minX, int minY, int maxX, int maxY,
                       int3* out, int* counter, int capacity,
                       cudaStream_t stream);

// 用於存放單一幀的計時數據
struct FrameTimeStats {
    double total_extraction = 0.0;
    double pyramid_construction = 0.0;
    double feature_detection = 0.0;
    double gpu_to_cpu_transfer = 0.0;
    double quadtree_distribution = 0.0;
    double cpu_to_gpu_transfer = 0.0;
    double orientation_computation = 0.0;
    double descriptor_computation = 0.0;
    double final_packing = 0.0;
};

#define BORDER_SKIP 20
#define CIRCULAR_HALF_PATCH_SIZE 15
#define NMS_WINDOW 3
#define NMS_WINDOW_HALF (NMS_WINDOW - 1 ) / 2


// ############## 從 ORB-SLAM2 ORBextractor 移植 ################
class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};


class ORB_GPU
{
public:

    ORB_GPU(int im_height, int im_width,
            int n_levels, float scale_factor,
            int FAST_N_MIN, int FAST_N_MAX,
            int th_FAST_MIN, int th_FAST_MAX,
            int tile_h, int tile_w,
            bool fixed_multi_scale_tile_size,
            bool apply_nms_ms, bool nms_ms_mode_gpu,
            std::string str_mask,
            int device_id,
            int n_total_features);

    ~ORB_GPU();

    void dummy_kernel_launch_tosetup_context();

    void Compute_pyramid(int height, int width,
                         int op_height, int op_width,
                         float inv_scale,
                         unsigned char* ip_image_data_gpu,
                         unsigned char* op_image_data_gpu,
                         cudaStream_t& cuda_stream);


    void Compute_pyramid(int height, int width,
                         int op_height, int op_width,
                         float inv_scale,
                         unsigned char* ip_image_data_gpu, int ip_pitch,
                         unsigned char* op_image_data_gpu, int op_pitch,
                         cudaStream_t& cuda_stream);

    void FAST_compute_score(int height, int width,
                            unsigned char* image_data_gpu,
                            int *threshold_tab,
                            int* score_data_gpu,
                            cudaStream_t& cuda_stream);

    void FAST_compute_score_lookpup(int height, int width,
                                    unsigned char* image_data_gpu,
                                    int image_pitch,
                                    int threshold_tab,
                                    int *lookup_table_gpu,
                                    int* score_data_gpu,
                                    int score_pitch,
                                    cudaStream_t& cuda_stream);

    void FAST_compute_score_lookpup_mask(int height, int width,
                                         unsigned char* image_data_gpu,
                                         int image_pitch,
                                         unsigned char* mask_data_gpu,
                                         int mask_pitch,
                                         int threshold_tab,
                                         int *lookup_table_gpu,
                                         int* score_data_gpu,
                                         int score_pitch,
                                         cudaStream_t& cuda_stream);

    void FAST_compute_score_lookpup_mask_v2(int height, int width,
                                            unsigned char* image_data_gpu,
                                            int image_pitch,
                                            unsigned char* mask_data_gpu,
                                            int mask_pitch,
                                            int *lookup_table_gpu,
                                            int* score_data_gpu,
                                            int score_pitch,
                                            cudaStream_t& cuda_stream);

    void FAST_apply_NMS_L(int height, int width,
                          int* score_data_gpu,
                          int score_pitch,
                          int *score_data_nms,
                          int score_nms_pitch,
                          cudaStream_t& cuda_stream);

    void FAST_obtain_keypoints(int height, int width,
                               int* score_data_nms_cpu,
                               int& n_keypoints,
                               int max_features,
                               int n_features_per_scale,
                               cv::Mat &mask,
                               SyncedMem<int> *keypoints_x_,
                               SyncedMem<int> *keypoints_y_);

    // 3x3 NMS
    void FAST_apply_NMS_global(int height, int width,
                               int* score_data_gpu,
                               int score_pitch,
                               int* score_data_nms,
                               int score_nms_pitch,
                               cudaStream_t& cuda_stream);

    void FAST_apply_NMS_G_unroll_reduce(int height, int width,
                                        int tile_h, int tile_w, int n_tiles_h, int n_tiles_w,
                                        int warp_tile_h, int warp_tile_w,
                                        int fuse_nms_L_with_nms_G,
                                        int* image_unroll_,
                                        int* score_data_nms_gpu,
                                        int* keypoints_x_,
                                        int* keypoints_y_,
                                        int* keypoints_score,
                                        cudaStream_t& cuda_stream);


    void FAST_apply_NMS_G_reduce_unroll_reduce(int height, int width,
                                               int tile_h, int tile_w,
                                               int n_tiles_h, int n_tiles_w,
                                               int warp_tile_h, int warp_tile_w,
                                               int fuse_nms_L_with_nms_G,
                                               int* image_unroll_,
                                               int* image_unroll_x_,
                                               int* image_unroll_y_,
                                               int* score_data_gpu, int score_pitch,
                                               int* keypoints_x,
                                               int* keypoints_y,
                                               int* keypoints_score,
                                               cudaStream_t& cuda_stream);

    void FAST_compute_orientation(int height, int width,
                                  unsigned char* image_data_gpu, int image_pitch,
                                  int  n_keypoints,
                                  int* keypoints_x_gpu,
                                  int* keypoints_y_gpu,
                                  int *keypoints_score,
                                  float* keypoints_angle_gpu,
                                  int* umax_gpu,
                                  cudaStream_t& cuda_stream);


    void compute_gaussian(int height, int width,
                          unsigned char* image_data_gpu, int image_pitch,
                          unsigned char *image_data_gaussian_gpu, int image_gaussian_pitch,
                          float* gaussian_weights_gpu,
                          cudaStream_t& cuda_stream);

    void compute_gaussian_v2(int height, int width,
                             unsigned char* image_data_gpu,
                             unsigned char* gaussian_unrolling,
                             unsigned char* image_data_gaussian_gpu,
                             float* gaussian_weights_gpu,
                             cudaStream_t& cuda_stream);


    void ORB_compute_descriptor(int height, int width,
                                unsigned char* image_data_gpu, int image_pitch,
                                signed char* pattern_x_gpu,
                                signed char* pattern_y_gpu,
                                int n_keypoints,
                                int* keypoints_x_gpu,
                                int* keypoints_y_gpu,
                                float* keypoints_angle_gpu,
                                unsigned char* keypoints_descriptor_gpu,
                                cudaStream_t& cuda_stream);

    void fill_bitpattern(signed char *pattern_x, signed char *pattern_y);

    void copy_output(const int n_keypoints,
                     const int octave,
                     const int width,
                     const float scale,
                     const int* keypoints_x,
                     const int* keypoints_y,
                     const int* keypoints_score,
                     const float* keypoints_angle,
                     int* keypoints_x_op,
                     int* keypoints_y_op,
                     int* keypoints_response_op,
                     float* keypoints_angle_op,
                     int* keypoints_octave_op,
                     int* keypoints_size_op,
                     cudaStream_t& cuda_stream);

    void extract(const cv::Mat &image,
                 SyncedMem<int> &out_keypoints,
                 SyncedMem<unsigned char> &out_keypoints_desc);




    void apply_NMS_MS_S_L(int height, int width,
                          int n_levels,
                          int n_keypoints,
                          int* keypoints_x,
                          int* keypoints_y,
                          int* keypoints_score,
                          int* keypoints_level,
                          float* keypoints_scale_factor,
                          int* s0_score_gpu,
                          int *nms_s_score,
                          int *nms_s_level,
                          cudaStream_t& cuda_stream);



    void NMS_S_L(int height, int width, int n_keypoints,
                 int *all_keypoints_x_,
                 int *all_keypoints_y_,
                 int *all_keypoints_score_,
                 int *all_keypoints_level_,
                 int* s0_score_gpu,
                 int* nms_s_score,
                 int* nms_s_level,
                 cudaStream_t& cuda_stream);


    void FAST_apply_NMS_MS_cpu(void);

    void FAST_obtain_keypoints();

    void patternCircle(int* pattern_circle, int rowStride, int patternSize);


    void ORB_compute_stereo_match(int ORB_TH_HIGH, int ORB_TH_LOW,
                                  float mb, float mbf,
                                  std::vector<int>& octave_height,
                                  std::vector<int>& octave_width,
                                  std::vector<cv::KeyPoint>& mvKeys,
                                  std::vector<cv::KeyPoint>& mvKeysRight,
                                  std::vector<float>& mvuRight,
                                  std::vector<float>& mvDepth,
                                  unsigned char* keypoint_descriptor_left,
                                  unsigned char* keypoint_descriptor_right,
                                  std::vector<SyncedMem<unsigned char> > &images_left_smem,
                                  std::vector<SyncedMem<unsigned char> > &images_right_smem);
    
    void ScaleKeypoints(int total_kps, SyncedMem<int>& keypoints, cudaStream_t& cuda_stream);

    // 用來取得 GPU 計算出的影像金字塔
    const std::vector<SyncedMem<unsigned char>>& GetImagePyramid() const { return image_; }

    // 計算運行時間
    void PrintTimings() const;


// private:
    // ############## 從ORB-SLAM移植過來的四叉樹分佈函式 ####################
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);
    
    // 與cv::FAST相同
    void AdaptiveFASTAndNMS(int level);

    void FAST_compute_score_single_thresh(int height, int width, const unsigned char* image_data_gpu, int image_pitch,
                                          const unsigned char* mask_data_gpu, int mask_pitch, const int* lookup_table_gpu,
                                          int* score_data_gpu, int score_pitch, int threshold, cudaStream_t& cuda_stream);
    void FAST_compute_score_tile_based_templated(int height, int width, unsigned char* image_data_gpu, int image_pitch, unsigned char* mask_data_gpu, int mask_pitch, int threshold_high, int threshold_low, int* lookup_table_gpu, int* score_data_gpu, int score_pitch, cudaStream_t& cuda_stream);

    void LaunchCheckAndSetBlockStatus(const int* score_high_nms, int* block_status, int score_pitch, int height, int width, cudaStream_t stream);
    void LaunchConditionalFASTScore(int imheight, int imwidth, const unsigned char* image_data, int image_pitch, const unsigned char* mask_data, int mask_pitch, const int* lookup_table, int* score_data, int score_pitch, const int* block_status, int threshold_low, cudaStream_t stream);
    void LaunchMergeResults(const int* score_high_nms, const int* score_low_nms, int* final_score_nms, int pitch, int height, int width, cudaStream_t stream);


    int patternSize_;

    int device_id_;

    std::vector<float> scale_;
    std::vector<float> inv_scale_;

    std::vector<int> height_;
    std::vector<int> width_;

    // ############## 用於特徵分佈的參數 ####################
    int n_features_;
    std::vector<int> n_features_per_level_;

    std::vector<int> n_keypoints_;

    std::vector<SyncedMem<unsigned char> > image_;
    std::vector<SyncedMem<unsigned char> > image_gaussian_;

    std::vector<SyncedMem<int> > score_;
    std::vector<SyncedMem<int> > score_nms_;

    // [20261116] 延遲優化：候選點壓縮用的常駐緩衝區（第一次 extract 時配置）
    std::vector<int3*> compact_gpu_;      // 每層的候選點清單（GPU）
    std::vector<int> compact_cap_;        // 每層容量
    int* compact_count_gpu_ = nullptr;    // 每層的計數器（GPU）
    int* compact_count_host_ = nullptr;   // 計數器（pinned host）
    int3* compact_host_ = nullptr;        // 候選點清單（pinned host，全部層級共用）
    int compact_host_cap_ = 0;

    SyncedMem<int> keypoints_;
    SyncedMem<unsigned char> keypoints_desc_;

    int x_offset_;
    int y_offset_;
    int s_offset_;
    int l_offset_;
    int a_offset_;

    int x_count_;
    int y_count_;
    int s_count_;
    int l_count_;
    int a_count_;

    int max_kp_count_;

    std::vector<int> level_offset_;

    SyncedMem<int> s0_score_;
    SyncedMem<int> nms_s_score_;
    SyncedMem<int> nms_s_level_;

    SyncedMem<int> grid_levels_;
    SyncedMem<float> grid_scale_factor_;

    std::vector<SyncedMem<signed char> > pattern_x_;
    std::vector<SyncedMem<signed char> > pattern_y_;

    std::vector<SyncedMem<int> > threshold_tab_;
    std::vector<SyncedMem<int> > umax_;
    std::vector<SyncedMem<float> > gaussian_weights_;

    std::vector<cudaStream_t> cuda_streams_;


    std::vector<SyncedMem<int> > image_unroll_x_;
    std::vector<SyncedMem<int> > image_unroll_y_;
    std::vector<SyncedMem<int> > image_unroll_;

    std::vector<int> tile_h_;
    std::vector<int> tile_w_;

    std::vector<int> n_tile_h_;
    std::vector<int> n_tile_w_;

    std::vector<int> warp_tile_h_;
    std::vector<int> warp_tile_w_;

    std::vector<SyncedMem<unsigned char>> masks_;

    int n_levels_;
    int th_FAST_MIN_;
    int th_FAST_MAX_;
    int FAST_N_MIN_;
    int FAST_N_MAX_;

    int threshold_;

    std::vector<std::vector<int> > nms_ms_cpu_x_;
    std::vector<std::vector<int> > nms_ms_cpu_y_;
    std::vector<std::vector<int> > nms_ms_cpu_score_;
    std::vector<std::vector<int> > nms_ms_cpu_level_;
    std::vector<std::vector<int> > nms_ms_cpu_idx_;

    bool apply_nms_ms_;
    bool nms_ms_mode_gpu_;

    int fuse_nms_L_with_nms_G_;

    std::vector<SyncedMem<int> > lookup_table_;

    // [新增] 用於儲存中間結果的 GPU 記憶體緩衝區
    std::vector<SyncedMem<int>> score_high_nms_;
    std::vector<SyncedMem<int>> score_low_temp_;
    std::vector<SyncedMem<int>> score_low_nms_;
    std::vector<SyncedMem<int>> block_status_;

    // <<<<< 為不同 Kernel 保存 Block 維度 >>>>>
    int fast_score_block_dim_x_;
    int fast_score_block_dim_y_;

    // CPU 版本的 umax 和 pattern，用於在 CPU 上計算方向和描述子
    std::vector<std::vector<int>> umax_cpu_;
    std::vector<cv::Point> pattern_cpu_;

    // 用於儲存每一幀處理後的計時統計數據
    mutable std::vector<FrameTimeStats> vFrameTimings; 
};


}





#endif
