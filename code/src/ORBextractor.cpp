/**
* This file is part of  Jetson-SLAM.
*/


#include "Ablation.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"

//#include<pcl/cnsole/time.h>

#include<ctime>

#include<chrono>

using namespace cv;
using namespace std;

namespace Jetson_SLAM
{

ORBExtractor::ORBExtractor(int im_height, int im_width,
                           float scale_factor, int n_levels,
                           int FAST_N_MIN,
                           int FAST_N_MAX,
                           int th_FAST_MIN,
                           int th_FAST_MAX,
                           string str_mask,
                           int tile_h, int tile_w,
                           bool fixed_multi_scale_tile_size,
                           bool apply_nms_ms, bool nms_ms_mode_gpu,
                           bool use_gpu,
                           int nfeatures_cpu)
    : use_gpu_(use_gpu), orb_gpu_(NULL), orb_cpu_(NULL)          
{
    n_levels_ = n_levels;
    scale_factor_ = scale_factor;
    use_gpu_ = use_gpu;


    scale_.resize(n_levels_);
    inv_scale_.resize(n_levels_);

    level_sigma2_.resize(n_levels_);
    inv_level_sigma2_.resize(n_levels_);

    scale_[0] = 1.0f;
    level_sigma2_[0] = 1.0f;

    for(int i=1; i<n_levels_; i++)
    {
        scale_[i] = scale_[i-1] * scale_factor_;
        level_sigma2_[i] = scale_[i] * scale_[i];
    }

    for(int i=0; i<n_levels_; i++)
    {
        inv_scale_[i] = 1.0f / scale_[i];
        inv_level_sigma2_[i] = 1.0f / level_sigma2_[i];
    }

    height_.resize(n_levels_);
    width_.resize(n_levels_);

    for(int i=0;i<n_levels_;i++)
    {
        height_[i] = im_height * inv_scale_[i];
        width_[i] = im_width * inv_scale_[i];
    }


    orb_gpu_ = NULL;
    orb_cpu_ = NULL;
    
    if(use_gpu)
    {
        orb_gpu_ = new ORB_GPU(im_height, im_width,
                                n_levels_, scale_factor_,
                                FAST_N_MIN, FAST_N_MAX,
                                th_FAST_MIN, th_FAST_MAX,
                                tile_h, tile_w,
                                fixed_multi_scale_tile_size,
                                apply_nms_ms, nms_ms_mode_gpu,
                                str_mask,
                                0, // device_id
                                nfeatures_cpu);
    }
    else
    {
        orb_cpu_ = new ORB_SLAM2::ORBextractorCPU(nfeatures_cpu, scale_factor, n_levels,
                                                  th_FAST_MAX, th_FAST_MIN);
    }

}

// [Test Extractor 用] 從設定檔初始化的新建構函式
ORBExtractor::ORBExtractor(cv::FileStorage& fSettings, bool use_gpu)
    : use_gpu_(use_gpu), orb_gpu_(nullptr), orb_cpu_(nullptr)
{
    // 從設定檔讀取所有需要的參數
    int nfeatures = fSettings["ORBextractor.nFeatures"];
    scale_factor_ = fSettings["ORBextractor.scaleFactor"];
    n_levels_ = fSettings["ORBextractor.nLevels"];
    int th_FAST_MAX = fSettings["ORBextractor.th_FAST_MAX"];
    int th_FAST_MIN = fSettings["ORBextractor.th_FAST_MIN"];
    if (!AblationOn(0)) th_FAST_MIN = th_FAST_MAX;  // [ABLATION]
    int tile_h = fSettings["ORBextractor.tile_h"];
    int tile_w = fSettings["ORBextractor.tile_w"];
    
    int fixed_multi_scale_tile_size, apply_nms_ms, nms_ms_mode_gpu;
    fSettings["ORBextractor.fixed_multi_scale_tile_size"] >> fixed_multi_scale_tile_size;
    fSettings["ORBextractor.apply_nms_ms"] >> apply_nms_ms;
    fSettings["ORBextractor.nms_ms_mode_gpu"] >> nms_ms_mode_gpu;
    
    int FAST_N_MIN, FAST_N_MAX;
    fSettings["ORBextractor.FAST_N_MIN"] >> FAST_N_MIN;
    fSettings["ORBextractor.FAST_N_MAX"] >> FAST_N_MAX;
    
    string str_mask_left;
    fSettings["mask.left"] >> str_mask_left;

    int im_width = fSettings["Camera.width"];
    int im_height = fSettings["Camera.height"];

    // 根據 use_gpu 旗標，建立對應的內部物件
    if(use_gpu_)
    {
        orb_gpu_ = new orb_cuda::ORB_GPU(im_height, im_width, n_levels_, scale_factor_,
                                         FAST_N_MIN, FAST_N_MAX, th_FAST_MIN, th_FAST_MAX,
                                         tile_h, tile_w, fixed_multi_scale_tile_size,
                                         apply_nms_ms, nms_ms_mode_gpu, str_mask_left,
                                         0, nfeatures);
    }
    else
    {
        orb_cpu_ = new ORB_SLAM2::ORBextractorCPU(nfeatures, scale_factor_, n_levels_,
                                                  th_FAST_MAX, th_FAST_MIN);
    }
}

ORBExtractor::~ORBExtractor()
{
    if(orb_gpu_) delete orb_gpu_;
    if(orb_cpu_) delete orb_cpu_;
}

void ORBExtractor::extract(const cv::Mat& image,
                           SyncedMem<int>& keypoints,
                           SyncedMem<unsigned char>& keypoints_desc)
{
    if(use_gpu_) 
    {
        orb_gpu_->extract(image,
                        keypoints,
                        keypoints_desc);
    }
    else
    {
        // 1. 準備 ORB-SLAM2 CPU extractor 需要的容器
        std::vector<cv::KeyPoint> cpu_keypoints;
        cv::Mat cpu_descriptors;

        // 2. 呼叫 CPU extractor
        // ORB-SLAM2 的 extractor 是以 operator() 的形式呼叫的
        (*orb_cpu_)(image, cv::Mat(), cpu_keypoints, cpu_descriptors);

        // 3. 將結果從 std::vector<cv::KeyPoint> 和 cv::Mat 轉換到 SyncedMem
        const int n_kps = cpu_keypoints.size();

        // 調整輸出 SyncedMem 的大小
        // Jetson-SLAM 的 keypoints 格式為 [x, y, score, angle, level, size]，共 6 個欄位
        keypoints.resize(n_kps * 6);
        keypoints_desc.resize(n_kps * 32);

        // 獲取 CPU 資料指標
        int* kp_data = keypoints.cpu_data();
        unsigned char* desc_data = keypoints_desc.cpu_data();

        const int out_x_offset  = 0 * n_kps;
        const int out_y_offset  = 1 * n_kps;
        const int out_s_offset  = 2 * n_kps; // score
        const int out_a_offset  = 3 * n_kps; // angle
        const int out_l_offset  = 4 * n_kps; // level (octave)
        const int out_sz_offset = 5 * n_kps; // size

        // [FIX] 建立指向各個 float 類型資料區塊的指標
        // SyncedMem<int> 的記憶體可以被重新解釋 (reinterpreted) 為 float 指標
        float* kp_data_float_score = (float*)(kp_data + out_s_offset);
        float* kp_data_float_angle = (float*)(kp_data + out_a_offset);
        float* kp_data_float_size  = (float*)(kp_data + out_sz_offset);

        for(int i = 0; i < n_kps; ++i)
        {
            const auto& kp = cpu_keypoints[i];

            // --- 寫入整數 (int) 型態的資料 ---
            kp_data[out_x_offset + i] = cvRound(kp.pt.x);
            kp_data[out_y_offset + i] = cvRound(kp.pt.y);
            kp_data[out_l_offset + i] = kp.octave;

            kp_data_float_score[i] = kp.response;
            kp_data_float_angle[i] = kp.angle;
            kp_data_float_size[i]  = kp.size;
        }

        if(n_kps > 0)
        {
            memcpy(desc_data, cpu_descriptors.data, n_kps * 32);
        }

        keypoints.to_gpu();
        keypoints_desc.to_gpu();
    }
}


// [Test Extractor 用] operator() 的實作
void ORBExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                              std::vector<cv::KeyPoint>& _keypoints,
                              cv::OutputArray _descriptors)
{
    if (use_gpu_ && orb_gpu_)
    {
        // --- GPU 路徑 ---
        orb_cuda::SyncedMem<int> keypoints_synced;
        orb_cuda::SyncedMem<unsigned char> descriptors_synced;

        // 1. 呼叫底層的 GPU 提取函式
        orb_gpu_->extract(_image.getMat(), keypoints_synced, descriptors_synced);

        // 2. 將 GPU 的 SyncedMem 結果轉換為標準的 CPU 資料結構
        keypoints_synced.to_cpu();
        descriptors_synced.to_cpu();
        
        const int n_kps = keypoints_synced.size() / 6;
        _keypoints.resize(n_kps);
        
        const int* kp_data = keypoints_synced.cpu_data();
        const unsigned char* desc_data = descriptors_synced.cpu_data();

        // 根據 Jetson-SLAM 的 6通道 格式解析資料
        const float* kp_data_float_score = (const float*)(kp_data + 2 * n_kps);
        const float* kp_data_float_angle = (const float*)(kp_data + 3 * n_kps);
        const float* kp_data_float_size = (const float*)(kp_data + 5 * n_kps);

        for(int i = 0; i < n_kps; ++i) {
            _keypoints[i].pt.x = kp_data[0 * n_kps + i];
            _keypoints[i].pt.y = kp_data[1 * n_kps + i];
            _keypoints[i].response = kp_data_float_score[i];
            _keypoints[i].angle = kp_data_float_angle[i];
            _keypoints[i].octave = kp_data[4 * n_kps + i];
            _keypoints[i].size = kp_data_float_size[i];
        }

        if (n_kps > 0) {
            cv::Mat desc_mat(n_kps, 32, CV_8U, (void*)desc_data);
            desc_mat.copyTo(_descriptors);
        } else {
            _descriptors.assign(cv::Mat());
        }

    }
    else if (orb_cpu_)
    {
        // --- CPU 路徑 ---
        // 直接呼叫原始 ORB-SLAM2 的 CPU 提取器
        (*orb_cpu_)(_image, _mask, _keypoints, _descriptors);
    }
}

const std::vector<orb_cuda::SyncedMem<unsigned char>>& ORBExtractor::GetGpuImagePyramid() const
{
    if (use_gpu_ && orb_gpu_) {
        return orb_gpu_->GetImagePyramid();
    }
    static const std::vector<orb_cuda::SyncedMem<unsigned char>> empty_vector;
    // std::cerr << "Warning: GetGpuImagePyramid() called in non-GPU mode." << std::endl;
    return empty_vector;
}

int ORBExtractor::GetGpuLevelWidth(int level) const
{
    if (use_gpu_ && orb_gpu_ && level < orb_gpu_->width_.size()) {
        return orb_gpu_->width_[level];
    }
    return 0;
}

int ORBExtractor::GetGpuLevelHeight(int level) const
{
     if (use_gpu_ && orb_gpu_ && level < orb_gpu_->height_.size()) {
        return orb_gpu_->height_[level];
    }
    return 0;
}

void ORBExtractor::PrintTimings() const
{
    if (use_gpu_ && orb_gpu_){
        orb_gpu_->PrintTimings();
    }
}


} //namespace ORB_SLAM
