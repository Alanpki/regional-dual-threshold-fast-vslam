/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* This file is part of  Jetson-SLAM and is taken from ORB-SLAM2 repo. See above.
*/



#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "DebugUtils.h"
#include "Ablation.h"
#include "ORBextractor.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>
#include <fstream>

#include <mutex>

#include <ctime>

#include <cuda/tracking_gpu.hpp>

#include <tictoc.hpp>

#include <unistd.h>
#include <opencv2/imgproc/types_c.h>

using namespace std;

namespace Jetson_SLAM
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file


    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    int im_width = fSettings["Camera.width"];
    int im_height = fSettings["Camera.height"];

    // 調整比例
    cv::FileNode nodeNewHeight = fSettings["Camera.newHeight"];
    cv::FileNode nodeNewWidth = fSettings["Camera.newWidth"];
    if (!(nodeNewHeight.empty() && nodeNewWidth.empty())) {
        int newWidth = nodeNewWidth.real();
        float scaleColFactor = (float)newWidth / (float)im_width;
        fx *= scaleColFactor;
        cx *= scaleColFactor;
        im_width = newWidth;
    
        int newHeight = nodeNewHeight.real();
        float scaleRowFactor = (float)newHeight / (float)im_height;
        fy *= scaleRowFactor;
        cy *= scaleRowFactor;
        im_height = newHeight;
    }

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    float scale_factor = fSettings["ORBextractor.scaleFactor"];
    int n_levels = fSettings["ORBextractor.nLevels"];
    int th_FAST_MAX = fSettings["ORBextractor.th_FAST_MAX"];
    int th_FAST_MIN = fSettings["ORBextractor.th_FAST_MIN"];
    if (!AblationOn(0)) th_FAST_MIN = th_FAST_MAX;  // [ABLATION] 關閉自適應FAST：單一門檻

    fSettings["gpu.use_gpu"] >> use_gpu_;

    std::string str_mask_left;
    std::string str_mask_right;

    fSettings["mask.left"] >> str_mask_left;
    fSettings["mask.right"] >> str_mask_right;

    cout << "use_gpu = " << use_gpu_ << endl;

    int tile_h = fSettings["ORBextractor.tile_h"];
    int tile_w = fSettings["ORBextractor.tile_w"];
    int nfeatures = fSettings["ORBextractor.nFeatures"];

    //    bool fixed_multi_scale_tile_size = fSettings["ORBextractor.fixed_multi_scale_tile_size"];
    //    bool apply_nms_ms = fSettings["ORBextractor.apply_nms_ms"];
    //    bool nms_ms_mode_gpu = fSettings["ORBextractor.nms_ms_mode_gpu"];

    int fixed_multi_scale_tile_size;
    int apply_nms_ms;
    int nms_ms_mode_gpu;

    fSettings["ORBextractor.fixed_multi_scale_tile_size"] >> fixed_multi_scale_tile_size;
    fSettings["ORBextractor.apply_nms_ms"] >> apply_nms_ms;
    fSettings["ORBextractor.nms_ms_mode_gpu"] >> nms_ms_mode_gpu;

    int FAST_N_MIN;
    int FAST_N_MAX;

    fSettings["ORBextractor.FAST_N_MIN"] >> FAST_N_MIN;
    fSettings["ORBextractor.FAST_N_MAX"] >> FAST_N_MAX;

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Scale Levels: " << n_levels << endl;
    cout << "- Scale Factor: " << scale_factor << endl;
    cout << "- th_FAST_MAX: " << th_FAST_MAX << endl;
    cout << "- th_FAST_MAX: " << th_FAST_MIN << endl;
    cout << "- FAST_N_MIN: " << FAST_N_MIN << endl;
    cout << "- FAST_N_MAX: " << FAST_N_MAX << endl;

    cout << "- Tile h: " << tile_h << endl;
    cout << "- Tile w: " << tile_w << endl;
    cout << "- nFeatures: " << nfeatures << endl;
    cout << "- fixed_multi_scale_tile_size: " << fixed_multi_scale_tile_size << endl;
    cout << "- apply_nms_multicale: " << apply_nms_ms << endl;
    cout << "- nms_multiscale_mode_gpu: " << nms_ms_mode_gpu << endl;

    mpORBextractorLeft = new ORBExtractor(im_height, im_width,
                                          scale_factor, n_levels,
                                          FAST_N_MIN,
                                          FAST_N_MAX,
                                          th_FAST_MIN,
                                          th_FAST_MAX,
                                          str_mask_left,
                                          tile_h, tile_w,
                                          fixed_multi_scale_tile_size,
                                          apply_nms_ms, nms_ms_mode_gpu,
                                          use_gpu_, nfeatures);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBExtractor(im_height, im_width,
                                               scale_factor, n_levels,
                                               FAST_N_MIN,
                                               FAST_N_MAX,
                                               th_FAST_MIN,
                                               th_FAST_MAX,
                                               str_mask_right,
                                               tile_h, tile_w,
                                               fixed_multi_scale_tile_size,
                                               apply_nms_ms, nms_ms_mode_gpu,
                                               use_gpu_, nfeatures);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBExtractor(im_height, im_width,
                                             scale_factor, n_levels,
                                             FAST_N_MIN,
                                             FAST_N_MAX,
                                             th_FAST_MIN,
                                             th_FAST_MAX,
                                             str_mask_left,
                                             tile_h, tile_w,
                                             fixed_multi_scale_tile_size,
                                             apply_nms_ms, nms_ms_mode_gpu,
                                             use_gpu_, nfeatures);


    mpFrameDrawer->tile_h_ = tile_h;
    mpFrameDrawer->tile_w_ = tile_w;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }


    // 初始化計數器
    mTotalInitialMatches = 0;
    mTotalInlierMatches = 0;
}


void Tracking::PrintFinalReport() const 
{
    std::streamsize original_precision = std::cout.precision();

    // 計算總體內點率
    float overallInlierRatio = 0.0f;
    if (mTotalInitialMatches > 0) {
        overallInlierRatio = static_cast<float>(mTotalInlierMatches) / mTotalInitialMatches;
    }

    std::cout << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "       Motion Model Tracking Final Report" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << " Total Initial Matches (Raw matcher output)      : " << mTotalInitialMatches << std::endl;
    std::cout << " Total Inlier Matches (Geometrically consistent) : " << mTotalInlierMatches << std::endl;
    std::cout << " Overall Inlier Ratio:                        " 
              << std::fixed << std::setprecision(4) << overallInlierRatio 
              << " (" << std::setprecision(2) << overallInlierRatio * 100.0f << "%)" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    // 呼叫特徵提取器的報告函式
    if (mpORBextractorLeft) {
        mpORBextractorLeft->PrintTimings();
    }
    
    std::cout.precision(original_precision);
    std::cout.unsetf(std::ios_base::floatfield);
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    //    std::cout << __func__ << " -color conversion time = ";  tt.toc_print();

    {
//        pcl::console::TicToc tt;tt.tic();

        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth, use_gpu_);

        static int avg_feature_count = 0;
        static int avg_feature_sum = 0;
        avg_feature_sum += mCurrentFrame.mvKeys.size();
        avg_feature_count++;

        std::cout << __func__ << " Num features = " << avg_feature_sum / avg_feature_count << "\n";
    }

    {
        //        pcl::console::TicToc tt;tt.tic();
        Track();
        //        std::cout << __func__ << " -track time = ";  tt.toc_print();

    }

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth, use_gpu_);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;


    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth, use_gpu_);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth, use_gpu_);

    Track();
    // usleep(100000);

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    mbOnlyTracking = false;

    // cout << "[Track] " << "Current Frame ID: " << mCurrentFrame.mnId << endl;

    // 若是第一幀，則初始化 SLAM
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed 獲取地圖鎖，避免地圖被其他模組修改
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        // 初始化 SLAM：單目或雙目
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();
        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
//        pcl::console::TicToc tt;           tt.tic();

        // System is initialized. Track Frame. 開始追蹤
        bool bOK; //  記錄 當前幀是否成功追蹤。

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost) 使用運動模型預測當前幀位姿
        if(!mbOnlyTracking)
        {
            //            pcl::console::TicToc tt;           tt.tic();

            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
//                    pcl::console::TicToc tt;
//                    tt.tic();
                    bOK = TrackReferenceKeyFrame();
//                    tt.toc_print();
                }
                else
                {
                    //                    pcl::console::TicToc tt;tt.tic();
                    bOK = TrackWithMotionModel();
                    //                    std::cout << __func__ << " -motion model time = ";tt.toc_print();
                    if (!bOK) {
                        cout << "==================== Fail to track with motion model ====================" << endl;
                    }

                    if(!bOK)
                    {

                        //                        pcl::console::TicToc tt;tt.tic();
                        bOK = TrackReferenceKeyFrame();
                        //                        std::cout << __func__ << " -track Reference keyframe time = ";tt.toc_print();
                        if (!bOK) {
                            cout << "==================== Fail to track reference key frame ====================" << endl;
                        }

                    }
                }
            }
            else
            {
                bOK = Relocalization();
                if (!bOK) {
                    cout << "Fail to relocalization" << endl;
                }
            }

            //            std::cout << __func__ << " -track localization time = ";tt.toc_print();

        }
        else
        {
            //            pcl::console::TicToc tt;           tt.tic();

            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }

            //            std::cout << "only Tracking time = ";tt.toc_print();

        }

        /* 儲存兩frame特徵點比較圖 
        std::string filename = "/root/OutputTest/MatchFrame/"
            + std::to_string(mCurrentFrame.mnId) + "_"
            + std::to_string(mLastFrame.mnId) + "_"
            + "amotion"
            + ".png";
        Jetson_SLAM::DebugMatches(mCurrentFrame, mLastFrame, mImGray, mImGrayLast, filename);*/

        mCurrentFrame.mpReferenceKF = mpReferenceKF;
        {
            //            pcl::console::TicToc tt;tt.tic();

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if(!mbOnlyTracking)
            {
                if (bOK) {
                    bOK = TrackLocalMap();
                }
                if (!bOK) {
                    cout << "Fail to track local map!" << endl;
                    cout << "State = " << mState << endl;
                }
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if(bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            //            std::cout << __func__ << " -tracking local map time = ";tt.toc_print();

        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        {

            // If tracking were good, check if we insert a keyframe
            if(bOK)
            {
                // Update motion model
                if(!mLastFrame.mTcw.empty())
                {
                    cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                    mVelocity = mCurrentFrame.mTcw*LastTwc;
                }
                else
                    mVelocity = cv::Mat();

                // ========== 運動類型分析與記錄 ==========
                if (!mVelocity.empty())
                {
                    const cv::Mat R = mVelocity.rowRange(0, 3).colRange(0, 3);
                    const cv::Mat t = mVelocity.rowRange(0, 3).col(3);
                    const double translation_norm = cv::norm(t, cv::NORM_L2);
                    double pitch_rad, yaw_rad, roll_rad;
                    getEulerAngles(R, pitch_rad, yaw_rad, roll_rad);

                    // ★★★ 將計算出的Yaw值儲存到Frame物件中 ★★★
                    mCurrentFrame.mYaw = yaw_rad;

                    // --- 1. 定義閾值 ---
                    const double YAW_THRESHOLD_RAD   = 3.0 * M_PI / 180.0; // 3度，這是我們的關鍵閾值
                    const double TRANS_THRESHOLD_M   = 0.02;

                    // --- 2. 實作「Yaw優先」的判斷邏輯 ---
                    // 只要 Yaw 運動超過閾值，就標記為旋轉主導，這是最高優先級
                    if (std::abs(yaw_rad) > YAW_THRESHOLD_RAD)
                    {
                        mCurrentFrame.mMotionType = ROTATION_DOMINATED;
                    }
                    // 如果 Yaw 不劇烈，再判斷是否有顯著平移
                    else if (translation_norm > TRANS_THRESHOLD_M)
                    {
                        mCurrentFrame.mMotionType = TRANSLATION;
                    }
                    // 如果 Yaw 和平移都不劇烈，則視為穩定
                    else
                    {
                        mCurrentFrame.mMotionType = STABLE;
                    }

                    // --- 3. (保留) 除錯日誌 ---
                    /*
                    const double yaw_deg = yaw_rad * 180.0 / M_PI;
                    const double pitch_deg = pitch_rad * 180.0 / M_PI;
                    const double roll_deg = roll_rad * 180.0 / M_PI;
                    printf("Motion_Debug: Yaw=%.2f°, Pitch=%.2f°, Roll=%.2f°, Trans=%.4f m\n", 
                        yaw_deg, pitch_deg, roll_deg, translation_norm);
                    const char* motion_str[] = {"STABLE", "TRANSLATION", "ROTATION_DOMINATED", "ROTATION_DOMINATED", "ROTATION_DOMINATED", "ROTATION_DOMINATED"};
                    printf("Motion_Decision: %s\n", motion_str[mCurrentFrame.mMotionType]);
                    */
                }
                else
                {
                    mCurrentFrame.mMotionType = STABLE;
                    mCurrentFrame.mYaw = 0.0;
                }
                // ===============================================
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matchesTrack()
                for(int i=0; i<mCurrentFrame.N; i++)
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if(pMP)
                        if(pMP->Observations()<1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
                {
                    MapPoint* pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                //                pcl::console::TicToc tt;           tt.tic();
                bool need_kf =  NeedNewKeyFrame();
                //                std::cout << __func__ << " - need key frame check time = ";tt.toc_print();

                if(need_kf)
                {
                    //                    tt.tic();
                    CreateNewKeyFrame();
                    //                    std::cout << __func__ << " - create key frame check time = ";tt.toc_print();

                }

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for(int i=0; i<mCurrentFrame.N;i++)
                {
                    if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                }

            }
        }
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        {
            /* 儲存兩frame特徵點比較圖 
            std::string filename = "/root/OutputTest/MatchFrame/"
                + std::to_string(mCurrentFrame.mnId) + "_"
                + std::to_string(mLastFrame.mnId) + "_"
                + "matches"
                + ".png";
            Jetson_SLAM::DebugMatches(mCurrentFrame, mLastFrame, mImGray, mImGrayLast, filename);*/
            
            //            pcl::console::TicToc tt;           tt.tic();
            mLastFrame = Frame(mCurrentFrame);
            //            std::cout << __func__ << " -framecopy time = ";tt.toc_print();

            // 複製前一frame的影像
            mImGrayLast = mImGray.clone();
        }
        //        std::cout << __func__ << " -track inner if else time = ";tt.toc_print();
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>50)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        // KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
        const int nGoodStateInliers = 100; // 定義一個代表良好狀態的數值
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,nGoodStateInliers,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        //        std::cout << "line " << __LINE__ << "\n";

        mpLocalMapper->InsertKeyFrame(pKFini);
        //        std::cout << "line " << __LINE__ << "\n";

        mLastFrame = Frame(mCurrentFrame);
        //        std::cout << "line " << __LINE__ << "\n";
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        //        std::cout << "line " << __LINE__ << "\n";
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        //        std::cout << "line " << __LINE__ << "\n";

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    const int np_min = 100;

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>np_min)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            cout << "Set Reference Frame " << mInitialFrame.mnId << endl;

            return;
        }
        cout << "Skip Frame " << mCurrentFrame.mnId 
             << " not enough keypoints: " << (int)mCurrentFrame.mvKeys.size() << endl;
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=np_min)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            cout << "Current Frame " << mCurrentFrame.mnId 
                 << "not enough keypoitns: " << (int)mCurrentFrame.mvKeys.size() << endl;
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,np_min);

        // Check if there are enough correspondences
        if(nmatches<50)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            cout << "Current Frame " << mCurrentFrame.mnId 
                 << " not enough matches: " << nmatches << endl;
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            cout << "Frame " << mInitialFrame.mnId << "->" << mCurrentFrame.mnId
                 << ", Create Initial Map" << endl;
            CreateInitialMapMonocular();
        }
        else {
            cout << "Frame " << mInitialFrame.mnId << "->" << mCurrentFrame.mnId
                 << ", Fail" << endl;
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    // KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    // KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
    const int nGoodStateInliers = 100;
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, nGoodStateInliers, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, nGoodStateInliers, mpMap, mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);
        pMP->source = MapPointSource::LAST_FRAME;

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15) {
        std::cout << "[TrackReferenceKeyFrame] nmatches=" << nmatches << std::endl;
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    std::cout << "[TrackReferenceKeyFrame] nmatches=" << nmatches << ", nmatchesMap=" << nmatchesMap << std::endl;
    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}


bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode

    //    pcl::console::TicToc ptt;
    // tictoc tt;

    //    ptt.tic();
    // tt.tic();
    UpdateLastFrame();
    //    std::cout <<  __func__ << " Update last frame time = ";tt.toc_print();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th = (mSensor != System::STEREO) ? 15 : 7;

    // v_final: 新增兩個 DMatch 向量來接收結果
    std::vector<cv::DMatch> vHighConfidenceMatches;
    std::vector<cv::DMatch> vLowConfidenceMatches;

    // int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);
    int nmatches = AblationOn(1) ?   // [ABLATION] 關閉兩階段匹配：用原版 SearchByProjection
                   matcher.SearchByProjection_v2(mCurrentFrame, mLastFrame, 
                                                 mImGray, mImGrayLast, 
                                                 th, mSensor==System::MONOCULAR,
                                                 vHighConfidenceMatches, vLowConfidenceMatches)
                 : matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // v_final: 呼叫新的除錯繪圖函式
    // std::string filename = "/root/OutputTest/MotionMatchFrame/"
    //         + std::to_string(mCurrentFrame.mnId) + "_"
    //         + std::to_string(mLastFrame.mnId) + "_"
    //         + "motion_model_matches"
    //         + ".png";
    // Jetson_SLAM::DebugMotionModelMatches(mLastFrame, mCurrentFrame, mImGrayLast, mImGray,
    //                                      vHighConfidenceMatches, vLowConfidenceMatches, filename);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        //        pcl::console::TicToc tt;tt.tic();
//        tictoc tt; tt.tic();
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        // nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
        nmatches = AblationOn(1) ?   // [ABLATION]
                   matcher.SearchByProjection_v2(mCurrentFrame, mLastFrame, 
                                                 mImGray, mImGrayLast, 
                                                 2*th, mSensor==System::MONOCULAR,
                                                 vHighConfidenceMatches, vLowConfidenceMatches)
                 : matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    
        // // v_final: 再次繪圖
        // std::string filename_w = "/root/OutputTest/MotionMatchFrame/"
        //     + std::to_string(mCurrentFrame.mnId) + "_"
        //     + std::to_string(mLastFrame.mnId) + "_"
        //     + "motion_model_matches_wide"
        //     + ".png";
        // Jetson_SLAM::DebugMotionModelMatches(mLastFrame, mCurrentFrame, mImGrayLast, mImGray,
        //                                      vHighConfidenceMatches, vLowConfidenceMatches, filename_w);

        //        std::cout <<  __func__ << " search by projection less matches time = "; tt.toc_print();
    }

   // std::cout <<  __func__ << " Tracking time = "; tt.toc_print();
    

    mTotalInitialMatches += nmatches;
    if(nmatches<20) // 把配對最低設1 原本20
        return false;
        
    {
        //        pcl::console::TicToc tt;tt.tic();
        // tictoc tt; tt.tic();
        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);
     //   std::cout << __func__ << " pose optimization time = "; tt.toc_print();
    }
    {
//        pcl::console::TicToc tt;tt.tic();

        // Discard outliers
        int nmatchesMap = 0;
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }


        mTotalInlierMatches += nmatchesMap;
        if(mbOnlyTracking)
        {
            mbVO = nmatchesMap<10;
            return nmatches>20;
        }

        //        std::cout <<  __func__ << " discard outliers time = "; tt.toc_print();


        // 把LastFrame得到的MapPoint全部指定來源Tracking
        for (int i = 0; i < mCurrentFrame.N; i++) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP) continue;
            pMP->source = MapPointSource::LAST_FRAME;
        }
        // cout << "[TrackWithMotionModel] " << nmatchesMap << " matches" << endl;
        
        return nmatchesMap>=10; // 把配對最低改成1 原本是10
    }
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    {
        //            pcl::console::TicToc tt;tt.tic();
        UpdateLocalMap();
        //        std::cout << __func__ << " update local map time = ";tt.toc_print();
    }

    {
        //        pcl::console::TicToc tt;tt.tic();
        SearchLocalPoints();
        //        std::cout << __func__ << "  search local points time = ";tt.toc_print();
    }

    {
//        pcl::console::TicToc tt;tt.tic();

//        tictoc tt; tt.tic();

        // Optimize Pose
        Optimizer::PoseOptimization(&mCurrentFrame);
        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                if(!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if(!mbOnlyTracking)
                    {
                        if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                            mnMatchesInliers++;
                    }
                    else
                        mnMatchesInliers++;
                }
                else if(mSensor==System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

            }
        }

        //        std::cout << __func__ << "  pose optimization time = ";tt.toc_print();

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        // cout << "[TrackLocalMap] " <<  mnMatchesInliers << " matches" << endl;
        
        if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
            return false;

        if(mnMatchesInliers<30) // 30
            return false;
        else
            return true;
    }
}


bool Tracking::NeedNewKeyFrame()
{
    // ================== 緊急關鍵幀插入機制 ==================
    // 當追蹤的內點數低於一個非常危險的閾值時，無論如何都要強制插入關鍵幀
    const int nEmergencyThresh = 50; // 這個值可以設在15-30之間，需要實驗調整

    if (AblationOn(2) && (mnMatchesInliers < nEmergencyThresh || mCurrentFrame.mMotionType == ROTATION_DOMINATED))  // [ABLATION] 關閉關鍵幀決策：不做緊急關鍵幀
    {
        // 向LocalMapping發送中斷當前優化任務的信號（這依然是個好習慣）
        mpLocalMapper->InterruptBA();
        
        // 在終端打印日誌，方便除錯
        std::cout << "[EMERGENCY KF] Tracking is critical "
                  << "(" << mnMatchesInliers << " inliers, " << "Yaw " << (mCurrentFrame.mYaw * 180.0 / M_PI) << " deg), "
                  << "forcing KeyFrame insertion at Frame " << mCurrentFrame.mnId << std::endl;

        // 立刻返回 true，繞過所有其他判斷，強制創建 KeyFrame！
        return true;
    }
    // ==========================================================

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f; // 0.9f

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    /* Test (15 < mnMatchesInliers < nRefMatches*0.9) */
    const bool c2a = (mnMatchesInliers<nRefMatches*thRefRatio);
    const bool c2b = (mnMatchesInliers>15);

    // cout << "[Need New Keyframe] "
    //      << "c1a: " << c1a << ", "
    //      << "c1b: " << c1b << ", "
    //      << "c1c: " << c1c << ", "
    //      << "c2a: " << c2a << ", "
    //      << "c2b: " << c2b << ", "
    //      << "c3: " << bLocalMappingIdle << ", ";
    // if (((c1a||c1b||c1c)&&c2)&&bLocalMappingIdle)
    //     cout << "True, ";
    // else
    //     cout << "False, ";
    // cout << "Frame: " << mCurrentFrame.mnId << endl;
    
    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
    // v35: 傳入當前的追蹤內點數 mnMatchesInliers
    KeyFrame* pKF = new KeyFrame(mCurrentFrame, mnMatchesInliers, mpMap, mpKeyFrameDB);
    cout << "[Track] " << "CurrentFrame->KeyFrame: " 
         << mCurrentFrame.mnId << "->" << pKF->mnId << ", "
         << "Median Depth: " << pKF->ComputeSceneMedianDepth(2) << endl;
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    {
        //        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Do not search map points already matched
        for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(pMP->isBad())
                {
                    *vit = static_cast<MapPoint*>(NULL);
                }
                else
                {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        //        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        //        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        //        std::cout << __func__ << "  search points time = " << ttrack  * 1000.0 << " ms\n";

    }

    {
        int nToMatch=0;
        int kl=0;
        {

            if(!use_gpu_)
            {
                //Project points in frame and check its visibility
                for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
                {
                    MapPoint* pMP = *vit;
                    if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                        continue;
                    if(pMP->isBad())
                        continue;
                    // Project (this fills MapPoint variables for matching)

                    if(mCurrentFrame.isInFrustum(pMP,0.5))
                    {
                        pMP->IncreaseVisible();
                        nToMatch++;
                    }
                }
            }
            else
            {
                //                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

#ifdef STATIC_MEM_IS_IN

                {
                    std::vector<MapPoint*> map_points;

                    // Project points in frame and check its visibility
                    for(int i =0;i<mvpLocalMapPoints.size(); i++)
                    {
                        MapPoint* pMP = mvpLocalMapPoints[i];
                        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                            continue;
                        if(pMP->isBad())
                            continue;
                        // Project (this fills MapPoint variables for matching)

                        map_points.push_back(pMP);
                    }


                    // prepare data for GPU
                    {
                        int n_points = map_points.size();

                        static SyncedMem<float> Px;
                        static SyncedMem<float> Py;
                        static SyncedMem<float> Pz;

                        static SyncedMem<float> Pnx;
                        static SyncedMem<float> Pny;
                        static SyncedMem<float> Pnz;

                        static SyncedMem<float> invz;
                        static SyncedMem<float> u;
                        static SyncedMem<float> v;

                        static SyncedMem<int> predictedlevel;
                        static SyncedMem<float> viewCos;

                        static SyncedMem<float> invariance_maxDistance;
                        static SyncedMem<float> invariance_minDistance;

                        static SyncedMem<float> MaxDistance;

                        static SyncedMem<float> Rcw;
                        static SyncedMem<float> tcw;
                        static SyncedMem<float> Ow;

                        Px.resize(n_points);
                        Py.resize(n_points);
                        Pz.resize(n_points);

                        Pnx.resize(n_points);
                        Pny.resize(n_points);
                        Pnz.resize(n_points);

                        invz.resize(n_points);
                        u.resize(n_points);
                        v.resize(n_points);

                        predictedlevel.resize(n_points);
                        viewCos.resize(n_points);

                        invariance_maxDistance.resize(n_points);
                        invariance_minDistance.resize(n_points);

                        MaxDistance.resize(n_points);

                        Rcw.resize(9);
                        tcw.resize(3);
                        Ow.resize(3);

                        float fx = mCurrentFrame.fx;
                        float fy = mCurrentFrame.fy;
                        float cx = mCurrentFrame.cx;
                        float cy = mCurrentFrame.cy;

                        int minX  = mCurrentFrame.mnMinX;
                        int maxX  = mCurrentFrame.mnMaxX;
                        int minY  = mCurrentFrame.mnMinY;
                        int maxY  = mCurrentFrame.mnMaxY;

                        int nScaleLevels = mCurrentFrame.mnScaleLevels;
                        float logScaleFactor = mCurrentFrame.mfLogScaleFactor;

                        float viewCosAngle = 0.5;

                        float* Rcw_cpu = Rcw.cpu_data();
                        float* tcw_cpu = tcw.cpu_data();
                        float* Ow_cpu = Ow.cpu_data();

                        for(int i=0;i<3;i++)
                            for(int j=0;j<3;j++)
                                Rcw_cpu[i*3+j] = ((float*)(mCurrentFrame.mRcw.data
                                                           + i * mCurrentFrame.mRcw.step[0]
                                                  + mCurrentFrame.mRcw.step[1] * j))[0];

                        for(int i=0;i<3;i++)
                            tcw_cpu[i] = ((float*)(mCurrentFrame.mtcw.data + i * mCurrentFrame.mtcw.step[0]))[0];

                        for(int i=0;i<3;i++)
                            Ow_cpu[i] = ((float*)(mCurrentFrame.mOw.data + i * mCurrentFrame.mOw.step[0]))[0];


                        float* Px_cpu = Px.cpu_data();
                        float* Py_cpu = Py.cpu_data();
                        float* Pz_cpu = Pz.cpu_data();
                        float* Pnx_cpu = Pnx.cpu_data();
                        float* Pny_cpu = Pny.cpu_data();
                        float* Pnz_cpu = Pnz.cpu_data();
                        float* invariance_minDistance_cpu = invariance_minDistance.cpu_data();
                        float* invariance_maxDistance_cpu = invariance_maxDistance.cpu_data();
                        float* MaxDistance_cpu = MaxDistance.cpu_data();

                        for(int i=0;i<n_points;i++)
                        {
                            MapPoint* pMP = map_points[i];

                            pMP->mbTrackInView = false;

                            // 3D in absolute coordinates
                            // Check viewing angle

                            pMP->GetWorldPosNormalExp(Px_cpu[i],
                                                      Py_cpu[i],
                                                      Pz_cpu[i],
                                                      Pnx_cpu[i],
                                                      Pny_cpu[i],
                                                      Pnz_cpu[i]);

                            pMP->GetDistanceInvariances(invariance_minDistance_cpu[i],
                                                        invariance_maxDistance_cpu[i],
                                                        MaxDistance_cpu[i]
                                                        );

                        }

                        Px.to_gpu_async();
                        Py.to_gpu_async();
                        Pz.to_gpu_async();
                        Pnx.to_gpu_async();
                        Pny.to_gpu_async();
                        Pnz.to_gpu_async();
                        MaxDistance.to_gpu_async();
                        invariance_maxDistance.to_gpu_async();
                        invariance_minDistance.to_gpu_async();
                        Rcw.to_gpu_async();
                        tcw.to_gpu_async();
                        Ow.to_gpu_async();

                        Px.sync_stream();
                        Py.sync_stream();
                        Pz.sync_stream();
                        Pnx.sync_stream();
                        Pny.sync_stream();
                        Pnz.sync_stream();
                        MaxDistance.sync_stream();
                        invariance_maxDistance.sync_stream();
                        invariance_minDistance.sync_stream();
                        Rcw.sync_stream();
                        tcw.sync_stream();
                        Ow.sync_stream();

                        static SyncedMem<unsigned char> isinfrustum;
                        isinfrustum.resize(n_points);

                        tracking_cuda::compute_isInFrustum_GPU(n_points,
                                                               Px.gpu_data(), Py.gpu_data(), Pz.gpu_data(),
                                                               Pnx.gpu_data(), Pny.gpu_data(), Pnz.gpu_data(),
                                                               MaxDistance.gpu_data(),
                                                               invariance_maxDistance.gpu_data(),
                                                               invariance_minDistance.gpu_data(),
                                                               Rcw.gpu_data(), tcw.gpu_data(), Ow.gpu_data(),
                                                               fx, fy, cx, cy,
                                                               minX, maxX, minY, maxY,
                                                               nScaleLevels,
                                                               logScaleFactor,
                                                               viewCosAngle,
                                                               invz.gpu_data(), u.gpu_data(), v.gpu_data(),
                                                               predictedlevel.gpu_data(),
                                                               viewCos.gpu_data(),
                                                               isinfrustum.gpu_data()
                                                               );

                        u.to_cpu_async();
                        v.to_cpu_async();
                        invz.to_cpu_async();
                        predictedlevel.to_cpu_async();
                        viewCos.to_cpu_async();
                        isinfrustum.to_cpu_async();

                        u.sync_stream();
                        v.sync_stream();
                        invz.sync_stream();
                        predictedlevel.sync_stream();
                        viewCos.sync_stream();
                        isinfrustum.sync_stream();

                        unsigned char* isinfrustum_cpu = isinfrustum.cpu_data();
                        float* u_cpu = u.cpu_data();
                        float* v_cpu = v.cpu_data();
                        float* invz_cpu = invz.cpu_data();
                        int* predictedlevel_cpu = predictedlevel.cpu_data();
                        float* viewCos_cpu = viewCos.cpu_data();

                        for(int i=0;i<map_points.size();i++)
                        {
                            if(isinfrustum_cpu[i])
                            {
                                MapPoint* pMP = map_points[i];

                                // Data used by the tracking
                                pMP->mbTrackInView = true;
                                pMP->mTrackProjX = u_cpu[i];
                                pMP->mTrackProjXR = u_cpu[i] - mCurrentFrame.mbf*invz_cpu[i];
                                pMP->mTrackProjY = v_cpu[i];
                                pMP->mnTrackScaleLevel= predictedlevel_cpu[i];
                                pMP->mTrackViewCos = viewCos_cpu[i];

                                pMP->IncreaseVisible();
                                nToMatch++;
                            }
                        }
                    }

#else

                    std::vector<MapPoint*> map_points;

                    // Project points in frame and check its visibility
                    for(int i =0;i<mvpLocalMapPoints.size(); i++)
                    {
                        MapPoint* pMP = mvpLocalMapPoints[i];
                        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                            continue;
                        if(pMP->isBad())
                            continue;
                        // Project (this fills MapPoint variables for matching)

                        map_points.push_back(pMP);
                    }


                    // prepare data for GPU
                    {
                        int n_points = map_points.size();

                        std::vector<float> Px(n_points);
                        std::vector<float> Py(n_points);
                        std::vector<float> Pz(n_points);

                        std::vector<float> Pnx(n_points);
                        std::vector<float> Pny(n_points);
                        std::vector<float> Pnz(n_points);

                        std::vector<float> invz(n_points);
                        std::vector<float> u(n_points);
                        std::vector<float> v(n_points);

                        std::vector<int> predictedlevel(n_points);
                        std::vector<float> viewCos(n_points);

                        std::vector<float> invariance_maxDistance(n_points);
                        std::vector<float> invariance_minDistance(n_points);

                        std::vector<float> MaxDistance(n_points);

                        std::vector<float> Rcw(9);
                        std::vector<float> tcw(3);
                        std::vector<float> Ow(3);

                        float fx = mCurrentFrame.fx;
                        float fy = mCurrentFrame.fy;
                        float cx = mCurrentFrame.cx;
                        float cy = mCurrentFrame.cy;

                        int minX  = mCurrentFrame.mnMinX;
                        int maxX  = mCurrentFrame.mnMaxX;
                        int minY  = mCurrentFrame.mnMinY;
                        int maxY  = mCurrentFrame.mnMaxY;

                        int nScaleLevels = mCurrentFrame.mnScaleLevels;
                        float logScaleFactor = mCurrentFrame.mfLogScaleFactor;

                        float viewCosAngle = 0.5;

                        for(int i=0;i<3;i++)
                            for(int j=0;j<3;j++)
                                Rcw[i*3+j] = ((float*)(mCurrentFrame.mRcw.data
                                                    + i * mCurrentFrame.mRcw.step[0]
                                            + mCurrentFrame.mRcw.step[1] * j))[0];

                        for(int i=0;i<3;i++)
                            tcw[i] = ((float*)(mC urrentFrame.mtcw.data + i * mCurrentFrame.mtcw.step[0]))[0];

                        for(int i=0;i<3;i++)
                            Ow[i] = ((float*)(mCurrentFrame.mOw.data + i * mCurrentFrame.mOw.step[0]))[0];

                        for(int i=0;i<n_points;i++)
                        {
                            MapPoint* pMP = map_points[i];

                            pMP->mbTrackInView = false;

                            // 3D in absolute coordinates
                            // Check viewing angle

                            pMP->GetWorldPosNormalExp(Px[i], Py[i], Pz[i],Pnx[i], Pny[i], Pnz[i]);

                            pMP->GetDistanceInvariances(invariance_minDistance[i],
                                                        invariance_maxDistance[i],
                                                        MaxDistance[i]
                                                        );

                        }

                        for(int i=0;i<n_points;i++)
                        {
                            MapPoint* pMP = map_points[i];

                            pMP->GetDistanceInvariances(invariance_minDistance[i],
                                                        invariance_maxDistance[i],
                                                        MaxDistance[i]
                                                        );
                        }

                        std::vector<unsigned char> isinfrustum(map_points.size());
                        tracking_cuda::compute_isInFrustum_GPU(Px, Py, Pz,
                                                            Pnx, Pny, Pnz,
                                                            MaxDistance,
                                                            invariance_maxDistance,
                                                            invariance_minDistance,
                                                            Rcw, tcw, Ow,
                                                            fx, fy, cx, cy,
                                                            minX, maxX, minY, maxY,
                                                            nScaleLevels,
                                                            logScaleFactor,
                                                            viewCosAngle,
                                                            invz, u, v,
                                                            predictedlevel,
                                                            viewCos,
                                                            isinfrustum
                                                            );



                        for(int i=0;i<map_points.size();i++)
                        {
                            if(isinfrustum[i])
                            {
                                MapPoint* pMP = map_points[i];

                                // Data used by the tracking
                                pMP->mbTrackInView = true;
                                pMP->mTrackProjX = u[i];
                                pMP->mTrackProjXR = u[i] - mCurrentFrame.mbf*invz[i];
                                pMP->mTrackProjY = v[i];
                                pMP->mnTrackScaleLevel= predictedlevel[i];
                                pMP->mTrackViewCos = viewCos[i];

                                pMP->IncreaseVisible();
                                nToMatch++;
                            }
                        }

                    //                    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                    //                    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                    //                    std::cout << __func__ << " " << "  filter and frustum check time = " << ttrack  * 1000.0 << " ms\n";

                    }

#endif
                }

                //            std::cout << "n match = " << nToMatch << "\n";
                //            exit(1);

            }


            {
                //            pcl::console::TicToc tt;tt.tic();
                if(nToMatch>0)
                {
                    //                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

                    ORBmatcher matcher(0.8);
                    int th = 1;
                    if(mSensor==System::RGBD)
                        th=3;
                    // If the camera has been relocalised recently, perform a coarser search
                    if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
                        th=5;
                    matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);

                    //                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                    //                double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                    //                std::cout << __func__ << "  project points in frame time = " << ttrack  * 1000.0 << " ms\n";

                }
            }
        }
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

// 從旋轉矩陣計算歐拉角(弧度)
// pitch: 俯仰角(點頭), yaw: 偏航角(搖頭), roll: 翻滾角(歪頭)
void Tracking::getEulerAngles(const cv::Mat &R, double &pitch, double &yaw, double &roll)
{
    assert(R.cols == 3 && R.rows == 3);

    yaw = asin(-R.at<float>(2,0));
    double cos_pitch = cos(pitch);

    if (fabs(cos_pitch) > 1e-6) {
        pitch = atan2(R.at<float>(2,1), R.at<float>(2,2));
        roll  = atan2(R.at<float>(1,0), R.at<float>(0,0));
    } else {
        // Gimbal lock
        pitch = 0;
        roll  = atan2(-R.at<float>(0,1), R.at<float>(1,1));
    }
}

} //namespace ORB_SLAM
