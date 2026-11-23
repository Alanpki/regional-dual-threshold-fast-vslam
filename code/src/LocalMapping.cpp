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

/*
修改內容:
- CreateNewMapPoints()
    重投影誤差卡方檢定閾值從 5.991 改為 7.815
    放寬尺度一致性ratiofactor 1.5f -> 2.0f
*/

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "DebugUtils.h"
#include "Ablation.h"
#include "PerformanceLogger.h"

#include<mutex>
#include <algorithm>
#include<unistd.h>

namespace Jetson_SLAM
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
    //////////TEST//////////
    for (int i = 0; i < 20; i++) {
        mNeighborRankCounts[i] = 0;
    }
    mKeyFramesCreated = 0;
    mKeyFramesCulled = 0;
}

void LocalMapping::PrintFinalReport() const
{
    {
        std::streamsize original_precision = std::cout.precision();

        long total_points = 0;
        for (const auto& pair : mNeighborRankCounts) {
            total_points += pair.second;
        }

        std::cout << std::endl;
        std::cout << "==============================================================" << std::endl;
        std::cout << "          Neighbor KeyFrame Contribution Final Report" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << " " << std::setw(6) << std::left << "Rank"
                << "| " << std::setw(23) << std::left << "Total MapPoints Created"
                << "| " << "Percentage" << std::endl;
        std::cout << "-------|------------------------|-------------" << std::endl;
        
        if (total_points == 0) {
            std::cout << " No points were created via triangulation." << std::endl;
        } else {
            for (const auto& pair : mNeighborRankCounts) {
                double percentage = (static_cast<double>(pair.second) / total_points) * 100.0;
                
                std::cout << " " << std::setw(6) << std::left << pair.first + 1
                        << "| " << std::setw(23) << std::left << pair.second
                        << "| " << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
            }
        }
        
        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << " Total points created via triangulation: " << total_points << std::endl;
        std::cout << "==============================================================" << std::endl;

        std::cout.precision(original_precision);
        std::cout.unsetf(std::ios_base::floatfield); // Reset float formatting
    }

    {
        std::cout << std::endl;
        std::cout << "======================================================" << std::endl;
        std::cout << "             KeyFrame Dynamics Final Report" << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << " KeyFrames Created (by Local Mapper): " << mKeyFramesCreated << std::endl;
        std::cout << " KeyFrames Culled (Redundancy)      : " << mKeyFramesCulled << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << " Final KeyFrames in Map             : " << mpMap->KeyFramesInMap() << std::endl;
        std::cout << "======================================================" << std::endl;
    }
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            if (mpCurrentKeyFrame != nullptr) {
                // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "Before ProcessNewKeyFrame");
            }
            ProcessNewKeyFrame();
            if (mpCurrentKeyFrame != nullptr) {
                // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After ProcessNewKeyFrame");
            }
            
            // Check recent MapPoints
            if (mpCurrentKeyFrame != nullptr) {
                // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "Before MapPointCulling");
            }
            MapPointCulling();
            if (mpCurrentKeyFrame != nullptr) {
                // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After MapPointCulling");
            }

            // Triangulate new MapPoints
            if (mpCurrentKeyFrame != nullptr) {
                // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "Before CreateNewMapPoints");
            }
            CreateNewMapPoints();
            if (mpCurrentKeyFrame != nullptr) {
                // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After CreateNewMapPoints");
            }
            // if (mpCurrentKeyFrame != nullptr) {
            //     // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After CreateNewMapPoints");
            // }
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                if (mpCurrentKeyFrame != nullptr) {
                    // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "Before SearchInNeighbors");
                }
                SearchInNeighbors();
                if (mpCurrentKeyFrame != nullptr) {
                    // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After SearchInNeighbors");
                }
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2) {
                    if (mpCurrentKeyFrame != nullptr) {
                        // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "Before LocalBundleAdjustment");
                    }
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);
                    if (mpCurrentKeyFrame != nullptr) {
                        // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After LocalBundleAdjustment");
                    }
                }
                // Check redundant local Keyframes
                if (mpCurrentKeyFrame != nullptr) {
                    // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "Before KeyFrameCulling");
                }
                KeyFrameCulling();
                if (mpCurrentKeyFrame != nullptr) {
                    // PerformanceLogger::logTegrastats("LocalMapping", "KeyFrameID", mpCurrentKeyFrame->mnId, "After KeyFrameCulling");
                }
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);

    //////////TEST//////////
    mKeyFramesCreated++;
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn = 20;
    
    // vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    
    // --- 階段一：獲取決策所需的兩個因子 ---

    // 因子1：追蹤是否虛弱
    bool is_weak = false;
    if(mpCurrentKeyFrame->GetParent())
    {
        const int nCurrentInliers = mpCurrentKeyFrame->mnTrackingInliers;
        const int nParentInliers = mpCurrentKeyFrame->GetParent()->mnTrackingInliers;
        const int delta = nCurrentInliers - nParentInliers;

        if (nCurrentInliers < 40 || delta < -100)
        {
            is_weak = true;
        }
    }

    // 因子2：運動類型是什麼 (從 KeyFrame 中直接讀取)
    MotionType motion_type = mpCurrentKeyFrame->mMotionType;

    // ★★★ 制定分級應對策略 ★★★
    // 只要「追蹤虛弱」且「主因是Yaw旋轉」，就觸發應急策略
    if (AblationOn(3) && is_weak && motion_type == ROTATION_DOMINATED)  // [ABLATION]
    {
        nn = 5; // 縮小鄰居範圍以求生
        std::cout << "[Adaptive nn] Weak tracking during YAW rotation detected. Setting nn = " << nn << std::endl;
    }

    // --- 階段三：使用修正後的、穩健的「混合填充」策略來構建鄰居列表 ---

    vector<KeyFrame*> vpNeighKFs;
    vpNeighKFs.reserve(nn);
    set<KeyFrame*> spAlreadyAdded;

    // 2a. 強制加入時序上最近的3個鄰居 (最高優先級) ******** KeyFrame的GetParent是按共視數量串接的不是時序
    const int nTemporalNeighbors = 3;

    vector<KeyFrame*> vpCovisibilityCandidates = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(20);
    // 對這個「共視候選池」按時間(mnId)進行降序排序
    sort(vpCovisibilityCandidates.begin(), vpCovisibilityCandidates.end(), [](KeyFrame* a, KeyFrame* b) {
        return a->mnId > b->mnId;
    });
    // 從排序後的列表中，選取時序最近的3個作為我們的「核心鄰居」
    for(KeyFrame* pKFi : vpCovisibilityCandidates)
    {
        if(vpNeighKFs.size() >= nTemporalNeighbors)
            break;
            
        if(!pKFi->isBad()) {
            vpNeighKFs.push_back(pKFi);
            spAlreadyAdded.insert(pKFi);
        }
    }

    // 2b. 如果名額未滿，按照共視分數，依次補充不重複的鄰居，直到滿額為止
    if(vpNeighKFs.size() < nn)
    {
        // 為了能補滿，我們獲取一個數量足夠多的共視鄰居列表 (例如40個)
        const int nCovisibilityNeighbors = nn + nTemporalNeighbors; 
        vector<KeyFrame*> vpCovisibilityNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nCovisibilityNeighbors);

        for(const auto& pKFCov : vpCovisibilityNeighKFs)
        {
            // 在迴圈的頂部檢查列表是否已滿
            if(vpNeighKFs.size() >= nn) {
                break;
            }
            
            // 使用 .count() 方法檢查元素是否存在，效率更高
            if(spAlreadyAdded.count(pKFCov))
                continue;

            // 如果不重複，則補上
            vpNeighKFs.push_back(pKFCov);
        }
    }

    // [ABLATION] 關閉自適應三角化：鄰近關鍵幀改回原版（共視分數前 nn 個，無時序優先）
    if (!AblationOn(3))
        vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor; // 1.5f

    int nnew=0;

    ////////// Test //////////
    int numTriangulatePoints = 0;
    int numTriangulateGood = 0;
    int numBadParallax = 0;
    vector<int> vSkipNeighKFid;
    int nTotalMatches = 0;
    map<int, int> NeighborMappointCount; // 用來記錄每個鄰居的 MapPoint 數量

    // Search matches with epipolar restriction and triangulate
    size_t nKFindex = 0;
    for(nKFindex=0; nKFindex<vpNeighKFs.size(); nKFindex++) {
        if (nKFindex>0 && CheckNewKeyFrames()) {
            nKFindex-1;
            return;
        }

        KeyFrame* pKF2 = vpNeighKFs[nKFindex];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        /* 這邊會把最近的 NeighborKeyFrame 過濾掉 Baddd 
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            if(ratioBaselineDepth<0.01)
            // if(ratioBaselineDepth<0.01 && pKF2->mnId+1 != mpCurrentKeyFrame->mnId && i > 2)
            {
                vSkipNeighKFid.push_back(pKF2->mnId);
                continue;
            }
        }*/
        // [ABLATION] 關閉自適應三角化：恢復原版基線檢查
        if (!AblationOn(3))
        {
            if(!mbMonocular)
            {
                if(baseline<pKF2->mb)
                    continue;
            }
            else
            {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline/medianDepthKF2;
                if(ratioBaselineDepth<0.01)
                    continue;
            }
        }
        
        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);
        // matcher.SearchForTriangulation_v12(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);
        
        //////////TEST////////
        nTotalMatches += vMatchedIndices.size();
        int points_created_with_this_neighbor = 0;

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            numTriangulatePoints++;
            double degreeMin = 0;
            double degreeMax = 90;
            double degree = std::acos(cosParallaxRays);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            // if (cosParallaxRays>0 && cosParallaxRays<0.9998)
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

                numTriangulateGood++;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else {
                numBadParallax++;
                continue; //No stereo and very low parallax
            }

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                // 對於非立體特徵點，卡方檢定閾值為5.991 (95%信賴區間), 7.815對應97.5%信賴區間
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
                // if((errX1*errX1+errY1*errY1)>7.815*sigmaSquare1)
                //     continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
                // if((errX2*errX2+errY2*errY2)>7.815*sigmaSquare2)
                //     continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
            pMP->source = MapPointSource::KEY_FRAME;

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
            points_created_with_this_neighbor++;
        }
        NeighborMappointCount[pKF2->mnId] = points_created_with_this_neighbor; // 記錄這個 MapPoint 是由哪個鄰居 KeyFrame 創建的
        mNeighborRankCounts[nKFindex] += points_created_with_this_neighbor;
    }


    // // 日誌
    // if (vpNeighKFs.size() > 0) 
    // {
    //     int nFiltered = nTotalMatches - nnew;

    //     std::cout << "--- CreateNewMapPoints for KF " << mpCurrentKeyFrame->mnId << " ---" << std::endl;
    //     std::cout << "  > Neighbors: ";
    //     for (size_t i = 0; i < vpNeighKFs.size(); i++) {
    //         if (i == 0) std::cout << " ";
    //         else std::cout << ", ";
    //         std::cout << vpNeighKFs[i]->mnId;
    //     } std::cout << std::endl;
        
    //     if (!vSkipNeighKFid.empty()) {
    //         std::cout << "  > Neighbors Skipped (short baseline): " << vSkipNeighKFid.size() << " [IDs: ";
    //         for (size_t i = 0; i < vSkipNeighKFid.size(); ++i) {
    //             std::cout << vSkipNeighKFid[i] << (i == vSkipNeighKFid.size() - 1 ? "" : ", ");
    //         }
    //         std::cout << "]" << std::endl;
    //     }
        
    //     std::cout << "  > Total 2D Candidate Pairs Found: " << nTotalMatches << std::endl;
    //     std::cout << "  > Triangulation Details (OK/Good/Bad Parallax): "
    //               << numTriangulatePoints << "/" << numTriangulateGood << "/" << numBadParallax
    //               << std::endl;
    //     std::cout << "  > Final New MapPoints Created: " << nnew << std::endl;
    //     std::cout << "  > Total Pairs Filtered Out: " << nFiltered << std::endl;
    //     std::cout << "-------------------------------------------------" << std::endl;
    // }

    // std::cout << "[MAPPOINT_GENERATION_LOG] KF " << mpCurrentKeyFrame->mnId << " report:" << std::endl;
    
    // // 獲取當前使用的鄰居列表，以便完整輸出
    // const auto& current_neighbors = vpNeighKFs; 
    // for (const auto& pKFN : current_neighbors) {
    //     if (!pKFN) continue;
        
    //     int neighbor_id = pKFN->mnId;
    //     int count = 0;
    //     // 在我們的計數 map 中查找該鄰居的計數結果
    //     if (NeighborMappointCount.count(neighbor_id)) {
    //         count = NeighborMappointCount[neighbor_id];
    //     }

    //     // 輸出格式化日誌，方便複製到論文表格
    //     std::cout << "[MAPPOINT_GENERATION_LOG] Covisible KF ID: " << neighbor_id 
    //               << ", New MapPoints Created: " << count << std::endl;
    // }
    // std::cout << "[MAPPOINT_GENERATION_LOG] Total New MapPoints: " << nnew << std::endl;
    // std::cout << "[MAPPOINT_GENERATION_LOG] --- End of report ---" << std::endl;
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn = 20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs) {
            pKF->SetBadFlag();
            //////////TEST//////////
            mKeyFramesCulled++;
        }
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
