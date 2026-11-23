#include <iostream>
#include <cstring> // for stderror()
#include <sstream>
#include <cmath>

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <libgen.h> // for dirname()

#include "DebugUtils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

namespace Jetson_SLAM
{

void DrawMatchingPoints(const Frame &lastFrame, const Frame &currentFrame, 
                        const cv::Mat &imLast, const cv::Mat &imCurrent, 
                        std::vector<cv::DMatch> &matches, cv::Mat &imgMatches)
{
    cv::drawMatches(imLast, lastFrame.mvKeys,
        imCurrent, currentFrame.mvKeys,
        matches, imgMatches,
        cv::Scalar(0, 255, 0),  // 綠色的連線
        cv::Scalar(255, 0, 0)); // 藍色的特徵點外框
}

cv::Mat CreateDepthOverlay(const Frame &frame, const cv::Mat &imGray)
{
    cv::Mat imgBGR;

    if (imGray.channels() == 1)
        cv::cvtColor(imGray, imgBGR, cv::COLOR_GRAY2BGR);
    else
        imgBGR = imGray.clone();
    // Track Fail = No Maches
    if (frame.mTcw.empty())
        return imgBGR;

    cv::Mat R = frame.mTcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = frame.mTcw.rowRange(0, 3).col(3);

    float minDepth = FLT_MAX, maxDepth = 0;
    // 先找出深度範圍
    for (int i = 0; i < frame.N; i++) {
        if (!frame.mvpMapPoints[i]) continue;
        MapPoint* pMP = frame.mvpMapPoints[i];
        cv::Mat Xw = pMP->GetWorldPos(); // 假設回傳 3×1 向量（單位需一致）
        cv::Mat Xc = R * Xw + t;
        float depth = Xc.at<float>(2);
        if (depth <= 0) continue;
        minDepth = std::min(minDepth, depth);
        maxDepth = std::max(maxDepth, depth);
    }
    // 如果沒有有效深度，給一個預設範圍, 假設 0 最近, 10 最遠
    if(minDepth == FLT_MAX) { minDepth = 0.0f; maxDepth = 3.0f; }

    auto depthToColor = [=](float depth) -> cv::Scalar {
        // 超過 10 用綠色
        if (depth > maxDepth)
            return cv::Scalar(0, 255, 0);
        float norm = depth / maxDepth;
        norm = std::max(0.f, std::min(1.f, norm));
        int red = static_cast<int>(255 * (1 - norm));
        int blue = static_cast<int>(255 * norm);
        return cv::Scalar(blue, 0, red); // BGR格式
    };
    
    // 為每個有 MapPoint 的 KeyPoint 上色
    for (int i = 0; i < frame.N; i++) {
        if (!frame.mvpMapPoints[i])
            continue;
        MapPoint* pMP = frame.mvpMapPoints[i];
        cv::Mat Xw = pMP->GetWorldPos();
        cv::Mat Xc = R * Xw + t;
        float depth = Xc.at<float>(2);
        cv::Scalar color = depthToColor(depth);
        cv::Point2f pt = frame.mvKeys[i].pt;
        cv::circle(imgBGR, pt, 4, color, -1);
    }

    return imgBGR;
}

cv::Mat CreateSourceOverlay(const Frame &frame, const cv::Mat &imGray)
{
    cv::Mat imgBGR;
    if(imGray.channels() == 1)
        cv::cvtColor(imGray, imgBGR, cv::COLOR_GRAY2BGR);
    else
        imgBGR = imGray.clone();

    for (int i = 0; i < frame.N; i++) {
        if (!frame.mvpMapPoints[i]) continue;

        MapPoint* pMP = frame.mvpMapPoints[i];
        cv::Point2f pt = frame.mvKeys[i].pt;
        
        // 根據 MapPoint 來源設定顏色
        cv::Scalar color;
        if (pMP->source == MapPointSource::LAST_FRAME) {
            color = cv::Scalar(0, 255, 255);  // 黃色 (Tracking)
            cv::circle(imgBGR, pt, 4, color, -1);
        } 
        else if (pMP->source == MapPointSource::COVISIBLE_FRAME) {
            color = cv::Scalar(255, 0, 255);    // 藍色 (LocalMapping)
            cv::circle(imgBGR, pt, 4, color, -1);
        } 
        else if (pMP->source == MapPointSource::KEY_FRAME) {
            color = cv::Scalar(255, 0, 0);    // 藍色 (LocalMapping)
            cv::circle(imgBGR, pt, 4, color, -1);
        } 
        else {
            continue;
        }
    }

    return imgBGR;
}

void CreateFolersRecursively(const std::string &path)
{
    std::stringstream ss(path);
    std::string item, current;

    while (std::getline(ss, item, '/')) {
        if (item.empty())
            continue;
        
        current += "/" + item;

        struct stat info;
        if (stat(current.c_str(), &info) != 0) {
            mkdir(current.c_str(), 0777);
        }
    }
}

void GetMatchingPoints(const Frame &lastFrame, const Frame &currentFrame, 
                        std::vector<cv::DMatch> &matches)
{
    for (int i = 0; i < lastFrame.N; i++) {
        MapPoint* pMP = lastFrame.mvpMapPoints[i];
        if (!pMP) continue;

        // 尋找 currentFrame 中是否有同一個 MapPoint
        int idxCur = -1;
        for (int j = 0; j < currentFrame.N; j++) {
            if (currentFrame.mvpMapPoints[j] == pMP) {
                idxCur = j;
                break;
            }
        }

        if (idxCur >= 0) {
            // 將匹配點存入 matches
            matches.push_back(cv::DMatch(i, idxCur, 1.0f));
        }
    }
}

void DebugMatches(const Frame &lastFrame, 
                const Frame &currentFrame,
                const cv::Mat &imLast,
                const cv::Mat &imCurrent,
                const std::string &filename)
{
    // ================================
    // **檢查影像是否有效**
    // ================================
    if (imCurrent.empty()) {
        std::cerr << "[DebugMatches] Error: imCurrent is empty! Skipping debug visualization." << std::endl;
        return;
    }
    if (imLast.empty()) {
        std::cerr << "[DebugMatches] Error: imLast is empty! Skipping debug visualization." << std::endl;
        return;
    }

    /* ================================
    畫 LastFrame 跟 CurrentFrame 的特徵點批配圖
    ================================ */
    std::vector<cv::DMatch> vMatches;
    vMatches.reserve(lastFrame.N);
    GetMatchingPoints(lastFrame, currentFrame, vMatches);
    if (vMatches.empty()) {
        // std::cerr << "[DebugMatches] No matching points found, skipping visualization." << std::endl;
        return;
    }

    cv::Mat imgMatches;
    DrawMatchingPoints(lastFrame, currentFrame, imLast, imCurrent, vMatches, imgMatches);
    
    /* ================================
                畫特徵點的深度
    ================================ */
    cv::Mat imgLastDepth = CreateDepthOverlay(lastFrame, imLast);
    cv::Mat imgCurrentDepth = CreateDepthOverlay(currentFrame, imCurrent);


    // cv::putText(imgLastDepth, "Last", cv::Point(15, 50),
    //             cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(247, 198, 24), 5);
    // cv::putText(imgCurrentDepth, "Current", cv::Point(15, 50),
    //             cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(246, 131, 229), 5);

    cv::Mat imgDepthCombined;
    cv::hconcat(imgLastDepth, imgCurrentDepth, imgDepthCombined);

    /* ================================
            畫出 MapPoint 的來源
    ================================ 
    cv::Mat imgLastSource = CreateSourceOverlay(lastFrame, imLast);
    cv::Mat imgCurrentSource = CreateSourceOverlay(currentFrame, imCurrent);

    if (currentFrame.mnId == mnLastKeyFrameId)
    {
        cv::putText(imgCurrentSource, "KF", cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 255, 0), 5);
    }
    if (lastFrame.mnId == mnLastKeyFrameId)
    {
        cv::putText(imgLastSource, "KF", cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 255, 0), 5);
    }

    cv::Mat imgSourceCombined;
    cv::hconcat(imgLastSource, imgCurrentSource, imgSourceCombined);*/

    /* ================================
                合併所有圖片
    ================================ */
    cv::Mat composite;
    // if (vMatches.empty()) {
    //     composite = imgMatches;
    // }
    // else {
        cv::vconcat(imgMatches, imgDepthCombined, composite);
        // cv::vconcat(composite, imgSourceCombined, composite);
    // }

    if (!filename.empty()) {
        std::string dirPath = filename.substr(0, filename.find_last_of("/"));

        struct stat info;
        if (stat(dirPath.c_str(), &info) != 0) {
            CreateFolersRecursively(dirPath);
        }
        
        cv::imwrite(filename, composite);
        // std::cout << "[DebugMatches] Image saved to: " << filename << std::endl;
    }
    else {
        cv::imshow("Debug Matches", composite);
        cv::waitKey(0);
    }
}

void DrawMapPointProjectionMatches(const Frame &currentFrame,
    const cv::Mat &imCurrent,
    const std::string &savePath)
{
    if (imCurrent.empty()) {
        std::cerr << "[DrawMapPointProjectionMatches] Error: imCurrent is empty" << std::endl;
        return;
    }

    const cv::Mat Rcw = currentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = currentFrame.mTcw.rowRange(0, 3).col(3);

    // MapPoint 投影點與 KeyPoint 的匹配對
    std::vector<cv::KeyPoint> projectedPoints;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::DMatch> matches;

    for (int i = 0, j = 0; i < currentFrame.N; i++) {

        MapPoint* pMP = currentFrame.mvpMapPoints[i];
        if (pMP && !currentFrame.mvbOutlier[i]) {
            cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Dc = Rcw*x3Dw + tcw;

            const float xc = x3Dc.at<float>(0);
            const float yc = x3Dc.at<float>(1);
            const float invzc = 1.0/x3Dc.at<float>(2);
            if (invzc < 0) continue;

            float u = currentFrame.fx*xc*invzc + currentFrame.cx;
            float v = currentFrame.fy*yc*invzc + currentFrame.cy;

            if (u < currentFrame.mnMinX || u > currentFrame.mnMaxX)
                continue;
            if (v < currentFrame.mnMinY || v > currentFrame.mnMaxY)
                continue;

            projectedPoints.push_back(cv::KeyPoint(u, v, 1.f));
            keypoints.push_back(currentFrame.mvKeys[i]);
            matches.push_back(cv::DMatch(j, j, 0.0f));

            j++;
        }
    }
    if (matches.empty()) {
        std::cout << "[DrawMapPointProjectionMatches] No MapPoints to draw." << std::endl;
        return;
    }
    // 確保 matches 中的索引有效，並且在範圍內
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].queryIdx >= projectedPoints.size() || matches[i].trainIdx >= keypoints.size()) {
            std::cerr << "[DrawMapPointProjectionMatches] Invalid match at index " << i << std::endl;
            return;
        }
    }
    cv::Mat imgMatches;
    cv::drawMatches(imCurrent, projectedPoints, 
                    imCurrent, keypoints, 
                    matches, imgMatches);
                    // cv::Scalar(0, 255, 0),  // 綠色的連線
                    // cv::Scalar(255, 255, 255)); // 藍色的特徵點外框

    if (!savePath.empty()) {
        std::string dirPath = savePath.substr(0, savePath.find_last_of("/"));

        struct stat info;
        if (stat(dirPath.c_str(), &info) != 0) {
            CreateFolersRecursively(dirPath);
        }
        
        cv::imwrite(savePath, imgMatches);
        std::cout << "[DrawMapPointProjectionMatches] Image saved to: " << savePath << std::endl;
    }
    else {
        cv::imshow("MapPoint Projections vs Keypoints", imgMatches);
        cv::waitKey(0);
    }
}

double AngleToCos(double angleInDegrees)
{
    double angleInRadians = angleInDegrees * (M_PI / 180.0);
    return std::cos(angleInRadians);
}

void DrawExtractorComparison(const cv::Mat& im, 
                                 const std::vector<cv::KeyPoint>& kpsGPU,
                                 const std::vector<cv::KeyPoint>& kpsCPU,
                                 const std::string& savePath)
{
    if (im.empty()) {
        std::cerr << "[DrawExtractorComparison] Error: Input image is empty!" << std::endl;
        return;
    }

    // 將灰階影像轉為彩色，方便繪製
    cv::Mat outputImage;
    if (im.channels() == 1) {
        cv::cvtColor(im, outputImage, cv::COLOR_GRAY2BGR);
    } else {
        outputImage = im.clone();
    }

    // 1. 繪製純 CPU 提取的特徵點 (紅色)
    for (const auto& kp : kpsCPU) {
        cv::circle(outputImage, kp.pt, 3, cv::Scalar(0, 0, 255), 1); // BGR，紅色，半徑為3，線寬為1
    }

    // 2. 繪製 GPU+CPU 混合模式提取的特徵點 (綠色)
    for (const auto& kp : kpsGPU) {
        cv::circle(outputImage, kp.pt, 2, cv::Scalar(0, 255, 0), -1); // BGR，綠色，半徑為2，實心
    }

    // 3. 加上圖例
    cv::putText(outputImage, "CPU-only KeyPoints " + to_string(kpsCPU.size()), cv::Point(15, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::putText(outputImage, "GPU+CPU KeyPoints " + to_string(kpsGPU.size()), cv::Point(15, 60), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 4. 儲存圖片
    if (!savePath.empty()) {
        std::string dirPath = savePath.substr(0, savePath.find_last_of("/"));
        CreateFolersRecursively(dirPath); // 使用您現有的函式建立資料夾
        cv::imwrite(savePath, outputImage);
        std::cout << "[DrawExtractorComparison] Comparison image saved to: " << savePath << std::endl;
    }
}

void DebugMotionModelMatches(const Frame &lastFrame, 
                             const Frame &currentFrame,
                             const cv::Mat &imLast,
                             const cv::Mat &imCurrent,
                             const std::vector<cv::DMatch> &highConfidenceMatches,
                             const std::vector<cv::DMatch> &lowConfidenceMatches,
                             const std::string &filename)
{
    if (imLast.empty() || imCurrent.empty()) {
        std::cerr << "[DebugMotionModelMatches] Error: Input image is empty." << std::endl;
        return;
    }
    if (highConfidenceMatches.empty() && lowConfidenceMatches.empty()) {
        return;
    }
    
    cv::Mat outImg;
    // 繪製高可信度匹配（綠色連線，藍色點）
    // 我們只繪製連線，特徵點可以稍後統一繪製以避免重疊
    cv::drawMatches(imLast, lastFrame.mvKeys, imCurrent, currentFrame.mvKeys,
                    highConfidenceMatches, outImg, 
                    cv::Scalar(0, 255, 0),       // high-confidence matchColor (Green)
                    cv::Scalar(255, 0, 0),       // singlePointColor (Blue)
                    std::vector<char>(),
                    cv::DrawMatchesFlags::DEFAULT);

    // 在同一張圖上疊加繪製低可信度匹配的連線（黃色）
    const int imLast_cols = imLast.cols;
    for (const auto& match : lowConfidenceMatches) {
        // 取得匹配點的座標
        cv::Point2f pt1 = lastFrame.mvKeys[match.queryIdx].pt;
        cv::Point2f pt2 = currentFrame.mvKeys[match.trainIdx].pt;
        pt2.x += imLast_cols; // 將 CurrentFrame 中的點的 x 座標平移

        // 繪製黃色連線
        cv::line(outImg, pt1, pt2, cv::Scalar(0, 0, 255), 1); // Red line
        // 繪製低可信度匹配點（紅色圓圈），以區分
        cv::circle(outImg, pt1, 3, cv::Scalar(0, 0, 255), 1);
        cv::circle(outImg, pt2, 3, cv::Scalar(0, 0, 255), 1);
    }

    // 儲存圖片
    if (!filename.empty()) {
        std::string dirPath = filename.substr(0, filename.find_last_of("/"));
        CreateFolersRecursively(dirPath); // 使用您現有的函式建立資料夾
        cv::imwrite(filename, outImg);
    }
}

    
} // namesapce Jetson_SLAM