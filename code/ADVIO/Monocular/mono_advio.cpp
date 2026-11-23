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


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <array>
#include <string>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <System.h>
#include <unistd.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
    vector<string> &vstrImages, vector<double> &vTimeStamps);
void LoadVideo(const string &strPathTimes, vector<double> &vTimeStamps);

void printVector(const vector<double> &vec);

int main(int argc, char **argv)
{
    if (argc != 5) {
        cerr << endl << "Usage: ./mono_advio path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file" << endl;
        return 1;
    }

    // Retrieve paths to timestamps
    vector<string> vstrImages;
    vector<double> vTimestamps;
    // LoadVideo(string(argv[4]), vTimestamps);
    LoadImages(argv[3], argv[4], vstrImages, vTimestamps);

    int nImages = vTimestamps.size();
    if (nImages <= 0) {
        cerr << "ERROR: Failed to load timestamps" << endl;
        return 1;
    }

    // // Open video file
    // cv::VideoCapture video(argv[3]);
    // if (!video.isOpened()) {
    //     cerr << "ERROR: Failed to load video" << endl;
    //     return 1;
    // }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    Jetson_SLAM::System SLAM(argv[1],argv[2],Jetson_SLAM::System::MONOCULAR, !(getenv("USE_VIEWER") && string(getenv("USE_VIEWER")) == "0"));  // [20261115] USE_VIEWER=0 關閉 viewer（批次執行用）

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vector<std::array<int,3>> vFrameStat;  // [20261115]
    vTimesTrack.resize(nImages);
 
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // 調整比例
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    cv::FileNode nodeNewHeight = fSettings["Camera.newHeight"];
    cv::FileNode nodeNewWidth = fSettings["Camera.newWidth"];
    bool needResize = false;
    if (!(nodeNewHeight.empty() && nodeNewWidth.empty())) {
        needResize = true;
    }

    // Main loop
    cv::Mat im;

    for (int ni = 0; ni < nImages; ni++) {
        // // Read image from video file
        // bool success = video.read(im);
        // if (!success) {
        //     cerr << "ERROR: Failed to read image from video" << endl;
        //     return 1;
        // }

        // Read image frome file
        im = cv::imread(vstrImages[ni], cv::IMREAD_UNCHANGED);
        if (im.empty()) {
            cerr << endl << "Failed to load image at: "
                         << vstrImages[ni] << endl;
            return 1;
        }
        
        double tframe = vTimestamps[ni];

        // 調整比例
        if (needResize) {
            cv::resize(im, im, cv::Size(nodeNewWidth.real(), nodeNewHeight.real()));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;
        { auto kps = SLAM.GetTrackedKeyPointsUn(); auto mps = SLAM.GetTrackedMapPoints(); int nmp = 0; for (auto* m : mps) if (m) nmp++;
          vFrameStat.push_back({(int)kps.size(), nmp, SLAM.GetTrackingState()}); }  // [20261115] 每幀特徵點數／地圖點數／狀態（計時之後）

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T) {
            usleep((T-ttrack)*1e6);
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    { const char* ft = getenv("FRAME_TIMES"); if (ft) { ofstream fo(ft); for (int k=0; k<nImages; k++) fo << setprecision(9) << vTimestamps[k] << " " << vTimesTrack[k] << " " << (k<(int)vFrameStat.size()?vFrameStat[k][0]:-1) << " " << (k<(int)vFrameStat.size()?vFrameStat[k][1]:-1) << " " << (k<(int)vFrameStat.size()?vFrameStat[k][2]:-1) << "\n"; } }  // [20261115]
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }

    // Input time
    float totalInputTime = 0;
    for (int i = 0; i < vTimestamps.size()-1; i++) {
        totalInputTime += (vTimestamps[i+1] - vTimestamps[i]);
    }

    // Variance
    double variance = 0.0;
    double mean = totaltime / nImages;
    for (double t : vTimesTrack) {
        variance += (t - mean) * (t - mean);
    }
    variance /= nImages;
    double stdDev = sqrt(variance);

    // printVector(vTimestamps);
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << " seconds" << endl;
    cout << "mean tracking time: " << totaltime/nImages << " seconds" << endl;
    cout << "standard deviation: " << stdDev << " seconds" << endl;
    cout << "Input FPS: " << 1.0/(totalInputTime/(vTimestamps.size()-1)) << endl;
    cout << "Output FPS: " << nImages/totaltime << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
    vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    cout << "Loading image... ";

    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);

    while(!fTimes.eof()) {
        string s;
        getline(fTimes,s);

        if(!s.empty()) {
            stringstream ss(s);
            string item;
            vector<string> items;

            while (getline(ss, item, ',')) {
                item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());  // 去除空格
                items.push_back(item);
            }

            // String to Double
            double timestamp = stod(items[0]);
            vTimeStamps.push_back(timestamp);
            vstrImages.push_back(strImagePath + "/" + items[1] + ".png");
        }
    }
}

void LoadVideo(const string &strPathTimes, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);

    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);

        if(!s.empty())
        {
            stringstream ss(s);
            string item;
            vector<string> items;
            
            // (timestamp, id)
            while (getline(ss, item, ',')) {
                items.push_back(item);
            }

            // String to Double
            double timestamp = stod(items[0]);
            vTimeStamps.push_back(timestamp); // 秒
        }
    }
}

void printVector(const vector<double> &vec)
{
    cout << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        cout << vec[i];
        if (i < vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}