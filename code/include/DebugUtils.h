#include <opencv2/core/core.hpp>
#include <string>
#include "Frame.h"

namespace Jetson_SLAM {

void DrawMatchingPoints(const Frame &lastFrame, const Frame &currentFrame, 
        const cv::Mat &imLast, const cv::Mat &imCurrent, 
        std::vector<cv::DMatch> &matches, cv::Mat &imgMatches);

cv::Mat CreateDepthOverlay(const Frame &frame, 
                            const cv::Mat &imGray);

cv::Mat CreateSourceOverlay(const Frame &frame, 
                            const cv::Mat &imGray);

void CreateFolersRecursively(const std::string &path);

void GetMatchingPoints(const Frame &lastFrame, 
                        const Frame &currentFrame, 
                        std::vector<cv::DMatch> &matches);

void DebugMatches(const Frame &lastFrame, 
                    const Frame &currentFrame,
                    const cv::Mat &imLast,
                    const cv::Mat &imCurrent,
                    const std::string &filename);

void DrawMapPointProjectionMatches(const Frame &currentFrame,
                                    const cv::Mat &imCurrent,
                                    const std::string &savePath);

double AngleToCos(double angleInDegrees);

void DrawExtractorComparison(const cv::Mat& im, 
                                const std::vector<cv::KeyPoint>& kpsGPU,
                                const std::vector<cv::KeyPoint>& kpsCPU,
                                const std::string& savePath);

void DebugMotionModelMatches(const Frame &lastFrame, 
                             const Frame &currentFrame,
                             const cv::Mat &imLast,
                             const cv::Mat &imCurrent,
                             const std::vector<cv::DMatch> &highConfidenceMatches,
                             const std::vector<cv::DMatch> &lowConfidenceMatches,
                             const std::string &filename);

} // namespace Jetson_SLAM