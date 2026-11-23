#ifndef JETSON_SLAM_TYPES_H
#define JETSON_SLAM_TYPES_H

namespace Jetson_SLAM
{

// 定義運動類型
enum MotionType {
    STABLE,                 // 穩定或低速運動
    TRANSLATION,            // 以平移為主的運動
    ROTATION_DOMINATED      // 偵測到需要特殊處理的旋轉（目前指Yaw）
};

} // namespace Jetson_SLAM

#endif // JETSON_SLAM_TYPES_H