#include "cuda/orb_gpu.hpp"
#include "cuda/orb_adaptive_kernels.cuh" // [新增]

#include<vector>
#include<list>      // 新增
#include<algorithm> // 新增
#include <numeric>

#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp> // cvRound


#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime_api.h>

#include<chrono>

#include<bitset>
#include <opencv2/imgproc/types_c.h>

using namespace cv;

namespace orb_cuda {

// 計時輔助類別
class Timer {
public:
    void Start() {
        start_time = std::chrono::steady_clock::now();
    }
    // 返回經過的時間（單位：毫秒）
    double Stop() {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count() * 1000.0;
    }
private:
    std::chrono::steady_clock::time_point start_time;
};

// ############## 從 ORBextractor_cpu.cpp 移植 ################

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


// (IC_Angle) 計算單一特徵點的方向
static float IC_Angle(const Mat& image, Point2f pt, const std::vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

// (computeOrientation) 遍歷 keypoints 計算方向
static void computeOrientation(const Mat& image, std::vector<KeyPoint>& keypoints, const std::vector<int>& umax)
{
    for (std::vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}


const float factorPI = (float)(CV_PI/180.f);
// (computeOrbDescriptor) 計算單一描述子
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const cv::Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}

// (computeDescriptors) 遍歷 keypoints 計算描述子
static void computeDescriptors(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors,
                               const std::vector<cv::Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;
}


std::vector<cv::KeyPoint> ORB_GPU::DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes
    const int nIni_raw = round(static_cast<float>(maxX-minX)/(maxY-minY));
    const int nIni = std::max(1, nIni_raw);

    const float hX = static_cast<float>(maxX-minX)/(nIni);

    std::list<ExtractorNode> lNodes;

    std::vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    std::list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    std::vector<std::pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                std::vector<std::pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(N);
    for(std::list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void CalculateAndPrintStats(const std::string& name, const std::vector<double>& timings) {
    if (timings.empty()) return;

    double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    double mean = sum / timings.size();

    double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / timings.size() - mean * mean);

    std::cout << " " << std::left << std::setw(25) << name << ": "
              << std::fixed << std::setprecision(4) << std::right << std::setw(8) << mean << " ms | "
              << std::setw(8) << std_dev << " ms" << std::endl;
}


void ORB_GPU::PrintTimings() const {
    if (vFrameTimings.empty()) {
        std::cout << "沒有記錄到任何特徵提取的計時資訊。" << std::endl;
        return;
    }

    // 將結構中的計時數據解開到獨立的 vector 中，方便處理
    std::vector<double> total, pyramid, detection, gpu2cpu, quadtree, cpu2gpu, orientation, descriptor, packing;
    for (const auto& stats : vFrameTimings) {
        total.push_back(stats.total_extraction);
        pyramid.push_back(stats.pyramid_construction);
        detection.push_back(stats.feature_detection);
        gpu2cpu.push_back(stats.gpu_to_cpu_transfer);
        quadtree.push_back(stats.quadtree_distribution);
        cpu2gpu.push_back(stats.cpu_to_gpu_transfer);
        orientation.push_back(stats.orientation_computation);
        descriptor.push_back(stats.descriptor_computation);
        packing.push_back(stats.final_packing);
    }
    
    std::cout << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "      orb_gpu final report (單位: 毫秒)      " << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << " 共處理了 " << vFrameTimings.size() << " 幀影像。" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << " Step                     |   Avg       |   Std " << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    CalculateAndPrintStats("Image Pyramid (GPU)", pyramid);
    CalculateAndPrintStats("FAST & Score (GPU)", detection);
    CalculateAndPrintStats("GPU -> CPU data transfer", gpu2cpu);
    CalculateAndPrintStats("DistributeOctTree (CPU)", quadtree);
    CalculateAndPrintStats("CPU -> GPU data transfer", cpu2gpu);
    CalculateAndPrintStats("Angle (GPU)", orientation);
    CalculateAndPrintStats("Descriptors (GPU)", descriptor);
    CalculateAndPrintStats("Packing (GPU)", packing);
    std::cout << "------------------------------------------------------" << std::endl;
    CalculateAndPrintStats("Total", total);
    std::cout << "======================================================" << std::endl;
}



/**/
ORB_GPU::ORB_GPU(int im_height, int im_width,
                 int n_levels, float scale_factor,
                 int FAST_N_MIN, int FAST_N_MAX,
                 int th_FAST_MIN, int th_FAST_MAX,
                 int tile_h, int tile_w,
                 bool fixed_multi_scale_tile_size,
                 bool apply_nms_ms, bool nms_ms_mode_gpu,
                 std::string str_mask,
                 int device_id,
                 int n_total_features)
{
    // 設置CUDA設備
    cudaSetDevice(device_id);

    
    // ##################################################################
    // ############## 初始化特徵分佈相關參數 ####################
    n_features_ = n_total_features; 
    
    n_features_per_level_.resize(n_levels);
    float factor = 1.0f / scale_factor;
    float nDesiredFeaturesPerScale = n_features_ * (1 - factor) / (1 - (float)pow((double)factor, (double)n_levels));
    int sumFeatures = 0;
    for (int level = 0; level < n_levels - 1; level++)
    {
        n_features_per_level_[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += n_features_per_level_[level];
        nDesiredFeaturesPerScale *= factor;
    }
    n_features_per_level_[n_levels - 1] = std::max(n_features_ - sumFeatures, 0);
    // ##################################################################

    this->fast_score_block_dim_x_ = tile_w;
    this->fast_score_block_dim_y_ = tile_h;

    float scale = scale_factor;
    n_levels_ = n_levels;

    apply_nms_ms_ = apply_nms_ms && (n_levels_ > 1);
    nms_ms_mode_gpu_ = nms_ms_mode_gpu;

    patternSize_ = 16;

    th_FAST_MIN_ = th_FAST_MIN;
    th_FAST_MAX_ = th_FAST_MAX;
    FAST_N_MIN_ = FAST_N_MIN;
    FAST_N_MAX_ = FAST_N_MAX;

    threshold_ = th_FAST_MAX;

    scale_.push_back(1.0);
    inv_scale_.push_back(1.0);

    height_.push_back(im_height);
    width_.push_back(im_width);

    // 初始化圖像金字塔和層級大小
    for(int i=1;i<n_levels;i++)
    {
        scale_.push_back(scale*scale_[i-1]);
        inv_scale_.push_back(1.0f/scale_[i]);

        // height_.push_back(im_height * inv_scale_[i]);
        // width_.push_back(im_width * inv_scale_[i]);
        // 使用 cvRound 來匹配 CPU 版本的取整策略
        height_.push_back(cvRound(im_height * inv_scale_[i]));
        width_.push_back(cvRound(im_width * inv_scale_[i]));
    }

    // 設置金字塔層的遮罩
    {
        masks_.resize(n_levels);

        std::cout << "讀取遮罩路徑: " << str_mask << std::endl;
        cv::Mat mask = cv::imread(str_mask, cv::IMREAD_GRAYSCALE);

        if(mask.empty())
        {
            std::cerr << "無法讀取遮罩圖，改用預設全白遮罩。" << std::endl;

            // 安全檢查高度與寬度
            if (height_.empty() || width_.empty() || height_[0] <= 0 || width_[0] <= 0)
            {
                std::cerr << "錯誤：圖像尺寸未正確初始化，無法建立遮罩！" << std::endl;
                exit(-1);
            }

            std::cout << "建立遮罩大小：" << width_[0] << " x " << height_[0] << std::endl;
            try {
                mask = cv::Mat(height_[0], width_[0], CV_8UC1, cv::Scalar(255));
            } catch (const std::exception& e) {
                std::cerr << "遮罩建立失敗: " << e.what() << std::endl;
                exit(-1);
            }
        }


        for(int i=0;i<n_levels_;i++)
        {
            cv::Mat resize_mask;
            cv::resize(mask, resize_mask, cv::Size(width_[i], height_[i]),0,0,CV_INTER_NN);
            cv::threshold(resize_mask, resize_mask, 10, 255, CV_THRESH_BINARY);

            masks_[i].resize(width_[i] * height_[i]);
            masks_[i].pitch_ = width_[i];

            memcpy(masks_[i].cpu_data(), resize_mask.data, width_[i] * height_[i]);

            masks_[i].to_gpu();
        }
    }


    {
        image_.resize(n_levels_);
        image_gaussian_.resize(n_levels_);

        score_.resize(n_levels_);
        score_nms_.resize(n_levels_);

        umax_.resize(n_levels_);
        threshold_tab_.resize(n_levels_);

        pattern_x_.resize(n_levels_);
        pattern_y_.resize(n_levels_);

        gaussian_weights_.resize(n_levels_);

        // [新增] 初始化新成員變數
        score_high_nms_.resize(n_levels_);
        score_low_temp_.resize(n_levels_);
        score_low_nms_.resize(n_levels_);
        block_status_.resize(n_levels_);
    }

    n_keypoints_.resize(n_levels);
    // 初始化CUDA流（CUDA Streams）
    cuda_streams_.resize(n_levels);

    for(int i=0;i<n_levels;i++)
    {
        // dont used pitched memory for SLAM
        // because in the stereo-match GPU,
        // currently pitched memory support is not there, only linear memory is there.

        //        image_[i].resize_pitched(width_[i], height_[i]);
        //        image_gaussian_[i].resize_pitched(width_[i], height_[i]);
        //        score_[i].resize_pitched(width_[i], height_[i]);
        //        score_nms_[i].resize_pitched(width_[i], height_[i]);

        image_[i].resize(width_[i] * height_[i]);
        image_[i].pitch_ = width_[i];

        image_gaussian_[i].resize(height_[i] * width_[i]);
        image_gaussian_[i].pitch_ = width_[i];

        score_[i].resize(height_[i] * width_[i]);
        score_[i].pitch_ = width_[i] * 4;

        score_nms_[i].resize(height_[i] * width_[i]);
        score_nms_[i].pitch_ = width_[i] * 4;

        umax_[i].resize(CIRCULAR_HALF_PATCH_SIZE + 1);// CIRCULAR_HALF_PATCH_SIZE = 15
        threshold_tab_[i].resize(512);

        pattern_x_[i].resize(512);
        pattern_y_[i].resize(512);

        gaussian_weights_[i].resize(7*7);

        n_keypoints_[i] = 0;

        cudaStreamCreate(&cuda_streams_[i]);

        // [新增]
        score_high_nms_[i].resize(height_[i] * width_[i]);
        score_high_nms_[i].pitch_ = width_[i] * 4;

        score_low_temp_[i].resize(height_[i] * width_[i]);
        score_low_temp_[i].pitch_ = width_[i] * 4;
        
        score_low_nms_[i].resize(height_[i] * width_[i]);
        score_low_nms_[i].pitch_ = width_[i] * 4;

        // block_status 的大小取決於 ConditionalFASTScore 的 Grid 維度
        int grid_dim_x = (width_[i] + this->fast_score_block_dim_x_ - 1) / this->fast_score_block_dim_x_;
        int grid_dim_y = (height_[i] + this->fast_score_block_dim_y_ - 1) / this->fast_score_block_dim_y_;
        block_status_[i].resize(grid_dim_x * grid_dim_y);
    }

    // 設置閾值表（Threshold Tables）
    // -255~255 -> 0~2
    // 0: 都不是, 1: 亮, 2: 暗
    for(int i=0;i<n_levels;i++)
    {
        int* threshold_tab_cpu_ = threshold_tab_[i].cpu_data();

        for(int j = -255; j <= 255; j++ )
            threshold_tab_cpu_[j+255] = (int)(j < -threshold_ ? 1 : j > threshold_ ? 2 : 0);

        threshold_tab_[i].to_gpu();
    }

    // 描述子用的圓形座標
    {
        /*
        for(int i=0;i<n_levels;i++)
        {
            int* umax_cpu_ = umax_[i].cpu_data();

            int v, v0, vmax = cvFloor(CIRCULAR_HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
            int vmin = cvCeil(CIRCULAR_HALF_PATCH_SIZE * sqrt(2.f) / 2);

            const double hp2 = CIRCULAR_HALF_PATCH_SIZE * CIRCULAR_HALF_PATCH_SIZE;

            for (v = 0; v <= vmax; ++v)
                umax_cpu_[v] = cvRound(sqrt(hp2 - v * v));

            // Make sure we are symmetric
            for (v = CIRCULAR_HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
            {
                while (umax_cpu_[v0] == umax_cpu_[v0 + 1])
                    ++v0;
                umax_cpu_[v] = v0;
                ++v0;
            }
            umax_[i].to_gpu();
        }*/
        // ========= 完整的 umax 初始化區塊 (同時包含 GPU 和 CPU 版本) =========
        // 1. 初始化供 CPU 使用的 umax_cpu_ 成員變數
        umax_cpu_.resize(n_levels_);
        for (int i = 0; i < n_levels_; i++) {
            umax_cpu_[i].resize(HALF_PATCH_SIZE + 1);
            int* umax_cpu_ptr = umax_cpu_[i].data();
            const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
            int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
            int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
            for (v = 0; v <= vmax; ++v) {
                umax_cpu_ptr[v] = cvRound(sqrt(hp2 - v * v));
            }
            for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
                while (umax_cpu_ptr[v0] == umax_cpu_ptr[v0 + 1])
                    ++v0;
                umax_cpu_ptr[v] = v0;
                ++v0;
            }
        }
        // 2. 初始化供 GPU 使用的 umax_ SyncedMem (您原有的邏輯)
        for(int i=0; i<n_levels; i++)
        {
            // 獲取 SyncedMem 的 CPU 緩衝區指標
            int* umax_gpu_buffer = umax_[i].cpu_data();

            // 將剛剛在 umax_cpu_ 中計算好的值，複製到 SyncedMem 的緩衝區中
            memcpy(umax_gpu_buffer, umax_cpu_[i].data(), (HALF_PATCH_SIZE + 1) * sizeof(int));
            
            // 將資料上傳到 GPU
            umax_[i].to_gpu();
        }
    }

    // 初始化隨機抽樣點
    {
        // 為 CPU 描述子計算初始化 pattern_cpu_
        const int npoints = 512;
        const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;
        pattern_cpu_.assign(pattern0, pattern0 + npoints);
        for(int i=0;i<n_levels;i++)
        {
            signed char* pattern_x_cpu_ = pattern_x_[i].cpu_data();
            signed char* pattern_y_cpu_ = pattern_y_[i].cpu_data();

            fill_bitpattern(pattern_x_cpu_, pattern_y_cpu_);

            pattern_x_[i].to_gpu();
            pattern_y_[i].to_gpu();
        }        
    }


    // 設置其他內部參數：
    //     包括每層圖像的高斯權重、角點的最大數量等。
    
    {
        for(int i=0;i<n_levels;i++)
        {
            float* gaussian_weights_cpu_ = gaussian_weights_[i].cpu_data();

            const float sigma = 2; //10
            const float sigma2 = sigma * sigma;

            float sum = 0;
            int count = 0;
            for(int j=-3;j<=3;j++)
                for(int k=-3;k<=3;k++)
                {
                    gaussian_weights_cpu_[count] =  exp(-(j*j+k*k)/(2*sigma2));
                    sum += gaussian_weights_cpu_[count];
                    count++;
                }
            
            // 正規劃 省略除以 2πσ^2
            for(int j=0;j<7*7;j++)
                gaussian_weights_cpu_[j] /= sum;

            gaussian_weights_[i].to_gpu();
        }

    }


    // NMS-G setup
    {
        tile_h_.resize(n_levels_);
        tile_w_.resize(n_levels_);

        n_tile_h_.resize(n_levels_);
        n_tile_w_.resize(n_levels_);

        warp_tile_h_.resize(n_levels_);
        warp_tile_w_.resize(n_levels_);

        tile_h_[0] = tile_h;
        tile_w_[0] = tile_w;

        image_unroll_.resize(n_levels_);
        image_unroll_x_.resize(n_levels_);
        image_unroll_y_.resize(n_levels_);

        for(int i=0;i<n_levels_;i++)
        {

            if(fixed_multi_scale_tile_size)

            {
                tile_h_[i] = tile_h_[0];
                tile_w_[i] = tile_w_[0];
            }
            else
            {
                tile_h_[i] = tile_h_[0] * inv_scale_[i];
                tile_w_[i] = tile_w_[0] * inv_scale_[i];
            }


            n_tile_h_[i] = ((height_[i] - 2*0) - 1) / tile_h_[i] + 1;
            n_tile_w_[i] = ((width_[i] - 2*0) - 1) / tile_w_[i] + 1;

            // 一個 warp 有 32 個 CUDA thread
            const int warp_size = 32;

            //// reduce_unroll_reduce requires less memory .
            //// disabled for experimenation insead allocating for unroll_reduce which covers all memory req.
            //            int tile_count_warped = ((tile_w_[i] - 1) / warp_size + 1) * warp_size; // simplified ceil(tile_count / warp_size);

            // unroll_reduce
            // 一個 cell 需要的 CUDA thread 數量 (warp 數量 * warp 大小)
            int tile_count_warped = ((tile_w_[i] * tile_h_[i] - 1) / warp_size + 1) * warp_size; // simplified ceil(tile_count / warp_size);

            int unroll_count = tile_count_warped * n_tile_h_[i]  * n_tile_w_[i];

            warp_tile_h_[i] = tile_count_warped / warp_size;
            warp_tile_w_[i] = warp_size;

            image_unroll_[i].resize(unroll_count);
            memset(image_unroll_[i].cpu_data(),0, sizeof(int)*unroll_count);
            image_unroll_[i].to_gpu();

            image_unroll_x_[i].resize(unroll_count);
            memset(image_unroll_x_[i].cpu_data(),0, sizeof(int)*unroll_count);
            image_unroll_x_[i].to_gpu();

            image_unroll_y_[i].resize(unroll_count);
            memset(image_unroll_y_[i].cpu_data(),0, sizeof(int)*unroll_count);
            image_unroll_y_[i].to_gpu();

            //            std::cout << tile_h_[i] << " " << tile_w_[i] << "\n";
            //            std::cout << n_tile_h_[i] << " " << n_tile_w_[i] << "\n";
            //            std::cout << warp_tile_h_[i] << " " << warp_tile_w_[i] << "\n";
            //            std::cout << tile_count_warped << " " << unroll_count << "\n";

        }
    }



    if(apply_nms_ms_)
    {
        nms_s_score_.resize(height_[0] * width_[0]);
        nms_s_level_.resize(height_[0] * width_[0]);

        // 每一層的 keypoint score 縮放到最底層
        s0_score_.resize(n_levels_ * height_[0] * width_[0]);
    }

    {
        int count =0;
        for(int i=0;i<n_levels_;i++)
        {
            level_offset_.push_back(count);
            count += n_tile_h_[i] * n_tile_w_[i];
        }

        x_offset_ = 0*count;
        y_offset_ = 1*count;
        s_offset_ = 2*count;
        l_offset_ = 3*count;
        a_offset_ = 4*count;

        max_kp_count_ = count;
        x_count_ = count;
        y_count_ = count;
        s_count_ = count;
        l_count_ = count;
        a_count_ = count;

        keypoints_.resize(x_count_ + y_count_ + s_count_ + l_count_ + a_count_);
        keypoints_desc_.resize(max_kp_count_ * 32);

        nms_ms_cpu_x_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_y_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_score_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_level_.resize(n_tile_h_[0]  * n_tile_w_[0]);
        nms_ms_cpu_idx_.resize(n_tile_h_[0]  * n_tile_w_[0]);
    }


    // setup for NMS-MS GPU
    {
        grid_levels_.resize(max_kp_count_);
        grid_scale_factor_.resize(max_kp_count_);

        int* level_data = grid_levels_.cpu_data();
        float* scale_factor_data = grid_scale_factor_.cpu_data();

        int count = 0;
        for(int i=0;i<n_levels_;i++)
        {
            int n_grids = n_tile_h_[i] * n_tile_w_[i];
            for(int j=0;j<n_grids;j++)
            {
                level_data[count] = i;
                scale_factor_data[count] = scale_[i];
                count++;
            }
        }

        grid_levels_.to_gpu();
        grid_scale_factor_.to_gpu();
    }


    s0_score_.set_zero_gpu();

    fuse_nms_L_with_nms_G_ = true;

    //lookpu tables init
    {
        lookup_table_.resize(n_levels_);

        for(int i=0;i<n_levels_;i++)
        {
            SyncedMem<int>& lut = lookup_table_[i];
            lut.resize(0x0000FFFF);

            int* data = lut.cpu_data();

            for(int j=0;j<0x0000FFFF;j++)
            {
                int num =  j;
                int n_valid = 0;
                int valid_bit = 0x00008000;

                int need_further_check = true;

                for(int k=0;k<16;k++)
                {
                    if(num & valid_bit)
                        n_valid++;
                    else
                    {
                        if(n_valid >= FAST_N_MIN_ && n_valid <= FAST_N_MAX_)
                        {
                            need_further_check = false;
                            break;
                        }
                        else
                            n_valid = 0;
                    }

                    valid_bit >>= 1;
                }

                //                std::cout <<  need_further_check << " " << n_valid << " ";

                if(need_further_check)
                {
                    int num = j;
                    int valid_bit = 0x00008000;

                    for(int k=0;k<16;k++)
                    {
                        if(num & valid_bit)
                            n_valid++;
                        else
                            break;

                        valid_bit >>= 1;
                    }
                }

                //                std::cout <<  n_valid << " " << std::bitset<16>(j);
                if(n_valid >= FAST_N_MIN_ && n_valid <= FAST_N_MAX_)
                {
                    data[j] = 1;
                    //                    std::cout <<  " " << "found 1\n";
                }
                else
                {
                    data[j] = 0;
                    //                    std::cout <<  " " << "found 0\n";
                }
            }

            lut.to_gpu();
        }
    }


    dummy_kernel_launch_tosetup_context();
}


ORB_GPU::~ORB_GPU()
{

    for(int i=0;i<n_levels_;i++)
    {
        cudaStreamDestroy(cuda_streams_[i]);
    }

}




class tictoc
{
public:
    void tic(void)
    {
        t1 = std::chrono::steady_clock::now();
    }

    float toc(void)
    {
        t2 = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() * 1000.0;
        return time_ms;
    }

    void toc_print(void)
    {
        t2 = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() * 1000.0;
        std::cout << "Time elapsed = " << time_ms << " ms\n";
    }

private:
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    float time_ms;
};


// [新增] 新的私有函式，封裝 GPU 處理流程
void ORB_GPU::AdaptiveFASTAndNMS(int level)
{
    cudaStream_t& stream = cuda_streams_[level];

    // === 1. 高門檻處理流程 ===
    // 1.1 計算高門檻原始分數 (輸出到 score_low_temp_ 作為臨時緩衝區)
    // 注意：您需要確保 FAST_compute_score_single_thresh 這個函式已經被正確實作
    FAST_compute_score_single_thresh(height_[level], width_[level],
                                     image_[level].gpu_data(), image_[level].pitch_,
                                     masks_[level].gpu_data(), masks_[level].pitch_,
                                     lookup_table_[level].gpu_data(),
                                     score_low_temp_[level].gpu_data(),
                                     score_low_temp_[level].pitch_ / 4,
                                     this->th_FAST_MAX_, // 只用高門檻
                                     stream);

    // 1.2 對高門檻分數做 NMS (輸出到 score_high_nms_)
    FAST_apply_NMS_global(height_[level], width_[level],
                          score_low_temp_[level].gpu_data(), score_low_temp_[level].pitch_ / 4,
                          score_high_nms_[level].gpu_data(), score_high_nms_[level].pitch_ / 4,
                          stream);

    // 使用計算出的大小來初始化狀態
    cudaMemsetAsync(block_status_[level].gpu_data(), 0, block_status_[level].size() * sizeof(int), stream);
    
    LaunchCheckAndSetBlockStatus(score_high_nms_[level].gpu_data(),
                                 (int*)block_status_[level].gpu_data(), // 確保型別正確
                                 score_high_nms_[level].pitch_ / 4,
                                 height_[level], width_[level],
                                 stream);

    // === 3. 低門檻處理流程 ===
    LaunchConditionalFASTScore(height_[level], width_[level],
                               image_[level].gpu_data(), image_[level].pitch_,
                               masks_[level].gpu_data(), masks_[level].pitch_,
                               lookup_table_[level].gpu_data(),
                               score_low_temp_[level].gpu_data(),
                               score_low_temp_[level].pitch_ / 4,
                               (const int*)block_status_[level].gpu_data(), // 確保型別正確
                               this->th_FAST_MIN_,
                               stream);
    
    FAST_apply_NMS_global(height_[level], width_[level],
                          score_low_temp_[level].gpu_data(), score_low_temp_[level].pitch_ / 4,
                          score_low_nms_[level].gpu_data(), score_low_nms_[level].pitch_ / 4,
                          stream);

    // === 4. 合併結果 ===
    LaunchMergeResults(score_high_nms_[level].gpu_data(),
                       score_low_nms_[level].gpu_data(), // [修正] 使用 score_low_nms_
                       score_nms_[level].gpu_data(),
                       score_nms_[level].pitch_ / 4,
                       height_[level], width_[level],
                       stream);
}

//v7
/*
// ##################################################################
// ############## 新的 extract 函式 (v4FAST->CPU方向->GPU描述子) ####################
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{
    // === Step 0-3: GPU 偵測 + CPU 四叉樹 (與之前版本相同) ===
    cudaMemcpy2D(image_[0].gpu_data(), image_[0].pitch_,
                 image.data, image.step,
                 image.cols * sizeof(unsigned char), image.rows,
                 cudaMemcpyHostToDevice);
    for (int i = 1; i < n_levels_; i++) {
        Compute_pyramid(height_[0], width_[0], height_[i], width_[i], inv_scale_[i],
                        image_[0].gpu_data(), image_[0].pitch_,
                        image_[i].gpu_data(), image_[i].pitch_, cuda_streams_[i]);
    }

    for (int level = 0; level < n_levels_; ++level) {
        AdaptiveFASTAndNMS(level);
    }

    std::vector<std::vector<cv::KeyPoint>> all_keypoints_hybrid(n_levels_);
    int total_keypoints = 0;
    for (int level = 0; level < n_levels_; ++level) {
        cudaStreamSynchronize(cuda_streams_[level]);
        cv::Mat score_map_cpu(height_[level], width_[level], CV_32S);
        cudaMemcpy2D(score_map_cpu.data, score_map_cpu.step,
                     score_nms_[level].gpu_data(), score_nms_[level].pitch_,
                     width_[level] * sizeof(int), height_[level],
                     cudaMemcpyDeviceToHost);

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = EDGE_THRESHOLD;
        const int maxBorderX = width_[level] - EDGE_THRESHOLD;
        const int maxBorderY = height_[level] - EDGE_THRESHOLD;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(n_features_per_level_[level] * 2);

        for (int y = minBorderY; y < maxBorderY; ++y) {
            for (int x = minBorderX; x < maxBorderX; ++x) {
                const int score = score_map_cpu.at<int>(y, x);
                if (score > 0) {
                    vToDistributeKeys.emplace_back(cv::Point2f(x - minBorderX, y - minBorderY), 7, -1, score);
                }
            }
        }
        
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        keypoints_this_level.reserve(n_features_per_level_[level]);
        
        if (!vToDistributeKeys.empty()) {
            keypoints_this_level = DistributeOctTree(vToDistributeKeys, 0, maxBorderX - minBorderX, 0, maxBorderY - minBorderY, n_features_per_level_[level], level);
        }

        for (auto& kp : keypoints_this_level) {
            kp.pt.x += minBorderX;
            kp.pt.y += minBorderY;
            kp.octave = level;
            kp.size = round(PATCH_SIZE * scale_[level]);
        }
        total_keypoints += keypoints_this_level.size();
    }

    if (total_keypoints == 0) {
        out_keypoints.resize(0);
        out_keypoints_desc.resize(0);
        return;
    }

    // === Step 4: 在 CPU 上計算方向 ===
    for (int level = 0; level < n_levels_; ++level) {
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        if(keypoints_this_level.empty()) continue;

        cv::Mat image_level_cpu(height_[level], width_[level], CV_8UC1);
        cudaMemcpy2D(image_level_cpu.data, image_level_cpu.step,
                     image_[level].gpu_data(), image_[level].pitch_,
                     width_[level] * sizeof(unsigned char), height_[level],
                     cudaMemcpyDeviceToHost);
        
        computeOrientation(image_level_cpu, keypoints_this_level, umax_cpu_[level]);
    }
    
    // === Step 5: 上傳資料並在 GPU 計算描述子 ===
    std::vector<int> kp_x_cpu(total_keypoints);
    std::vector<int> kp_y_cpu(total_keypoints);
    std::vector<float> kp_angle_cpu(total_keypoints);
    std::vector<int> kp_level_offset_cpu(n_levels_);
    
    int offset = 0;
    for (int level = 0; level < n_levels_; ++level) {
        kp_level_offset_cpu[level] = offset;
        for (size_t i = 0; i < all_keypoints_hybrid[level].size(); ++i) {
            kp_x_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.x;
            kp_y_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.y;
            kp_angle_cpu[offset + i] = all_keypoints_hybrid[level][i].angle;
        }
        offset += all_keypoints_hybrid[level].size();
    }

    keypoints_.resize(total_keypoints * 5); // x, y, score(dummy), dummy, angle
    cudaMemcpy(keypoints_.gpu_data() + 0*total_keypoints, kp_x_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 1*total_keypoints, kp_y_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((float*)(keypoints_.gpu_data() + 4*total_keypoints), kp_angle_cpu.data(), total_keypoints * sizeof(float), cudaMemcpyHostToDevice);

    for(int i = 0; i < n_levels_; ++i) {
        compute_gaussian(height_[i], width_[i], image_[i].gpu_data(), image_[i].pitch_,
                         image_gaussian_[i].gpu_data(), image_gaussian_[i].pitch_,
                         gaussian_weights_[i].gpu_data(), cuda_streams_[i]);
    }

    for (int level = 0; level < n_levels_; ++level) {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level > 0) {
            int level_offset = kp_level_offset_cpu[level];
            ORB_compute_descriptor(height_[level], width_[level],
                                   image_gaussian_[level].gpu_data(), image_gaussian_[level].pitch_,
                                   pattern_x_[level].gpu_data(), pattern_y_[level].gpu_data(),
                                   n_kps_level,
                                   keypoints_.gpu_data() + 0*total_keypoints + level_offset,
                                   keypoints_.gpu_data() + 1*total_keypoints + level_offset,
                                   (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset,
                                   keypoints_desc_.gpu_data() + (level_offset * 32),
                                   cuda_streams_[level]);
        }
    }
    
    // === [修改] Step 6: 將所有結果下載回 CPU，並在 CPU 上進行最終打包 ===
    
    // 6a. 從 GPU 下載計算好的描述子
    // 確保所有描述子計算都已完成
    for(int i = 0; i < n_levels_; ++i) cudaStreamSynchronize(cuda_streams_[i]);
    // 準備 CPU 端的描述子儲存空間
    cv::Mat all_descriptors_cpu(total_keypoints, 32, CV_8U);
    // 執行下載
    cudaMemcpy(all_descriptors_cpu.data, keypoints_desc_.gpu_data(), total_keypoints * 32, cudaMemcpyDeviceToHost);

    // 6b. 使用與 v6 版本相同的 CPU 端打包邏輯，確保座標縮放等細節正確無誤
    out_keypoints.resize(total_keypoints * 6);
    out_keypoints_desc.resize(total_keypoints * 32);

    int* kp_data = out_keypoints.cpu_data();
    unsigned char* desc_data = out_keypoints_desc.cpu_data();

    const int out_x_offset  = 0 * total_keypoints;
    const int out_y_offset  = 1 * total_keypoints;
    const int out_s_offset  = 2 * total_keypoints;
    const int out_a_offset  = 3 * total_keypoints;
    const int out_l_offset  = 4 * total_keypoints;
    const int out_sz_offset = 5 * total_keypoints;

    float* kp_data_float_s = reinterpret_cast<float*>(kp_data + out_s_offset);
    float* kp_data_float_a = reinterpret_cast<float*>(kp_data + out_a_offset);
    float* kp_data_float_sz = reinterpret_cast<float*>(kp_data + out_sz_offset);
    
    int current_kp_offset = 0;
    for(int level = 0; level < n_levels_; ++level) {
        const auto& keypoints_this_level = all_keypoints_hybrid[level];
        if(keypoints_this_level.empty()) continue;

        for(size_t i = 0; i < keypoints_this_level.size(); ++i) {
            const auto& kp = keypoints_this_level[i];
            const int target_idx = current_kp_offset + i;
            
            // 進行座標縮放，將所有層級的座標統一到第0層
            kp_data[out_x_offset + target_idx] = cvRound(kp.pt.x * scale_[level]);
            kp_data[out_y_offset + target_idx] = cvRound(kp.pt.y * scale_[level]);
            
            kp_data[out_l_offset + target_idx] = kp.octave;
            kp_data_float_s[target_idx] = kp.response;
            kp_data_float_a[target_idx] = kp.angle;
            kp_data_float_sz[target_idx] = kp.size;
        }
        current_kp_offset += keypoints_this_level.size();
    }
    
    // 將下載回來的描述子複製到最終的輸出緩衝區
    memcpy(desc_data, all_descriptors_cpu.data, total_keypoints * 32);

    // 通知 SyncedMem CPU 端的資料是最新，若後續有 GPU 操作需同步
    out_keypoints.to_gpu();
    out_keypoints_desc.to_gpu();
}
// ##################################################################
*/

//v6
/*
// ##################################################################
// ############## 新的 extract 函式 (v4FAST->GPU方向->CPU描述子) ####################
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{
    // === Step 0, 1, 2, 3: GPU 偵測 + CPU 四叉樹 (與上一版完全相同) ===
    // 建立圖像金字塔
    cudaMemcpy2D(image_[0].gpu_data(), image_[0].pitch_,
                 image.data, image.step,
                 image.cols * sizeof(unsigned char), image.rows,
                 cudaMemcpyHostToDevice);
    for (int i = 1; i < n_levels_; i++)
    {
        Compute_pyramid(height_[0], width_[0],
                        height_[i], width_[i],
                        inv_scale_[i],
                        image_[0].gpu_data(), image_[0].pitch_,
                        image_[i].gpu_data(), image_[i].pitch_,
                        cuda_streams_[i]);
    }

    // 在 GPU 上執行自適應 FAST + NMS
    for (int level = 0; level < n_levels_; ++level) {
        AdaptiveFASTAndNMS(level);
    }

    // 下載結果到 CPU 並用四元樹篩選
    std::vector<std::vector<cv::KeyPoint>> all_keypoints_hybrid(n_levels_);
    int total_keypoints = 0;
    for (int level = 0; level < n_levels_; ++level) {
        cudaStreamSynchronize(cuda_streams_[level]);
        cv::Mat score_map_cpu(height_[level], width_[level], CV_32S);
        cudaMemcpy2D(score_map_cpu.data, score_map_cpu.step,
                     score_nms_[level].gpu_data(), score_nms_[level].pitch_,
                     width_[level] * sizeof(int), height_[level],
                     cudaMemcpyDeviceToHost);

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = EDGE_THRESHOLD;
        const int maxBorderX = width_[level] - EDGE_THRESHOLD;
        const int maxBorderY = height_[level] - EDGE_THRESHOLD;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(n_features_per_level_[level] * 2);

        for (int y = minBorderY; y < maxBorderY; ++y) {
            for (int x = minBorderX; x < maxBorderX; ++x) {
                const int score = score_map_cpu.at<int>(y, x);
                if (score > 0) {
                    vToDistributeKeys.emplace_back(cv::Point2f(x - minBorderX, y - minBorderY), 7, -1, score);
                }
            }
        }
        
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        keypoints_this_level.reserve(n_features_per_level_[level]);
        
        if (!vToDistributeKeys.empty()) {
            keypoints_this_level = DistributeOctTree(vToDistributeKeys, 0, maxBorderX - minBorderX, 0, maxBorderY - minBorderY, n_features_per_level_[level], level);
        }

        for (auto& kp : keypoints_this_level) {
            kp.pt.x += minBorderX;
            kp.pt.y += minBorderY;
            kp.octave = level;
            kp.size = round(PATCH_SIZE * scale_[level]);
        }
        total_keypoints += keypoints_this_level.size();
    }


    // 如果沒有檢測到任何特徵點，則提前退出
    if (total_keypoints == 0) {
        out_keypoints.resize(0);
        out_keypoints_desc.resize(0);
        return;
    }

    // === [修改] Step 4: 上傳 Keypoints 並在 GPU 計算方向 ===
    
    // 4a. 彙總所有層級的特徵點，並準備上傳 GPU
    std::vector<int> kp_x_cpu(total_keypoints);
    std::vector<int> kp_y_cpu(total_keypoints);
    std::vector<int> kp_score_cpu(total_keypoints);
    std::vector<int> kp_level_offset_cpu(n_levels_);
    
    int offset = 0;
    for (int level = 0; level < n_levels_; ++level)
    {
        kp_level_offset_cpu[level] = offset;
        for (size_t i = 0; i < all_keypoints_hybrid[level].size(); ++i)
        {
            kp_x_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.x;
            kp_y_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.y;
            kp_score_cpu[offset + i] = all_keypoints_hybrid[level][i].response;
        }
        offset += all_keypoints_hybrid[level].size();
    }

    // 4b. 將特徵點座標上傳到一個臨時的 GPU 緩衝區
    // 這裡我們重新使用 keypoints_ 成員變數
    keypoints_.resize(total_keypoints * 5); // x, y, score, level(dummy), angle
    cudaMemcpy(keypoints_.gpu_data() + 0*total_keypoints, kp_x_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 1*total_keypoints, kp_y_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 2*total_keypoints, kp_score_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);

    // 4c. 在 GPU 上計算方向 (重新啟用)
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level > 0)
        {
            int level_offset = kp_level_offset_cpu[level];
            // 呼叫 GPU 方向計算函式
            FAST_compute_orientation(height_[level], width_[level],
                                     image_[level].gpu_data(), image_[level].pitch_,
                                     n_kps_level,
                                     keypoints_.gpu_data() + 0*total_keypoints + level_offset, // in x
                                     keypoints_.gpu_data() + 1*total_keypoints + level_offset, // in y
                                     keypoints_.gpu_data() + 2*total_keypoints + level_offset, // in score
                                     (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset, // out angle
                                     umax_[level].gpu_data(), // umax table on GPU
                                     cuda_streams_[level]);
        }
    }

    // === [修改] Step 5: 下載角度，並在 CPU 計算描述子 ===

    // 5a. 從 GPU 下載計算好的角度
    std::vector<float> angles_from_gpu(total_keypoints);
    // 確保所有方向計算都已完成
    for(int i = 0; i < n_levels_; ++i) cudaStreamSynchronize(cuda_streams_[i]);
    // 執行下載
    cudaMemcpy(angles_from_gpu.data(), (float*)(keypoints_.gpu_data() + 4*total_keypoints), total_keypoints * sizeof(float), cudaMemcpyDeviceToHost);

    // 5b. 將下載的角度更新到 CPU 的 KeyPoint 結構中
    int angle_offset = 0;
    for (int level = 0; level < n_levels_; ++level) {
        for(size_t i = 0; i < all_keypoints_hybrid[level].size(); ++i) {
            all_keypoints_hybrid[level][i].angle = angles_from_gpu[angle_offset + i];
        }
        angle_offset += all_keypoints_hybrid[level].size();
    }
    
    // 5c. 遍歷每一層，在 CPU 上計算描述子 (與上一版相同)
    std::vector<cv::Mat> all_descriptors(n_levels_);
    for (int level = 0; level < n_levels_; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        if(keypoints_this_level.empty()) continue;

        cv::Mat image_level_cpu(height_[level], width_[level], CV_8UC1);
        cudaMemcpy2D(image_level_cpu.data, image_level_cpu.step,
                     image_[level].gpu_data(), image_[level].pitch_,
                     width_[level] * sizeof(unsigned char), height_[level],
                     cudaMemcpyDeviceToHost);
        
        cv::Mat blurred_image;
        cv::GaussianBlur(image_level_cpu, blurred_image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        computeDescriptors(blurred_image, keypoints_this_level, all_descriptors[level], pattern_cpu_);
    }

    // === [修改] Step 6: 將最終結果打包到 SyncedMem 中 (與上一版相同) ===

    out_keypoints.resize(total_keypoints * 6);
    out_keypoints_desc.resize(total_keypoints * 32);

    int* kp_data = out_keypoints.cpu_data();
    unsigned char* desc_data = out_keypoints_desc.cpu_data();

    const int out_x_offset  = 0 * total_keypoints;
    const int out_y_offset  = 1 * total_keypoints;
    const int out_s_offset  = 2 * total_keypoints;
    const int out_a_offset  = 3 * total_keypoints;
    const int out_l_offset  = 4 * total_keypoints;
    const int out_sz_offset = 5 * total_keypoints;

    float* kp_data_float_s = reinterpret_cast<float*>(kp_data + out_s_offset);
    float* kp_data_float_a = reinterpret_cast<float*>(kp_data + out_a_offset);
    float* kp_data_float_sz = reinterpret_cast<float*>(kp_data + out_sz_offset);
    
    int current_kp_offset = 0;
    for(int level = 0; level < n_levels_; ++level)
    {
        const auto& keypoints_this_level = all_keypoints_hybrid[level];
        if(keypoints_this_level.empty()) continue;

        for(size_t i = 0; i < keypoints_this_level.size(); ++i)
        {
            const auto& kp = keypoints_this_level[i];
            const int target_idx = current_kp_offset + i;
            
            kp_data[out_x_offset + target_idx] = cvRound(kp.pt.x * scale_[level]);
            kp_data[out_y_offset + target_idx] = cvRound(kp.pt.y * scale_[level]);
            kp_data[out_l_offset + target_idx] = kp.octave;

            kp_data_float_s[target_idx] = kp.response;
            kp_data_float_a[target_idx] = kp.angle;
            kp_data_float_sz[target_idx] = kp.size;
        }

        if(!all_descriptors[level].empty())
        {
             memcpy(desc_data + (current_kp_offset * 32), 
                    all_descriptors[level].data,
                    all_descriptors[level].total() * all_descriptors[level].elemSize());
        }
        current_kp_offset += keypoints_this_level.size();
    }

    out_keypoints.to_gpu();
    out_keypoints_desc.to_gpu();
}
// ##################################################################
*/

//v5
/*
// ##################################################################
// ############## 新的 extract 函式 (v4FAST->CPU方向+描述子) ####################
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{
    // === Step 0: Build Image Pyramid on GPU (與原版相同) ===
    cudaMemcpy2D(image_[0].gpu_data(), image_[0].pitch_,
                 image.data, image.step,
                 image.cols * sizeof(unsigned char), image.rows,
                 cudaMemcpyHostToDevice);
    for (int i = 1; i < n_levels_; i++)
    {
        Compute_pyramid(height_[0], width_[0],
                        height_[i], width_[i],
                        inv_scale_[i],
                        image_[0].gpu_data(), image_[0].pitch_,
                        image_[i].gpu_data(), image_[i].pitch_,
                        cuda_streams_[i]);
    }

    // === Step 1 & 2: GPU 上執行自適應 FAST + NMS (與原版相同) ===
    for (int level = 0; level < n_levels_; ++level) {
        AdaptiveFASTAndNMS(level);
    }

    // === Step 3: 下載結果到 CPU 並用四元樹篩選 (與原版相同) ===
    std::vector<std::vector<cv::KeyPoint>> all_keypoints_hybrid(n_levels_);
    for (int level = 0; level < n_levels_; ++level) {
        cudaStreamSynchronize(cuda_streams_[level]);
        cv::Mat score_map_cpu(height_[level], width_[level], CV_32S);
        cudaMemcpy2D(score_map_cpu.data, score_map_cpu.step,
                     score_nms_[level].gpu_data(), score_nms_[level].pitch_,
                     width_[level] * sizeof(int), height_[level],
                     cudaMemcpyDeviceToHost);

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = EDGE_THRESHOLD;
        const int maxBorderX = width_[level] - EDGE_THRESHOLD;
        const int maxBorderY = height_[level] - EDGE_THRESHOLD;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(n_features_per_level_[level] * 2);

        for (int y = minBorderY; y < maxBorderY; ++y) {
            for (int x = minBorderX; x < maxBorderX; ++x) {
                const int score = score_map_cpu.at<int>(y, x);
                if (score > 0) {
                    vToDistributeKeys.emplace_back(cv::Point2f(x - minBorderX, y - minBorderY), 7, -1, score);
                }
            }
        }
        
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        keypoints_this_level.reserve(n_features_per_level_[level]);
        
        if (!vToDistributeKeys.empty()) {
            keypoints_this_level = DistributeOctTree(vToDistributeKeys, 0, maxBorderX - minBorderX, 0, maxBorderY - minBorderY, n_features_per_level_[level], level);
        }

        for (auto& kp : keypoints_this_level) {
            kp.pt.x += minBorderX;
            kp.pt.y += minBorderY;
            kp.octave = level;
            kp.size = round(PATCH_SIZE * scale_[level]); // ORB-SLAM2 CPU 版的 size 計算方式
        }
    }

    // === [修改] Step 4: Finalization (Orientation & Descriptor on CPU) ===
    
    // 4a. 準備 CPU 描述子容器
    std::vector<cv::Mat> all_descriptors(n_levels_);
    int total_keypoints = 0;

    // 4b. 遍歷每一層，在 CPU 上計算方向和描述子
    for (int level = 0; level < n_levels_; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        if(keypoints_this_level.empty()) {
            continue;
        }

        total_keypoints += keypoints_this_level.size();

        // (1) 從 GPU 下載該層級的圖像
        cv::Mat image_level_cpu(height_[level], width_[level], CV_8UC1);
        cudaMemcpy2D(image_level_cpu.data, image_level_cpu.step,
                     image_[level].gpu_data(), image_[level].pitch_,
                     width_[level] * sizeof(unsigned char), height_[level],
                     cudaMemcpyDeviceToHost);
        
        // (2) 計算方向
        computeOrientation(image_level_cpu, keypoints_this_level, umax_cpu_[level]);

        // (3) 準備計算描述子 (高斯模糊)
        cv::Mat blurred_image;
        cv::GaussianBlur(image_level_cpu, blurred_image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // (4) 計算描述子
        computeDescriptors(blurred_image, keypoints_this_level, all_descriptors[level], pattern_cpu_);
    }

    // === [修改] Step 5: 將 CPU 結果打包到 SyncedMem 中 ===

    // 如果沒有檢測到任何特徵點，則提前退出
    if (total_keypoints == 0) {
        out_keypoints.resize(0);
        out_keypoints_desc.resize(0);
        return;
    }

    // 5a. 調整輸出 SyncedMem 的大小
    out_keypoints.resize(total_keypoints * 6); // x, y, score, angle, level, size
    out_keypoints_desc.resize(total_keypoints * 32);

    // 5b. 獲取 CPU 資料指標
    int* kp_data = out_keypoints.cpu_data();
    unsigned char* desc_data = out_keypoints_desc.cpu_data();

    // 5c. 定義輸出格式的位移
    const int out_x_offset  = 0 * total_keypoints;
    const int out_y_offset  = 1 * total_keypoints;
    const int out_s_offset  = 2 * total_keypoints; // score
    const int out_a_offset  = 3 * total_keypoints; // angle
    const int out_l_offset  = 4 * total_keypoints; // level (octave)
    const int out_sz_offset = 5 * total_keypoints; // size

    // 為了正確寫入 float，需要將 int 指標重新解釋為 float 指標
    float* kp_data_float_s = reinterpret_cast<float*>(kp_data + out_s_offset);
    float* kp_data_float_a = reinterpret_cast<float*>(kp_data + out_a_offset);
    float* kp_data_float_sz = reinterpret_cast<float*>(kp_data + out_sz_offset);
    
    int current_kp_offset = 0; // 追蹤在最終輸出緩衝區中的位移
    for(int level = 0; level < n_levels_; ++level)
    {
        const auto& keypoints_this_level = all_keypoints_hybrid[level];
        if(keypoints_this_level.empty()) continue;

        // 填充 keypoint 資料
        for(size_t i = 0; i < keypoints_this_level.size(); ++i)
        {
            const auto& kp = keypoints_this_level[i];
            const int target_idx = current_kp_offset + i;

            kp_data[out_x_offset + target_idx] = cvRound(kp.pt.x * scale_[level]); // 縮放到 level 0
            kp_data[out_y_offset + target_idx] = cvRound(kp.pt.y * scale_[level]); // 縮放到 level 0
            kp_data[out_l_offset + target_idx] = kp.octave;

            kp_data_float_s[target_idx] = kp.response;
            kp_data_float_a[target_idx] = kp.angle;
            kp_data_float_sz[target_idx] = kp.size;
        }

        // 填充 descriptor 資料
        if(!all_descriptors[level].empty())
        {
             memcpy(desc_data + (current_kp_offset * 32), 
                    all_descriptors[level].data,
                    all_descriptors[level].total() * all_descriptors[level].elemSize());
        }

        current_kp_offset += keypoints_this_level.size();
    }

    // 5d. 通知 SyncedMem CPU 端的資料是最新，若後續有 GPU 操作需同步
    out_keypoints.to_gpu();
    out_keypoints_desc.to_gpu();
}
// ##################################################################
*/

//v4
/**/
// ##################################################################
// ############## 新的 extract 函式 (FAST高閾值->NMS->FAST低閾值->NMS) ####################
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{
    // +++ 加入計時基礎設施 +++
    FrameTimeStats current_stats;
    Timer timer;
    Timer total_timer;
    total_timer.Start();


    // Step 0: Build Image Pyramid on GPU
    timer.Start();
    cudaMemcpy2D(image_[0].gpu_data(), image_[0].pitch_,
                 image.data, image.step,
                 image.cols * sizeof(unsigned char), image.rows,
                 cudaMemcpyHostToDevice);
    for (int i = 1; i < n_levels_; i++)
    {
        Compute_pyramid(height_[0], width_[0],
                        height_[i], width_[i],
                        inv_scale_[i],
                        image_[0].gpu_data(), image_[0].pitch_,
                        image_[i].gpu_data(), image_[i].pitch_,
                        cuda_streams_[i]);
    }
    current_stats.pyramid_construction = timer.Stop();


    // Step 1 & 2: 在 GPU 上執行新的、更魯棒的特徵點檢測流程
    timer.Start();
    for (int level = 0; level < n_levels_; ++level) {
        AdaptiveFASTAndNMS(level);
    }
    // 同步所有 CUDA stream，確保偵測已完成
    for (int level = 0; level < n_levels_; ++level) cudaStreamSynchronize(cuda_streams_[level]);
    current_stats.feature_detection = timer.Stop();


    // Step 3 & 4: [20261116 延遲優化] GPU 上壓縮候選點 → 只傳清單回 CPU → 依 (y,x) 排序（＝原本逐列掃描的順序）→ 四元樹篩選
    //   原本：每層整張分數圖 cudaMemcpy2D 回 CPU（8 層約 3.7 MB）＋CPU 逐像素掃描＋每層 cudaDeviceSynchronize
    //   選點邏輯、參數、四元樹的輸入內容與順序都不變
    double gpu_to_cpu_time = 0.0;
    double quadtree_time = 0.0;
    std::vector<std::vector<cv::KeyPoint>> all_keypoints_hybrid(n_levels_);
    const int EDGE_THRESHOLD = 19;
    if (compact_gpu_.empty()) {  // 第一次呼叫時配置常駐緩衝區
        compact_gpu_.assign(n_levels_, nullptr);
        compact_cap_.assign(n_levels_, 0);
        int maxcap = 1;
        for (int level = 0; level < n_levels_; ++level) {
            const int w = std::max(0, width_[level] - 2 * EDGE_THRESHOLD);
            const int h = std::max(0, height_[level] - 2 * EDGE_THRESHOLD);
            compact_cap_[level] = std::max(1, w * h);  // 最壞情況：每個像素都是候選點 → 不會溢位
            cudaMalloc(&compact_gpu_[level], compact_cap_[level] * sizeof(int3));
            maxcap = std::max(maxcap, compact_cap_[level]);
        }
        cudaMalloc(&compact_count_gpu_, n_levels_ * sizeof(int));
        cudaMallocHost(&compact_count_host_, n_levels_ * sizeof(int));
        cudaMallocHost(&compact_host_, maxcap * sizeof(int3));
        compact_host_cap_ = maxcap;
    }
    timer.Start();
    for (int level = 0; level < n_levels_; ++level) {  // 和偵測在同一個 stream 上 → 保證偵測完成後才壓縮
        cudaMemsetAsync(compact_count_gpu_ + level, 0, sizeof(int), cuda_streams_[level]);
        compact_score_map(score_nms_[level].gpu_data(), score_nms_[level].pitch_ / (int)sizeof(int),
                          EDGE_THRESHOLD, EDGE_THRESHOLD, width_[level] - EDGE_THRESHOLD, height_[level] - EDGE_THRESHOLD,
                          compact_gpu_[level], compact_count_gpu_ + level, compact_cap_[level], cuda_streams_[level]);
        cudaMemcpyAsync(compact_count_host_ + level, compact_count_gpu_ + level, sizeof(int), cudaMemcpyDeviceToHost, cuda_streams_[level]);
    }
    for (int level = 0; level < n_levels_; ++level) cudaStreamSynchronize(cuda_streams_[level]);
    gpu_to_cpu_time += timer.Stop();

    for (int level = 0; level < n_levels_; ++level) {
        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = EDGE_THRESHOLD;
        const int maxBorderX = width_[level] - EDGE_THRESHOLD;
        const int maxBorderY = height_[level] - EDGE_THRESHOLD;

        timer.Start();
        const int n_cand = std::min(compact_count_host_[level], compact_cap_[level]);
        if (n_cand > 0)
            cudaMemcpy(compact_host_, compact_gpu_[level], n_cand * sizeof(int3), cudaMemcpyDeviceToHost);
        std::sort(compact_host_, compact_host_ + n_cand,
                  [](const int3& a, const int3& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
        gpu_to_cpu_time += timer.Stop();

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(std::max(n_cand, n_features_per_level_[level] * 2));
        for (int k = 0; k < n_cand; ++k)
            vToDistributeKeys.emplace_back(cv::Point2f(compact_host_[k].x, compact_host_[k].y), 7, -1, compact_host_[k].z);

        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        keypoints_this_level.reserve(n_features_per_level_[level]);
        
        timer.Start();
        if (!vToDistributeKeys.empty()) {
            keypoints_this_level = DistributeOctTree(vToDistributeKeys, 0, maxBorderX - minBorderX, 0, maxBorderY - minBorderY, n_features_per_level_[level], level);
        }
        quadtree_time += timer.Stop();

        for (auto& kp : keypoints_this_level) {
            kp.pt.x += minBorderX;
            kp.pt.y += minBorderY;
            kp.octave = level;
            kp.size = round(CIRCULAR_HALF_PATCH_SIZE * 2 * scale_[level]);
        }
    }
    current_stats.gpu_to_cpu_transfer = gpu_to_cpu_time;
    current_stats.quadtree_distribution = quadtree_time;


    // === Step 4: Finalization (Orientation & Descriptor on GPU) ===
    
    // 4a. 彙總所有層級的特徵點，並準備上傳 GPU
    int total_keypoints = 0;
    for (int level = 0; level < n_levels_; ++level) {
        total_keypoints += all_keypoints_hybrid[level].size();
    }

    // 如果沒有檢測到任何特徵點，則提前退出
    if (total_keypoints == 0) {
        out_keypoints.resize(0);
        out_keypoints_desc.resize(0);
        return;
    }
    
    // Allocate temporary CPU memory to linearize keypoint data for upload
    timer.Start();
    std::vector<int> kp_x_cpu(total_keypoints);
    std::vector<int> kp_y_cpu(total_keypoints);
    std::vector<int> kp_score_cpu(total_keypoints);
    std::vector<int> kp_level_offset_cpu(n_levels_);
    
    int offset = 0;
    for (int level = 0; level < n_levels_; ++level)
    {
        kp_level_offset_cpu[level] = offset;
        for (size_t i = 0; i < all_keypoints_hybrid[level].size(); ++i)
        {
            kp_x_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.x;
            kp_y_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.y;
            kp_score_cpu[offset + i] = all_keypoints_hybrid[level][i].response;
        }
        offset += all_keypoints_hybrid[level].size();
    }
    
    // 4b. 將最終的特徵點座標上傳到一個臨時的 GPU 緩衝區
    // Using the existing 'keypoints_' buffer for this temporary storage
    keypoints_.resize(total_keypoints * 5);// x, y, score, level, angle

    cudaMemcpy(keypoints_.gpu_data() + 0*total_keypoints, kp_x_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 1*total_keypoints, kp_y_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 2*total_keypoints, kp_score_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    current_stats.cpu_to_gpu_transfer = timer.Stop();


    // 4c. 在 GPU 上計算方向
    timer.Start();
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level > 0)
        {
            int level_offset = kp_level_offset_cpu[level];
            FAST_compute_orientation(height_[level], width_[level],
                                     image_[level].gpu_data(), image_[level].pitch_,
                                     n_kps_level,
                                     keypoints_.gpu_data() + 0*total_keypoints + level_offset,
                                     keypoints_.gpu_data() + 1*total_keypoints + level_offset,
                                     keypoints_.gpu_data() + 2*total_keypoints + level_offset,
                                     (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset,
                                     umax_[level].gpu_data(),
                                     cuda_streams_[level]);
        }
    }
    current_stats.orientation_computation = timer.Stop();

    
    // 4d. 在 GPU 上計算描述子
    // First, apply Gaussian blur to pyramid images
    timer.Start();
    for(int i = 0; i < n_levels_; ++i) {
        compute_gaussian(height_[i], width_[i],
                         image_[i].gpu_data(), image_[i].pitch_,
                         image_gaussian_[i].gpu_data(), image_gaussian_[i].pitch_,
                         gaussian_weights_[i].gpu_data(),
                         cuda_streams_[i]);
    }

    // Then compute descriptors
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level > 0)
        {
            int level_offset = kp_level_offset_cpu[level];
             ORB_compute_descriptor(height_[level], width_[level],
                                   image_gaussian_[level].gpu_data(), image_gaussian_[level].pitch_,
                                   pattern_x_[level].gpu_data(), pattern_y_[level].gpu_data(),
                                   n_kps_level,
                                   keypoints_.gpu_data() + 0*total_keypoints + level_offset,
                                   keypoints_.gpu_data() + 1*total_keypoints + level_offset,
                                   (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset,
                                   keypoints_desc_.gpu_data() + (level_offset * 32),
                                   cuda_streams_[level]);
        }
    }
    current_stats.descriptor_computation = timer.Stop();


    // 4e. 準備最終輸出緩衝區
    timer.Start();
    out_keypoints.resize(total_keypoints * 6); // x, y, score, angle, level, size
    out_keypoints_desc.resize(total_keypoints * 32);

    int kp_out_offset = 0; // 追蹤在最終輸出緩衝區中的位移
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level == 0) continue;

        // 獲取當前層級在臨時緩衝區中的數據起始位移
        int level_offset_gpu = kp_level_offset_cpu[level];

        // 定義最終輸出緩衝區的各個數據區塊的起始位移
        const int out_x_offset  = 0 * total_keypoints;
        const int out_y_offset  = 1 * total_keypoints;
        const int out_s_offset  = 2 * total_keypoints;
        const int out_a_offset  = 3 * total_keypoints;
        const int out_l_offset  = 4 * total_keypoints;
        const int out_sz_offset = 5 * total_keypoints;
        
        // 呼叫您 v2 版本中的 copy_output 函式，它會在 GPU 上完成複製和座標縮放
        copy_output(n_kps_level,
                    level, // octave
                    width_[level], // width (在 kernel 中可能沒用到)
                    scale_[level], // <<<<<<< 傳入縮放因子
                    keypoints_.gpu_data() + 0*total_keypoints + level_offset_gpu, // in x
                    keypoints_.gpu_data() + 1*total_keypoints + level_offset_gpu, // in y
                    keypoints_.gpu_data() + 2*total_keypoints + level_offset_gpu, // in score
                    (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset_gpu, // in angle
                    out_keypoints.gpu_data() + out_x_offset + kp_out_offset, // out x
                    out_keypoints.gpu_data() + out_y_offset + kp_out_offset, // out y
                    out_keypoints.gpu_data() + out_s_offset + kp_out_offset, // out score (response)
                    (float*)out_keypoints.gpu_data() + out_a_offset + kp_out_offset, // out angle
                    out_keypoints.gpu_data() + out_l_offset + kp_out_offset, // out level
                    out_keypoints.gpu_data() + out_sz_offset + kp_out_offset, // out size
                    cuda_streams_[level]
                    );
        
        // 異步複製對應的描述子
        cudaMemcpyAsync(out_keypoints_desc.gpu_data() + kp_out_offset * 32,
                        keypoints_desc_.gpu_data() + level_offset_gpu * 32,
                        n_kps_level * 32,
                        cudaMemcpyDeviceToDevice,
                        cuda_streams_[level]);

        // 更新下一個層級在輸出緩衝區中的寫入位置
        kp_out_offset += n_kps_level;
    }

    // 等待所有串流中的異步操作全部完成
    for(int i = 0; i < n_levels_; ++i)
    {
        cudaStreamSynchronize(cuda_streams_[i]);
    }
    current_stats.final_packing = timer.Stop();

    // 計算總時間並儲存
    current_stats.total_extraction = total_timer.Stop();
    vFrameTimings.push_back(current_stats);
}
// ##################################################################


//v3
/*
// ##################################################################
// ############## 新的 extract 函式 (混合式方法) ####################
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{
    // Step 0: Build Image Pyramid on GPU
    // cudaMemcpy2D(image_[0].gpu_data(), image_[0].pitch_, image.data, width_[0], width_[0], height_[0], cudaMemcpyHostToDevice);
    cudaMemcpy2D(image_[0].gpu_data(), image_[0].pitch_,
                 image.data, image.step,
                 image.cols * sizeof(unsigned char), image.rows,
                 cudaMemcpyHostToDevice);
    for (int i = 1; i < n_levels_; i++)
    {
        Compute_pyramid(height_[0], width_[0],
                        height_[i], width_[i],
                        inv_scale_[i],
                        image_[0].gpu_data(), image_[0].pitch_,
                        image_[i].gpu_data(), image_[i].pitch_,
                        cuda_streams_[i]);
    }

    // This vector will hold the distributed keypoints from all levels
    std::vector<std::vector<cv::KeyPoint>> all_keypoints_hybrid(n_levels_);

    // Process each pyramid level
    for (int level = 0; level < n_levels_; ++level)
    {
        // === Step 1: GPU Score Calculation ===
        // 計算出原始的、密集的響應分數圖，存入 score_[level]
        FAST_compute_score_lookpup_mask_v2(height_[level], width_[level],
                                           image_[level].gpu_data(), image_[level].pitch_,
                                           masks_[level].gpu_data(), masks_[level].pitch_,
                                           lookup_table_[level].gpu_data(),
                                           score_[level].gpu_data(),
                                           score_[level].pitch_ / 4,
                                           cuda_streams_[level]);

        // === Step 2: GPU 3x3 Non-Maximal Suppression (NMS) ===
        // 對原始分數圖執行 NMS，結果存入 score_nms_[level]
        FAST_apply_NMS_global(height_[level], width_[level],
                        score_[level].gpu_data(), score_[level].pitch_ / 4,
                        score_nms_[level].gpu_data(), score_nms_[level].pitch_ / 4,
                        cuda_streams_[level]);

        // === Step 3: Data Transfer (GPU -> CPU) ===
        // 只下載經過 NMS 處理後的、稀疏的分數圖
        cudaStreamSynchronize(cuda_streams_[level]); // Wait for score calculation to finish
        cv::Mat score_map_cpu(height_[level], width_[level], CV_32S);
        cudaMemcpy2D(score_map_cpu.data, score_map_cpu.step,
             score_nms_[level].gpu_data(), score_nms_[level].pitch_, // <-- 使用 NMS 後的結果
             width_[level] * sizeof(int), height_[level],
             cudaMemcpyDeviceToHost);

        // === Step 4: CPU Quadtree Distribution ===
        // 4a. 從稀疏的分數圖中產生候選點
        const int EDGE_THRESHOLD = 19; // From ORBextractor_cpu.cpp
        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = EDGE_THRESHOLD;
        const int maxBorderX = width_[level] - EDGE_THRESHOLD;
        const int maxBorderY = height_[level] - EDGE_THRESHOLD;

        // 4b. 從下載的分數圖中，只提取有效區域內的候選點
        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(n_features_per_level_[level] * 2); // Reserve some memory

        for (int y = minBorderY; y < maxBorderY; ++y)
        {
            for (int x = minBorderX; x < maxBorderX; ++x)
            {
                const int score = score_map_cpu.at<int>(y, x);
                if (score > 0) // 現在只有 NMS 後的局部最大值分數 > 0
                {
                    // 使用相對於 ROI 左上角 (minBorderX, minBorderY) 的座標
                    vToDistributeKeys.emplace_back(cv::Point2f(x - minBorderX, y - minBorderY), 7, -1, score);
                }
            }
        }
        
        std::vector<cv::KeyPoint>& keypoints_this_level = all_keypoints_hybrid[level];
        keypoints_this_level.reserve(n_features_per_level_[level]);
        // 3c. 將 ROI 的邊界傳入 DistributeOctTree
        if (!vToDistributeKeys.empty())
        {
            keypoints_this_level = DistributeOctTree(vToDistributeKeys, 
                                                     0, maxBorderX - minBorderX, // 使用絕對邊界
                                                     0, maxBorderY - minBorderY, // 使用絕對邊界
                                                     n_features_per_level_[level], 
                                                     level);
        }

        for (auto& kp : keypoints_this_level)
        {
            kp.pt.x += minBorderX;
            kp.pt.y += minBorderY;
            kp.octave = level;
            kp.size = round(CIRCULAR_HALF_PATCH_SIZE * 2 * scale_[level]);
        }
    }

    // === Step 4: Finalization (Orientation & Descriptor on GPU) ===
    
    // 4a. 彙總所有層級的特徵點，並準備上傳 GPU
    int total_keypoints = 0;
    for (int level = 0; level < n_levels_; ++level) {
        total_keypoints += all_keypoints_hybrid[level].size();
    }

    // 如果沒有檢測到任何特徵點，則提前退出
    if (total_keypoints == 0) {
        out_keypoints.resize(0);
        out_keypoints_desc.resize(0);
        return;
    }
    
    // Allocate temporary CPU memory to linearize keypoint data for upload
    std::vector<int> kp_x_cpu(total_keypoints);
    std::vector<int> kp_y_cpu(total_keypoints);
    std::vector<int> kp_score_cpu(total_keypoints);
    std::vector<int> kp_level_offset_cpu(n_levels_);
    
    int offset = 0;
    for (int level = 0; level < n_levels_; ++level)
    {
        kp_level_offset_cpu[level] = offset;
        for (size_t i = 0; i < all_keypoints_hybrid[level].size(); ++i)
        {
            kp_x_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.x;
            kp_y_cpu[offset + i] = all_keypoints_hybrid[level][i].pt.y;
            kp_score_cpu[offset + i] = all_keypoints_hybrid[level][i].response;
        }
        offset += all_keypoints_hybrid[level].size();
    }
    
    // 4b. 將最終的特徵點座標上傳到一個臨時的 GPU 緩衝區
    // Using the existing 'keypoints_' buffer for this temporary storage
    keypoints_.resize(total_keypoints * 5);// x, y, score, level, angle

    cudaMemcpy(keypoints_.gpu_data() + 0*total_keypoints, kp_x_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 1*total_keypoints, kp_y_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(keypoints_.gpu_data() + 2*total_keypoints, kp_score_cpu.data(), total_keypoints * sizeof(int), cudaMemcpyHostToDevice);
    
    // 4c. 在 GPU 上計算方向
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level > 0)
        {
            int level_offset = kp_level_offset_cpu[level];
            FAST_compute_orientation(height_[level], width_[level],
                                     image_[level].gpu_data(), image_[level].pitch_,
                                     n_kps_level,
                                     keypoints_.gpu_data() + 0*total_keypoints + level_offset,
                                     keypoints_.gpu_data() + 1*total_keypoints + level_offset,
                                     keypoints_.gpu_data() + 2*total_keypoints + level_offset,
                                     (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset,
                                     umax_[level].gpu_data(),
                                     cuda_streams_[level]);
        }
    }

    // 4d. 在 GPU 上計算描述子
    // First, apply Gaussian blur to pyramid images
    for(int i = 0; i < n_levels_; ++i) {
        compute_gaussian(height_[i], width_[i],
                         image_[i].gpu_data(), image_[i].pitch_,
                         image_gaussian_[i].gpu_data(), image_gaussian_[i].pitch_,
                         gaussian_weights_[i].gpu_data(),
                         cuda_streams_[i]);
    }
    // Then compute descriptors
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level > 0)
        {
            int level_offset = kp_level_offset_cpu[level];
             ORB_compute_descriptor(height_[level], width_[level],
                                   image_gaussian_[level].gpu_data(), image_gaussian_[level].pitch_,
                                   pattern_x_[level].gpu_data(), pattern_y_[level].gpu_data(),
                                   n_kps_level,
                                   keypoints_.gpu_data() + 0*total_keypoints + level_offset,
                                   keypoints_.gpu_data() + 1*total_keypoints + level_offset,
                                   (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset,
                                   keypoints_desc_.gpu_data() + (level_offset * 32),
                                   cuda_streams_[level]);
        }
    }

    // 4e. 準備最終輸出緩衝區
    out_keypoints.resize(total_keypoints * 6); // x, y, score, angle, level, size
    out_keypoints_desc.resize(total_keypoints * 32);

    int kp_out_offset = 0; // 追蹤在最終輸出緩衝區中的位移
    for (int level = 0; level < n_levels_; ++level)
    {
        int n_kps_level = all_keypoints_hybrid[level].size();
        if (n_kps_level == 0) continue;

        // 獲取當前層級在臨時緩衝區中的數據起始位移
        int level_offset_gpu = kp_level_offset_cpu[level];

        // 定義最終輸出緩衝區的各個數據區塊的起始位移
        const int out_x_offset  = 0 * total_keypoints;
        const int out_y_offset  = 1 * total_keypoints;
        const int out_s_offset  = 2 * total_keypoints;
        const int out_a_offset  = 3 * total_keypoints;
        const int out_l_offset  = 4 * total_keypoints;
        const int out_sz_offset = 5 * total_keypoints;
        
        // 呼叫您 v2 版本中的 copy_output 函式，它會在 GPU 上完成複製和座標縮放
        copy_output(n_kps_level,
                    level, // octave
                    width_[level], // width (在 kernel 中可能沒用到)
                    scale_[level], // <<<<<<< 傳入縮放因子
                    keypoints_.gpu_data() + 0*total_keypoints + level_offset_gpu, // in x
                    keypoints_.gpu_data() + 1*total_keypoints + level_offset_gpu, // in y
                    keypoints_.gpu_data() + 2*total_keypoints + level_offset_gpu, // in score
                    (float*)(keypoints_.gpu_data() + 4*total_keypoints) + level_offset_gpu, // in angle
                    out_keypoints.gpu_data() + out_x_offset + kp_out_offset, // out x
                    out_keypoints.gpu_data() + out_y_offset + kp_out_offset, // out y
                    out_keypoints.gpu_data() + out_s_offset + kp_out_offset, // out score (response)
                    (float*)out_keypoints.gpu_data() + out_a_offset + kp_out_offset, // out angle
                    out_keypoints.gpu_data() + out_l_offset + kp_out_offset, // out level
                    out_keypoints.gpu_data() + out_sz_offset + kp_out_offset, // out size
                    cuda_streams_[level]
                    );
        
        // 異步複製對應的描述子
        cudaMemcpyAsync(out_keypoints_desc.gpu_data() + kp_out_offset * 32,
                        keypoints_desc_.gpu_data() + level_offset_gpu * 32,
                        n_kps_level * 32,
                        cudaMemcpyDeviceToDevice,
                        cuda_streams_[level]);

        // 更新下一個層級在輸出緩衝區中的寫入位置
        kp_out_offset += n_kps_level;
    }

    // 等待所有串流中的異步操作全部完成
    for(int i = 0; i < n_levels_; ++i)
    {
        cudaStreamSynchronize(cuda_streams_[i]);
    }
}
// ##################################################################
*/

//v2
/*
// ##################################################################
// ############## 原始方法 ####################
void ORB_GPU::extract(const cv::Mat& image,
                      SyncedMem<int>& out_keypoints,
                      SyncedMem<unsigned char>& out_keypoints_desc)
{

    // compute pyramid for all levels async
    {
        //        cudaMemcpy(image_[0].gpu_data(), image.data, height_[0] * width_[0], cudaMemcpyHostToDevice);
        cudaMemcpy2D(image_[0].gpu_data(),image_[0].pitch_, image.data, width_[0], width_[0],height_[0], cudaMemcpyHostToDevice);

        // resize each image async
        for(int i=1;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];

            Compute_pyramid(height_[0], width_[0],
                    height_[i], width_[i],
                    inv_scale_[i],
                    image_[0].gpu_data(),
                    image_[0].pitch_,
                    image_[i].gpu_data(),
                    image_[i].pitch_,
                    cuda_stream);
        }

        //        for(int i=0;i<n_levels_;i++)
        //        {
        //            cudaStream_t& cuda_stream = cuda_streams_[i];
        //            cudaStreamSynchronize(cuda_stream);

        //            cv::Mat img(height_[i], width_[i], CV_8UC1);
        //            cudaMemcpy2D(img.data, width_[i], image_[i].gpu_data(),image_[i].pitch_, width_[i], height_[i], cudaMemcpyDeviceToHost);

        //            {
        //                cudaError_t cuerror = cudaGetLastError();
        //                if(cuerror) std::cout <<  cuerror << " error incurred 1\n";
        //            }

        //            std::stringstream winname;
        //            winname << "img" << i;

        //            cv::imshow(winname.str(),img);
        //            cv::waitKey(0);

        //        }
    }

    // compute FAST score for all levels async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        //// no need to synchronize because different scales have different stream
        //// and same scale stream always queue the jobs launched in them
        //// also its overhead intriduces a delay of us
        //   cudaStreamSynchronize(cuda_stream);

        //                        FAST_compute_score(height_[i], width_[i],
        //                                           image_[i].gpu_data(),
        //                                           threshold_tab_[i].gpu_data(),
        //                                           score_[i].gpu_data(),
        //                                           cuda_stream);

        //        FAST_compute_score_lookpup(height_[i], width_[i],
        //                                   image_[i].gpu_data(),
        //                                   image_[i].pitch_,
        //                                   threshold_,
        //                                   lookup_table_[i].gpu_data(),
        //                                   score_[i].gpu_data(),
        //                                   score_[i].pitch_ / 4,
        //                                   cuda_stream);

        // Jetson-SLAM 原始方法
        FAST_compute_score_lookpup_mask(height_[i], width_[i],
                                        image_[i].gpu_data(),
                                        image_[i].pitch_,
                                        masks_[i].gpu_data(),
                                        masks_[i].pitch_,
                                        threshold_,
                                        lookup_table_[i].gpu_data(),
                                        score_[i].gpu_data(),
                                        score_[i].pitch_ / 4,
                                        cuda_stream);

        // // 兩個閾值一起跑
        // FAST_compute_score_lookpup_mask_v2(height_[i], width_[i],
        //                                         image_[i].gpu_data(),
        //                                         image_[i].pitch_,
        //                                         masks_[i].gpu_data(),
        //                                         masks_[i].pitch_,
        //                                         lookup_table_[i].gpu_data(),
        //                                         score_[i].gpu_data(),
        //                                         score_[i].pitch_ / 4,
        //                                         cuda_stream);
    }


    if(!fuse_nms_L_with_nms_G_)
    {
        // apply NMS-L for all levels  async
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];
            FAST_apply_NMS_L(height_[i], width_[i],
                             score_[i].gpu_data(),
                             score_[i].pitch_ / 4,
                             score_nms_[i].gpu_data(),
                             score_nms_[i].pitch_ / 4,
                             cuda_stream);


        }
    }


    // compute keypoints on CPU for all levels sync and send to GPU async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        if(fuse_nms_L_with_nms_G_)
        {
            FAST_apply_NMS_G_reduce_unroll_reduce(height_[i], width_[i],
                                                  tile_h_[i], tile_w_[i],
                                                  n_tile_h_[i], n_tile_w_[i],
                                                  warp_tile_h_[i], warp_tile_w_[i],
                                                  fuse_nms_L_with_nms_G_,
                                                  image_unroll_[i].gpu_data(),
                                                  image_unroll_x_[i].gpu_data(),
                                                  image_unroll_y_[i].gpu_data(),
                                                  score_[i].gpu_data(),
                                                  score_[i].pitch_ / 4,
                                                  keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                                                  cuda_stream);

            //            FAST_apply_NMS_G_unroll_reduce(height_[i], width_[i],
            //                                           tile_h_[i], tile_w_[i],
            //                                           n_tile_h_[i], n_tile_w_[i],
            //                                           warp_tile_h_[i], warp_tile_w_[i],
            //                                           fuse_nms_L_with_nms_G_,
            //                                           image_unroll_[i].gpu_data(),
            //                                           score_[i].gpu_data(),
            //                score_[i].pitch_,
            //                                           keypoints_.gpu_data() + x_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + y_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + s_offset_ + level_offset_[i],
            //                                           cuda_stream);
        }
        else
        {
            FAST_apply_NMS_G_reduce_unroll_reduce(height_[i], width_[i],
                                                  tile_h_[i], tile_w_[i],
                                                  n_tile_h_[i], n_tile_w_[i],
                                                  warp_tile_h_[i], warp_tile_w_[i],
                                                  fuse_nms_L_with_nms_G_,
                                                  image_unroll_[i].gpu_data(),
                                                  image_unroll_x_[i].gpu_data(),
                                                  image_unroll_y_[i].gpu_data(),
                                                  score_nms_[i].gpu_data(),
                                                  score_nms_[i].pitch_ / 4,
                                                  keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                                                  keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                                                  cuda_stream);

            //            FAST_apply_NMS_G_unroll_reduce(height_[i], width_[i],
            //                                           tile_h_[i], tile_w_[i],
            //                                           n_tile_h_[i], n_tile_w_[i],
            //                                           warp_tile_h_[i], warp_tile_w_[i],
            //                                           fuse_nms_L_with_nms_G_,
            //                                           image_unroll_[i].gpu_data(),
            //                                           score_nms_[i].gpu_data(),
            //                score_nms_[i].pitch_,
            //                                           keypoints_.gpu_data() + x_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + y_offset_ + level_offset_[i],
            //                                           keypoints_.gpu_data() + s_offset_ + level_offset_[i],
            //                                           cuda_stream);
        }



    }
    

    //// NMS-MS
    if(apply_nms_ms_)
    {
        if(nms_ms_mode_gpu_)
        {
            // NMS-MS GPU
            {
                // 等待每一層金字塔的計算完成
                for(int i=0;i<n_levels_;i++)
                {
                    cudaStream_t& cuda_stream = cuda_streams_[i];
                    cudaStreamSynchronize(cuda_stream);
                }

                cudaStream_t& cuda_stream = cuda_streams_[0];

                nms_s_score_.set_zero_gpu();

                apply_NMS_MS_S_L(height_[0], width_[0],
                        n_levels_,
                        max_kp_count_,
                        keypoints_.gpu_data() + x_offset_,
                        keypoints_.gpu_data() + y_offset_,
                        keypoints_.gpu_data() + s_offset_,
                        grid_levels_.gpu_data(),
                        grid_scale_factor_.gpu_data(),
                        s0_score_.gpu_data(),
                        nms_s_score_.gpu_data(),
                        nms_s_level_.gpu_data(),
                        cuda_stream);

                cudaStreamSynchronize(cuda_stream);
            }

        }
        else
        {
            // NMS-MS CPU
            for(int i=0;i<n_levels_;i++)
            {
                cudaStream_t& cuda_stream = cuda_streams_[i];
                cudaStreamSynchronize(cuda_stream);
            }

            keypoints_.to_cpu(x_count_ + y_count_ + s_count_);

            FAST_apply_NMS_MS_cpu();

            keypoints_.to_gpu(x_count_ + y_count_ + s_count_);
        }
    }


    // obtain keypoints
    {

        // keypoints 資訊從 GPU 複製到 CPU，且將 score 非零的 keypoints 向前移動，並紀錄各層 keypoints 數量
        FAST_obtain_keypoints();

        keypoints_.to_gpu(x_count_ + y_count_ + s_count_);
    }


    // compute orientations async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        FAST_compute_orientation(height_[i], width_[i],
                                 image_[i].gpu_data(),
                                 image_[i].pitch_,
                                 n_keypoints_[i],
                                 keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                                 keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                                 keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                                 (float*)keypoints_.gpu_data() + a_offset_ + level_offset_[i],
                                 umax_[i].gpu_data(),
                                 cuda_stream);
    }



    //compute gaussian async
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        compute_gaussian(height_[i], width_[i],
                         image_[i].gpu_data(),
                         image_[i].pitch_,
                         image_gaussian_[i].gpu_data(),
                         image_gaussian_[i].pitch_,
                         gaussian_weights_[i].gpu_data(),
                         cuda_stream);
    }


    //compute descriptor
    for(int i=0;i<n_levels_;i++)
    {
        cudaStream_t& cuda_stream = cuda_streams_[i];

        ORB_compute_descriptor(height_[i], width_[i],
                               image_gaussian_[i].gpu_data(),
                               image_gaussian_[i].pitch_,
                               pattern_x_[i].gpu_data(),
                               pattern_y_[i].gpu_data(),
                               n_keypoints_[i],
                               keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                               keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                               (float*)keypoints_.gpu_data() + a_offset_ + level_offset_[i],
                               keypoints_desc_.gpu_data() + 32 * level_offset_[i],
                               cuda_stream);
    }

    //copy to output
    {
        int n_total_keypoints = 0;
        for(int i=0;i<n_levels_;i++)
            n_total_keypoints += n_keypoints_[i];

        out_keypoints.resize(n_total_keypoints * (1 + 1 + 1 + 1 + 1 + 1)); // x y s a l size
        out_keypoints_desc.resize(32 * n_total_keypoints);

        int kp_offset = 0;
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];

            const int out_x_offset  = 0 * n_total_keypoints;
            const int out_y_offset  = 1 * n_total_keypoints;
            const int out_s_offset  = 2 * n_total_keypoints; // score or response
            const int out_a_offset  = 3 * n_total_keypoints; // angle
            const int out_l_offset  = 4 * n_total_keypoints; // level or octave
            const int out_sz_offset = 5 * n_total_keypoints; // size

            const int out_level_offset = kp_offset;

            copy_output(n_keypoints_[i],
                        i, width_[i], scale_[i],
                        keypoints_.gpu_data() + x_offset_ + level_offset_[i],
                        keypoints_.gpu_data() + y_offset_ + level_offset_[i],
                        keypoints_.gpu_data() + s_offset_ + level_offset_[i],
                        (float*)keypoints_.gpu_data() + a_offset_ + level_offset_[i],
                        out_keypoints.gpu_data() + out_x_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_y_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_s_offset + out_level_offset,
                        (float*)out_keypoints.gpu_data() + out_a_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_l_offset + out_level_offset,
                        out_keypoints.gpu_data() + out_sz_offset + out_level_offset,
                        cuda_stream
                        );

            kp_offset += n_keypoints_[i];
        }

        int offset =  0;
        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];

            cudaMemcpyAsync(out_keypoints_desc.gpu_data() + offset,
                            keypoints_desc_.gpu_data() + 32 * level_offset_[i],
                            n_keypoints_[i]*32,
                            cudaMemcpyDeviceToDevice,
                            cuda_stream);

            offset += 32 * n_keypoints_[i];
        }

        for(int i=0;i<n_levels_;i++)
        {
            cudaStream_t& cuda_stream = cuda_streams_[i];
            cudaStreamSynchronize(cuda_stream);
        }

    }

}
*/
}

