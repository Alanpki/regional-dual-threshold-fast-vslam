#include "cuda/orb_gpu.hpp"


#include<chrono>


namespace orb_cuda {


#define CUDA_NUM_THREADS_PER_BLOCK 512

#define CUDA_NUM_THREADS_PER_BLOCK_x 32
#define CUDA_NUM_THREADS_PER_BLOCK_y 8

#define GRID_LAUNCH


__global__ void imresize_GPU_pitched(int n_threads,
                                     int height, int width,
                                     int op_height, int op_width,
                                     float inv_scale,
                                     unsigned char* ip_image,int ip_pitch,
                                     unsigned char* op_image,int op_pitch)
{
#ifdef GRID_LAUNCH

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < op_width && h < op_height)
    {
        const int offset = h * op_pitch + w;

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index <  n_threads)
    {
        int h = index / op_width;
        int w = index % op_width;

        const int offset = h * op_pitch + w;

#endif
        float scale = 1.0f / inv_scale;

        float ip_h = scale * h;
        float ip_w = scale * w;

        int xl = floorf(ip_w);
        int xr = xl + 1;

        int yt = floorf(ip_h);
        int yb = yt + 1;

        float w_xl = xr - ip_w;
        float w_xr = 1  - w_xl;
        float w_yt = yb - ip_h;
        float w_yb = 1  - w_yt;

        op_image[offset] = w_yt * w_xl * ip_image[yt * ip_pitch + xl]
                + w_yt * w_xr * ip_image[yt * ip_pitch + xr]
                + w_yb * w_xl * ip_image[yb * ip_pitch + xl]
                + w_yb * w_xr * ip_image[yb * ip_pitch + xr];

    }

}


__global__ void imresize_GPU_pitched_robust(int op_height, int op_width,
                                            int ip_height, int ip_width,
                                            float scale_x, float scale_y, // 使用獨立的 x, y 縮放比例
                                            const unsigned char* __restrict__ ip_image, int ip_pitch,
                                            unsigned char* __restrict__ op_image, int op_pitch)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if(w < op_width && h < op_height)
    {
        // 像素中心對齊座標對應公式
        const float ip_w_f = (w + 0.5f) * scale_x - 0.5f;
        const float ip_h_f = (h + 0.5f) * scale_y - 0.5f;

        // 計算整數座標
        int xl = floorf(ip_w_f);
        int yt = floorf(ip_h_f);

        // 使用更穩健的邊界鉗制 (clamping)
        int xr = min(xl + 1, ip_width - 1);
        int yb = min(yt + 1, ip_height - 1);
        xl = max(0, xl);
        yt = max(0, yt);

        // 計算權重
        const float w_x = ip_w_f - xl;
        const float w_y = ip_h_f - yt;

        // 讀取四個點的像素值
        const unsigned char p1 = ip_image[yt * ip_pitch + xl]; // Top-Left
        const unsigned char p2 = ip_image[yt * ip_pitch + xr]; // Top-Right
        const unsigned char p3 = ip_image[yb * ip_pitch + xl]; // Bottom-Left
        const unsigned char p4 = ip_image[yb * ip_pitch + xr]; // Bottom-Right

        // 進行雙線性內插
        op_image[h * op_pitch + w] = (unsigned char)(
            (1.0f - w_y) * ((1.0f - w_x) * p1 + w_x * p2) +
                 (w_y) * ((1.0f - w_x) * p3 + w_x * p4)
        );
    }
}

void ORB_GPU::Compute_pyramid(int height, int width,
                              int op_height, int op_width,
                              float inv_scale,
                              unsigned char* ip_image_data_gpu,
                              int ip_pitch,
                              unsigned char* op_image_data_gpu,
                              int op_pitch,
                              cudaStream_t& cuda_stream)
{

    int n_threads =  op_height * op_width;

#ifdef GRID_LAUNCH

    int CUDA_NUM_BLOCKS_x = (op_width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
    int CUDA_NUM_BLOCKS_y = (op_height - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

    dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
    dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);

    // imresize_GPU_pitched<<<grid_dim, block_dim, 0, cuda_stream>>>(
    //                                                                 n_threads,
    //                                                                 height, width,
    //                                                                 op_height, op_width,
    //                                                                 inv_scale,
    //                                                                 ip_image_data_gpu, ip_pitch,
    //                                                                 op_image_data_gpu, op_pitch);

    // 計算獨立的 x, y 縮放比例，以獲得最高精度
    const float scale_x = (float)width / op_width;
    const float scale_y = (float)height / op_height;
    
    imresize_GPU_pitched_robust<<<grid_dim, block_dim, 0, cuda_stream>>>(
        op_height, op_width,
        height, width,
        scale_x, scale_y,
        ip_image_data_gpu, ip_pitch,
        op_image_data_gpu, op_pitch);

#else
    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;


    imresize_GPU_pitched<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                            n_threads,
                                                                                            height, width,
                                                                                            op_height, op_width,
                                                                                            inv_scale,
                                                                                            ip_image_data_gpu, ip_pitch,
                                                                                            op_image_data_gpu, op_pitch);

#endif

}





__global__ void imresize_GPU(int n_threads,
                             int height, int width,
                             int op_height, int op_width,
                             float inv_scale,
                             unsigned char* ip_image,
                             unsigned char* op_image)
{
    int index = blockDim.x * blockIdx.x +  threadIdx.x;

    if(index < n_threads)
    {
        int h = index / op_width;
        int w = index % op_width;

        float scale = 1.0f / inv_scale;

        float ip_h = scale * h;
        float ip_w = scale * w;

        int xl = floorf(ip_w);
        int xr = xl + 1;

        int yt = floorf(ip_h);
        int yb = yt + 1;

        float w_xl = xr - ip_w;
        float w_xr = 1  - w_xl;
        float w_yt = yb - ip_h;
        float w_yb = 1  - w_yt;

        op_image[index] = w_yt * w_xl * ip_image[yt * width + xl]
                + w_yt * w_xr * ip_image[yt * width + xr]
                + w_yb * w_xl * ip_image[yb * width + xl]
                + w_yb * w_xr * ip_image[yb * width + xr];

    }

}



void ORB_GPU::Compute_pyramid(int height, int width,
                              int op_height, int op_width,
                              float inv_scale,
                              unsigned char* ip_image_data_gpu,
                              unsigned char* op_image_data_gpu,
                              cudaStream_t& cuda_stream)
{
    int n_threads =  op_height * op_width;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;


    imresize_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_stream>>>(
                                                                                    n_threads,
                                                                                    height, width,
                                                                                    op_height, op_width,
                                                                                    inv_scale,
                                                                                    ip_image_data_gpu,
                                                                                    op_image_data_gpu);
}




__global__ void dummy_GPU()
{

}

// without this first time kernel takes roughly 2uS more
void ORB_GPU::dummy_kernel_launch_tosetup_context()
{
    int n_threads =  512 * 512;

    int CUDA_NUM_BLOCKS = (n_threads + CUDA_NUM_THREADS_PER_BLOCK) / CUDA_NUM_THREADS_PER_BLOCK;

    for(int i=0;i<n_levels_;i++)
        dummy_GPU<<<CUDA_NUM_BLOCKS, CUDA_NUM_THREADS_PER_BLOCK, 0, cuda_streams_[i] >>>();

    for(int i=0;i<n_levels_;i++)
        cudaStreamSynchronize(cuda_streams_[i]);
}

}
