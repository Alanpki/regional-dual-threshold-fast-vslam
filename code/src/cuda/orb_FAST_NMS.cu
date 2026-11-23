#include "cuda/orb_gpu.hpp"

namespace orb_cuda {

#define CUDA_NUM_THREADS_PER_BLOCK_x 32
#define CUDA_NUM_THREADS_PER_BLOCK_y 8
#define NMS_WINDOW_HALF 1 // For 3x3 window

__global__ void FASTapplyNMSGlobalGPU(int imheight, int imwidth,
                                      const int* score_data_gpu,
                                      int score_pitch,
                                      int* score_data_nms,
                                      int score_nms_pitch)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    // Standard boundary check
    if (w >= NMS_WINDOW_HALF && w < imwidth - NMS_WINDOW_HALF && 
        h >= NMS_WINDOW_HALF && h < imheight - NMS_WINDOW_HALF)
    {
        int score = score_data_gpu[h * score_pitch + w];

        // If the score is already zero, no need to check
        if (score == 0) {
            score_data_nms[h * score_nms_pitch + w] = 0;
            return;
        }

        // 3x3 NMS check
        // This unrolled version is fast and efficient
        int row_offset = (h - 1) * score_pitch;
        int col_offset = w - 1;
        int total_offset = row_offset + col_offset;
        if (score < score_data_gpu[total_offset] || score < score_data_gpu[total_offset+1] || score < score_data_gpu[total_offset+2]) {
             score_data_nms[h * score_nms_pitch + w] = 0;
             return;
        }

        row_offset = h * score_pitch;
        total_offset = row_offset + col_offset;
        if (score < score_data_gpu[total_offset] || score < score_data_gpu[total_offset+2]) { // Skip checking against itself
             score_data_nms[h * score_nms_pitch + w] = 0;
             return;
        }

        row_offset = (h + 1) * score_pitch;
        total_offset = row_offset + col_offset;
        if (score < score_data_gpu[total_offset] || score < score_data_gpu[total_offset+1] || score < score_data_gpu[total_offset+2]) {
             score_data_nms[h * score_nms_pitch + w] = 0;
             return;
        }

        // If it's a local maximum, write the score
        score_data_nms[h * score_nms_pitch + w] = score;
    }
    // Optional: Handle border pixels if necessary, here we just leave them as they are or zero them out
    else if (w < imwidth && h < imheight) {
         score_data_nms[h * score_nms_pitch + w] = 0;
    }
}

void ORB_GPU::FAST_apply_NMS_global(int height, int width,
                                    int* score_data_gpu,
                                    int score_pitch,
                                    int* score_data_nms,
                                    int score_nms_pitch,
                                    cudaStream_t& cuda_stream)
{
    int CUDA_NUM_BLOCKS_x = (width  - 1) / CUDA_NUM_THREADS_PER_BLOCK_x + 1;
    int CUDA_NUM_BLOCKS_y = (height - 1) / CUDA_NUM_THREADS_PER_BLOCK_y + 1;

    dim3 grid_dim(CUDA_NUM_BLOCKS_x, CUDA_NUM_BLOCKS_y, 1);
    dim3 block_dim(CUDA_NUM_THREADS_PER_BLOCK_x, CUDA_NUM_THREADS_PER_BLOCK_y, 1);

    // Initialize the output NMS buffer with zeros
    cudaMemsetAsync(score_data_nms, 0, height * score_nms_pitch, cuda_stream);

    FASTapplyNMSGlobalGPU<<<grid_dim, block_dim, 0, cuda_stream>>>(
                                                               height, width,
                                                               score_data_gpu,
                                                               score_pitch,
                                                               score_data_nms,
                                                               score_nms_pitch);
}

} // namespace orb_cuda