// [20261116] 延遲優化：在 GPU 上把 NMS 後分數 > 0 的點壓縮成清單，只傳清單回 CPU（取代整張分數圖的 GPU→CPU 複製＋CPU 逐像素掃描）
// 輸出格式與原本 CPU 掃描相同：座標已減去邊界（x - minBorder, y - minBorder），分數為 NMS 後的值
// 順序：GPU 寫入順序不固定 → CPU 端依 (y, x) 排序，和原本逐列掃描的順序完全相同，QuadTree 輸入不變
#include "cuda/orb_gpu.hpp"

namespace orb_cuda {

__global__ void CompactScoreKernel(const int* score, int pitch_elems,
                                   int minX, int minY, int maxX, int maxY,
                                   int3* out, int* counter, int capacity)
{
    const int x = minX + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = minY + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= maxX || y >= maxY) return;
    const int s = score[y * pitch_elems + x];
    if (s > 0) {
        const int k = atomicAdd(counter, 1);
        if (k < capacity) out[k] = make_int3(x - minX, y - minY, s);
    }
}

void compact_score_map(const int* score, int pitch_elems,
                       int minX, int minY, int maxX, int maxY,
                       int3* out, int* counter, int capacity,
                       cudaStream_t stream)
{
    if (maxX <= minX || maxY <= minY) return;
    dim3 block(32, 8);
    dim3 grid((maxX - minX + block.x - 1) / block.x, (maxY - minY + block.y - 1) / block.y);
    CompactScoreKernel<<<grid, block, 0, stream>>>(score, pitch_elems, minX, minY, maxX, maxY, out, counter, capacity);
}

} // namespace orb_cuda
