#include <cuda.h>
#include <cub/cub.cuh>

inline int getdim(int num) {
  if (num > 31) return 32;
  else if (num > 15) return 16;
  else if (num > 7) return 8;
  return 4;
}

inline int getT(int num) {
  if (num > 128) return 4;
  else if (num > 16) return 2;
  return 1;
}

template <typename T, typename Tind>
__global__ void warpGatherKernel_v1(T const *input, Tind const *indices, T *output, int stride, int indSize)
{
    int otherIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * indSize;
    output[tid + index * stride] = input[tid + indices[index] * stride];
}



template <typename T, typename Tind>
void gatherLaunch_v1(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    int BLOCK_DIM_x = getdim(othersize);
    int BLOCK_DIM_y = getdim(indSize);
    int num_block_x = (othersize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_block_y = (indSize + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    warpGatherKernel_v1<T, Tind>
        <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize);
}

template <typename T, typename Tind>
__global__ void warpGatherKernel_v2(T const *input, Tind const *indices, T *output, int stride, int indSize, int TM, int TN)
{
    int otherIdx = (blockIdx.x * blockDim.x + threadIdx.x) * TM;
    int index = (blockIdx.y * blockDim.y + threadIdx.y) * TN;
    
    __shared__ int s_indices[1024];
    #pragma unroll 
    for (int cycle = 0; cycle < (TN - 1)/ blockDim.x; cycle++) {
        int indx = cycle * blockDim.x + threadIdx.x;
        s_indices[threadIdx.y * TN + indx] = indices[index + indx];
    }

    if (threadIdx.x < TN % blockDim.x) {
        s_indices[threadIdx.y * TN + threadIdx.x] = indices[index + threadIdx.x];
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int otherIdx_ = otherIdx + i;
        int tid = otherIdx_ % stride + (otherIdx_ - otherIdx_ % stride) * indSize; // tid = s + i(BCD)
        #pragma unroll
        for (int j = 0; j < TN; j++) {
	  int index_ = index + j;
           output[tid + index_ * stride] = input[tid + s_indices[threadIdx.y * TN + j] * stride];
	}
    }
}



template <typename T, typename Tind>
void gatherLaunch_v2(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    int TM = getT(othersize);
    int TN = getT(indSize);
    int othersize_ = othersize / TM;
    int indSize_ = indSize / TN;
    int BLOCK_DIM_x = getdim(othersize_);
    int BLOCK_DIM_y = getdim(indSize_);
    int num_block_x = (othersize_ + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_block_y = (indSize_ + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    warpGatherKernel_v2<T, Tind>
        <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize, TM, TN);
}

template <typename T, typename Tind>
__global__ void warpGatherKernel_v3(T const *input, Tind const *indices, T *output, int stride, int indSize, int TM, int TN)
{
    int otherIdx = (blockIdx.x * blockDim.x + threadIdx.x) * TM;
    int index = (blockIdx.y * blockDim.y + threadIdx.y) * TN;
    
    __shared__ int s_indices[1024];
    #pragma unroll 
    for (int cycle = 0; cycle < (TN - 1)/ blockDim.x; cycle++) {
        int indx = cycle * blockDim.x + threadIdx.x;
        s_indices[threadIdx.y * TN + indx] = indices[index + indx];
    }

    if (threadIdx.x < TN % blockDim.x) {
        s_indices[threadIdx.y * TN + threadIdx.x] = indices[index + threadIdx.x];
    }
    __syncthreads();

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * indSize; // tid = s + i(BCD)
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int index_ = index + j;
      if constexpr (std::is_same<T, float>::value) {
          if (TM == 4)(float4 &)output[tid + index_ * stride] = (float4 &)input[tid + s_indices[threadIdx.y * TN + j] * stride];
          else if (TM == 2)(float2 &)output[tid + index_ * stride] = (float2 &)input[tid + s_indices[threadIdx.y * TN + j] * stride];
          else (float &)output[tid + index_ * stride] = (float &)input[tid + s_indices[threadIdx.y * TN + j] * stride];
      } else {
          if (TM == 4)(float2 &)output[tid + index_ * stride] = (float2 &)input[tid + s_indices[threadIdx.y * TN + j] * stride];
          else if (TM == 2)(half2 &)output[tid + index_ * stride] = (half2 &)input[tid + s_indices[threadIdx.y * TN + j] * stride];
          else (half &)output[tid + index_ * stride] = (half &)input[tid + s_indices[threadIdx.y * TN + j] * stride];
      }
    }
}

template <typename T, typename Tind>
void gatherLaunch_v3(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    int TM = getT(othersize);
    int TN = getT(indSize);
    // printf("stride: %d, indSize: %d, othersize: %d, TM: %d, TN: %d\n", stride, indSize, othersize, TM, TN);
    int othersize_ = othersize / TM;
    int indSize_ = indSize / TN;
    int BLOCK_DIM_x = getdim(othersize_);
    int BLOCK_DIM_y = getdim(indSize_);
    int num_block_x = (othersize_ + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_block_y = (indSize_ + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    warpGatherKernel_v3<T, Tind>
        <<<grid_dim, block_dim>>>((T *)input, (Tind *)indices, (T *)output, stride, indSize, TM, TN);
}


extern "C" void gather_nv_f32(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    gatherLaunch_v3<float, uint64_t>(input, indices, output, stride, indSize, othersize);
}
extern "C" void gather_nv_f16(void const *input, void const *indices, void *output, int stride, int indSize, int othersize)
{
    gatherLaunch_v3<half, uint64_t>(input, indices, output, stride, indSize, othersize);
}
