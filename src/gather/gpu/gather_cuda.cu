#include <cuda_fp16.h>
#include <stdio.h>
template <typename T>
__global__ void gatherKernel0_v2(T const*input, T* output, int const* indices, int inputShape0, int inputShape1, int indicesShape0, int indicesShape1, int inputAxis, int indicesAxis)
{
  int outputShape[2];
  outputShape[0] = indicesShape0;
  outputShape[1] = indicesShape1 * inputShape1;
  
  int j = threadIdx.x + blockIdx.x*blockDim.x;
  int k = (threadIdx.y + blockIdx.y*blockDim.y) << 2;

  int t0 = j * inputShape1;
  for (int i = 0; i < indicesShape0; i += 4) {
      int indice_0 = indices[(i+0) * indicesShape1 + j];
      int t1_0 = indice_0 * inputShape1;
      int t2_0 = (i+0) * outputShape[1];
      if constexpr (std::is_same<T, float>::value) (float4 &)output[t2_0 + (t0 + k)] = (float4 &)input[t1_0 + k];
      else if constexpr (std::is_same<T, half>::value) (float2 &)output[t2_0 + (t0 + k)] = (float2 &)input[t1_0 + k];

      int indice_1 = indices[(i+1) * indicesShape1 + j];
      int t1_1 = indice_1 * inputShape1;
      int t2_1 = (i+1) * outputShape[1];
      if constexpr (std::is_same<T, float>::value) (float4 &)output[t2_1 + (t0 + k)] = (float4 &)input[t1_1 + k];
      else if constexpr (std::is_same<T, half>::value) (float2 &)output[t2_1 + (t0 + k)] = (float2 &)input[t1_1 + k];

      int indice_2 = indices[(i+2) * indicesShape1 + j];
      int t1_2 = indice_2 * inputShape1;
      int t2_2 = (i+2) * outputShape[1];
      if constexpr (std::is_same<T, float>::value) (float4 &)output[t2_2 + (t0 + k)] = (float4 &)input[t1_2 + k];
      else if constexpr (std::is_same<T, half>::value) (float2 &)output[t2_2 + (t0 + k)] = (float2 &)input[t1_2 + k];

      int indice_3 = indices[(i+3) * indicesShape1 + j];
      int t1_3 = indice_3 * inputShape1;
      int t2_3 = (i+3) * outputShape[1];
      if constexpr (std::is_same<T, float>::value) (float4 &)output[t2_3 + (t0 + k)] = (float4 &)input[t1_3 + k];
      else if constexpr (std::is_same<T, half>::value) (float2 &)output[t2_3 + (t0 + k)] = (float2 &)input[t1_3 + k];
  }
}

template <typename T>
__global__ void gatherKernel01(T const*input, T* output, int const* indices, int inputShape0, int inputShape1, int indicesShape0, int indicesShape1, int inputAxis, int indicesAxis)
{
  int outputShape[2];
  int indice;
  outputShape[0] = indicesShape0;
  outputShape[1] = indicesShape1 * inputShape1;
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  for (int j = 0; j < indicesShape1; j++) {
    indice = indices[i * indicesShape1 + j];
    for (int k = 0; k < inputShape1; k++) {
      output[i * outputShape[1] + j * inputShape1 + k] = input[(indice * inputShape1) + k];
    }
  }
}

template <typename T>
__global__ void gatherKernel1(T const*input, T* output, int const* indices, int inputShape0, int inputShape1, int indicesShape0, int indicesShape1, int inputAxis, int indicesAxis)
{
  int outputShape[2];
  int indice;
  outputShape[0] = indicesShape0 * inputShape0;
  outputShape[1] = indicesShape1;
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  for (int j = 0; j < indicesShape1; j++) {
    indice = indices[i * indicesShape1 + j];
    for (int k = 0; k < inputShape0; k++) {
      output[(i * inputShape0 + k) * outputShape[1] + j] = input[(k * inputShape1) + indice];
    }
  }
}

template <typename T>
void gatherLaunch(void const*input, void* output, void const* indices, int inputShape0, int inputShape1, int indicesShape0, int indicesShape1, int inputAxis, int indicesAxis, int axis) {
  if (axis == 0) {
    if (inputShape1 > 32) {
    int BLOCK_DIM_x = 16;
    int num_block_x = (indicesShape1 + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int BLOCK_DIM_y = 32;
    int num_block_y = ((inputShape1>>2) + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    int blockSize = BLOCK_DIM_x * BLOCK_DIM_y;
  
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
  
    gatherKernel0_v2<T><<<grid_dim, block_dim>>>((T const*)input, (T*)output, (int const*)indices, inputShape0, inputShape1, indicesShape0, indicesShape1, inputAxis, indicesAxis);
    }
    else {
      int BLOCK_DIM_x = 1;
      int num_block_x = (indicesShape1 + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
      int blockSize = BLOCK_DIM_x;
  
      dim3 grid_dim(num_block_x, 1, 1);
      dim3 block_dim(BLOCK_DIM_x, 1, 1);
 
      gatherKernel01<T><<<grid_dim, block_dim>>>((T const*)input, (T*)output, (int const*)indices, inputShape0, inputShape1, indicesShape0, indicesShape1, inputAxis, indicesAxis);
    }
  }
  else{ 
    int BLOCK_DIM_x = 32;
    int num_block_x = (indicesShape1 + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int blockSize = BLOCK_DIM_x;
  
    dim3 grid_dim(num_block_x, 1, 1);
    dim3 block_dim(BLOCK_DIM_x, 1, 1);
 
    gatherKernel1<T><<<grid_dim, block_dim>>>((T const*)input, (T*)output, (int const*)indices, inputShape0, inputShape1, indicesShape0, indicesShape1, inputAxis, indicesAxis);
  }
  cudaDeviceSynchronize();

  // int device;
  // cudaGetDevice(&device); 
  // int maxActiveBlocks;
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, (void*)gatherKernel0_v2<float>, blockSize, 0);
  // int blockThreads;
  // int smThreads;
  // cudaOccupancyMaxPotentialBlockSize(&blockThreads, &smThreads, (void*)gatherKernel0_v2<float>, 0, 0);
  // printf("Max Active Blocks per SM: %d\n", maxActiveBlocks);
  // printf("Threads per Block: %d\n", blockSize);
  // printf("Max Threads per SM: %d\n" , smThreads);
  // float occupancy = (float)(maxActiveBlocks * blockSize) / smThreads;
  // printf("Occupancy: %f \n", occupancy * 100);
  //cudaError_t err = cudaGetLastError();
  //int sharedMemPerBlock;
  //cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  //printf("max shared memory: %d\n", sharedMemPerBlock);
  //if (err != cudaSuccess) {
  //    printf("CUDA error: %s\n", cudaGetErrorString(err));
  //} else {
  //    printf("CUDA kernel launched successfully!\n");
  //}
}

extern "C" void gather_nv_f32(void const*input, void* output, void const* indices, int inputShape0, int inputShape1, int indicesShape0, int indicesShape1, int inputAxis, int indicesAxis, int axis)
{
    gatherLaunch<float>(input, output, indices, inputShape0, inputShape1, indicesShape0, indicesShape1, inputAxis, indicesAxis, axis);
}
extern "C" void gather_nv_f16(void const*input, void* output, void const* indices, int inputShape0, int inputShape1, int indicesShape0, int indicesShape1, int inputAxis, int indicesAxis, int axis)
{
     gatherLaunch<half>(input, output, indices, inputShape0, inputShape1, indicesShape0, indicesShape1, inputAxis, indicesAxis, axis);
}
