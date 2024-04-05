#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

__global__ void MemoryIntensiveKernel(float *input, int data_size);
__global__ void RegisterIntensiveKernel(float *input, float *output, int data_size);

#endif // KERNELS_CUH
