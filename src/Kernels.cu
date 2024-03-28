#include "../include/Kernels.cuh"

// Kernel 1: Performs trigonometric operations on a float value.
__global__ void kernel1()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(idx) * 1.5f;
    for (int i = 0; i < 1e3; i++) {
        value = sin(value) * cos(value);
    }
}

// Kernel 2: Performs logarithmic and exponential operations on a double value.
__global__ void kernel2()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = static_cast<double>(idx) * 2.5;
    for (int i = 0; i < 8e2; i++) {
        value = log(value) * exp(value);
    }
}

// Kernel 3: Performs square root and power operations on a double value.
__global__ void kernel3()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = static_cast<double>(idx) * 3.5;
    for (int i = 0; i < 1.2e3; i++) {
        value = sqrt(value) * pow(value, 2.0);
    }
}

// Kernel 4: Performs trigonometric operations on a float value using shared memory.
__global__ void kernel4()
{
    __shared__ float sharedMem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedMem[threadIdx.x] = static_cast<float>(idx) * 4.5f;
    __syncthreads();
    for (int i = 0; i < 6e2; i++) {
        sharedMem[threadIdx.x] = sin(sharedMem[threadIdx.x]) * cos(sharedMem[threadIdx.x]);
    }
}

// Kernel 5: Performs logarithmic and exponential operations on a double value using shared memory.
__global__ void kernel5()
{
    __shared__ double sharedMem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedMem[threadIdx.x] = static_cast<double>(idx) * 5.5;
    __syncthreads();
    for (int i = 0; i < 4e2; i++) {
        sharedMem[threadIdx.x] = log(sharedMem[threadIdx.x]) * exp(sharedMem[threadIdx.x]);
    }
}

// Kernel 6: Performs square root and power operations on a float value.
__global__ void kernel6()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(idx) * 6.5f;
    for (int i = 0; i < 9e2; i++) {
        value = sqrt(value) * pow(value, 3.0f);
    }
}