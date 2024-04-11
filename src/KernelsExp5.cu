#include "../include/KernelsExp5.cuh"
#include <device_launch_parameters.h>
#include <math_constants.h>

const unsigned long long target_duration = 4000000000ULL; // 4e9 clock ticks ≈ 1 second

// Kernel 1: Performs trigonometric operations on a float value.
__global__ void kernel1()
{
    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(idx) * 1.5f;

    while (clock64() < end_clock) {
        value = sinf(value) * cosf(value);
    }
}

// Kernel 2: Performs logarithmic and exponential operations on a double value.
__global__ void kernel2()
{
    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = static_cast<double>(idx) * 2.5;

    while (clock64() < end_clock) {
        value = log(value) * exp(value);
    }
}

// Kernel 3: Performs square root and power operations on a double value.
__global__ void kernel3()
{
    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = static_cast<double>(idx) * 3.5;

    while (clock64() < end_clock) {
        value = sqrt(value) * pow(value, 2.0);
    }
}

// Kernel 4: Performs trigonometric operations on a float value using shared memory.
__global__ void kernel4()
{
    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration;

    __shared__ float sharedMem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedMem[threadIdx.x] = static_cast<float>(idx) * 4.5f;
    __syncthreads();

    while (clock64() < end_clock) {
        sharedMem[threadIdx.x] = sinf(sharedMem[threadIdx.x]) * cosf(sharedMem[threadIdx.x]);
    }
}

// Kernel 5: Performs logarithmic and exponential operations on a double value using shared memory.
__global__ void kernel5()
{
    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration;

    __shared__ double sharedMem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sharedMem[threadIdx.x] = static_cast<double>(idx) * 5.5;
    __syncthreads();

    while (clock64() < end_clock) {
        sharedMem[threadIdx.x] = log(sharedMem[threadIdx.x]) * exp(sharedMem[threadIdx.x]);
    }
}

// Kernel 6: Performs square root and power operations on a float value.
__global__ void kernel6()
{
    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(idx) * 6.5f;

    while (clock64() < end_clock) {
        value = sqrtf(value) * powf(value, 3.0f);
    }
}