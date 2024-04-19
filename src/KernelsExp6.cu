#include "../include/KernelsExp6.cuh"
#include <device_launch_parameters.h>
#include <math_constants.h>

// Kernel 1: Performs trigonometric operations on a float value.
__global__ void kernel1()
{
    const unsigned long long target_duration1 = 4000000000ULL; // 4e9 clock ticks

    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(idx) * 1.5f;

    while (clock64() < end_clock) {
        value = value * value;
    }
}

// Kernel 2: Performs trigonometric operations on a float value.
__global__ void kernel2()
{
    const unsigned long long target_duration2 = 2000000000ULL; // 2e9 clock ticks

    unsigned long long start_clock = clock64();
    unsigned long long end_clock = start_clock + target_duration2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(idx) * 1.5f;

    while (clock64() < end_clock) {
        value = value * value;
    }
}