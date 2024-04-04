#include "../include/KernelsExp4.cuh"

__global__ void MemoryIntensiveKernel(float *input, float *output, int data_size)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = 5; // Convolution kernel radius
    int diameter = 2 * radius + 1;

    // Load data into shared memory
    if (idx < data_size) {
        shared[tid] = input[idx];
        __syncthreads();

        float sum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            int sharedIdx = tid + i;
            if (sharedIdx >= 0 && sharedIdx < blockDim.x) {
                sum += shared[sharedIdx] * (radius - abs(i) + 1);
            }
        }
        output[idx] = sum / diameter;

        __syncthreads();
    }
}

__global__ void RegisterIntensiveKernel(float *input, float *output, int data_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        float a = input[idx];
        float b = sin(a) + cos(a);
        float c = b * tan(a);
        float d = sqrt(abs(b - c));
        float e = log(d + 1.0f);
        float f = e * sin(b) * cos(c) * tan(d);
        output[idx] = f;
    }
}

