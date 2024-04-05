#include "../include/KernelsExp4.cuh"

__global__ void MemoryIntensiveKernel(float *input, int data_size) {
    extern __shared__ float shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < data_size) {
        shared[tid] = input[idx];

        __syncthreads();

        float temp = shared[tid];
        temp = exp(sin(temp) + cos(temp) + log(fabs(temp) + 1.0f) + exp(temp));

        input[idx] = temp;
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

