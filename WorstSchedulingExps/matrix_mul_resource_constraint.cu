#include <cuda_runtime.h>
#include "gpu_sm_masking.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

#define kMatrixSize 1024  // Matrix size
#define kNumIterations 100  // Number of iterations for each experiment

__global__ void MatrixMulKernel(float* c, float* a, float* b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        for (int e = 0; e < n; ++e)
            value += a[row * n + e] * b[e * n + col];
        c[row * n + col] = value;
    }
}

void InitializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i)
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
}

void RunKernel(dim3 threads_per_block, dim3 num_blocks, std::vector<double>& times) {
    // Allocate matrices
    float *a, *b, *c;
    cudaMallocManaged(&a, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&b, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&c, kMatrixSize * kMatrixSize * sizeof(float));

    // Initialize matrices
    InitializeMatrix(a, kMatrixSize);
    InitializeMatrix(b, kMatrixSize);

    // Run the kernel multiple times and record the execution time
    for (int i = 0; i < kNumIterations; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        MatrixMulKernel<<<num_blocks, threads_per_block>>>(c, a, b, kMatrixSize);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

void RunMultipleKernels(dim3 threads_per_block, dim3 num_blocks, int num_streams, std::vector<double>& times) {
    // Allocate matrices
    float *a, *b, *c;
    cudaMallocManaged(&a, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&b, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&c, kMatrixSize * kMatrixSize * sizeof(float));

    // Initialize matrices
    InitializeMatrix(a, kMatrixSize);
    InitializeMatrix(b, kMatrixSize);

    // Create streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Run the kernels multiple times and record the execution time
    for (int i = 0; i < kNumIterations; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int s = 0; s < num_streams; ++s) {
            MatrixMulKernel<<<num_blocks, threads_per_block, 0, streams[s]>>>(c, a, b, kMatrixSize);
        }
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    // Destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

void CalculateStatistics(const std::vector<double>& times, const char* description) {
    int num_samples = times.size();
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / num_samples;

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median = (num_samples % 2 == 0) ? (sorted_times[num_samples / 2 - 1] + sorted_times[num_samples / 2]) / 2.0
                                          : sorted_times[num_samples / 2];

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / num_samples - mean * mean);

    std::cout << description << std::endl;
    std::cout << "# Samples: " << num_samples << std::endl;
    std::cout << "Min: " << min_time << " ms" << std::endl;
    std::cout << "Max: " << max_time << " ms" << std::endl;
    std::cout << "Median: " << median << " ms" << std::endl;
    std::cout << "Arithmetic Mean: " << mean << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdev << " ms" << std::endl;
}

void QueryCudaProperties() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
}

int main() {
    QueryCudaProperties();

    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    int max_threads_per_block = prop.maxThreadsPerBlock;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    int warp_size = prop.warpSize;

    std::vector<double> times;

    // Experiment 1: Different Thread and Block Numbers
    std::cout << "Experiment 1: Different Thread and Block Numbers" << std::endl;

    // Underutilized Kernel
    dim3 underutilized_threads_per_block = dim3((max_threads_per_sm / 2) + warp_size, 1);
    dim3 underutilized_num_blocks = dim3((kMatrixSize + underutilized_threads_per_block.x - 1) / underutilized_threads_per_block.x,
                                       (kMatrixSize + underutilized_threads_per_block.y - 1) / underutilized_threads_per_block.y);
    RunKernel(underutilized_threads_per_block, underutilized_num_blocks, times);
    CalculateStatistics(times, "Underutilized Kernel");

    // Fully Utilized Kernel
    times.clear();
    dim3 fully_utilized_threads_per_block = dim3(sqrt(max_threads_per_block), sqrt(max_threads_per_block));
    dim3 fully_utilized_num_blocks = dim3((kMatrixSize + fully_utilized_threads_per_block.x - 1) / fully_utilized_threads_per_block.x,
                                       (kMatrixSize + fully_utilized_threads_per_block.y - 1) / fully_utilized_threads_per_block.y);
    RunKernel(fully_utilized_threads_per_block, fully_utilized_num_blocks, times);
    CalculateStatistics(times, "Fully Utilized Kernel");

    // Experiment 2: Single Large Kernel vs. Multiple Small Kernels
    std::cout << "Experiment 2: Single Large Kernel vs. Multiple Small Kernels" << std::endl;

    // Single large kernel
    times.clear();
    RunKernel(dim3(32, 32), dim3((kMatrixSize + 31) / 32, (kMatrixSize + 31) / 32), times);
    CalculateStatistics(times, "Single large kernel");

    // Multiple small kernels with 1 stream
    times.clear();
    RunMultipleKernels(dim3(32, 32), dim3((kMatrixSize + 31) / 32, (kMatrixSize + 31) / 32), 1, times);
    CalculateStatistics(times, "Multiple small kernels with 1 stream");

    // Multiple small kernels with 2 streams
    times.clear();
    RunMultipleKernels(dim3(32, 32), dim3((kMatrixSize + 31) / 32, (kMatrixSize + 31) / 32), 2, times);
    CalculateStatistics(times, "Multiple small kernels with 2 streams");

    // Multiple small kernels with 4 streams
    times.clear();
    RunMultipleKernels(dim3(32, 32), dim3((kMatrixSize + 31) / 32, (kMatrixSize + 31) / 32), 4, times);
    CalculateStatistics(times, "Multiple small kernels with 4 streams");

    return 0;
}