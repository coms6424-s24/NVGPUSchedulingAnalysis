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

#define N 1024  // Matrix size
#define NUM_ITERATIONS 1  // Number of iterations for each experiment

__global__ void matrixMulKernel(float* C, float* A, float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        for (int e = 0; e < n; ++e)
            value += A[row * n + e] * B[e * n + col];
        C[row * n + col] = value;
    }
}

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i)
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
}

void runKernel(int numThreadsPerBlock, std::vector<double>& times, uint64_t mask = 0) {
    // Allocate matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, N * N * sizeof(float));
    cudaMallocManaged(&B, N * N * sizeof(float));
    cudaMallocManaged(&C, N * N * sizeof(float));

    // Initialize matrices
    initializeMatrix(A, N);
    initializeMatrix(B, N);

    // Configure kernel launch parameters
    dim3 threadsPerBlock(numThreadsPerBlock, numThreadsPerBlock);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Set the TPC mask if provided
    // if (mask != 0) 
    {
        SetTPCMask(mask);
    }

    // Run the kernel multiple times and record the execution time
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        matrixMulKernel<<<numBlocks, threadsPerBlock>>>(C, A, B, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void calculateStatistics(const std::vector<double>& times, const char* description) {
    int numSamples = times.size();
    double minTime = *std::min_element(times.begin(), times.end());
    double maxTime = *std::max_element(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / numSamples;

    std::vector<double> sortedTimes = times;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    double median = (numSamples % 2 == 0) ? (sortedTimes[numSamples / 2 - 1] + sortedTimes[numSamples / 2]) / 2.0
                                          : sortedTimes[numSamples / 2];

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / numSamples - mean * mean);

    std::cout << description << std::endl;
    std::cout << "# Samples: " << numSamples << std::endl;
    std::cout << "Min: " << minTime << " ms" << std::endl;
    std::cout << "Max: " << maxTime << " ms" << std::endl;
    std::cout << "Median: " << median << " ms" << std::endl;
    std::cout << "Arithmetic Mean: " << mean << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdev << " ms" << std::endl;
}

int main() {
    std::vector<double> timesOriginal, timesMasked;
    uint32_t numTpcs;
    GetTpcInfo(numTpcs, 0);
    uint64_t halfMask = (1ULL << (numTpcs / 2)) - 1;

    // Run original kernel
    std::cout << "Running original kernel..." << std::endl;
    runKernel(32, timesOriginal);
    calculateStatistics(timesOriginal, "Original Kernel");

    // Run masked kernel
    std::cout << "Running masked kernel (half TPCs)..." << std::endl;
    runKernel(32, timesMasked, halfMask);
    calculateStatistics(timesMasked, "Masked Kernel (Half TPCs)");

    return 0;
}
