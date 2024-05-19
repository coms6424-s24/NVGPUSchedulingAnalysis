#include <cuda_runtime.h>
#include "gpu_sm_masking.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>

#define kMatrixSize 1024  // Matrix size
#define kNumIterations 100  // Number of iterations for each experiment

__global__ void MatrixMulKernel(float* c, float* a, float* b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        for (int e = 0; e < n; e++)
            value += a[row * n + e] * b[e * n + col];
        c[row * n + col] = value;
    }
}

void InitializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; i++)
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
}

void RunExperiment(int num_threads_per_block1, int num_threads_per_block2,
                   std::vector<double>& times, uint64_t mask1, uint64_t mask2, bool concurrent = true) {
    // Allocate matrices
    float *a1, *b1, *c1;
    float *a2, *b2, *c2;
    cudaMallocManaged(&a1, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&b1, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&c1, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&a2, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&b2, kMatrixSize * kMatrixSize * sizeof(float));
    cudaMallocManaged(&c2, kMatrixSize * kMatrixSize * sizeof(float));

    // Initialize matrices
    InitializeMatrix(a1, kMatrixSize);
    InitializeMatrix(b1, kMatrixSize);
    InitializeMatrix(a2, kMatrixSize);
    InitializeMatrix(b2, kMatrixSize);

    // Configure kernel launch parameters
    dim3 threads_per_block1(num_threads_per_block1, num_threads_per_block1);
    dim3 num_blocks1((kMatrixSize + threads_per_block1.x - 1) / threads_per_block1.x,
                        (kMatrixSize + threads_per_block1.y - 1) / threads_per_block1.y);
    dim3 threads_per_block2(num_threads_per_block2, num_threads_per_block2);
    dim3 num_blocks2((kMatrixSize + threads_per_block2.x - 1) / threads_per_block2.x,
                        (kMatrixSize + threads_per_block2.y - 1) / threads_per_block2.y);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Run the kernel multiple times and record the execution time
    for (int i = 0; i < kNumIterations; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        if (concurrent) {
            SetTPCMask(mask1);
            MatrixMulKernel<<<num_blocks1, threads_per_block1, 0, stream1>>>(c1, a1, b1, kMatrixSize);
            SetTPCMask(mask2);
            MatrixMulKernel<<<num_blocks2, threads_per_block2, 0, stream2>>>(c2, a2, b2, kMatrixSize);
        } else {
            SetTPCMask(mask1);
            MatrixMulKernel<<<num_blocks1, threads_per_block1, 0, stream1>>>(c1, a1, b1, kMatrixSize);
            cudaStreamSynchronize(stream1);
            SetTPCMask(mask2);
            MatrixMulKernel<<<num_blocks2, threads_per_block2, 0, stream2>>>(c2, a2, b2, kMatrixSize);
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
    cudaFree(a1);
    cudaFree(b1);
    cudaFree(c1);
    cudaFree(a2);
    cudaFree(b2);
    cudaFree(c2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
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

void RunAllScenarios() {
    std::cout << "Running experiments..." << std::endl;
    std::vector<double> times;
    uint32_t num_tpcs;
    GetTpcInfo(num_tpcs, 0);
    uint64_t half_mask = (1ULL << (num_tpcs / 2)) - 1;
    uint64_t quarter_mask = (1ULL << (num_tpcs / 4)) - 1;
    uint64_t three_quarter_mask = (1ULL << (3 * num_tpcs / 4)) - 1;

    // Independent Execution Baselines
    std::cout << "Running baseline for isolated MM1024..." << std::endl;
    RunExperiment(32, 32, times, 0, 0, false);
    CalculateStatistics(times, "Isolated MM1024");

    std::cout << "Running baseline for isolated MM256..." << std::endl;
    RunExperiment(16, 16, times, 0, 0, false);
    CalculateStatistics(times, "Isolated MM256");

    // Full Sharing Baseline
    std::cout << "Running baseline for MM1024 and MM256 - Full Sharing..." << std::endl;
    RunExperiment(32, 16, times, 0, 0, true);
    CalculateStatistics(times, "MM1024 and MM256 - Full Sharing");

    // Partitioning Scenarios
    auto run_partitioning_scenarios = [&](int num_threads_per_block1, int num_threads_per_block2,
                                          uint64_t mask1, uint64_t mask2,
                                          std::vector<double>& times,
                                          const char* description) {
        RunExperiment(num_threads_per_block1, num_threads_per_block2, times, mask1, mask2);
        CalculateStatistics(times, description);
    };

    // MM1024 (vs. MM1024) - Even Partitioning
    run_partitioning_scenarios(32, 32, half_mask, ~half_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM1024 (vs. MM1024) - Even Partitioning");

    // MM1024 (vs. MM1024) - Quarter Partitioning
    run_partitioning_scenarios(32, 32, quarter_mask, ~quarter_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM1024 (vs. MM1024) - Quarter Partitioning");

    // MM1024 (vs. MM256) - Even Partitioning
    run_partitioning_scenarios(32, 16, half_mask, ~half_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM1024 (vs. MM256) - Even Partitioning");

    // MM1024 (vs. MM256) - Quarter Partitioning
    run_partitioning_scenarios(32, 16, quarter_mask, ~quarter_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM1024 (vs. MM256) - Quarter Partitioning");

    // MM256 (vs. MM256) - Even Partitioning
    run_partitioning_scenarios(16, 16, half_mask, ~half_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM256 (vs. MM256) - Even Partitioning");

    // MM256 (vs. MM256) - Quarter Partitioning
    run_partitioning_scenarios(16, 16, quarter_mask, ~quarter_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM256 (vs. MM256) - Quarter Partitioning");

    // MM256 (vs. MM1024) - Even Partitioning
    run_partitioning_scenarios(16, 32, half_mask, ~half_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM256 (vs. MM1024) - Even Partitioning");

    // MM256 (vs. MM1024) - Quarter Partitioning
    run_partitioning_scenarios(16, 32, quarter_mask, ~quarter_mask & ((1ULL << num_tpcs) - 1),
                               times, "MM256 (vs. MM1024) - Quarter Partitioning");

    std::cout << "Experiments completed." << std::endl;
}

int main() {
    RunAllScenarios();
    return 0;
}
