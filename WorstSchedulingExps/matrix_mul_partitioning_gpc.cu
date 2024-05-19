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
        for (int e = 0; e < n; ++e)
            value += a[row * n + e] * b[e * n + col];
        c[row * n + col] = value;
    }
}

void InitializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i)
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
}

void CheckCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void RunExperiment(int num_threads_per_block, std::vector<double>& times, uint64_t mask) {
    // Allocate matrices
    float *a, *b, *c;
    CheckCudaError(cudaMallocManaged(&a, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating A");
    CheckCudaError(cudaMallocManaged(&b, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating B");
    CheckCudaError(cudaMallocManaged(&c, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating C");

    // Initialize matrices
    InitializeMatrix(a, kMatrixSize);
    InitializeMatrix(b, kMatrixSize);

    // Configure kernel launch parameters
    dim3 threads_per_block(num_threads_per_block, num_threads_per_block);
    dim3 num_blocks((kMatrixSize + threads_per_block.x - 1) / threads_per_block.x,
                    (kMatrixSize + threads_per_block.y - 1) / threads_per_block.y);

    // Print the mask being used
    std::cout << "Running experiment with mask: " << std::hex << mask << std::dec << std::endl;

    // Ensure the mask enables at least one TPC
    if (mask == ~0ULL) {
        std::cerr << "Error: Mask disables all TPCs. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Run the kernel multiple times and record the execution time
    for (int i = 0; i < kNumIterations; ++i) {
        // Set the TPC mask
        SetTPCMask(mask);

        cudaEvent_t start, stop;
        CheckCudaError(cudaEventCreate(&start), "Creating start event");
        CheckCudaError(cudaEventCreate(&stop), "Creating stop event");

        CheckCudaError(cudaEventRecord(start), "Recording start event");
        MatrixMulKernel<<<num_blocks, threads_per_block>>>(c, a, b, kMatrixSize);
        CheckCudaError(cudaEventRecord(stop), "Recording stop event");

        CheckCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

        float milliseconds = 0;
        CheckCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
        times.push_back(milliseconds);

        CheckCudaError(cudaEventDestroy(start), "Destroying start event");
        CheckCudaError(cudaEventDestroy(stop), "Destroying stop event");
    }

    // Free memory
    CheckCudaError(cudaFree(a), "Freeing A");
    CheckCudaError(cudaFree(b), "Freeing B");
    CheckCudaError(cudaFree(c), "Freeing C");
}

void RunConcurrentExperiment(int num_threads_per_block_1, int num_threads_per_block_2, 
                             std::vector<double>& times, uint64_t mask1, uint64_t mask2) {
    // Allocate matrices
    float *a1, *b1, *c1, *a2, *b2, *c2;
    CheckCudaError(cudaMallocManaged(&a1, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating A1");
    CheckCudaError(cudaMallocManaged(&b1, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating B1");
    CheckCudaError(cudaMallocManaged(&c1, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating C1");
    CheckCudaError(cudaMallocManaged(&a2, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating A2");
    CheckCudaError(cudaMallocManaged(&b2, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating B2");
    CheckCudaError(cudaMallocManaged(&c2, kMatrixSize * kMatrixSize * sizeof(float)), "Allocating C2");

    // Initialize matrices
    InitializeMatrix(a1, kMatrixSize);
    InitializeMatrix(b1, kMatrixSize);
    InitializeMatrix(a2, kMatrixSize);
    InitializeMatrix(b2, kMatrixSize);

    // Configure kernel launch parameters
    dim3 threads_per_block_1(num_threads_per_block_1, num_threads_per_block_1);
    dim3 num_blocks_1((kMatrixSize + threads_per_block_1.x - 1) / threads_per_block_1.x,
                      (kMatrixSize + threads_per_block_1.y - 1) / threads_per_block_1.y);

    dim3 threads_per_block_2(num_threads_per_block_2, num_threads_per_block_2);
    dim3 num_blocks_2((kMatrixSize + threads_per_block_2.x - 1) / threads_per_block_2.x,
                      (kMatrixSize + threads_per_block_2.y - 1) / threads_per_block_2.y);

    // Print the masks being used
    std::cout << "Running concurrent experiment with masks: " << std::hex << mask1 << " and " << mask2 << std::dec << std::endl;

    // Ensure the masks enable at least one TPC each
    if (mask1 == ~0ULL || mask2 == ~0ULL) {
        std::cerr << "Error: Mask disables all TPCs. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Run the kernels multiple times and record the execution time
    for (int i = 0; i < kNumIterations; ++i) {
        cudaEvent_t start, stop;
        CheckCudaError(cudaEventCreate(&start), "Creating start event");
        CheckCudaError(cudaEventCreate(&stop), "Creating stop event");

        CheckCudaError(cudaEventRecord(start), "Recording start event");

        // Launch first kernel with mask1
        SetTPCMask(mask1);
        MatrixMulKernel<<<num_blocks_1, threads_per_block_1>>>(c1, a1, b1, kMatrixSize);

        // Launch second kernel with mask2
        SetTPCMask(mask2);
        MatrixMulKernel<<<num_blocks_2, threads_per_block_2>>>(c2, a2, b2, kMatrixSize);

        CheckCudaError(cudaEventRecord(stop), "Recording stop event");
        CheckCudaError(cudaEventSynchronize(stop), "Synchronizing stop event");

        float milliseconds = 0;
        CheckCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Calculating elapsed time");
        times.push_back(milliseconds);

        CheckCudaError(cudaEventDestroy(start), "Destroying start event");
        CheckCudaError(cudaEventDestroy(stop), "Destroying stop event");
    }

    // Free memory
    CheckCudaError(cudaFree(a1), "Freeing A1");
    CheckCudaError(cudaFree(b1), "Freeing B1");
    CheckCudaError(cudaFree(c1), "Freeing C1");
    CheckCudaError(cudaFree(a2), "Freeing A2");
    CheckCudaError(cudaFree(b2), "Freeing B2");
    CheckCudaError(cudaFree(c2), "Freeing C2");
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
    std::vector<double> times;

    // Define the CU masks for different partitioning strategies
    uint64_t full_shared_mask = 0x0ULL;  // No TPCs disabled for full GPU sharing
    uint64_t packed_equal_mask1 = 0x5555555555555555ULL; // 0101 pattern, enable every even TPC
    uint64_t packed_equal_mask2 = 0xAAAAAAAAAAAAAAAAULL; // 1010 pattern, enable every odd TPC
    uint64_t packed_unequal_mask1 = 0x5555555555555554ULL; // 0101...0100 pattern, add one more TPC to first
    uint64_t packed_unequal_mask2 = 0xAAAAAAAAAAAAAAAAULL; // 1010 pattern, enable every odd TPC
    uint64_t distributed_equal_mask1 = 0x00000000FFFFFFFFULL; // 0000...1111 pattern, enable lower half
    uint64_t distributed_equal_mask2 = 0xFFFFFFFF00000000ULL; // 1111...0000 pattern, enable upper half
    uint64_t distributed_unequal_mask1 = 0x00000000FFFFFFFEULL; // 0000...1111...1110 pattern, add one more TPC to first
    uint64_t distributed_unequal_mask2 = 0xFFFFFFFF00000000ULL; // 1111...0000 pattern, enable upper half

    std::cout << "Packed Equal Mask 1: " << std::hex << packed_equal_mask1 << std::dec << std::endl;
    std::cout << "Packed Equal Mask 2: " << std::hex << packed_equal_mask2 << std::dec << std::endl;
    std::cout << "Packed Unequal Mask 1: " << std::hex << packed_unequal_mask1 << std::dec << std::endl;
    std::cout << "Packed Unequal Mask 2: " << std::hex << packed_unequal_mask2 << std::dec << std::endl;
    std::cout << "Distributed Equal Mask 1: " << std::hex << distributed_equal_mask1 << std::dec << std::endl;
    std::cout << "Distributed Equal Mask 2: " << std::hex << distributed_equal_mask2 << std::dec << std::endl;
    std::cout << "Distributed Unequal Mask 1: " << std::hex << distributed_unequal_mask1 << std::dec << std::endl;
    std::cout << "Distributed Unequal Mask 2: " << std::hex << distributed_unequal_mask2 << std::dec << std::endl;

    // Function to run concurrent experiment with description
    auto run_concurrent_with_description = [&](int num_threads_per_block_1, int num_threads_per_block_2, 
                                               std::vector<double>& times, uint64_t mask1, uint64_t mask2, 
                                               const char* description) {
        std::cout << "Running concurrent experiment for " << description << std::endl;
        RunConcurrentExperiment(num_threads_per_block_1, num_threads_per_block_2, times, mask1, mask2);
        CalculateStatistics(times, description);
    };

    // Full GPU Sharing (Unpartitioned)
    run_concurrent_with_description(32, 16, times, full_shared_mask, full_shared_mask, "Full GPU Sharing (Unpartitioned)");

    // GPC-packed, Equal Partitions
    run_concurrent_with_description(32, 16, times, packed_equal_mask1, packed_equal_mask2, "GPC-packed, Equal Partitions");

    // GPC-packed, Unequal Partitions
    run_concurrent_with_description(32, 16, times, packed_unequal_mask1, packed_unequal_mask2, "GPC-packed, Unequal Partitions");

    // GPC-distributed, Equal Partitions
    run_concurrent_with_description(32, 16, times, distributed_equal_mask1, distributed_equal_mask2, "GPC-distributed, Equal Partitions");

    // GPC-distributed, Unequal Partitions
    run_concurrent_with_description(32, 16, times, distributed_unequal_mask1, distributed_unequal_mask2, "GPC-distributed, Unequal Partitions");

    std::cout << "Experiments completed." << std::endl;
}

int main() {
    RunAllScenarios();
    return 0;
}
