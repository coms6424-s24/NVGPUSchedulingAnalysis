#ifndef KERNEL_GENERATOR_HPP
#define KERNEL_GENERATOR_HPP

#include <random>
#include <vector>
#include <optional>
#include <type_traits>
#include <memory>
#include "StreamManager.hpp"

struct KernelSetting {
    std::optional<int> threads_per_block; // Optional number of threads per block
    std::optional<int> blocks;          // Optional number of blocks
    std::optional<int> shared_mem_size;   // Optional shared memory size in bytes
    std::optional<float> execution_time; // Optional execution time in seconds

    KernelSetting() = default;
};

class KernelGenerator {
public:
    KernelGenerator(int num_streams, int max_threads, int max_blocks, int max_shared_mem);

    template<typename... Args>
    void GenerateAndLaunchKernels(int num_kernels, Args&&... args);

    static __global__ void TestKernel(int *smids, int *block_ids, int *thread_ids, int *block_dims, int *thread_dims,
                           int *shared_mem_sizes, float *kernel_durations, clock_t clock_rate);

private:
    std::vector<StreamManager> stream_manager_;
    int max_threads_;
    int max_blocks_;
    int max_shared_mem_;
    std::mt19937 rng_;

    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, T> GetRandomNumber(T min, T max);
};

#endif // KERNEL_GENERATOR_HPP