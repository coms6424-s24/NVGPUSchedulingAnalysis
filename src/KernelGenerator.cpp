#include <chrono>
#include "../include/KernelGenerator.hpp"

KernelGenerator::KernelGenerator(int num_streams, int max_threads, int max_blocks, int max_shared_mem)
    : max_threads_(max_threads), max_blocks_(max_blocks), max_shared_mem_(max_shared_mem),
    rng_(std::random_device{}())
{
    stream_manager_.reserve(num_streams);
    for (int i = 0; i < num_streams; i++) {
        stream_manager_.emplace_back();
    }
}


template<typename T>
std::enable_if_t<std::is_integral_v<T>, T> KernelGenerator::GetRandomNumber(T min, T max)
{
    std::uniform_int_distribution<T> dist(min, max);
    return dist(rng_);
}

template<typename... Args>
void KernelGenerator::GenerateAndLaunchKernels(int num_kernels, Args&&... args)
{
    auto settings = std::vector<KernelSetting>{std::forward<Args>(args)...};
    for (int i = 0; i < num_kernels; i++) {
        int stream_index = i % stream_manager_.size();
        KernelSetting setting = i < settings.size() ? settings[i] : KernelSetting();

        int thread_per_block = setting.threads_per_block.value_or(GetRandomNumber(1, max_threads_));
        int blocks = setting.blocks.value_or(GetRandomNumber(1, max_blocks_));
        int shared_mem_size = setting.shared_mem_size.value_or(GetRandomNumber(0, max_shared_mem_));

        // Allocate memory for kernel parameters
        auto d_smids = std::make_unique<int[]>(blocks);
        auto d_block_ids = std::make_unique<int[]>(blocks);
        auto d_thread_ids = std::make_unique<int[]>(blocks * thread_per_block);
        auto d_block_dims = std::make_unique<int[]>(blocks);
        auto d_thread_dims = std::make_unique<int[]>(blocks * thread_per_block);
        auto d_shared_mem_sizes = std::make_unique<int[]>(blocks);
        auto d_kernel_durations = std::make_unique<float[]>(blocks);

        auto kernel_durations = std::vector<float>(blocks, setting.execution_time.value_or(0.1f));
        cudaMemcpy(d_kernel_durations.get(), kernel_durations.data(), blocks * sizeof(float), cudaMemcpyHostToDevice);

        auto kernelFunction = [d_smids = std::move(d_smids), d_block_ids = std::move(d_block_ids),
                               d_thread_ids = std::move(d_thread_ids), d_block_dims = std::move(d_block_dims),
                               d_thread_dims = std::move(d_thread_dims), d_shared_mem_sizes = std::move(d_shared_mem_sizes),
                               d_kernel_durations = std::move(d_kernel_durations), blocks, thread_per_block, shared_mem_size](cudaStream_t stream) {
            TestKernel<<<blocks, thread_per_block, shared_mem_size, stream>>>(d_smids.get(), d_block_ids.get(), d_thread_ids.get(),
                                                                             d_block_dims.get(), d_thread_dims.get(), d_shared_mem_sizes.get(),
                                                                             d_kernel_durations.get(), 900);
        };

        stream_manager_[stream_index].AddKernel("TestKernel" + std::to_string(i), dim3(blocks), dim3(thread_per_block),
                                                shared_mem_size, std::move(kernelFunction));
        stream_manager_[stream_index].ScheduleKernelExecution("TestKernel" + std::to_string(i));
    }
}

__global__ void KernelGenerator::TestKernel(int *smids, int *block_ids, int *thread_ids, int *block_dims, int *thread_dims,
                           int *shared_mem_sizes, float *kernel_durations, clock_t clock_rate)
{
    // Get the block ID and thread ID
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    // Get the SM ID using inline assembly
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));

    // Store the SM ID, block ID, and thread ID in global memory
    if (thread_id == 0) {
        smids[block_id] = smid;
        block_ids[block_id] = block_id;
        block_dims[block_id] = blockDim.x;
        shared_mem_sizes[block_id] = (int)blockDim.x * sizeof(int);
    }
    thread_ids[block_id * blockDim.x + thread_id] = thread_id;
    thread_dims[block_id * blockDim.x + thread_id] = threadIdx.x;

    // Allocate shared memory
    extern __shared__ int shared_mem[];

    // Initialize shared memory with thread IDs
    shared_mem[thread_id] = thread_id;
    __syncthreads();

    // Perform some computations using registers and shared memory
    int reg_val = thread_id;
    for (int i = 0; i < blockDim.x; i++) {
        reg_val += shared_mem[i];
    }

    // Introduce a delay based on the kernel duration
    clock_t start_time = clock64();
    float kernel_duration = kernel_durations[block_id];
    while ((clock64() - start_time) / (clock_rate * 1e-3f) < kernel_duration) {
        // Busy wait
    }
}