#include "../include/KernelGenerator.cuh"

__global__ void TestKernel(int *smids, int *block_ids, int *thread_ids, int *block_dims, int *thread_dims,
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
    const int regs_count = 256;
    const int smem_count = shared_mem_sizes[block_id] / sizeof(int);
    int regs[regs_count];

    #pragma unroll 256
    for (int i = 0; i < smem_count; i++) {
        regs[i % regs_count] += shared_mem[i];
    }

    for (int i = 0; i < regs_count && i < smem_count; i++) {
        shared_mem[i] = regs[i];
    }

    // Introduce a delay based on the kernel duration
    clock_t start_time = clock64();
    float kernel_duration = 1.0f;

    float kernel_duration_ticks = kernel_duration * clock_rate * 1e-3f;
    while ((clock64() - start_time) < kernel_duration_ticks) {
        // Busy wait
    }
}

__global__ void kernel1() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d\n", idx);
}

KernelGenerator::KernelGenerator(int num_streams, int max_threads, int max_blocks, int max_shared_mem)
    : max_threads_(max_threads), max_blocks_(max_blocks), max_shared_mem_(max_shared_mem),
    rng_(std::random_device{}())
{
    stream_managers_.reserve(num_streams);
    for (int i = 0; i < num_streams; i++) {
        stream_managers_.emplace_back(std::make_unique<StreamManager>());
    }
}