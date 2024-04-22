#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <execution>
#include "CudaKernel.hpp"

// Custom deleter for device memory
struct cudaDeleter {
    void operator()(void* ptr) {
        cudaFree(ptr);
    }
};

// Using a unique_ptr with the custom deleter
template<typename T>
using unique_cuda_ptr = std::unique_ptr<T, cudaDeleter>;

// Manages CUDA streams and kernel execution.
class StreamManager {
public:
    enum class Priority {
        High,
        Low
    };

    // Constructs a StreamManager object and creates a CUDA stream.
    StreamManager(bool use_null_stream = false, Priority priority = Priority::Low)
    {
        if (!use_null_stream) {
            int least_priority, greatest_priority;
            cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

            int cuda_priority = (priority == Priority::High) ? least_priority : greatest_priority;

            cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, cuda_priority);
        } else {
            stream_ = nullptr;
        }
    }

    // Destroys the StreamManager object and the associated CUDA stream.
    ~StreamManager()
    {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    // Allocates device memory for the stream manager.
    void AllocateDeviceMemory(int max_blocks, int max_threads_per_block)
    {
        int* int_raw_ptr;
        float* float_raw_ptr;

        cudaMalloc(reinterpret_cast<void**>(&int_raw_ptr), max_blocks * sizeof(int));
        d_smids_.reset(int_raw_ptr);

        cudaMalloc(reinterpret_cast<void**>(&int_raw_ptr), max_blocks * sizeof(int));
        d_block_ids_.reset(int_raw_ptr);

        cudaMalloc(reinterpret_cast<void**>(&int_raw_ptr), max_blocks * max_threads_per_block * sizeof(int));
        d_thread_ids_.reset(int_raw_ptr);

        cudaMalloc(reinterpret_cast<void**>(&int_raw_ptr), max_blocks * sizeof(int));
        d_block_dims_.reset(int_raw_ptr);

        cudaMalloc(reinterpret_cast<void**>(&int_raw_ptr), max_blocks * max_threads_per_block * sizeof(int));
        d_thread_dims_.reset(int_raw_ptr);

        cudaMalloc(reinterpret_cast<void**>(&int_raw_ptr), max_blocks * sizeof(int));
        d_shared_mem_sizes_.reset(int_raw_ptr);

        cudaMalloc(reinterpret_cast<void**>(&float_raw_ptr), max_blocks * sizeof(float));
        d_kernel_durations_.reset(float_raw_ptr);

        std::cout << "Size of d_kernel_durations: " << max_blocks * sizeof(float) << " bytes" << std::endl;
    }

    // Getter methods to access the device memory pointers.
    int* GetDeviceSmids() { return d_smids_.get(); }
    int* GetDeviceBlockIds() { return d_block_ids_.get(); }
    int* GetDeviceThreadIds() { return d_thread_ids_.get(); }
    int* GetDeviceBlockDims() { return d_block_dims_.get(); }
    int* GetDeviceThreadDims() { return d_thread_dims_.get(); }
    int* GetDeviceSharedMemSizes() { return d_shared_mem_sizes_.get(); }
    float* GetDeviceKernelDurations() { return d_kernel_durations_.get(); }

    // Adds a new kernel to the stream manager.
    //
    // @tparam Func The type of the kernel function.
    // @tparam Args The types of the kernel function arguments.
    // @param name The name of the kernel.
    // @param grid_size The grid size for launching the kernel.
    // @param block_size The block size for launching the kernel.
    // @param shared_mem_size The amount of shared memory to allocate for the kernel.
    // @param func The kernel function to be executed.
    // @param args The arguments to be passed to the kernel function.
    template<typename Func, typename... Args>
    void AddKernel(const std::string& name, dim3 grid_size, dim3 block_size, size_t shared_mem_size, Func func, Args... args)
    {
        CudaKernel::KernelFunc kernel_func = [func, args..., grid_size, block_size, shared_mem_size](cudaStream_t stream) mutable {
            func<<<grid_size, block_size, shared_mem_size, stream>>>(args...);
        };

        auto kernel = std::make_unique<CudaKernel>(name, grid_size, block_size, shared_mem_size, kernel_func);
        kernel_map_[name] = std::move(kernel);
    }

    // Schedules a previously added kernel for execution.
    //
    // @param name The name of the kernel to schedule.
    void ScheduleKernelExecution(const std::string& name)
    {
        if (auto it = kernel_map_.find(name); it != kernel_map_.end()) {
            CudaKernel* kernel_ptr = it->second.get();
            operations_.emplace_back([this, kernel_ptr]() { 
                std::cout << "Executing kernel: " << kernel_ptr->GetName() << std::endl;
                kernel_ptr->Run(stream_); 
            });
        } else {
            std::cerr << "Kernel " << name << " not found for scheduling.\n";
        }
    }

    // Adds a copy operation to be performed at a specific time in the sequence.
    //
    // @param dst The destination pointer for the copy operation.
    // @param src The source pointer for the copy operation.
    // @param count The number of bytes to copy.
    // @param kind The type of memory copy operation (e.g., host-to-device, device-to-host).
    void AddCopyOperation(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    {
        operations_.emplace_back([=] { cudaMemcpyAsync(dst, src, count, kind, stream_); });
    }

    // Executes all scheduled operations in the sequence they were added.
    void ExecuteScheduledOperations()
    {
        for (auto& operation: operations_) {
            operation();
        }
        std::cout << "Size of operation: " << operations_.size() << std::endl;
        operations_.clear();
    }

    // Runs a specific kernel by name.
    //
    // @param name The name of the kernel to run.
    void RunKernel(const std::string& name)
    {
        if (auto it = kernel_map_.find(name); it != kernel_map_.end()) {
            it->second->Run(stream_);
        } else {
            std::cerr << "Kernel " << name << " not found.\n";
        }
    }

    // Synchronizes the CUDA stream, blocking until all operations are complete.
    void Synchronize()
    {
        if (stream_ == nullptr) {
            cudaDeviceSynchronize();
        } else {
            cudaStreamSynchronize(stream_);
        }
        if (stream_ == nullptr) {
            cudaDeviceSynchronize();
        } else {
            cudaStreamSynchronize(stream_);
        }
    }

private:
    cudaStream_t stream_;  // The CUDA stream associated with the manager.
    std::vector<std::function<void()>> operations_;  // Stores all scheduled operations.
    std::unordered_map<std::string, std::unique_ptr<CudaKernel>> kernel_map_;  // Stores kernels by name for potential rescheduling.

    // Device memory pointers
    unique_cuda_ptr<int> d_smids_;
    unique_cuda_ptr<int> d_block_ids_;
    unique_cuda_ptr<int> d_thread_ids_;
    unique_cuda_ptr<int> d_block_dims_;
    unique_cuda_ptr<int> d_thread_dims_;
    unique_cuda_ptr<int> d_shared_mem_sizes_;
    unique_cuda_ptr<float> d_kernel_durations_;
};

#endif