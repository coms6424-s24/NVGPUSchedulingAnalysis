#ifndef CUDA_KERNEL_HPP
#define CUDA_KERNEL_HPP

#include <iostream>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <cuda_runtime.h>

// Represents a CUDA kernel with its associated metadata and launch configuration.
class CudaKernel {
public:
    // Function type for the kernel function.
    using KernelFunc = std::function<void(cudaStream_t)>;

    // Constructs a CudaKernel object with the given parameters.
    //
    // @param name The name of the kernel.
    // @param grid_size The grid size for launching the kernel.
    // @param block_size The block size for launching the kernel.
    // @param shared_mem_size The amount of shared memory to allocate for the kernel.
    // @param kernel_func The kernel function to be executed.
    CudaKernel(std::string_view name, dim3 grid_size, dim3 block_size, size_t shared_mem_size, KernelFunc kernel_func)
        : kernel_name_(name),
          grid_size_(grid_size),
          block_size_(block_size),
          shared_mem_size_(shared_mem_size),
          kernel_func_(std::move(kernel_func)) {}

    // Runs the kernel on the specified CUDA stream.
    //
    // @param stream The CUDA stream on which to run the kernel.
    void Run(cudaStream_t stream) const
    {
        kernel_func_(stream);
    }

    // Returns the name of the kernel.
    //
    // @return The name of the kernel.
    [[nodiscard]] std::string_view GetName() const noexcept
    {
        return kernel_name_;
    }

private:
    std::string kernel_name_;   // The name of the kernel.
    dim3 grid_size_;            // The grid size for launching the kernel.
    dim3 block_size_;           // The block size for launching the kernel.
    size_t shared_mem_size_;    // The amount of shared memory to allocate for the kernel.
    KernelFunc kernel_func_;    // The kernel function to be executed.
};

#endif // CUDA_KERNEL_HPP