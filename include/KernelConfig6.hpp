#ifndef KERNEL_CONFIG_HPP
#define KERNEL_CONFIG_HPP

#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <algorithm>

// Stores and calculates configuration parameters for CUDA kernels.
class KernelConfig {
public:
    KernelConfig() = default;
    explicit KernelConfig(const cudaDeviceProp& prop)
    {
        SetParameter(prop);
    }

    // Sets the configuration parameters based on the given CUDA device properties.
    //
    // @param prop The CUDA device properties.
    void SetParameter(const cudaDeviceProp& prop)
    {
        int sm_count = prop.multiProcessorCount;
        int max_thread_per_sm = prop.maxThreadsPerMultiProcessor;

        // Calculate grid size for kernel1
        grid_size1 = CalculateGridSize(0.25, sm_count, max_thread_per_sm, block_size1.x);

        // Calculate grid size for kernel2
        grid_size2 = CalculateGridSize(0.25, sm_count, max_thread_per_sm, block_size2.x);

        // Calculate grid size for kernel3
        grid_size3 = CalculateGridSize(0.25, sm_count, max_thread_per_sm, block_size3.x);

        // Calculate grid size for kernel4
        grid_size4 = CalculateGridSize(0.5, sm_count, max_thread_per_sm, block_size4.x);

        // Calculate grid size for kernel5
        grid_size5 = CalculateGridSize(0.25, sm_count, max_thread_per_sm, block_size5.x);
    }

    static constexpr std::size_t MegaBytes(std::size_t size)
    {
        return size * 1024 * 1024;
    }

    static constexpr std::size_t KiloBytes(std::size_t size)
    {
        return size * 1024;
    }


    dim3 grid_size1{6, 1, 1};
    dim3 block_size1{512, 1, 1};

    dim3 grid_size2{2, 1, 1};
    dim3 block_size2{512, 1, 1};

    dim3 grid_size3{2, 1, 1};
    dim3 block_size3{512, 1, 1};

    dim3 grid_size4{4, 1, 1};
    dim3 block_size4{1024, 1, 1};

    dim3 grid_size5{2, 1, 1};
    dim3 block_size5{256, 1, 1};

private:
    // Calculates the grid size based on the given factor, SM count, maximum threads per SM, and block size.
    //
    // @tparam T The type of the factor.
    // @param factor The factor to scale the grid size.
    // @param sm_count The number of streaming multiprocessors (SMs) on the device.
    // @param max_thread_per_sm The maximum number of threads per SM.
    // @param block_size_x The block size in the x-dimension.
    // @return The calculated grid size.
    template <typename T>
    static dim3 CalculateGridSize(T factor, int sm_count, int max_thread_per_sm, int block_size_x)
    {
        assert(factor > 0 && "Factor must be greater than zero");
        int grid_size = static_cast<int>(factor * sm_count * max_thread_per_sm / block_size_x);
        return {static_cast<unsigned int>(std::max(1, grid_size)), 1u, 1u};
    }

};

#endif