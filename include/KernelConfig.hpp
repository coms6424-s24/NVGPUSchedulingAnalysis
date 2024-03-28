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
        // Ensure the number of threads surpasses the maximum allowable limit for concurrent scheduling
        grid_size1 = CalculateGridSize(1.5, sm_count, max_thread_per_sm, block_size1.x);

        // Calculate grid size for kernel2
        // Ensure the number of threads remains below the maximum supported by each SM
        grid_size2 = CalculateGridSize(0.5, 1, max_thread_per_sm, block_size2.x);

        // Calculate grid size for kernel3
        // Ensure the number of threads is equal to the maximum supported by each SM
        grid_size3 = CalculateGridSize(1.0, 1, max_thread_per_sm, block_size3.x);

        // Calculate grid size for kernel4
        // Ensure the number of threads is equal to half of the maximum supported by each SM
        grid_size4 = CalculateGridSize(0.5, 1, max_thread_per_sm, block_size4.x);

        // Calculate grid size for kernel5
        // Ensure the number of threads is equal to 1/4 of the maximum supported by each SM
        grid_size5 = CalculateGridSize(0.25, 1, max_thread_per_sm, block_size5.x);

        // Calculate grid size for kernel6
        // Ensure the number of threads is equal to half of the maximum supported by each SM
        grid_size6 = CalculateGridSize(0.5, 1, max_thread_per_sm, block_size6.x);
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
    dim3 block_size1{1024, 1, 1};

    dim3 grid_size2{2, 1, 1};
    dim3 block_size2{512, 1, 1};
    size_t copy_size2 = MegaBytes(256);

    dim3 grid_size3{2, 1, 1};
    dim3 block_size3{1024, 1, 1};
    size_t copy_size3 = MegaBytes(256);

    dim3 grid_size4{4, 1, 1};
    dim3 block_size4{256, 1, 1};
    size_t shared_mem_size4 = KiloBytes(32);

    dim3 grid_size5{2, 1, 1};
    dim3 block_size5{256, 1, 1};
    size_t shared_mem_size5 = KiloBytes(32);
    size_t copy_size5 = MegaBytes(256);

    dim3 grid_size6{2, 1, 1};
    dim3 block_size6{512, 1, 1};
    size_t copy_size6 = MegaBytes(256);

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