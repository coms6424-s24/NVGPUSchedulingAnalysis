#ifndef KERNEL_GENERATOR_CUH
#define KERNEL_GENERATOR_CUH

#include <random>
#include <vector>
#include <fstream>
#include <optional>
#include <type_traits>
#include <device_launch_parameters.h>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include "../include/json.hpp"
#include "StreamManager.hpp"

struct KernelSetting {
    std::optional<int> threads_per_block; // Optional number of threads per block
    std::optional<int> blocks; // Optional number of blocks
    std::optional<int> shared_mem_size; // Optional shared memory size in bytes
    std::optional<float> execution_time; // Optional execution time in seconds

    KernelSetting() = default;
};

extern __global__ void TestKernel(int *smids, int *block_ids, int *thread_ids, int *block_dims, int *thread_dims,
                                  int *shared_mem_sizes, float *kernel_durations, clock_t clock_rate);

__global__ void kernel1();

class KernelGenerator {
public:
    KernelGenerator(int num_streams, int max_threads, int max_blocks, int max_shared_mem);

    template<typename... Args>
    void GenerateAndLaunchKernels(int num_kernels_per_stream, Args... args) {
        auto settings = std::vector<KernelSetting>{std::forward<Args>(args)...};
        nlohmann::json training_data;

        KernelSetting default_setting;

        int num_streams = stream_managers_.size();
        std::cout << "Initializing with " << num_streams << " streams.\n";

        // Declare and initialize the host-side vectors outside the stream loop
        std::vector<std::vector<int>> block_to_sm(num_streams, std::vector<int>(max_blocks_, 0));
        std::vector<std::vector<int>> block_ids(num_streams, std::vector<int>(max_blocks_, 0));
        std::vector<std::vector<int>> block_dims(num_streams, std::vector<int>(max_blocks_, 0));
        std::vector<std::vector<int>> thread_dims(num_streams, std::vector<int>(max_blocks_ * max_threads_, 0));
        std::vector<std::vector<int>> shared_mem_sizes(num_streams, std::vector<int>(max_blocks_, 0));
        std::vector<std::vector<float>> kernel_durations(num_streams, std::vector<float>(max_blocks_, 0.0f));

        std::vector<std::vector<int>> threads_per_block_array(num_streams, std::vector<int>(num_kernels_per_stream, 0));
        std::vector<std::vector<int>> blocks_array(num_streams, std::vector<int>(num_kernels_per_stream, 0));


        // Allocate device memory for each stream manager
        for (int stream_index = 0; stream_index < num_streams; stream_index++) {
            std::cout << "Allocating device memory for stream " << stream_index << ".\n";
            stream_managers_[stream_index]->AllocateDeviceMemory(max_blocks_, max_threads_);
        }

        // Configure the kernel parameters outside the stream loop
        for (int stream_index = 0; stream_index < num_streams; stream_index++) {
            for (int kernel_index = 0; kernel_index < num_kernels_per_stream; kernel_index++) {
                KernelSetting setting = settings.empty() ? default_setting : settings[kernel_index % settings.size()];
                threads_per_block_array[stream_index][kernel_index] = setting.threads_per_block.value_or(GetRandomNumber(1, max_threads_));
                blocks_array[stream_index][kernel_index] = setting.blocks.value_or(GetRandomNumber(1, max_blocks_));

            }
        }

        // Loop through each stream and launch the kernels
        for (int stream_index = 0; stream_index < num_streams; stream_index++) {
            auto& manager = stream_managers_[stream_index];
            std::cout << "Adding kernels in stream " << stream_index << ".\n";

            int* d_smids = manager->GetDeviceSmids();
            int* d_block_ids = manager->GetDeviceBlockIds();
            int* d_thread_ids = manager->GetDeviceThreadIds();
            int* d_block_dims = manager->GetDeviceBlockDims();
            int* d_thread_dims = manager->GetDeviceThreadDims();
            int* d_shared_mem_sizes = manager->GetDeviceSharedMemSizes();
            float* d_kernel_durations = manager->GetDeviceKernelDurations();

            std::cout << "Get device memory.\n";

            for (int kernel_index = 0; kernel_index < num_kernels_per_stream; kernel_index++) {
                KernelSetting setting = settings.empty() ? default_setting : settings[kernel_index % settings.size()];

                int thread_per_block = threads_per_block_array[stream_index][kernel_index];
                int blocks = blocks_array[stream_index][kernel_index];
                int shared_mem_size = max_shared_mem_;

                std::cout << "Stream " << stream_index << ", Configuring kernel " << kernel_index
                        << ": " << blocks << " blocks, " << thread_per_block << std::endl;

                auto kernel_exec_durations = std::vector<float>(max_blocks_, 1.0f);
                std::cout << "Size of kernel_exec_durations: " << kernel_exec_durations.size() * sizeof(float) << " bytes" << std::endl;

                if (max_blocks_ * sizeof(float) != kernel_exec_durations.size() * sizeof(float)) {
                    std::cerr << "Sizes of d_kernel_durations and kernel_exec_durations do not match." << std::endl;
                    // Handle the error, e.g., throw an exception or return an error code
                }

                manager->AddCopyOperation(d_kernel_durations, kernel_exec_durations.data(), max_blocks_ * sizeof(float), cudaMemcpyHostToDevice);

                auto kernel_name = "TestKernel" + std::to_string(stream_index * num_kernels_per_stream + kernel_index);
                manager->AddKernel(kernel_name, dim3(blocks), dim3(thread_per_block), shared_mem_size, TestKernel,
                    d_smids, d_block_ids, d_thread_ids,
                    d_block_dims, d_thread_dims, d_shared_mem_sizes,
                    d_kernel_durations, 1320);

                manager->ScheduleKernelExecution(kernel_name);
            }
        }

        for (int i = 0; i < stream_managers_.size(); i++) {
            stream_managers_[i]->ExecuteScheduledOperations();
            std::cout << "Executing all scheduled operations for stream " << i << ". \n";
        }

        for (int i = 0; i < stream_managers_.size(); i++) {
            std::cout << "Synchronizing after execution for stream  " << i << ". \n";
            stream_managers_[i]->Synchronize();
        }

        // After synchronization, safely transfer data from device to host
        for (int stream_index = 0; stream_index < num_streams; stream_index++) {
            auto& manager = stream_managers_[stream_index];
            int* d_smids = manager->GetDeviceSmids();
            int* d_block_ids = manager->GetDeviceBlockIds();
            int* d_block_dims = manager->GetDeviceBlockDims();
            int* d_thread_ids = manager->GetDeviceThreadIds();
            int* d_shared_mem_sizes = manager->GetDeviceSharedMemSizes();
            float* d_kernel_durations = manager->GetDeviceKernelDurations();

            cudaError_t error;
            
            // Perform cudaMemcpy to fetch data back to host
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA Kernel error: " << cudaGetErrorString(error) << std::endl;
                throw std::runtime_error("Kernel execution failed.");
            }

            error = cudaMemcpy(block_to_sm[stream_index].data(), d_smids, max_blocks_ * sizeof(int), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy block_to_sm: " << cudaGetErrorString(error) << std::endl;
            }

            error = cudaMemcpy(block_ids[stream_index].data(), d_block_ids, max_blocks_ * sizeof(int), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy block_ids: " << cudaGetErrorString(error) << std::endl;
            }

            error = cudaMemcpy(block_dims[stream_index].data(), d_block_dims, max_blocks_ * sizeof(int), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy block_dims: " << cudaGetErrorString(error) << std::endl;
            }

            error = cudaMemcpy(thread_dims[stream_index].data(), d_thread_ids, max_blocks_ * max_threads_ * sizeof(int), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy thread_dims: " << cudaGetErrorString(error) << std::endl;
            }

            error = cudaMemcpy(shared_mem_sizes[stream_index].data(), d_shared_mem_sizes, max_blocks_ * sizeof(int), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy shared_mem_sizes: " << cudaGetErrorString(error) << std::endl;
            }

            error = cudaMemcpy(kernel_durations[stream_index].data(), d_kernel_durations, max_blocks_ * sizeof(float), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy kernel_durations: " << cudaGetErrorString(error) << std::endl;
            }
    
            // Store the block data in the JSON training data
            std::cout << "Preparing to write training data to JSON file." << std::endl;

            error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
                throw std::runtime_error("CUDA error.");
            }

            for (int kernel_index = 0; kernel_index < num_kernels_per_stream; kernel_index++) {
                int thread_per_block = threads_per_block_array[stream_index][kernel_index];
                int blocks = blocks_array[stream_index][kernel_index];
        
                auto kernel_name = "TestKernel" + std::to_string(stream_index * num_kernels_per_stream + kernel_index);
        
                nlohmann::json kernel_data;
                kernel_data["stream"] = stream_index;
                kernel_data["kernel"] = kernel_name;
        
                for (int j = 0; j < blocks; j++) {
                    nlohmann::json block_data;
                    block_data["block_id"] = block_ids[stream_index][j];
                    block_data["sm_id"] = block_to_sm[stream_index][j];
                    block_data["block_dim"] = block_dims[stream_index][j];
                    block_data["thread_dim"] = thread_dims[stream_index][j * thread_per_block];
                    block_data["shared_mem_size"] = shared_mem_sizes[stream_index][j];
                    block_data["kernel_duration"] = 1.0;
                    kernel_data["blocks"].push_back(block_data);
                }
        
                training_data.push_back(kernel_data);
            }
        }

        std::cout << "Finished processing all kernels." << std::endl;

        // After all streams have completed, save the training data
        SaveTrainingData(training_data);
    }

    void SaveTrainingData(const nlohmann::json& training_data) {
        std::string path = "../data/training_data.json";
        std::ofstream output_file(path, std::ios::out);

        if (!output_file.is_open()) {
            std::cerr << "Failed to open the output file for writing: " << path << std::endl;
            return;
        }

        output_file << training_data.dump(4);
        output_file.close();

        std::cout << "Training data successfully written to " << path << std::endl;
    }

private:
    std::vector<std::unique_ptr<StreamManager>> stream_managers_;
    int max_threads_;
    int max_blocks_;
    int max_shared_mem_;
    std::mt19937 rng_;

    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, T> GetRandomNumber(T min, T max)
    {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng_);
    }
};

#endif // KERNEL_GENERATOR_CUH