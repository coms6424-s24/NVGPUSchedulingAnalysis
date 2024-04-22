#include <iostream>
#include <fstream>
#include <limits>
#include <cuda_runtime.h>
#include "../include/KernelGenerator.cuh"
#include "../include/SchedulingPredictor.hpp"

int main() {
    try {
        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);
        if (num_devices == 0) {
            std::cerr << "No CUDA-capable devices found." << std::endl;
            return 1;
        }

        int device_id = 0;
        cudaSetDevice(device_id);

        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, device_id);

        const int num_streams = 4;
        const int num_kernels_per_stream = 10;
        
        int max_blocks = std::min(144, device_props.maxGridSize[0]);  // Safeguard to not exceed device capability
        int max_threads = std::min(256, device_props.maxThreadsPerBlock);  // Adjust based on device limits
        int max_shared_mem = device_props.sharedMemPerBlock;  // Use a safe limit for shared memory

        const int num_sms = device_props.multiProcessorCount;

        
        std::cout << "Configuring kernels with " << max_blocks << " blocks, "
                  << max_threads << " threads per block and " << max_shared_mem << " bytes shared memory." << std::endl;


        KernelGenerator kernel_generator(num_streams, max_threads, max_blocks, max_shared_mem);
        SchedulingPredictor scheduling_predictor(num_sms);

        kernel_generator.GenerateAndLaunchKernels(num_kernels_per_stream);

        std::ifstream input_file("../data/training_data.json");
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open the training data file.");
        }

        nlohmann::json training_data;
        input_file >> training_data;
        input_file.close();

        scheduling_predictor.Predict(num_streams, num_kernels_per_stream, max_blocks);

        std::cout << "Kernels executed and data processed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
