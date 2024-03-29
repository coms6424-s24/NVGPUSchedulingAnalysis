#include <Windows.h>
#include <chrono>
#include <thread>
#include "../include/KernelsExp1.cuh"
#include "../include/StreamManager.hpp"
#include "../include/TimerSpin.cuh"
#include "../include/KernelConfig1.hpp"

KernelConfig g_kernel_config;

void CPU0Behavior(StreamManager& stream1, void* device_data, void* host_data)
{
    stream1.ScheduleKernelExecution("kernel1");
    stream1.ScheduleKernelExecution("kernel2");
    stream1.AddCopyOperation(device_data, host_data, g_kernel_config.copy_size2, cudaMemcpyDeviceToHost); // Copy after K2
    stream1.AddCopyOperation(host_data, device_data, g_kernel_config.copy_size3, cudaMemcpyHostToDevice); // Copy before K3
    stream1.ScheduleKernelExecution("kernel3");
    stream1.AddCopyOperation(device_data, host_data, g_kernel_config.copy_size3, cudaMemcpyDeviceToHost); // Copy after K3

    std::cout << "Executing scheduled kernel1, 2, 3 on stream1..." << std::endl;
    stream1.ExecuteScheduledOperations();
    stream1.Synchronize();
    std::cout << "kernel1, 2 and 3 on stream1 completed" << std::endl;
}

void CPU1Behavior(StreamManager& stream2, StreamManager& stream3, void* device_data, void* host_data)
{
    int wait_time_ms1 = 200;
    std::cout << "Sleep waiting for "<< wait_time_ms1 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms1));

    stream2.ScheduleKernelExecution("kernel4");

    int wait_time_ms2 = 200;
    std::cout << "Sleep waiting for "<< wait_time_ms2 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms2));

    stream3.ScheduleKernelExecution("kernel5");
    stream3.AddCopyOperation(device_data, host_data, g_kernel_config.copy_size5, cudaMemcpyDeviceToHost);

    std::cout << "Executing scheduled kernel4 on stream2, kernel5 on stream3..." << std::endl;
    stream2.ExecuteScheduledOperations();
    stream3.ExecuteScheduledOperations();

    stream2.Synchronize();
    stream3.Synchronize();
    std::cout << "kernel4 on stream2, kernel5 on stream3 completed" << std::endl;

    int wait_time_ms3 = 1400;
    std::cout << "Sleep waiting for " << wait_time_ms3 <<"ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms3));

    stream2.ScheduleKernelExecution("kernel6");
    stream2.AddCopyOperation(device_data, host_data, g_kernel_config.copy_size6, cudaMemcpyDeviceToHost);
    std::cout << "Executing kernel6 on stream2..." << std::endl;
    stream2.ExecuteScheduledOperations();
    stream2.Synchronize();
    std::cout << "Additional stream2 operations completed." << std::endl;
}

int main()
{
    int device;
    cudaDeviceProp properties;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&properties, device);
    std::cout << "Device Number: " << device << std::endl
              << "  Device name: " << properties.name << std::endl
              << "  Number of multiprocessors: " << properties.multiProcessorCount << std::endl
              << "  Maximum number of threads per multiprocessor: " << properties.maxThreadsPerMultiProcessor << std::endl
              << "  Maximum number of warps per multiprocessor: " << properties.maxThreadsPerMultiProcessor / 32 << std::endl;

    g_kernel_config.SetParameter(properties);

    void* device_data = nullptr;
    void* host_data = nullptr;

    cudaMalloc(&device_data, 256 * 1024 * 1024);
    host_data = malloc(256 * 1024 * 1024);

    StreamManager stream1;
    StreamManager stream2;
    StreamManager stream3;

    std::cout << "Adding kernels to stream managers..." << std::endl;

    // Add all kernels into the stream manager kernel map
    stream1.AddKernel("kernel1", g_kernel_config.grid_size1, g_kernel_config.block_size1, 0, kernel1);
    stream1.AddKernel("kernel2", g_kernel_config.grid_size2, g_kernel_config.block_size2, 0, kernel2);
    stream1.AddKernel("kernel3", g_kernel_config.grid_size3, g_kernel_config.block_size3, 0, kernel3);
    
    stream2.AddKernel("kernel4", g_kernel_config.grid_size4, g_kernel_config.block_size4, g_kernel_config.shared_mem_size4, kernel4);
    stream2.AddKernel("kernel6", g_kernel_config.grid_size6, g_kernel_config.block_size6, 0, kernel6);

    stream3.AddKernel("kernel5", g_kernel_config.grid_size5, g_kernel_config.block_size5, g_kernel_config.shared_mem_size5, kernel5);

    // Simulate 2 CPUs environment in the paper
    std::thread cpu0(CPU0Behavior, std::ref(stream1), device_data, host_data);
    std::thread cpu1(CPU1Behavior, std::ref(stream2), std::ref(stream3), device_data, host_data);

    cpu0.join();
    cpu1.join();

    std::cout << "All operations completed." << std::endl;

    return 0;
}
