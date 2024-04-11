// Stream priority
#include <Windows.h>
#include <chrono>
#include "../include/KernelsExp6.cuh"
#include "../include/StreamManager.hpp"
#include "../include/TimerSpin.cuh"
#include "../include/KernelConfig6.hpp"

KernelConfig g_kernel_config;

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

    StreamManager stream1(false, StreamManager::Priority::Low);
    StreamManager stream2(false, StreamManager::Priority::Low);
    StreamManager stream3(false, StreamManager::Priority::Low);
    StreamManager stream4(false, StreamManager::Priority::High);
    StreamManager stream5(false, StreamManager::Priority::Low);

    std::cout << "Adding kernels to stream managers..." << std::endl;

    // Add all kernels into the stream manager kernel map
    stream1.AddKernel("kernel1", g_kernel_config.grid_size1, g_kernel_config.block_size1, 0, kernel1);
    stream2.AddKernel("kernel1", g_kernel_config.grid_size2, g_kernel_config.block_size2, 0, kernel1);
    stream3.AddKernel("kernel1", g_kernel_config.grid_size3, g_kernel_config.block_size3, 0, kernel1);
    stream4.AddKernel("kernel2", g_kernel_config.grid_size4, g_kernel_config.block_size4, 0, kernel2);
    stream5.AddKernel("kernel1", g_kernel_config.grid_size5, g_kernel_config.block_size5, 0, kernel1);
    
    // Schedule and execute stored kernels
    stream1.ScheduleKernelExecution("kernel1");
    stream1.ExecuteScheduledOperations();

    int wait_time_ms1 = 100;
    std::cout << "Sleep waiting for "<< wait_time_ms1 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms1));

    stream2.ScheduleKernelExecution("kernel1");
    stream2.ExecuteScheduledOperations();

    std::cout << "Sleep waiting for "<< wait_time_ms1 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms1));

    stream3.ScheduleKernelExecution("kernel1");
    stream3.ExecuteScheduledOperations();

    int wait_time_ms2 = 50;
    std::cout << "Sleep waiting for "<< wait_time_ms2 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms2));

    stream4.ScheduleKernelExecution("kernel2");
    stream4.ExecuteScheduledOperations();

    std::cout << "Sleep waiting for "<< wait_time_ms1 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms1));

    stream5.ScheduleKernelExecution("kernel1");
    stream5.ExecuteScheduledOperations();

    stream1.Synchronize();
    stream2.Synchronize();
    stream3.Synchronize();
    stream4.Synchronize();
    stream5.Synchronize();

    std::cout << "All operations completed." << std::endl;

    // *******************************************************
    return 0;
}
