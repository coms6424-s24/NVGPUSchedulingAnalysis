// Null stream
#include <Windows.h>
#include <chrono>
#include "../include/KernelsExp5.cuh"
#include "../include/StreamManager.hpp"
#include "../include/TimerSpin.cuh"
#include "../include/KernelConfig5.hpp"

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

    StreamManager stream_null(true);
    StreamManager stream1;
    StreamManager stream2;
    StreamManager stream3;

    std::cout << "Adding kernels to stream managers..." << std::endl;

    // Add all kernels into the stream manager kernel map
    stream1.AddKernel("kernel1", g_kernel_config.grid_size1, g_kernel_config.block_size1, 0, kernel1);

    stream2.AddKernel("kernel3", g_kernel_config.grid_size3, g_kernel_config.block_size3, 0, kernel3);
    stream2.AddKernel("kernel4", g_kernel_config.grid_size4, g_kernel_config.block_size4, g_kernel_config.shared_mem_size4, kernel4);

    stream3.AddKernel("kernel6", g_kernel_config.grid_size6, g_kernel_config.block_size6, 0, kernel6);

    stream_null.AddKernel("kernel2", g_kernel_config.grid_size2, g_kernel_config.block_size2, 0, kernel2);
    stream_null.AddKernel("kernel5", g_kernel_config.grid_size5, g_kernel_config.block_size5, g_kernel_config.shared_mem_size5, kernel5);

    // Schedule and execute stored kernels
    stream1.ScheduleKernelExecution("kernel1");
    stream1.ExecuteScheduledOperations();

    int wait_time_ms1 = 200;
    std::cout << "Sleep waiting for "<< wait_time_ms1 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms1));

    stream2.ScheduleKernelExecution("kernel3");
    stream_null.ScheduleKernelExecution("kernel2");

    stream2.ExecuteScheduledOperations();
    stream_null.ExecuteScheduledOperations();

    int wait_time_ms2 = 200;
    std::cout << "Sleep waiting for "<< wait_time_ms2 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms2));

    stream2.ScheduleKernelExecution("kernel4");
    stream2.ExecuteScheduledOperations();

    int wait_time_ms3 = 200;
    std::cout << "Sleep waiting for "<< wait_time_ms3 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms3));

    stream_null.ScheduleKernelExecution("kernel5");
    stream_null.ExecuteScheduledOperations();

    int wait_time_ms4 = 200;
    std::cout << "Sleep waiting for "<< wait_time_ms4 << "ms" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms4));

    stream_null.ScheduleKernelExecution("kernel5");
    stream_null.ExecuteScheduledOperations();

    stream1.Synchronize();
    stream2.Synchronize();
    stream3.Synchronize();
    stream_null.Synchronize();

    std::cout << "All operations completed." << std::endl;

    // *******************************************************



    return 0;
}
