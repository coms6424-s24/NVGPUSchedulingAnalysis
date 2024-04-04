// Copy Engine Testing

#include <Windows.h>
#include <chrono>
#include "../include/StreamManager.hpp"
#include "../include/TimerSpin.cuh"

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

    size_t copy_size = 1024 * 1024 * 1024 * 1024;

    void* host_data_src1 = nullptr;
    void* device_data_dst1 = nullptr;

    void* host_data_src2 = nullptr;
    void* device_data_dst2 = nullptr;

    void* host_data_src3 = nullptr;
    void* device_data_dst3 = nullptr;

    void* host_data_src4 = nullptr;
    void* device_data_dst4 = nullptr;

    void* host_data_src5 = nullptr;
    void* device_data_dst5 = nullptr;

    host_data_src1 = malloc(copy_size);
    cudaMalloc(&device_data_dst1, copy_size);

    host_data_src2= malloc(copy_size);
    cudaMalloc(&device_data_dst2, copy_size);

    host_data_src3 = malloc(copy_size);
    cudaMalloc(&device_data_dst3, copy_size);

    host_data_src4 = malloc(copy_size);
    cudaMalloc(&device_data_dst4, copy_size);

    host_data_src5 = malloc(copy_size);
    cudaMalloc(&device_data_dst5, copy_size);

    StreamManager stream1;
    StreamManager stream2;
    StreamManager stream3;
    StreamManager stream4;
    StreamManager stream5;

    std::cout << "Adding copy operations to stream managers..." << std::endl;


    stream1.AddCopyOperation(host_data_src1, device_data_dst1, copy_size, cudaMemcpyHostToDevice);
    stream2.AddCopyOperation(host_data_src2, device_data_dst2, copy_size, cudaMemcpyHostToDevice);
    stream3.AddCopyOperation(host_data_src3, device_data_dst3, copy_size, cudaMemcpyHostToDevice);
    stream4.AddCopyOperation(host_data_src4, device_data_dst4, copy_size, cudaMemcpyHostToDevice);
    stream5.AddCopyOperation(host_data_src5, device_data_dst5, copy_size, cudaMemcpyHostToDevice);

    stream1.ExecuteScheduledOperations();
    stream2.ExecuteScheduledOperations();
    stream3.ExecuteScheduledOperations();
    stream4.ExecuteScheduledOperations();
    stream5.ExecuteScheduledOperations();

    stream1.Synchronize();
    stream2.Synchronize();
    stream3.Synchronize();
    stream4.Synchronize();
    stream5.Synchronize();

    std::cout << "Host to device copy compeleted." << std::endl;

    stream1.AddCopyOperation(device_data_dst1, host_data_src1, copy_size, cudaMemcpyDeviceToHost);
    stream2.AddCopyOperation(device_data_dst2, host_data_src2, copy_size, cudaMemcpyDeviceToHost);
    stream3.AddCopyOperation(device_data_dst3, host_data_src3, copy_size, cudaMemcpyDeviceToHost);
    stream4.AddCopyOperation(device_data_dst4, host_data_src4, copy_size, cudaMemcpyDeviceToHost);
    stream5.AddCopyOperation(device_data_dst5, host_data_src5, copy_size, cudaMemcpyDeviceToHost);

    stream1.ExecuteScheduledOperations();
    stream2.ExecuteScheduledOperations();
    stream3.ExecuteScheduledOperations();
    stream4.ExecuteScheduledOperations();
    stream5.ExecuteScheduledOperations();

    stream1.Synchronize();
    stream2.Synchronize();
    stream3.Synchronize();
    stream4.Synchronize();
    stream5.Synchronize();
    
    std::cout << "Device to host copy compeleted." << std::endl;


    std::cout << "All operations completed." << std::endl;

    return 0;
}
