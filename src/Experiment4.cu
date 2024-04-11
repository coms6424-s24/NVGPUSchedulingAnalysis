// Shared Memory Testing

#include <Windows.h>
#include <chrono>
#include "../include/KernelsExp4.cuh"
#include "../include/StreamManager.hpp"
#include "../include/TimerSpin.cuh"
#include "../include/KernelConfig4.hpp"

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
              << "  Maximum number of warps per multiprocessor: " << properties.maxThreadsPerMultiProcessor / 32 << std::endl
              << "  Shared Memory per SM: " << properties.sharedMemPerMultiprocessor << " bytes" << std::endl
              << "  Shared Memory per Block: " << properties.sharedMemPerBlock << " bytes" << std::endl;

    g_kernel_config.SetParameter(properties);

    int data_size = 1024;
    size_t bytes = data_size * sizeof(float);

    float *input1, *output1;
    float *d_input1, *d_output1;

    float *input2, *output2;
    float *d_input2, *d_output2;

    input1 = new float[data_size];
    output1 = new float[data_size];

    input2 = new float[data_size];
    output2 = new float[data_size];

    for (int i = 0; i < data_size; ++i) {
        input1[i] = float(i);
        input2[i] = float(i);
    }

    cudaMalloc(&d_input1, bytes);
    cudaMalloc(&d_output1, bytes);

    cudaMalloc(&d_input2, bytes);
    cudaMalloc(&d_output2, bytes);

    StreamManager stream1;
    StreamManager stream2;

    std::cout << "Adding kernels to stream managers..." << std::endl;

    stream1.AddCopyOperation(d_input1, input1, bytes, cudaMemcpyHostToDevice);
    stream2.AddCopyOperation(d_input2, input2, bytes, cudaMemcpyHostToDevice);

    stream1.ExecuteScheduledOperations();
    stream2.ExecuteScheduledOperations();

    stream1.Synchronize();
    stream2.Synchronize();

    stream1.AddKernel("shared1", g_kernel_config.grid_size1, g_kernel_config.block_size1, g_kernel_config.data_size1, 
    MemoryIntensiveKernel, d_input1, g_kernel_config.block_size1.x * sizeof(float));

    stream2.AddKernel("shared2", g_kernel_config.grid_size2, g_kernel_config.block_size2, g_kernel_config.data_size2, 
    MemoryIntensiveKernel, d_input2, g_kernel_config.block_size2.x * sizeof(float));

    stream2.AddKernel("shared3", g_kernel_config.grid_size2, g_kernel_config.block_size2, 0, 
    MemoryIntensiveKernel, d_input2, 256);

    stream1.ScheduleKernelExecution("shared1");
    stream2.ScheduleKernelExecution("shared2");
    stream2.ScheduleKernelExecution("shared3");

    stream1.ExecuteScheduledOperations();
    stream2.ExecuteScheduledOperations();

    stream1.Synchronize();
    stream2.Synchronize();

    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaFree(d_input2);
    cudaFree(d_output2);
    delete[] input1;
    delete[] output1;
    delete[] input2;
    delete[] output2;

    std::cout << "All operations completed." << std::endl;

    return 0;
}
