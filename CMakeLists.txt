cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MyCUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("../include")

set(CMAKE_CUDA_COMPILER "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64")

set(SOURCE_FILES
    ../src/Experiment1.cu
    ../src/Kernels.cu
    ../src/TimerSpin.cu
)

set(CUDA_ARCH_FLAGS "-gencode arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} -lineinfo -Xptxas -v")

add_executable(NVGPUSchedulingAnalysis ${SOURCE_FILES})

target_link_libraries(NVGPUSchedulingAnalysis cudart)
