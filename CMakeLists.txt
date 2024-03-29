cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MyCUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("../include")

set(CMAKE_CUDA_COMPILER "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64")
set(CUDA_ARCH_FLAGS "-gencode arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} -lineinfo -Xptxas -v")

set(COMMON_CUDA_SOURCES
    ../src/KernelsExp1.cu
    ../src/TimerSpin.cu
)

# Experiment 1
set(EXPERIMENT1_SOURCES
    ../src/Experiment1.cu
)
add_executable(Experiment1 ${EXPERIMENT1_SOURCES} ${COMMON_CUDA_SOURCES})
target_link_libraries(Experiment1 cudart)

# Experiment 2
set(EXPERIMENT2_SOURCES
    ../src/Experiment2.cu
)
add_executable(Experiment2 ${EXPERIMENT2_SOURCES} ${COMMON_CUDA_SOURCES})
target_link_libraries(Experiment2 cudart)

# Experiment 3
set(EXPERIMENT3_SOURCES
    ../src/Experiment3.cu
)
add_executable(Experiment3 ${EXPERIMENT3_SOURCES} ${COMMON_CUDA_SOURCES})
target_link_libraries(Experiment3 cudart)
