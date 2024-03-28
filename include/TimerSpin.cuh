#ifndef TIMER_SPIN_CUH
#define TIMER_SPIN_CUH

#include <cuda_runtime.h>
#include <cstdint>

__global__ void TimerSpin(uint64_t wait_time);

#endif // TIMER_SPIN_CUH