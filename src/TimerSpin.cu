# include "../include/TimerSpin.cuh"

// Kernel function that spins for a specified amount of time.
//
// @param wait_time The amount of time to spin, in clock cycles.
__global__ void TimerSpin(uint64_t wait_time)
{
    uint64_t start = clock64();
    uint64_t elapsed = 0;

    while (elapsed < wait_time) {
        uint64_t now = clock64();
        elapsed = now - start;
    }
}