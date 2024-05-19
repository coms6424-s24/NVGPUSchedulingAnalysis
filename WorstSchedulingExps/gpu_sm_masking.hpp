

#include <stdint.h>
#include <memory>

extern void SetTPCMask(uint64_t mask);
extern int GetGpcInfo(uint32_t& num_enabled_gpcs, std::unique_ptr<uint64_t[]>& tpcs_for_gpc, int dev);
extern int GetTpcInfo(uint32_t& num_tpcs, int cuda_dev);

