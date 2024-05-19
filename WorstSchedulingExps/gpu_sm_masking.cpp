#include <cuda.h>
#include <cerrno>
#include <fcntl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <filesystem>
#include <iostream>
#include "gpu_sm_masking.hpp"

namespace fs = std::filesystem;

#define HANDLE_ERROR(ret, errnum, ...) \
    do { \
        fprintf(stderr, __VA_ARGS__); \
        exit(ret); \
    } while (0)

struct SMControl {
    uint32_t is_enabled;
    uint64_t mask_value;
} __attribute__((packed));

constexpr CUuuid SM_CONTROL_CALLBACK_ID = {
    static_cast<int8_t>(0x2c), static_cast<int8_t>(0x8e), static_cast<int8_t>(0x0a), static_cast<int8_t>(0xd8),
    static_cast<int8_t>(0x07), static_cast<int8_t>(0x10), static_cast<int8_t>(0xab), static_cast<int8_t>(0x4e),
    static_cast<int8_t>(0x90), static_cast<int8_t>(0xdd), static_cast<int8_t>(0x54), static_cast<int8_t>(0x71),
    static_cast<int8_t>(0x9f), static_cast<int8_t>(0xe5), static_cast<int8_t>(0xf7), static_cast<int8_t>(0x4b)
};

constexpr int DOMAIN_LAUNCH = 0x3;
constexpr int PRE_UPLOAD_LAUNCH = 0x3;

static std::atomic<uint64_t> sm_mask{0};
static std::atomic<bool> sm_control_initialized{false};

void KernelLaunchCallback(void* user_data, int domain, int callback_id, const void* params) {
    if (*(static_cast<const uint32_t*>(params)) < 0x50) {
        std::cerr << "CUDA version too old for callback-based SM masking. Terminating...\n";
        return;
    }

    auto param_ptr = static_cast<uintptr_t**>(const_cast<void*>(params)) + 8;
    if (!*param_ptr) {
        std::cerr << "Null halLaunchDataAllocation encountered\n";
        return;
    }

    auto lower_mask_ptr = reinterpret_cast<uint32_t*>(**reinterpret_cast<char***>(param_ptr) + 84);
    auto upper_mask_ptr = reinterpret_cast<uint32_t*>(**reinterpret_cast<char***>(param_ptr) + 88);
    if (!*lower_mask_ptr && !*upper_mask_ptr) {
        *lower_mask_ptr = static_cast<uint32_t>(sm_mask.load());
        *upper_mask_ptr = static_cast<uint32_t>(sm_mask.load() >> 32);
    }
}

void InitializeSmControl() {
    if (sm_control_initialized.exchange(true)) {
        return;
    }

    using CallbackType = int (*)(uint32_t* handle, void (*callback)(void*, int, int, const void*), void* user_data);
    using EnableType = int (*)(uint32_t enable, uint32_t handle, int domain, int callback_id);
    
    const uintptr_t* export_table = nullptr;
    uint32_t handle = 0;
    cuGetExportTable(reinterpret_cast<const void**>(&export_table), &SM_CONTROL_CALLBACK_ID);

    auto subscribe_callback = reinterpret_cast<CallbackType>(export_table[3]);
    auto enable_callback = reinterpret_cast<EnableType>(export_table[6]);

    int result = subscribe_callback(&handle, KernelLaunchCallback, nullptr);
    if (result) {
        std::cerr << "Error subscribing to launch callback. Code " << result << "\n";
        return;
    }
    result = enable_callback(1, handle, DOMAIN_LAUNCH, PRE_UPLOAD_LAUNCH);
    if (result) {
        std::cerr << "Error enabling launch callback. Code " << result << "\n";
    }
}

void SetTPCMask(uint64_t mask) {
    int driver_version;
    cuDriverGetVersion(&driver_version);
    if (driver_version > 10020) {
        if (!sm_control_initialized.load()) {
            InitializeSmControl();
        }
        sm_mask.store(mask);
    } else {
        HANDLE_ERROR(1, ENOSYS, "TPC masking requires CUDA 10.2 or higher; this application is using CUDA %d.%d\n",
                     driver_version / 1000, driver_version % 100);
    }
}

std::optional<uint64_t> ReadProcFile(const std::string& filename) {
    char file_data[18] = {0};
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        return std::nullopt;
    }
    read(fd, file_data, sizeof(file_data));
    close(fd);
    return strtoll(file_data, nullptr, 16);
}

int GetGpcInfo(uint32_t& num_enabled_gpcs, std::unique_ptr<uint64_t[]>& tpcs_for_gpc, int dev) {
    uint32_t gpc_count = 0, tpc_count_per_gpc = 0, max_gpcs = 0;
    uint64_t gpc_mask = 0, gpc_tpc_mask = 0;

    num_enabled_gpcs = 0;

    std::string filename = "/proc/gpu" + std::to_string(dev) + "/num_gpcs";
    auto max_gpcs_opt = ReadProcFile(filename);
    if (!max_gpcs_opt) {
        std::cerr << "nvdebug module must be loaded before using gpu_sm_masking_get_*_info() functions\n";
        return ENOENT;
    }
    max_gpcs = static_cast<uint32_t>(*max_gpcs_opt);

    filename = "/proc/gpu" + std::to_string(dev) + "/gpc_mask";
    auto gpc_mask_opt = ReadProcFile(filename);
    if (!gpc_mask_opt) {
        return ENOENT;
    }
    gpc_mask = *gpc_mask_opt;

    filename = "/proc/gpu" + std::to_string(dev) + "/num_tpc_per_gpc";
    auto tpc_count_per_gpc_opt = ReadProcFile(filename);
    if (!tpc_count_per_gpc_opt) {
        return ENOENT;
    }
    tpc_count_per_gpc = static_cast<uint32_t>(*tpc_count_per_gpc_opt);

    tpcs_for_gpc = std::make_unique<uint64_t[]>(max_gpcs);
    uint32_t virtual_tpc_index = 0;
    for (uint32_t i = 0; i < max_gpcs; i++) {
        if ((1 << i) & gpc_mask) {
            continue;
        }
        num_enabled_gpcs++;
        filename = "/proc/gpu" + std::to_string(dev) + "/gpc" + std::to_string(i) + "_tpc_mask";
        auto gpc_tpc_mask_opt = ReadProcFile(filename);
        if (!gpc_tpc_mask_opt) {
            return ENOENT;
        }
        gpc_tpc_mask = *gpc_tpc_mask_opt;
        tpcs_for_gpc[num_enabled_gpcs - 1] = 0;
        for (uint32_t j = 0; j < tpc_count_per_gpc; j++) {
            if ((1 << j) & gpc_tpc_mask) {
                continue;
            }
            tpcs_for_gpc[num_enabled_gpcs - 1] |= (1ull << virtual_tpc_index);
            virtual_tpc_index++;
        }
    }

    return 0;
}

int GetTpcInfo(uint32_t& num_tpcs, int cuda_dev) {
    int num_sms = 0, major = 0, minor = 0;
    CUresult cuda_result = cuInit(0);
    if (cuda_result != CUDA_SUCCESS) {
        goto handle_cuda_error;
    }
    if ((cuda_result = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev)) != CUDA_SUCCESS) {
        goto handle_cuda_error;
    }
    if ((cuda_result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev)) != CUDA_SUCCESS) {
        goto handle_cuda_error;
    }
    if ((cuda_result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev)) != CUDA_SUCCESS) {
        goto handle_cuda_error;
    }

    if (major < 3 || (major == 3 && minor < 5)) {
        return ENOTSUP;
    }

    num_tpcs = (major > 6 || (major == 6 && minor == 0)) ? num_sms / 2 : num_sms;
    return 0;

handle_cuda_error:
    const char* error_string;
    cuGetErrorName(cuda_result, &error_string);
    std::cerr << "CUDA call failed: " << error_string << ". Returning EIO...\n";
    return EIO;
}
