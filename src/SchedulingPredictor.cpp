#include <functional>
#include <iostream>
#include "../include/SchedulingPredictor.hpp"

SchedulingPredictor::SchedulingPredictor(int num_SMs) : num_SMs_(num_SMs)
{
    for (int i = 0; i < num_SMs_; i++) {
        round_robin_queue_.push(i);
    }
}

void SchedulingPredictor::RoundRobinPredict(const nlohmann::json& training_data)
{
    predictions_.clear();

    // Initialize variables to count the number of streams, kernels, and blocks
    int num_streams = 0;
    int num_kernels_per_stream = 0;
    int num_blocks_per_kernel = 0;
    std::unordered_map<int, int> stream_kernel_count;
    std::unordered_map<std::string, int> kernel_block_count;

    // Traverse the JSON object to count streams, kernels, and blocks
    for (const auto& kernel_data : training_data) {
        int stream_id = kernel_data["stream"];
        std::string kernel_name = kernel_data["kernel"];
        const auto& blocks_data = kernel_data["blocks"];
        
        // Update stream count
        if (stream_kernel_count.find(stream_id) == stream_kernel_count.end()) {
            stream_kernel_count[stream_id] = 0;
        }
        stream_kernel_count[stream_id]++;

        // Count kernels and blocks
        kernel_block_count[kernel_name] = blocks_data.size();
    }

    // Determine number of streams and maximum kernels per stream
    num_streams = stream_kernel_count.size();
    for (const auto& sk : stream_kernel_count) {
        if (sk.second > num_kernels_per_stream) {
            num_kernels_per_stream = sk.second;
        }
    }

    // Determine maximum blocks per kernel
    for (const auto& kb : kernel_block_count) {
        if (kb.second > num_blocks_per_kernel) {
            num_blocks_per_kernel = kb.second;
        }
    }

    // Run the Round Robin scheduling prediction
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
        auto& stream_predictions = predictions_[stream_id];

        for (int kernel_id = 0; kernel_id < num_kernels_per_stream; kernel_id++) {
            std::string kernel_key = "Kernel" + std::to_string(kernel_id);
            auto& block_predictions = stream_predictions[kernel_key];
            block_predictions.resize(num_blocks_per_kernel);

            std::generate(block_predictions.begin(), block_predictions.end(), [this]() {
                int sm_id = round_robin_queue_.front();
                round_robin_queue_.push(sm_id);
                round_robin_queue_.pop();
                return sm_id;
            });
        }
    }
}

void SchedulingPredictor::Predict(const nlohmann::json& training_data)
{
    predictions_.clear();
    auto used_sms = std::vector<int>(num_SMs_);
    

    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, 0);

    int stack_frame = 2064; // bytes, from ptxas output
    int spill_stores = 1128; // bytes, from ptxas output
    int spill_loads = 1668; // bytes, from ptxas output

    if (status != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(status) << std::endl;
        return;
    }

    for (const auto& kernel_data : training_data) {
        int stream_id = kernel_data["stream"];
        std::string kernel_name = kernel_data["kernel"];
        const auto& blocks_data = kernel_data["blocks"];

        std::vector<int>& block_predictions = predictions_[stream_id][kernel_name];
        block_predictions.resize(blocks_data.size());

        for (size_t block_index = 0; block_index < blocks_data.size(); block_index++) {
            const auto& block_data = blocks_data[block_index];

            // Extract block information
            int TB_t = block_data["thread_dim"];
            int TB_r = 128; // registers per thread
            int TB_s = block_data["shared_mem_size"];
            int TB_config_s = prop.sharedMemPerBlock; // shared memory configuration
            int TB_config_l = (stack_frame + spill_stores + spill_loads) / blocks_data.size(); // local memory configuration

            // Extract GPGPU hardware specifications
            int GPGPU_gran_r = 8; // register allocation granularity
            int GPGPU_gran_s = 128; // shared memory allocation granularity
            int GPGPU_cudart_s = 10000; // CUDA runtime shared memory

            int GPGPU_config_l = stack_frame + spill_stores + spill_loads; // GPGPU local memory configuration

            int max_thread_blocks = 0;
            int smid = 0;

            for (int i = 0; i < num_SMs_; i += 2) {
                // Extract SM resource status
                int SM_b = 16; // available thread block slots
                int SM_TPC_config_s = prop.sharedMemPerBlock; // TPC shared memory configuration
                int SM_s = prop.maxBlocksPerMultiProcessor * prop.sharedMemPerBlock; // available shared memory

                // Calculate limits
                int limit_b = SM_b;
                std::vector<int> limit_pb_w(4, 32); // available warp slots
                std::vector<int> limit_pb_r(4, 65536); // available registers
                for (int j = 0; j < 4; ++j) {
                    int r = std::ceil(TB_r * 32.0 / GPGPU_gran_r) * GPGPU_gran_r;
                    limit_pb_w[j] = std::min(limit_pb_w[j], static_cast<int>(std::floor(limit_pb_r[j] / static_cast<double>(r))));
                }
                int limit_w = *std::min_element(limit_pb_w.begin(), limit_pb_w.end()) * 4;
                int limit_s = SM_s / (TB_s + GPGPU_cudart_s);

                int cur_thread_blocks = std::min({limit_b, limit_w, limit_s});
                
                if (cur_thread_blocks > max_thread_blocks) {
                    max_thread_blocks = cur_thread_blocks;
                    smid = i;
                }
            }

            // Assign block to the selected SM
            block_predictions[block_index] = smid;
        }
    }
}

std::vector<int> SchedulingPredictor::GetPredictions(int stream_id, const std::string& kernel_id)
{
    return predictions_[stream_id][kernel_id];
}

void SchedulingPredictor::ComparePredictions(const nlohmann::json& training_data)
{
    for (const auto& kernel_data : training_data) {
        int stream_id = kernel_data["stream"];
        std::string kernel_name = kernel_data["kernel"];
        const auto& blocks_data = kernel_data["blocks"];

        std::vector<int> actual_mappings;
        for (const auto& block_data : blocks_data) {
            actual_mappings.push_back(block_data["sm_id"]);
        }

        std::vector<int> predicted_mappings = GetPredictions(stream_id, kernel_name);
        int num_correct_predictions = 0;
        std::cout << "Predictions for Stream " << stream_id << ", Kernel " << kernel_name << ":\n";
        
        for (size_t i = 0; i < actual_mappings.size(); ++i) {
            if (i < predicted_mappings.size()) {
                if (actual_mappings[i] == predicted_mappings[i]) {
                    num_correct_predictions++;
                } else {
                    std::cout << "Block " << i << ": Incorrect prediction, Actual SM: " << actual_mappings[i]
                              << ", Predicted SM: " << predicted_mappings[i] << std::endl;
                }
            }
        }
    }
}