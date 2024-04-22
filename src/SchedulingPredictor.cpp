#include <functional>
#include <iostream>
#include "../include/SchedulingPredictor.hpp"

SchedulingPredictor::SchedulingPredictor(int num_SMs) : num_SMs_(num_SMs)
{
    for (int i = 0; i < num_SMs_; i++) {
        round_robin_queue_.push(i);
    }
}

void SchedulingPredictor::Predict(int num_streams, int num_kernels_per_stream, int num_blocks_per_kernel)
{
    predictions_.clear();

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

std::vector<int> SchedulingPredictor::GetPredictions(int stream_id, const std::string& kernel_id)
{
    return predictions_[stream_id][kernel_id];
}

void SchedulingPredictor::OutputPredictions() const
{
    for (const auto& [stream_id, stream_predictions] : predictions_) {
        for (const auto& [kernel_id, sm_predictions] : stream_predictions) {
            std::cout << "Predictions for Stream " << stream_id << ", " << kernel_id << ": ";
            for (int sm_id : sm_predictions) {
                std::cout << sm_id << " ";
            }
            std::cout << std::endl;
        }
    }
}