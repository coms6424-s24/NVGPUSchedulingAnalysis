#ifndef SCHEDULING_PREDICTOR_HPP
#define SCHEDULING_PREDICTOR_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "../include/json.hpp"

// Class to predict thread block scheduling on SMs
class SchedulingPredictor {
public:
    explicit SchedulingPredictor(int num_SMs);

    // Initialize round-robin predictions for each block in each kernel on each stream
    void RoundRobinPredict(const nlohmann::json& training_data);

    // Predict each block in each kernel on each stream
    void Predict(const nlohmann::json& training_data);

    // Get predictions for a specific stream and kernel
    std::vector<int> GetPredictions(int stream_id, const std::string& kernel_id);

    // compare its prediction results with the ground truth
    void ComparePredictions(const nlohmann::json& training_data);

private:
    int num_SMs_;  // Number of Streaming Multiprocessors
    // Maps to store predictions for each stream and kernel
    std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>> predictions_;
    std::queue<int> round_robin_queue_;  // Queue to implement round-robin scheduling
};

#endif // SCHEDULING_PREDICTOR_HPP