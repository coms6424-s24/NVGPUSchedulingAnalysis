#ifndef SCHEDULING_PREDICTOR_HPP
#define SCHEDULING_PREDICTOR_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <string>

// Class to predict thread block scheduling on SMs
class SchedulingPredictor {
public:
    explicit SchedulingPredictor(int num_SMs);

    // Initialize round-robin predictions for each block in each kernel on each stream
    void Predict(int num_streams, int num_kernels_per_stream, int num_blocks_per_kernel);

    // Get predictions for a specific stream and kernel
    std::vector<int> GetPredictions(int stream_id, const std::string& kernel_id);

    // Output current predictions for manual verification
    void OutputPredictions() const;

private:
    int num_SMs_;  // Number of Streaming Multiprocessors
    // Maps to store predictions for each stream and kernel
    std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>> predictions_;
    std::queue<int> round_robin_queue_;  // Queue to implement round-robin scheduling
};

#endif // SCHEDULING_PREDICTOR_HPP