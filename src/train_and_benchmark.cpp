#include "utils/training_utils.hpp"
#include "utils/benchmark_utils.hpp"
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::thread> threads;
    //threads.emplace_back(train_2_layer_model);
    //threads.emplace_back(train_3_layer_model);
    //threads.emplace_back(train_cnn_model);
    threads.emplace_back(train_2_layer_model_all);
    threads.emplace_back(train_3_layer_model_all);
    threads.emplace_back(train_cnn_model_all);

    for (auto& t : threads) {
        t.join();
    }

    run_benchmarks(argc, argv);
    return 0;
}
