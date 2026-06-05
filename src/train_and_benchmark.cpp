#include "utils/training_utils.hpp"
#include "utils/benchmark_utils.hpp"

int main(int argc, char** argv) {
    train_2_layer_model();
    train_3_layer_model();
    train_cnn_model();
    run_benchmarks(argc, argv);
    return 0;
}