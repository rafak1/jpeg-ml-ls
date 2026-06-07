#include "utils/benchmark_utils.hpp"
#include "jpegls_ai.hpp"
#include "random_ai_logic.hpp"
#include "deterministic_best_logic.hpp"
#include "non_ai_best_logic.hpp"
#include "selected_one_logic.hpp"
#include "ai/2_layer_chooser_ai_logic.hpp"
#include "ai/3_layer_chooser_ai_logic.hpp"
#include "ai/cnn_chooser_ai_logic.hpp"
#include "config.hpp"
#include <charls/charls.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <fstream>

namespace {
    void benchmark_logic(const std::string& name, std::unique_ptr<AILogic> logic, const std::vector<unsigned char>& original_image, int width, int height,
                         std::vector<std::pair<std::string, double>>& times, std::vector<std::pair<std::string, size_t>>& sizes) {
        std::cout << "\n--- Compressing with " << name << " ---" << std::endl;
        const JpeglsAI ctx(std::move(logic), CHUNK_SIZE, CHUNK_SIZE);

        const auto start = std::chrono::high_resolution_clock::now();
        const std::vector<uint8_t> encoded_data = ctx.encode(original_image, width, height);
        const auto end = std::chrono::high_resolution_clock::now();

        const std::vector<uint8_t> decoded_image = ctx.decode(encoded_data, width, height);
        assert(original_image == decoded_image && "Round trip failed!");
        std::cout << name << " round trip successful." << std::endl;

        const std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back({name, duration.count()});
        sizes.push_back({name, encoded_data.size()});
    }
}

void run_benchmarks(int argc, char** argv) {
    std::cout << "\n=== Running Benchmarks ===" << std::endl;
    int width = 512;
    int height = 512;
    std::vector<unsigned char> original_image;

    if (argc > 3) {
        width = std::stoi(argv[2]);
        height = std::stoi(argv[3]);
        std::ifstream ifs(argv[1], std::ios::binary);
        if (!ifs) {
            std::cerr << "Failed to open file: " << argv[1] << std::endl;
            return;
        }
        original_image.resize(width * height);
        ifs.read(reinterpret_cast<char *>(original_image.data()), width * height);
        if (ifs.gcount() != static_cast<std::streamsize>(width * height)) {
            std::cerr << "Warning: Only read " << ifs.gcount() << " of " << width * height << " bytes." << std::endl;
        }
    } else {
        original_image.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                original_image[y * width + x] = static_cast<unsigned char>((x + y) % 256);
            }
        }
    }

    std::vector<std::pair<std::string, double>> times;
    std::vector<std::pair<std::string, size_t>> sizes;

    benchmark_logic("RandomAILogic", std::make_unique<RandomAILogic>(), original_image, width, height, times, sizes);
    benchmark_logic("DeterministicBestLogic", std::make_unique<DeterministicBestLogic>(), original_image, width, height, times, sizes);
    benchmark_logic("NonAIBestLogic", std::make_unique<NonAIBestLogic>(), original_image, width, height, times, sizes);
    benchmark_logic("MLP Predictor", std::make_unique<SelectedOneLogic>(PredictorType::MLP), original_image, width, height, times, sizes);
    benchmark_logic("MLP 5x5 Predictor", std::make_unique<SelectedOneLogic>(PredictorType::MLP_5X5), original_image, width, height, times, sizes);
    benchmark_logic("Neural Blender", std::make_unique<SelectedOneLogic>(PredictorType::NEURAL_BLENDER), original_image, width, height, times, sizes);
    benchmark_logic("2-Layer Chooser AI", std::make_unique<TwoLayerChooserAILogic>(false), original_image, width, height, times, sizes);
    benchmark_logic("3-Layer Chooser AI", std::make_unique<ThreeLayerChooserAILogic>(false), original_image, width, height, times, sizes);
    benchmark_logic("CNN Chooser AI", std::make_unique<CnnChooserAILogic>(false), original_image, width, height, times, sizes);

    // --- CharLS Implementation ---
    std::cout << "\n--- Compressing with CharLS Library ---" << std::endl;
    std::vector<uint8_t> charls_encoded_data(width * height * 2);

    auto charls_start = std::chrono::high_resolution_clock::now();
    charls::jpegls_encoder encoder;
    encoder.destination(charls_encoded_data);
    encoder.frame_info({static_cast<uint32_t>(width), static_cast<uint32_t>(height), 8, 1});
    const auto charls_bytes_written = encoder.encode(original_image);
    auto charls_end = std::chrono::high_resolution_clock::now();

    charls_encoded_data.resize(charls_bytes_written);
    std::cout << "CharLS encoded successfully." << std::endl;

    std::chrono::duration<double, std::milli> charls_duration = charls_end - charls_start;
    times.push_back({"CharLS", charls_duration.count()});
    sizes.push_back({"CharLS", charls_encoded_data.size()});

    // --- Comparison ---
    std::cout << "\n--- Compression Results ---" << std::endl;
    std::cout << "Original Size:                 " << original_image.size() << " bytes" << std::endl;
    for (const auto& size : sizes) {
        std::cout << size.first << " Compressed Size: " << size.second << " bytes" << std::endl;
    }

    // --- Timing Results ---
    std::cout << "\n--- Speed Results (Encode) ---" << std::endl;
    for (const auto& time : times) {
        std::cout << time.first << " Time: " << time.second << " ms" << std::endl;
    }
}