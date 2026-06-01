#include "jpegls_ai.hpp"
#include "random_ai_logic.hpp"
#include "deterministic_best_logic.hpp"
#include "selected_one_logic.hpp"
#include <charls/charls.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <chrono>

int main() {
    // Create a larger dummy image for better timing measurements
    constexpr int width = 512;
    constexpr int height = 512;
    std::vector<unsigned char> original_image(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            original_image[y * width + x] = static_cast<unsigned char>((x + y) % 256);
        }
    }

    // --- Run with RandomAILogic ---
    std::cout << "--- Compressing with RandomAILogic ---" << std::endl;
    auto random_ai_logic = std::make_unique<RandomAILogic>();
    JpeglsAI random_ai_ctx(std::move(random_ai_logic), 128, 128);

    auto random_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> random_encoded_data = random_ai_ctx.encode(original_image, width, height);
    auto random_end = std::chrono::high_resolution_clock::now();

    const std::vector<uint8_t> random_decoded_image = random_ai_ctx.decode(random_encoded_data, width, height);
    assert(original_image == random_decoded_image && "RandomAILogic failed round trip!");
    std::cout << "RandomAILogic round trip successful." << std::endl;

    // --- Run with DeterministicBestLogic ---
    std::cout << "\n--- Compressing with DeterministicBestLogic ---" << std::endl;
    auto deterministic_ai_logic = std::make_unique<DeterministicBestLogic>();
    JpeglsAI deterministic_ai_ctx(std::move(deterministic_ai_logic), 128, 128);

    auto deterministic_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> deterministic_encoded_data = deterministic_ai_ctx.encode(original_image, width, height);
    auto deterministic_end = std::chrono::high_resolution_clock::now();

    const std::vector<uint8_t> deterministic_decoded_image = deterministic_ai_ctx.decode(deterministic_encoded_data, width, height);
    assert(original_image == deterministic_decoded_image && "DeterministicBestLogic failed round trip!");
    std::cout << "DeterministicBestLogic round trip successful." << std::endl;

    // --- Run with MLPPredictor ---
    std::cout << "\n--- Compressing with MLP Predictor ---" << std::endl;
    auto mlp_ai_logic = std::make_unique<SelectedOneLogic>(PredictorType::MLP);
    JpeglsAI mlp_ai_ctx(std::move(mlp_ai_logic), 128, 128);

    auto mlp_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> mlp_encoded_data = mlp_ai_ctx.encode(original_image, width, height);
    auto mlp_end = std::chrono::high_resolution_clock::now();

    const std::vector<uint8_t> mlp_decoded_image = mlp_ai_ctx.decode(mlp_encoded_data, width, height);
    assert(original_image == mlp_decoded_image && "MLPPredictor failed round trip!");
    std::cout << "MLPPredictor round trip successful." << std::endl;


    // --- CharLS Implementation ---
    std::cout << "\n--- Compressing with CharLS Library ---" << std::endl;
    std::vector<uint8_t> charls_encoded_data(width * height * 2); // Pre-allocate buffer

    auto charls_start = std::chrono::high_resolution_clock::now();
    charls::jpegls_encoder encoder;
    encoder.destination(charls_encoded_data);
    encoder.frame_info({static_cast<uint32_t>(width), static_cast<uint32_t>(height), 8, 1});
    const auto charls_bytes_written = encoder.encode(original_image);
    auto charls_end = std::chrono::high_resolution_clock::now();

    charls_encoded_data.resize(charls_bytes_written);
    std::cout << "CharLS encoded successfully." << std::endl;

    // --- Comparison ---
    std::cout << "\n--- Compression Results ---" << std::endl;
    std::cout << "Original Size:                 " << original_image.size() << " bytes" << std::endl;
    std::cout << "Random Logic Compressed Size:    " << random_encoded_data.size() << " bytes" << std::endl;
    std::cout << "Deterministic Compressed Size: " << deterministic_encoded_data.size() << " bytes" << std::endl;
    std::cout << "MLP Compressed Size:           " << mlp_encoded_data.size() << " bytes" << std::endl;
    std::cout << "CharLS Compressed Size:        " << charls_encoded_data.size() << " bytes" << std::endl;

    // --- Timing Results ---
    std::chrono::duration<double, std::milli> random_duration = random_end - random_start;
    std::chrono::duration<double, std::milli> deterministic_duration = deterministic_end - deterministic_start;
    std::chrono::duration<double, std::milli> mlp_duration = mlp_end - mlp_start;
    std::chrono::duration<double, std::milli> charls_duration = charls_end - charls_start;

    std::cout << "\n--- Speed Results (Encode) ---" << std::endl;
    std::cout << "Random Logic Time:        " << random_duration.count() << " ms" << std::endl;
    std::cout << "Deterministic Logic Time: " << deterministic_duration.count() << " ms" << std::endl;
    std::cout << "MLP Predictor Time:       " << mlp_duration.count() << " ms" << std::endl;
    std::cout << "CharLS Time:              " << charls_duration.count() << " ms" << std::endl;

    return 0;
}
