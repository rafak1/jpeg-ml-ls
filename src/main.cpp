#include "jpegls_ai.hpp"
#include "random_ai_logic.hpp"
#include <charls/charls.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdint>
#include <chrono>

int main() {
    auto ai_logic = std::make_unique<RandomAILogic>();
    JpeglsAI ai_ctx(std::move(ai_logic), 16); // Chunk height of 16

    // Create a larger dummy image for better timing measurements
    constexpr int width = 512;
    constexpr int height = 512;
    std::vector<unsigned char> original_image(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            original_image[y * width + x] = static_cast<unsigned char>((x + y) % 256);
        }
    }

    // --- Our Implementation ---
    std::cout << "--- Compressing with Our Implementation ---" << std::endl;

    auto our_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> our_encoded_data = ai_ctx.encode(original_image, width, height);
    auto our_end = std::chrono::high_resolution_clock::now();

    if (our_encoded_data.empty()) {
        std::cerr << "Our encoding failed, aborting." << std::endl;
        return 1;
    }

    auto our_decode_start = std::chrono::high_resolution_clock::now();
    const std::vector<uint8_t> our_decoded_image = ai_ctx.decode(our_encoded_data, width, height);
    auto our_decode_end = std::chrono::high_resolution_clock::now();

    assert(original_image == our_decoded_image && "Our implementation failed round trip!");
    std::cout << "Our implementation round trip successful." << std::endl;


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
    std::cout << "Original Size:      " << original_image.size() << " bytes" << std::endl;
    std::cout << "Our Compressed Size:  " << our_encoded_data.size() << " bytes" << std::endl;
    std::cout << "CharLS Compr. Size: " << charls_encoded_data.size() << " bytes" << std::endl;

    if (our_encoded_data.size() < charls_encoded_data.size()) {
        std::cout << "Congratulations! Our implementation produced a smaller file!" << std::endl;
    } else {
        std::cout << "CharLS produced a smaller or equal size file. Lots of room for our AI to improve!" << std::endl;
    }

    // --- Timing Results ---
    std::chrono::duration<double, std::milli> our_duration = our_end - our_start;
    std::chrono::duration<double, std::milli> our_decode_duration = our_decode_end - our_decode_start;
    std::chrono::duration<double, std::milli> charls_duration = charls_end - charls_start;

    std::cout << "\n--- Speed Results ---" << std::endl;
    std::cout << "Our Encode Time:    " << our_duration.count() << " ms" << std::endl;
    std::cout << "Our Decode Time:    " << our_decode_duration.count() << " ms" << std::endl;
    std::cout << "CharLS Encode Time: " << charls_duration.count() << " ms" << std::endl;

    return 0;
}
