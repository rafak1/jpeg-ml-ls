#include "jpegls_ai.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstdint>

int main() {
    auto ai_logic = std::make_unique<RandomAILogic>();
    JpeglsAI ai_ctx(std::move(ai_logic), 16); // Chunk height of 16

    // Create a more interesting dummy image with some gradients and flat areas
    constexpr int width = 32;
    constexpr int height = 32;
    std::vector<unsigned char> original_image(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            original_image[y * width + x] = static_cast<unsigned char>((x + y) % 256);
        }
    }

    // 1. Encode the image
    const std::vector<uint8_t> encoded_data = ai_ctx.encode(original_image, width, height);

    if (encoded_data.empty()) {
        std::cerr << "Encoding failed, aborting." << std::endl;
        return 1;
    }

    // 2. Decode the image
    const std::vector<uint8_t> decoded_image = ai_ctx.decode(encoded_data, width, height);

    if (decoded_image.empty()) {
        std::cerr << "Decoding failed, aborting." << std::endl;
        return 1;
    }

    // 3. Verify the round trip
    std::cout << "Verifying the encode/decode round trip..." << std::endl;
    assert(original_image.size() == decoded_image.size() && "Size mismatch after decoding!");

    bool match = true;
    for (size_t i = 0; i < original_image.size(); ++i) {
        if (original_image[i] != decoded_image[i]) {
            std::cerr << "Mismatch found at pixel " << i << "! "
                      << "Original: " << (int)original_image[i]
                      << ", Decoded: " << (int)decoded_image[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Success! Decoded image is identical to the original." << std::endl;
    } else {
        std::cerr << "Error! The decoded image does not match the original." << std::endl;
        return 1;
    }

    std::cout << "JPEG-LS with chunked AI Context executed successfully." << std::endl;
    return 0;
}
