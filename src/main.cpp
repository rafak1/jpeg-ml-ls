#include "../include/jpegls_ai.hpp"
#include <iostream>
#include <vector>

int main() {
    AIContext ai_ctx(AIChooserType::NeuralNet);

    // Dummy image data
    std::vector<unsigned char> image_data(100);
    for (int i = 0; i < 100; ++i) {
        image_data[i] = static_cast<unsigned char>(i);
    }

    if (ai_ctx.encode(image_data, 10, 10) != 0) {
        std::cerr << "Failed to encode image with AI chooser" << std::endl;
        return 1;
    }

    std::cout << "JPEG-LS with AI chooser executed successfully." << std::endl;
    return 0;
}
