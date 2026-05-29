#include "../include/jpegls_ai.hpp"
#include <iostream>

AIContext::AIContext(AIChooserType type) : type_(type), model_id_(0) {
    std::cout << "AI Chooser initialized with type: " << static_cast<int>(type_) << std::endl;
}

AIContext::~AIContext() {
    std::cout << "AI Chooser context freed." << std::endl;
}

int AIContext::encode(const std::vector<unsigned char>& image_data, int width, int height) {
    std::cout << "Encoding image with AI chooser type: " << static_cast<int>(type_) << std::endl;
    // Placeholder for actual encoding logic
    return 0;
}
