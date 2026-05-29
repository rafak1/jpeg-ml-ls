#include "jpegls_ai.hpp"
#include "med_predictor.hpp"
#include "paeth_predictor.hpp"
#include "left_predictor.hpp"
#include "top_predictor.hpp"
#include "top_left_predictor.hpp"
#include "average_left_top_predictor.hpp"
#include "dpcm_predictor.hpp"
#include "gradient_predictor.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

JpeglsAI::JpeglsAI(std::unique_ptr<AILogic> ai_logic, const int chunk_height)
    : ai_logic_(std::move(ai_logic)), chunk_height_(chunk_height) {
    std::cout << "JpeglsAI initialized with chunk height: " << chunk_height_ << std::endl;
    encoder_ = std::make_unique<LocoIEncoder>();
    std::cout << "Using LOCO-I Encoder implementation." << std::endl;
}

JpeglsAI::~JpeglsAI() {
    std::cout << "JpeglsAI freed." << std::endl;
}

std::unique_ptr<Predictor> JpeglsAI::getPredictorForType(const PredictorType type) {
    switch (type) {
        case PredictorType::MED:
            return std::make_unique<MEDPredictor>();
        case PredictorType::PAETH:
            return std::make_unique<PaethPredictor>();
        case PredictorType::LEFT:
            return std::make_unique<LeftPredictor>();
        case PredictorType::TOP:
            return std::make_unique<TopPredictor>();
        case PredictorType::TOP_LEFT:
            return std::make_unique<TopLeftPredictor>();
        case PredictorType::AVERAGE_LEFT_TOP:
            return std::make_unique<AverageLeftTopPredictor>();
        case PredictorType::DPCM:
            return std::make_unique<DPCMPredictor>();
        case PredictorType::GRADIENT:
            return std::make_unique<GradientPredictor>();
        default:
            std::cerr << "Unknown predictor type, falling back to MED." << std::endl;
            return std::make_unique<MEDPredictor>();
    }
}

std::vector<uint8_t> JpeglsAI::encode(const std::vector<unsigned char>& image_data, const int width, const int height) {
    if (!encoder_) {
        std::cerr << "No encoder implementation available!" << std::endl;
        return {};
    }

    std::vector<uint8_t> final_encoded_data;

    for (int y = 0; y < height; y += chunk_height_) {
        const int current_chunk_height = std::min(chunk_height_, height - y);
        const size_t chunk_start_index = y * width;
        const size_t chunk_size = current_chunk_height * width;

        std::vector<unsigned char> chunk_data(image_data.begin() + chunk_start_index, image_data.begin() + chunk_start_index + chunk_size);

        // AI chooses a predictor for the chunk
        std::unique_ptr<Predictor> predictor = ai_logic_->getPredictor(chunk_data);
        PredictorType predictor_type = predictor->getType();
        encoder_->setPredictor(std::move(predictor));

        std::vector<uint8_t> encoded_chunk = encoder_->encode(chunk_data, width, current_chunk_height);

        // Prepend the predictor type to the encoded chunk data
        final_encoded_data.push_back(static_cast<uint8_t>(predictor_type));

        // Append the size of the encoded chunk (as a 4-byte integer)
        const uint32_t chunk_data_size = encoded_chunk.size();
        final_encoded_data.push_back((chunk_data_size >> 24) & 0xFF);
        final_encoded_data.push_back((chunk_data_size >> 16) & 0xFF);
        final_encoded_data.push_back((chunk_data_size >> 8) & 0xFF);
        final_encoded_data.push_back(chunk_data_size & 0xFF);

        final_encoded_data.insert(final_encoded_data.end(), encoded_chunk.begin(), encoded_chunk.end());
    }

    std::cout << "Successfully encoded " << image_data.size() << " bytes into "
              << final_encoded_data.size() << " bytes of chunked bitstream." << std::endl;

    return final_encoded_data;
}

std::vector<uint8_t> JpeglsAI::decode(const std::vector<uint8_t>& encoded_data, const int width, const int height) {
    if (!encoder_) {
        std::cerr << "No decoder implementation available!" << std::endl;
        return {};
    }

    std::vector<uint8_t> decoded_image;
    size_t current_pos = 0;
    int current_height = 0;

    while (current_pos < encoded_data.size() && current_height < height) {
        // Read the predictor type
        const auto predictor_type = static_cast<PredictorType>(encoded_data[current_pos++]);

        // Read the chunk data size
        const uint32_t chunk_data_size = (encoded_data[current_pos] << 24) |
                                   (encoded_data[current_pos + 1] << 16) |
                                   (encoded_data[current_pos + 2] << 8) |
                                   encoded_data[current_pos + 3];
        current_pos += 4;

        // Set the predictor for the current chunk based on the read type
        encoder_->setPredictor(getPredictorForType(predictor_type));

        const int chunk_height = std::min(chunk_height_, height - current_height);

        std::vector<uint8_t> encoded_chunk(encoded_data.begin() + current_pos, encoded_data.begin() + current_pos + chunk_data_size);

        std::vector<uint8_t> decoded_chunk = encoder_->decode(encoded_chunk, width, chunk_height);
        decoded_image.insert(decoded_image.end(), decoded_chunk.begin(), decoded_chunk.end());

        current_pos += chunk_data_size;
        current_height += chunk_height;
    }

    std::cout << "Successfully decoded into " << decoded_image.size() << " bytes." << std::endl;
    return decoded_image;
}
