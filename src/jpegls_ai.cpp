#include "jpegls_ai.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include "predictors/predictor_factory.hpp"

JpeglsAI::JpeglsAI(std::unique_ptr<AILogic> ai_logic, int chunk_width, int chunk_height)
    : ai_logic_(std::move(ai_logic)), chunk_width_(chunk_width), chunk_height_(chunk_height) {
    std::cout << "JpeglsAI initialized with chunk size: " << chunk_width_ << "x" << chunk_height_ << std::endl;
    encoder_ = std::make_unique<LocoIEncoder>();
    std::cout << "Using LOCO-I Encoder implementation." << std::endl;
}

JpeglsAI::~JpeglsAI() {
    std::cout << "JpeglsAI freed." << std::endl;
}

std::vector<uint8_t> JpeglsAI::encode(const std::vector<unsigned char>& image_data, const int width, const int height) const {
    if (!encoder_) {
        std::cerr << "No encoder implementation available!" << std::endl;
        return {};
    }

    std::vector<uint8_t> final_encoded_data;

    for (int y = 0; y < height; y += chunk_height_) {
        for (int x = 0; x < width; x += chunk_width_) {
            const int current_chunk_width = std::min(chunk_width_, width - x);
            const int current_chunk_height = std::min(chunk_height_, height - y);

            // Extract the 2D chunk data
            std::vector<unsigned char> chunk_data;
            chunk_data.reserve(current_chunk_width * current_chunk_height);
            for (int row = 0; row < current_chunk_height; ++row) {
                const int start_index = (y + row) * width + x;
                chunk_data.insert(chunk_data.end(), image_data.begin() + start_index, image_data.begin() + start_index + current_chunk_width);
            }

            // AI chooses a predictor for the chunk.
            std::unique_ptr<Predictor> predictor;
            if (current_chunk_width < chunk_width_ || current_chunk_height < chunk_height_) {
                std::vector<unsigned char> padded_chunk(chunk_width_ * chunk_height_, 0);
                for (int row = 0; row < current_chunk_height; ++row) {
                    std::copy(chunk_data.begin() + row * current_chunk_width,
                              chunk_data.begin() + row * current_chunk_width + current_chunk_width,
                              padded_chunk.begin() + row * chunk_width_);
                }
                predictor = ai_logic_->getPredictor(padded_chunk, chunk_width_, chunk_height_);
            } else {
                predictor = ai_logic_->getPredictor(chunk_data, current_chunk_width, current_chunk_height);
            }

            PredictorType predictor_type = predictor->getType();
            encoder_->setPredictor(std::move(predictor));

            std::vector<uint8_t> encoded_chunk = encoder_->encode(chunk_data, current_chunk_width, current_chunk_height);

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
    }

    std::cout << "Successfully encoded " << image_data.size() << " bytes into "
              << final_encoded_data.size() << " bytes of chunked bitstream." << std::endl;

    return final_encoded_data;
}

std::vector<uint8_t> JpeglsAI::decode(const std::vector<uint8_t>& encoded_data, const int width, const int height) const {
    if (!encoder_) {
        std::cerr << "No decoder implementation available!" << std::endl;
        return {};
    }

    std::vector<uint8_t> decoded_image(width * height);
    size_t current_pos = 0;

    for (int y = 0; y < height; y += chunk_height_) {
        for (int x = 0; x < width; x += chunk_width_) {
            // Read the predictor type
            const auto predictor_type = static_cast<PredictorType>(encoded_data[current_pos++]);

            // Read the chunk data size
            const uint32_t chunk_data_size = (encoded_data[current_pos] << 24) |
                                       (encoded_data[current_pos + 1] << 16) |
                                       (encoded_data[current_pos + 2] << 8) |
                                       encoded_data[current_pos + 3];
            current_pos += 4;

            // Set the predictor for the current chunk based on the read type
            encoder_->setPredictor(PredictorFactory::create(predictor_type));

            const int current_chunk_width = std::min(chunk_width_, width - x);
            const int current_chunk_height = std::min(chunk_height_, height - y);

            std::vector<uint8_t> encoded_chunk(encoded_data.begin() + current_pos, encoded_data.begin() + current_pos + chunk_data_size);

            std::vector<uint8_t> decoded_chunk = encoder_->decode(encoded_chunk, current_chunk_width, current_chunk_height);

            // Place the decoded chunk back into the full image
            for (int row = 0; row < current_chunk_height; ++row) {
                const int dest_start_index = (y + row) * width + x;
                const int src_start_index = row * current_chunk_width;
                std::copy(decoded_chunk.begin() + src_start_index, decoded_chunk.begin() + src_start_index + current_chunk_width, decoded_image.begin() + dest_start_index);
            }

            current_pos += chunk_data_size;
        }
    }

    std::cout << "Successfully decoded into " << decoded_image.size() << " bytes." << std::endl;
    return decoded_image;
}
