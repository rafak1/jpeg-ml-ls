#ifndef JPEGLS_AI_HPP
#define JPEGLS_AI_HPP

#include "loco_i.hpp"
#include "ai_logic.hpp"
#include <vector>
#include <memory>

class JpeglsAI {
public:
    explicit JpeglsAI(std::unique_ptr<AILogic> ai_logic, int chunk_height = 16);
    ~JpeglsAI();

    std::vector<uint8_t> encode(const std::vector<unsigned char>& image_data, int width, int height);
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded_data, int width, int height);

private:
    std::unique_ptr<LocoIEncoder> encoder_;
    std::unique_ptr<AILogic> ai_logic_;
    int chunk_height_;

    // Helper to get predictor based on type
    static std::unique_ptr<Predictor> getPredictorForType(PredictorType type);
};

#endif // JPEGLS_AI_HPP
