#ifndef JPEGLS_AI_HPP
#define JPEGLS_AI_HPP

#include "loco_i.hpp"
#include "ai_logic.hpp"
#include <vector>
#include <memory>

class JpeglsAI {
public:
    explicit JpeglsAI(std::unique_ptr<AILogic> ai_logic, int chunk_width = 16, int chunk_height = 16);
    ~JpeglsAI();

    std::vector<uint8_t> encode(const std::vector<unsigned char>& image_data, int width, int height) const;
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded_data, int width, int height) const;

private:
    std::unique_ptr<LocoIEncoder> encoder_;
    std::unique_ptr<AILogic> ai_logic_;
    int chunk_width_;
    int chunk_height_;
};

#endif // JPEGLS_AI_HPP
