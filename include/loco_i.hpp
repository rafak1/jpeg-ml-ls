#ifndef LOCO_I_HPP
#define LOCO_I_HPP

#include "bit_stream.hpp"
#include "predictor.hpp"
#include <vector>
#include <cstdint>
#include <memory>

// Core JPEG-LS (LOCO-I) Encoder/Decoder Architecture
class LocoIEncoder {
public:
    LocoIEncoder();
    ~LocoIEncoder() = default;

    std::vector<uint8_t> encode(const std::vector<uint8_t>& image_data, int width, int height);
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded_data, int width, int height);

    void setPredictor(std::unique_ptr<Predictor> predictor) { predictor_ = std::move(predictor); }
    Predictor* getPredictor() const { return predictor_.get(); }

private:
    // Constants for 8-bit images as per JPEG-LS standard
    static constexpr int MAXVAL = 255;
    static constexpr int RANGE = (MAXVAL + 1);
    static constexpr int qbpp = 8;
    static constexpr int bpp = 8;
    static constexpr int LIMIT = 2 * (bpp + std::max(8, bpp));

    // Default thresholds for 8-bit
    static constexpr int T1 = 3;
    static constexpr int T2 = 7;
    static constexpr int T3 = 21;

    static constexpr int RESET = 64;
    static constexpr int NUM_CONTEXTS = 365;

    // Context state arrays
    std::vector<int> N_;
    std::vector<int> A_;
    std::vector<int> B_;
    std::vector<int> C_;

    std::unique_ptr<Predictor> predictor_;

    void resetContexts();

    static int quantizeGradient(int gradient);

    static void getContext(int d1, int d2, int d3, int& Q, int& sign);

    [[nodiscard]] int computeGolombK(int Q) const;

    static void encodeMappedError(BitWriter& writer, int mapped_error, int k);

    static int decodeMappedError(BitReader& reader, int k);
    void updateContext(int Q, int error);
};

#endif // LOCO_I_HPP
