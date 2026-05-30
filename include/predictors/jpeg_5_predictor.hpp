#ifndef JPEG_5_PREDICTOR_HPP
#define JPEG_5_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"

class Jpeg5Predictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        return std::clamp(a + (b - c) / 2, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::JPEG5; }
};

#endif // JPEG_5_PREDICTOR_HPP