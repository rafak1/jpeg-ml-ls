#ifndef JPEG_6_PREDICTOR_HPP
#define JPEG_6_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"


class Jpeg6Predictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        return std::clamp(b + (a - c) / 2, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::JPEG6; }
};



#endif //JPEG_6_PREDICTOR_HPP
