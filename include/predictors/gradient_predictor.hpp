#ifndef GRADIENT_PREDICTOR_HPP
#define GRADIENT_PREDICTOR_HPP

#include "predictor.hpp"
#include <algorithm>

// A simple gradient predictor
class GradientPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        return std::clamp(a + b - c, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::GRADIENT; }
};

#endif // GRADIENT_PREDICTOR_HPP
