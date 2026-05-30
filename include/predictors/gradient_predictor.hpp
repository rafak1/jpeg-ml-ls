#ifndef GRADIENT_PREDICTOR_HPP
#define GRADIENT_PREDICTOR_HPP

#include "predictor.hpp"
#include <algorithm>

// A simple gradient predictor
class GradientPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        if (c >= std::max(a, b)) {
            return std::min(a, b);
        }
        if (c <= std::min(a, b)) {
            return std::max(a, b);
        }
        return a + b - c;
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::GRADIENT; }
};

#endif // GRADIENT_PREDICTOR_HPP
