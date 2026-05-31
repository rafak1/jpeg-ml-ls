#ifndef MULTIPLICATIVE_PREDICTOR_HPP
#define MULTIPLICATIVE_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"

class MultiplicativePredictor final : public Predictor {
public:
    int predict(const int a, const int b, int c) override {
        if (c == 0) c = 1; // Prevent division by zero
        return std::clamp((a * b) / c, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::MULTIPLICATIVE; }
};



#endif //MULTIPLICATIVE_PREDICTOR_HPP
