#ifndef HARMONIC_MEAN_PREDICTOR_HPP
#define HARMONIC_MEAN_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"


class HarmonicMeanPredictor final : public Predictor {
public:
    int predict(const int a, const int b, int c) override {
        if (a + b == 0) return 0; // Prevent division by zero
        return std::clamp((2 * a * b) / (a + b), 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::HARMONIC_MEAN; }
};



#endif //HARMONIC_MEAN_PREDICTOR_HPP
