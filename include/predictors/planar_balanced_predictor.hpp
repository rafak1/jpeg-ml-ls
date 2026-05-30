#ifndef PLANAR_BALANCED_PREDICTOR_HPP
#define PLANAR_BALANCED_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"


class PlanarBalancedPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        return std::clamp((2 * a + 2 * b - c) / 3, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::PLANAR_BALANCED; }
};



#endif //PLANAR_BALANCED_PREDICTOR_HPP
