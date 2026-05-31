#ifndef TOP_PREDICTOR_HPP
#define TOP_PREDICTOR_HPP

#include "predictor.hpp"

// Predicts exactly the pixel to the top (b)
class TopPredictor final : public Predictor {
public:
    int predict(int a, const int b, int c) override { return b; }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::TOP; }
};

#endif // TOP_PREDICTOR_HPP
