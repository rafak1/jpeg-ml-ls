#ifndef TOP_LEFT_PREDICTOR_HPP
#define TOP_LEFT_PREDICTOR_HPP

#include "predictor.hpp"

// Predicts exactly the pixel to the top-left (c)
class TopLeftPredictor final : public Predictor {
public:
    int predict(int a, int b, const int c) override { return c; }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::TOP_LEFT; }
};

#endif // TOP_LEFT_PREDICTOR_HPP
