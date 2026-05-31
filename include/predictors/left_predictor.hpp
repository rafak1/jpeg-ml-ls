#ifndef LEFT_PREDICTOR_HPP
#define LEFT_PREDICTOR_HPP

#include "predictor.hpp"

// Predicts exactly the pixel to the left (a)
class LeftPredictor final : public Predictor {
public:
    int predict(const int a, int b, int c) override { return a; }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::LEFT; }
};

#endif // LEFT_PREDICTOR_HPP
