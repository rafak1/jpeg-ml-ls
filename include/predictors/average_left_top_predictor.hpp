#ifndef AVERAGE_LEFT_TOP_PREDICTOR_HPP
#define AVERAGE_LEFT_TOP_PREDICTOR_HPP

#include "predictor.hpp"

// Predicts the average of left and top pixels
class AverageLeftTopPredictor final : public Predictor {
public:
    int predict(const int a, const int b, int c) override { return (a + b) / 2; }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::AVERAGE_LEFT_TOP; }
};

#endif // AVERAGE_LEFT_TOP_PREDICTOR_HPP
