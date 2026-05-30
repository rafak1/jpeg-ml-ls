#ifndef DPCM_PREDICTOR_HPP
#define DPCM_PREDICTOR_HPP

#include "predictor.hpp"

// Predicts using basic DPCM (Differential Pulse-Code Modulation)
class DPCMPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override { return a + b - c; }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::DPCM; }
};

#endif // DPCM_PREDICTOR_HPP
