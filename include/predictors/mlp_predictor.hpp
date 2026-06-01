#ifndef MLP_PREDICTOR_HPP
#define MLP_PREDICTOR_HPP

#include "predictor.hpp"
#include "mlp_weights.hpp"
#include <algorithm>

class MLPPredictor final : public Predictor {
public:
    int predict(int a, int b, int c) override {
        // Hidden layer (3 -> 8)
        int hidden[8];
        for (int i = 0; i < 8; ++i) {
            long long sum = (long long)a * MLPWeights::w1[i * 3 + 0] +
                            (long long)b * MLPWeights::w1[i * 3 + 1] +
                            (long long)c * MLPWeights::w1[i * 3 + 2] +
                            (long long)MLPWeights::b1[i] * 1; // Bias (scaled)
            
            // ReLU and scale down
            hidden[i] = (sum > 0) ? (int)(sum / MLPWeights::SCALE) : 0;
        }

        // Output layer (8 -> 1)
        long long out_sum = 0;
        for (int i = 0; i < 8; ++i) {
            out_sum += (long long)hidden[i] * MLPWeights::w2[i];
        }
        out_sum += (long long)MLPWeights::b2[0]; // Output bias (scaled)

        int result = (int)(out_sum / MLPWeights::SCALE);
        return std::clamp(result, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::MLP; }
};

#endif // MLP_PREDICTOR_HPP
