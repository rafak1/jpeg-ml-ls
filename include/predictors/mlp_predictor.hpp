#ifndef MLP_PREDICTOR_HPP
#define MLP_PREDICTOR_HPP

#include "predictor.hpp"
#include "mlp_weights.hpp"
#include <algorithm>

class MLPPredictor final : public Predictor {
public:
    int predict(int a, int b, int c) override {
        // Hidden layer (3 -> 32)
        long long h[32];
        for (int i = 0; i < 32; ++i) {
            long long sum = (long long)a * MLPWeights::w1[i * 3 + 0] +
                            (long long)b * MLPWeights::w1[i * 3 + 1] +
                            (long long)c * MLPWeights::w1[i * 3 + 2] +
                            (long long)MLPWeights::b1[i];
            h[i] = (sum > 0) ? sum : 0; // ReLU
        }

        // Output layer (32 -> 1)
        long long out = 0;
        for (int i = 0; i < 32; ++i) {
            out += h[i] * MLPWeights::w2[i]; // Result is SCALE^2 scaled
        }
        out += (long long)MLPWeights::b2[0] * MLPWeights::SCALE;

        long long scale2 = (long long)MLPWeights::SCALE * MLPWeights::SCALE;
        int result = (int)((out + scale2 / 2) / scale2);
        
        return std::clamp(result, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::MLP; }
};

#endif // MLP_PREDICTOR_HPP
