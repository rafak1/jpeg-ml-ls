#ifndef MLP_PREDICTOR_HPP
#define MLP_PREDICTOR_HPP

#include "predictor.hpp"
#include "mlp_weights.hpp"
#include <algorithm>

class MLPPredictor final : public Predictor {
public:
    int predict(int a, int b, int c) override {
        long long in[3] = { (long long)a - 128, (long long)b - a, (long long)c - a };
        long long h[64];
        for (int i = 0; i < 64; ++i) {
            long long sum = (long long)MLPWeights::b1[i];
            for (int j = 0; j < 3; ++j) {
                sum += in[j] * MLPWeights::w1[i * 3 + j];
            }
            h[i] = (sum > 0) ? sum : sum / 10;
        }

        long long out = 0;
        for (int i = 0; i < 64; ++i) {
            out += h[i] * MLPWeights::w2[i];
        }
        out += (long long)MLPWeights::b2[0] * MLPWeights::SCALE;

        long long scale2 = (long long)MLPWeights::SCALE * MLPWeights::SCALE;
        int delta = (int)((out + scale2 / 2) / scale2);
        return std::clamp(a + delta, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::MLP; }
};

#endif
