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
            long long sum = in[0] * MLPWeights::w1[i * 3 + 0] +
                            in[1] * MLPWeights::w1[i * 3 + 1] +
                            in[2] * MLPWeights::w1[i * 3 + 2] +
                            (long long)MLPWeights::b1[i];
            h[i] = (sum > 0) ? sum : 0;
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
