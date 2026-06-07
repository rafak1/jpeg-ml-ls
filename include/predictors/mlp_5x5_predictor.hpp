#ifndef MLP_5X5_PREDICTOR_HPP
#define MLP_5X5_PREDICTOR_HPP

#include "predictor.hpp"
#include "mlp_5x5_weights.hpp"
#include <algorithm>

class MLP5x5Predictor final : public Predictor {
public:
    int predict(int a, int b, int c) override { return predict(a, b, c, nullptr, 0, 0, 0); }

    int predict(int a, int, int, const uint8_t* data, int x, int y, int width) override {
        if (!data) return a;
        long long n[12];
        int idx = 0;
        for (int dy = -2; dy <= -1; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int nx = x + dx, ny = y + dy;
                n[idx++] = (ny >= 0 && nx >= 0 && nx < width) ? (int)data[ny * width + nx] - a : -a;
            }
        }
        n[idx++] = (x - 2 >= 0) ? (int)data[y * width + x - 2] - a : -a;
        n[idx++] = (long long)a - 128;

        long long h[64];
        for (int i = 0; i < 64; ++i) {
            long long sum = (long long)MLP5x5Weights::b1[i];
            for (int j = 0; j < 12; ++j) {
                sum += n[j] * MLP5x5Weights::w1[i * 12 + j];
            }
            h[i] = (sum > 0) ? sum : 0;
        }

        long long out = (long long)MLP5x5Weights::b2[0] * MLP5x5Weights::SCALE;
        for (int i = 0; i < 64; ++i) {
            out += h[i] * MLP5x5Weights::w2[i];
        }

        long long scale2 = (long long)MLP5x5Weights::SCALE * MLP5x5Weights::SCALE;
        int delta = (int)((out + scale2 / 2) / scale2);
        return std::clamp(a + delta, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::MLP_5X5; }
};

#endif
