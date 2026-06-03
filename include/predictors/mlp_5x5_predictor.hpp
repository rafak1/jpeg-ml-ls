#ifndef MLP_5X5_PREDICTOR_HPP
#define MLP_5X5_PREDICTOR_HPP

#include "predictor.hpp"
#include "mlp_5x5_weights.hpp"
#include <algorithm>

class MLP5x5Predictor final : public Predictor {
public:
    int predict(int a, int b, int c) override { return predict(a, b, c, nullptr, 0, 0, 0); }

    int predict(int /*a*/, int /*b*/, int /*c*/, const uint8_t* data, int x, int y, int width) override {
        if (!data) return 0;

        long long n[12];
        int idx = 0;

        // Row y-2
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = x + dx, ny = y - 2;
            n[idx++] = (ny >= 0 && nx >= 0 && nx < width) ? data[ny * width + nx] : 0;
        }
        // Row y-1
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = x + dx, ny = y - 1;
            n[idx++] = (ny >= 0 && nx >= 0 && nx < width) ? data[ny * width + nx] : 0;
        }
        // Row y (left pixels)
        for (int dx = -2; dx <= -1; ++dx) {
            int nx = x + dx;
            n[idx++] = (nx >= 0) ? data[y * width + nx] : 0;
        }

        // Hidden layer (12 -> 64)
        long long h[64];
        for (int i = 0; i < 64; ++i) {
            long long sum = (long long)MLP5x5Weights::b1[i];
            for (int j = 0; j < 12; ++j) {
                sum += n[j] * MLP5x5Weights::w1[i * 12 + j];
            }
            h[i] = (sum > 0) ? sum : 0; // ReLU
        }

        // Output layer (64 -> 1)
        long long out = (long long)MLP5x5Weights::b2[0] * MLP5x5Weights::SCALE;
        for (int i = 0; i < 64; ++i) {
            out += h[i] * MLP5x5Weights::w2[i];
        }

        long long scale2 = (long long)MLP5x5Weights::SCALE * MLP5x5Weights::SCALE;
        int result = (int)((out + scale2 / 2) / scale2);
        return std::clamp(result, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::MLP_5X5; }
};

#endif // MLP_5X5_PREDICTOR_HPP
