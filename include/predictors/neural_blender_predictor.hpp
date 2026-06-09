#ifndef NEURAL_BLENDER_PREDICTOR_HPP
#define NEURAL_BLENDER_PREDICTOR_HPP

#include "predictor.hpp"
#include "med_predictor.hpp"
#include "mlp_predictor.hpp"
#include "gradient_predictor.hpp"
#include "blender_weights.hpp"
#include <algorithm>

class NeuralBlenderPredictor final : public Predictor {
private:
    MEDPredictor med_;
    MLPPredictor mlp_;
    GradientPredictor grad_;

public:
    int predict(int a, int b, int c) override {
        long long in[3] = { (long long)a - 128, (long long)b - a, (long long)c - a };
        long long h[16];
        for (int i = 0; i < 16; ++i) {
            long long sum = in[0] * BlenderWeights::w1[i * 3 + 0] +
                            in[1] * BlenderWeights::w1[i * 3 + 1] +
                            in[2] * BlenderWeights::w1[i * 3 + 2] +
                            (long long)BlenderWeights::b1[i];
            h[i] = (sum > 0) ? sum : 0;
        }

        long long w[3];
        for (int i = 0; i < 3; ++i) {
            w[i] = (long long)BlenderWeights::b2[i] * BlenderWeights::SCALE;
            for (int j = 0; j < 16; ++j) {
                w[i] += h[j] * BlenderWeights::w2[i * 16 + j];
            }
            if (w[i] < 0) w[i] = 0;
        }

        long long total_w = w[0] + w[1] + w[2] + 1;
        long long p0 = med_.predict(a, b, c);
        long long p1 = mlp_.predict(a, b, c);
        long long p2 = grad_.predict(a, b, c);

        long long val = (w[0] * p0 + w[1] * p1 + w[2] * p2 + (total_w / 2)) / total_w;
        return std::clamp((int)val, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::MED; }
};

#endif
