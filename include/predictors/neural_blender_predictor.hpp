#ifndef NEURAL_BLENDER_PREDICTOR_HPP
#define NEURAL_BLENDER_PREDICTOR_HPP

#include "predictor.hpp"
#include "med_predictor.hpp"
#include "mlp_predictor.hpp"
#include "gradient_predictor.hpp"
#include "blender_weights.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>

class NeuralBlenderPredictor final : public Predictor {
private:
    MEDPredictor med_;
    MLPPredictor mlp_;
    GradientPredictor grad_;

    static inline long double total_w0 = 0, total_w1 = 0, total_w2 = 0;
    static inline long long total_pixels = 0;

    struct StatsReporter {
        ~StatsReporter() {
            if (total_pixels == 0) return;
            std::cout << "\n--- Statistics for Neural Blender ---" << std::endl;
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  Average Weight MED:      " << (total_w0 / total_pixels) * 100.0 << "%" << std::endl;
            std::cout << "  Average Weight MLP:      " << (total_w1 / total_pixels) * 100.0 << "%" << std::endl;
            std::cout << "  Average Weight Gradient: " << (total_w2 / total_pixels) * 100.0 << "%" << std::endl;
            std::cout << "  Total Pixels Processed:  " << total_pixels << std::endl;
        }
    };
    static inline StatsReporter reporter;

public:
    int predict(int a, int b, int c) override {
        long long in[3] = { (long long)a - 128, (long long)b - a, (long long)c - a };
        long long h[16];
        for (int i = 0; i < 16; ++i) {
            long long sum = (long long)BlenderWeights::b1[i];
            for (int j = 0; j < 3; ++j) {
                sum += in[j] * BlenderWeights::w1[i * 3 + j];
            }
            h[i] = (sum > 0) ? sum : sum / 10;
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
        
        // Accumulate relative weights for statistics
        total_w0 += (long double)w[0] / total_w;
        total_w1 += (long double)w[1] / total_w;
        total_w2 += (long double)w[2] / total_w;
        total_pixels++;

        long long p0 = med_.predict(a, b, c);
        long long p1 = mlp_.predict(a, b, c);
        long long p2 = grad_.predict(a, b, c);

        long long val = (w[0] * p0 + w[1] * p1 + w[2] * p2 + (total_w / 2)) / total_w;
        return std::clamp((int)val, 0, 255);
    }

    [[nodiscard]] PredictorType getType() const override { return PredictorType::NEURAL_BLENDER; }
};

#endif
