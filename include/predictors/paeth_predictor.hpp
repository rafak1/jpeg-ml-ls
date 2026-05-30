#ifndef PAETH_PREDICTOR_HPP
#define PAETH_PREDICTOR_HPP

#include "predictor.hpp"
#include <cmath>

// Paeth predictor (from PNG)
class PaethPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        const int p = a + b - c; // Initial estimate
        const int pa = std::abs(p - a);
        const int pb = std::abs(p - b);
        const int pc = std::abs(p - c);

        // Return the neighbor closest to the estimate
        if (pa <= pb && pa <= pc) {
            return a;
        } else if (pb <= pc) {
            return b;
        } else {
            return c;
        }
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::PAETH; }
};

#endif // PAETH_PREDICTOR_HPP
