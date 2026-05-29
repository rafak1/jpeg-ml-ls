#ifndef MED_PREDICTOR_HPP
#define MED_PREDICTOR_HPP

#include "predictor.hpp"
#include <algorithm>

// Median Edge Detection (JPEG-LS default)
class MEDPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        if (c >= std::max(a, b)) {
            return std::min(a, b);
        } else if (c <= std::min(a, b)) {
            return std::max(a, b);
        } else {
            return a + b - c;
        }
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::MED; }
};

#endif // MED_PREDICTOR_HPP
