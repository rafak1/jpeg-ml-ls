#ifndef HORIZONTAL_SHARP_EDGE_PREDICTOR_HPP
#define HORIZONTAL_SHARP_EDGE_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"

class HorizontalSharpEdgePredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        // If top-left and top are almost identical, follow the left pixel's row trend
        if (std::abs(c - b) < 4) {
            return a;
        }
        return std::clamp((2 * a + b) / 3, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::HORIZ_SHARP; }
};



#endif //HORIZONTAL_SHARP_EDGE_PREDICTOR_HPP
