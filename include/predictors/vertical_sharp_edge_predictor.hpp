#ifndef VERTICAL_SHARP_EDGE_PREDICTOR_HPP
#define VERTICAL_SHARP_EDGE_PREDICTOR_HPP
#include <algorithm>
#include <cmath>

#include "predictor.hpp"


class VerticalSharpEdgePredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        // If top-left and left are almost identical, follow the column trend
        if (std::abs(c - a) < 4) {
            return b;
        }
        return std::clamp((2 * b + a) / 3, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::VERT_SHARP; }
};



#endif //VERTICAL_SHARP_EDGE_PREDICTOR_HPP
