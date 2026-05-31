#ifndef MID_RANGE_PREDICTOR_HPP
#define MID_RANGE_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"


class MidrangePredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        const int highest = std::max({a, b, c});
        const int lowest = std::min({a, b, c});
        return std::clamp((highest + lowest) / 2, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::MIDRANGE; }
};


#endif //MID_RANGE_PREDICTOR_HPP
