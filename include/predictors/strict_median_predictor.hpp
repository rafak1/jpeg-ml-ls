#ifndef STRICT_MEDIAN_PREDICTOR_HPP
#define STRICT_MEDIAN_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"


class StrictMedianPredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        int vals[3] = {a, b, c};
        std::sort(vals, vals + 3);
        return vals[1];
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::STRICT_MEDIAN; }
};



#endif //STRICT_MEDIAN_PREDICTOR_HPP
