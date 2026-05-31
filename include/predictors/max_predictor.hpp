#ifndef MAX_PREDICTOR_HPP
#define MAX_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"

class MaxPredictor final : public Predictor {
public:
    int predict(const int a, const int b, int c) override {
        return std::max(a, b);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::MAX_FILTER; }
};



#endif //MAX_PREDICTOR_HPP
