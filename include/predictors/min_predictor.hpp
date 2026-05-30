#ifndef MIN_PREDICTOR_HPP
#define MIN_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"

class MinPredictor final : public Predictor {
public:
    int predict(const int a, const int b, int c) override {
        return std::min(a, b);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::MIN_FILTER; }
};



#endif //MIN_PREDICTOR_HPP
