#ifndef ASYMMETRIC_TEXTURE_PREDICTOR_HPP
#define ASYMMETRIC_TEXTURE_PREDICTOR_HPP
#include <algorithm>

#include "predictor.hpp"


class AsymmetricTexturePredictor final : public Predictor {
public:
    int predict(const int a, const int b, const int c) override {
        return std::clamp((3 * a + 3 * b - 2 * c) / 4, 0, 255);
    }
    [[nodiscard]] PredictorType getType() const override { return PredictorType::ASYMMETRIC_TEXTURE; }
};



#endif //ASYMMETRIC_TEXTURE_PREDICTOR_HPP
