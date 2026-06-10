#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <cstdint>

enum class PredictorType {
    MED = 0,
    PAETH = 1,
    LEFT = 2,
    TOP = 3,
    TOP_LEFT = 4,
    AVERAGE_LEFT_TOP = 5,
    DPCM = 6,
    GRADIENT = 7,
    JPEG5 = 8,
    JPEG6 = 9,
    STRICT_MEDIAN = 10,
    MULTIPLICATIVE = 11,
    PLANAR_BALANCED = 12,
    ASYMMETRIC_TEXTURE = 13,
    HORIZ_SHARP = 14,
    VERT_SHARP = 15,
    MAX_FILTER = 16,
    MIN_FILTER = 17,
    HARMONIC_MEAN = 18,
    MIDRANGE = 19,
    MLP = 20,
    MLP_5X5 = 21,
    NEURAL_BLENDER = 22,
    COUNT = 22
};

inline const char* predictorTypeToString(PredictorType type) {
    switch (type) {
        case PredictorType::MED: return "MED";
        case PredictorType::PAETH: return "PAETH";
        case PredictorType::LEFT: return "LEFT";
        case PredictorType::TOP: return "TOP";
        case PredictorType::TOP_LEFT: return "TOP_LEFT";
        case PredictorType::AVERAGE_LEFT_TOP: return "AVG_LEFT_TOP";
        case PredictorType::DPCM: return "DPCM";
        case PredictorType::GRADIENT: return "GRADIENT";
        case PredictorType::JPEG5: return "JPEG5";
        case PredictorType::JPEG6: return "JPEG6";
        case PredictorType::STRICT_MEDIAN: return "STRICT_MED";
        case PredictorType::MULTIPLICATIVE: return "MULTIPLICATIVE";
        case PredictorType::PLANAR_BALANCED: return "PLANAR_BAL";
        case PredictorType::ASYMMETRIC_TEXTURE: return "ASYM_TEXTURE";
        case PredictorType::HORIZ_SHARP: return "HORIZ_SHARP";
        case PredictorType::VERT_SHARP: return "VERT_SHARP";
        case PredictorType::MAX_FILTER: return "MAX_FILTER";
        case PredictorType::MIN_FILTER: return "MIN_FILTER";
        case PredictorType::HARMONIC_MEAN: return "HARMONIC_MEAN";
        case PredictorType::MIDRANGE: return "MIDRANGE";
        case PredictorType::MLP: return "MLP";
        case PredictorType::MLP_5X5: return "MLP_5X5";
        case PredictorType::NEURAL_BLENDER: return "NEURAL_BLENDER";
        default: return "Unknown";
    }
}

class Predictor {
public:
    virtual ~Predictor() = default;
    virtual int predict(int a, int b, int c) = 0;
    virtual int predict(int a, int b, int c, const uint8_t* /*data*/, int /*x*/, int /*y*/, int /*width*/) {
        return predict(a, b, c);
    }
    [[nodiscard]] virtual PredictorType getType() const = 0;
};

#endif
