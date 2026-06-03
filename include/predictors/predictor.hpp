#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <cstdint>

// Enum to identify predictor types, used by the AI context
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
    COUNT = 22
};

// Abstract base class for all predictors
class Predictor {
public:
    virtual ~Predictor() = default;
    virtual int predict(int a, int b, int c) = 0;
    
    // Advanced prediction with more context (default calls simple predict)
    virtual int predict(int a, int b, int c, const uint8_t* /*data*/, int /*x*/, int /*y*/, int /*width*/) {
        return predict(a, b, c);
    }

    [[nodiscard]] virtual PredictorType getType() const = 0;
};

#endif // PREDICTOR_HPP
