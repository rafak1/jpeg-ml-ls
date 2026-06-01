#ifndef MLP_WEIGHTS_HPP
#define MLP_WEIGHTS_HPP
#include <array>

namespace MLPWeights {
    static constexpr int SCALE = 1024;
    static constexpr std::array<int, 24> w1 = { 497, 444, 696, -197, -118, 456, -487, -123, 421, -546, -511, 202, 636, -372, 315, -181, -55, 301, 97, 493, -225, 58, -61, 121 };
    static constexpr std::array<int, 8> b1 = { 544, 54, 479, 34, 81, -419, 912, -812 };
    static constexpr std::array<int, 8> w2 = { 322, 343, 185, -84, 210, -71, 542, -43 };
    static constexpr std::array<int, 1> b2 = { 695 };
}
#endif
