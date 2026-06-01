#ifndef PREDICTOR_FACTORY_HPP
#define PREDICTOR_FACTORY_HPP

#include "predictor.hpp"

#include "med_predictor.hpp"
#include "paeth_predictor.hpp"
#include "left_predictor.hpp"
#include "top_predictor.hpp"
#include "top_left_predictor.hpp"
#include "average_left_top_predictor.hpp"
#include "dpcm_predictor.hpp"
#include "gradient_predictor.hpp"

#include "jpeg_5_predictor.hpp"
#include "jpeg_6_predictor.hpp"
#include "strict_median_predictor.hpp"
#include "multiplicative_predictor.hpp"
#include "planar_balanced_predictor.hpp"
#include "asymmetric_texture_predictor.hpp"
#include "horizontal_sharp_edge_predictor.hpp"
#include "vertical_sharp_edge_predictor.hpp"

#include "max_predictor.hpp"
#include "min_predictor.hpp"
#include "harmonic_mean_predictor.hpp"
#include "mid_range_predictor.hpp"
#include "mlp_predictor.hpp"

#include <array>
#include <memory>
#include <iostream>

class PredictorFactory {
private:
    using CreatorFn = std::unique_ptr<Predictor> (*)();

    template <typename T>
    static std::unique_ptr<Predictor> make() {
        return std::make_unique<T>();
    }

    // Must match the exact order of the PredictorType enum!
    static constexpr std::array<CreatorFn, static_cast<size_t>(PredictorType::COUNT)> registry_ = {
        &make<MEDPredictor>,                  // 0
        &make<PaethPredictor>,                // 1
        &make<LeftPredictor>,                 // 2
        &make<TopPredictor>,                  // 3
        &make<TopLeftPredictor>,              // 4
        &make<AverageLeftTopPredictor>,       // 5
        &make<DPCMPredictor>,                 // 6
        &make<GradientPredictor>,             // 7
        &make<Jpeg5Predictor>,                // 8
        &make<Jpeg6Predictor>,                // 9
        &make<StrictMedianPredictor>,         // 10
        &make<MultiplicativePredictor>,       // 11
        &make<PlanarBalancedPredictor>,       // 12
        &make<AsymmetricTexturePredictor>,    // 13
        &make<HorizontalSharpEdgePredictor>,  // 14
        &make<VerticalSharpEdgePredictor>,    // 15
        &make<MaxPredictor>,                  // 16
        &make<MinPredictor>,                  // 17
        &make<HarmonicMeanPredictor>,         // 18
        &make<MidrangePredictor>,             // 19
        &make<MLPPredictor>                   // 20
    };

public:
    static std::unique_ptr<Predictor> create(const PredictorType type) {
        const auto index = static_cast<size_t>(type);
        if (index < registry_.size()) {
            return registry_[index]();
        }

        std::cerr << "Unknown predictor type (" << index << "), falling back to MED." << std::endl;
        return make<MEDPredictor>();
    }

    static constexpr int getCount() {
        return registry_.size();
    }
};



#endif //PREDICTOR_FACTORY_HPP
