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
#include "mlp_5x5_predictor.hpp"
#include "neural_blender_predictor.hpp"

#include <array>
#include <memory>
#include <iostream>

class PredictorFactory {
private:
    using CreatorFn = std::unique_ptr<Predictor> (*)();
    template <typename T> static std::unique_ptr<Predictor> make() { return std::make_unique<T>(); }

    static constexpr std::array<CreatorFn, 23> registry_ = {
        &make<MEDPredictor>, &make<PaethPredictor>, &make<LeftPredictor>, &make<TopPredictor>,
        &make<TopLeftPredictor>, &make<AverageLeftTopPredictor>, &make<DPCMPredictor>, &make<GradientPredictor>,
        &make<Jpeg5Predictor>, &make<Jpeg6Predictor>, &make<StrictMedianPredictor>, &make<MultiplicativePredictor>,
        &make<PlanarBalancedPredictor>, &make<AsymmetricTexturePredictor>, &make<HorizontalSharpEdgePredictor>,
        &make<VerticalSharpEdgePredictor>, &make<MaxPredictor>, &make<MinPredictor>, &make<HarmonicMeanPredictor>,
        &make<MidrangePredictor>, &make<MLPPredictor>, &make<MLP5x5Predictor>, &make<NeuralBlenderPredictor>
    };

public:
    static std::unique_ptr<Predictor> create(const PredictorType type) {
        const auto index = static_cast<size_t>(type);
        if (index < registry_.size()) return registry_[index]();
        return make<MEDPredictor>();
    }
    static constexpr int getCount() { return registry_.size(); }
};

#endif
