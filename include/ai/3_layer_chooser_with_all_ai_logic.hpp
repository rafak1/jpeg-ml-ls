//
// Created by mattt on 12.06.2026.
//

#ifndef INC_3_LAYER_CHOOSER_WITH_ALL_AI_LOGIC_HPP
#define INC_3_LAYER_CHOOSER_WITH_ALL_AI_LOGIC_HPP

#include "../ai_logic.hpp"
#include "../predictors/predictor.hpp"
#include "../config.hpp"
#include <vector>
#include <memory>
#include <map>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using three_layer_net_type_all = dlib::loss_multiclass_log<
                            dlib::fc<static_cast<int>(PredictorType::COUNT),
                            dlib::relu<dlib::fc<128,
                            dlib::relu<dlib::fc<CHUNK_SIZE,
                            dlib::input<dlib::matrix<float>>
                            >>>>>>;

class ThreeLayerChooserWithAllAILogic final : public AILogic {
private:
    void load_model();
    mutable std::map<PredictorType, int> choice_counts_;
public:
    explicit ThreeLayerChooserWithAllAILogic(bool for_training);
    ~ThreeLayerChooserWithAllAILogic() override;

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) override;

    three_layer_net_type_all model;
};

#endif //INC_3_LAYER_CHOOSER_WITH_ALL_AI_LOGIC_HPP
