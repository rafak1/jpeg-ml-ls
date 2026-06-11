//
// Created by mattt on 11.06.2026.
//

#ifndef INC_2_LAYER_CHOOSER_WITH_ALL_AI_LOGIC_HPP
#define INC_2_LAYER_CHOOSER_WITH_ALL_AI_LOGIC_HPP

#include "../ai_logic.hpp"
#include "../predictors/predictor.hpp"
#include "../config.hpp"
#include <vector>
#include <memory>
#include <map>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using net_type_all = dlib::loss_multiclass_log<
                            dlib::fc<static_cast<int>(PredictorType::COUNT),
                            dlib::relu<dlib::fc<CHUNK_SIZE,
                            dlib::input<dlib::matrix<float>>
                            >>>>;

class TwoLayerChooserWithAllAILogic final : public AILogic {
private:
    void load_model();
    mutable std::map<PredictorType, int> choice_counts_;
public:
    explicit TwoLayerChooserWithAllAILogic(bool for_training);
    ~TwoLayerChooserWithAllAILogic() override;

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) override;

    net_type_all model;
};

#endif //INC_2_LAYER_CHOOSER_WITH_ALL_AI_LOGIC_HPP
