#ifndef THREE_LAYER_CHOOSER_AI_LOGIC_HPP
#define THREE_LAYER_CHOOSER_AI_LOGIC_HPP

#include "../ai_logic.hpp"
#include "../predictors/predictor.hpp"
#include "../config.hpp"
#include <vector>
#include <memory>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using net_type_3_layer = dlib::loss_multiclass_log<
                            dlib::fc<20,
                            dlib::relu<dlib::fc<128,
                            dlib::relu<dlib::fc<CHUNK_SIZE,
                            dlib::input<dlib::matrix<float>>
                            >>>>>>;

class ThreeLayerChooserAILogic final : public AILogic {
private:
    void load_model();
public:
    explicit ThreeLayerChooserAILogic(bool loadModel);

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) override;

    net_type_3_layer model;
};

#endif // THREE_LAYER_CHOOSER_AI_LOGIC_HPP