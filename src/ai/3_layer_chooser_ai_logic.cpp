#include "ai/3_layer_chooser_ai_logic.hpp"
#include "predictors/predictor_factory.hpp"
#include "config.hpp"
#include <iostream>
#include <filesystem>

ThreeLayerChooserAILogic::ThreeLayerChooserAILogic(bool loadModel) {
    if (!loadModel)
            return;
    load_model();
}

void ThreeLayerChooserAILogic::load_model() {
    if (const std::string path = "3_layer_fc_logic_model.dat"; std::filesystem::exists(path)) {
        dlib::deserialize(path) >> model;
    } else {
        std::cerr << "Warning: Could not load model from " << path << std::endl;
    }
}

std::unique_ptr<Predictor> ThreeLayerChooserAILogic::getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) {
    dlib::matrix<float> input_mat(CHUNK_SIZE * CHUNK_SIZE, 1);
    for (size_t i = 0; i < chunk_content.size(); ++i) {
        input_mat(i, 0) = static_cast<float>(chunk_content[i]);
    }

    unsigned long predicted_class = model(input_mat);
    return PredictorFactory::create(static_cast<PredictorType>(predicted_class));
}