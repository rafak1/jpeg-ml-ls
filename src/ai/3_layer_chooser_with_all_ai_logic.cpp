#include "../../include/ai/3_layer_chooser_with_all_ai_logic.hpp"
#include "../../include/predictors/predictor_factory.hpp"
#include "../../include/config.hpp"
#include <iostream>
#include <filesystem>

ThreeLayerChooserWithAllAILogic::ThreeLayerChooserWithAllAILogic(const bool for_training) {
    if (for_training)
        return;
    load_model();
}

ThreeLayerChooserWithAllAILogic::~ThreeLayerChooserWithAllAILogic() {
    if (choice_counts_.empty()) return;
    std::cout << "\n--- Statistics for 3-Layer Chooser AI For All predictors ---" << std::endl;
    int total = 0;
    for (auto const& [type, count] : choice_counts_) total += count;
    for (auto const& [type, count] : choice_counts_) {
        std::cout << "  " << predictorTypeToString(type) << ": " << count
                  << " (" << (100.0 * count / total) << "%)" << std::endl;
    }
}

void ThreeLayerChooserWithAllAILogic::load_model() {
    if (const std::string path = "3_layer_fc_for_all_logic_model.dat"; std::filesystem::exists(path)) {
        dlib::deserialize(path) >> model;
    } else {
        std::cerr << "Warning: Could not load model from " << path << std::endl;
    }
}

std::unique_ptr<Predictor> ThreeLayerChooserWithAllAILogic::getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) {
    dlib::matrix<float> input_mat(CHUNK_SIZE * CHUNK_SIZE, 1);
    for (size_t i = 0; i < chunk_content.size(); ++i) {
        input_mat(i, 0) = static_cast<float>(chunk_content[i]);
    }

    unsigned long predicted_class = model(input_mat);
    PredictorType type = static_cast<PredictorType>(predicted_class);
    choice_counts_[type]++;
    return PredictorFactory::create(type);
}
