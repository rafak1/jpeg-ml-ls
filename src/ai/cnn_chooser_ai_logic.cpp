#include "ai/cnn_chooser_ai_logic.hpp"
#include "predictors/predictor_factory.hpp"
#include "config.hpp"
#include <iostream>
#include <filesystem>

CnnChooserAILogic::CnnChooserAILogic(bool loadModel) {
    if (!loadModel)
        return;
    load_model();
}

CnnChooserAILogic::~CnnChooserAILogic() {
    if (choice_counts_.empty()) return;
    std::cout << "\n--- Statistics for CNN Chooser AI ---" << std::endl;
    int total = 0;
    for (auto const& [type, count] : choice_counts_) total += count;
    for (auto const& [type, count] : choice_counts_) {
        std::cout << "  " << predictorTypeToString(type) << ": " << count 
                  << " (" << (100.0 * count / total) << "%)" << std::endl;
    }
}

void CnnChooserAILogic::load_model() {
    if (const std::string path = "cnn_predictor_model.dat"; std::filesystem::exists(path)) {
        dlib::deserialize(path) >> model;
    } else {
        std::cerr << "Warning: Could not load model from " << path << std::endl;
    }
}

std::unique_ptr<Predictor> CnnChooserAILogic::getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) {
    dlib::matrix<unsigned char> input_mat(CHUNK_SIZE, CHUNK_SIZE);
    for (int r = 0; r < CHUNK_SIZE; ++r) {
        for (int c = 0; c < CHUNK_SIZE; ++c) {
            input_mat(r, c) = chunk_content[r * CHUNK_SIZE + c];
        }
    }

    unsigned long predicted_class = model(input_mat);
    PredictorType type = static_cast<PredictorType>(predicted_class);
    choice_counts_[type]++;
    return PredictorFactory::create(type);
}