#ifndef SELECTED_ONE_LOGIC_HPP
#define SELECTED_ONE_LOGIC_HPP
#include "ai_logic.hpp"
#include "predictors/predictor_factory.hpp"


class SelectedOneLogic final : public AILogic {
private:
    PredictorType type_;

public:
    explicit SelectedOneLogic(const PredictorType type): type_(type) {  }
    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& /*chunk_content*/, int /*width*/, int /*height*/) override {
        return PredictorFactory::create(type_);
    }
};



#endif //SELECTED_ONE_LOGIC_HPP
