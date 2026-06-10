#ifndef BASE_MED_LOGIC_HPP
#define BASE_MED_LOGIC_HPP

#include "ai_logic.hpp"
#include "predictors/predictor_factory.hpp"

class BaseMEDLogic final : public AILogic {
public:
    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& /*chunk_content*/, int /*width*/, int /*height*/) override {
        return PredictorFactory::create(PredictorType::MED);
    }
};

#endif // BASE_MED_LOGIC_HPP
