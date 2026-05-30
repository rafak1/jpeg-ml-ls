#ifndef RANDOM_AI_LOGIC_HPP
#define RANDOM_AI_LOGIC_HPP

#include "ai_logic.hpp"
#include "predictors/predictor_factory.hpp"
#include <random>
#include <memory>
#include <vector>

// Simple AI that randomly chooses a predictor
class RandomAILogic final : public AILogic {
private:
    std::mt19937 gen_;
    std::uniform_int_distribution<> distrib_;

public:
    RandomAILogic()
        : gen_(std::random_device{}()),
          distrib_(0, PredictorFactory::getCount() - 1)
    {}

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content) override {
        const auto random_type = static_cast<PredictorType>(distrib_(gen_));
        return PredictorFactory::create(random_type);
    }
};

#endif // RANDOM_AI_LOGIC_HPP