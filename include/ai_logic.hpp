#ifndef AI_LOGIC_HPP
#define AI_LOGIC_HPP

#include "predictor.hpp"
#include "med_predictor.hpp"
#include "paeth_predictor.hpp"
#include "left_predictor.hpp"
#include "top_predictor.hpp"
#include "top_left_predictor.hpp"
#include "average_left_top_predictor.hpp"
#include "dpcm_predictor.hpp"
#include "gradient_predictor.hpp"
#include <vector>
#include <memory>
#include <random>

// Abstract base class for AI logic
class AILogic {
public:
    virtual ~AILogic() = default;
    virtual std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content) = 0;
};

// Simple AI that randomly chooses a predictor
class RandomAILogic final : public AILogic {
public:
    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content) override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 7);

        switch (distrib(gen)) {
            case 0: return std::make_unique<MEDPredictor>();
            case 1: return std::make_unique<PaethPredictor>();
            case 2: return std::make_unique<LeftPredictor>();
            case 3: return std::make_unique<TopPredictor>();
            case 4: return std::make_unique<TopLeftPredictor>();
            case 5: return std::make_unique<AverageLeftTopPredictor>();
            case 6: return std::make_unique<DPCMPredictor>();
            case 7: return std::make_unique<GradientPredictor>();
            default: return std::make_unique<MEDPredictor>(); // Fallback
        }
    }
};

#endif // AI_LOGIC_HPP
