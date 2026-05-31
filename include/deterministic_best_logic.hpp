#ifndef DETERMINISTIC_BEST_LOGIC_HPP
#define DETERMINISTIC_BEST_LOGIC_HPP

#include "ai_logic.hpp"
#include "loco_i.hpp"
#include <vector>
#include <memory>
#include <limits>

#include "predictors/predictor_factory.hpp"

class DeterministicBestLogic final : public AILogic {
private:
    std::vector<PredictorType> predictor_types_;

public:
    DeterministicBestLogic() : predictor_types_(static_cast<int>(PredictorType::COUNT)),
    trial_encoder_(std::make_unique<LocoIEncoder>()) {
        for (int type = 0; type < static_cast<int>(PredictorType::COUNT); ++type) {
            predictor_types_[type] = static_cast<PredictorType>(type);
        }
    }

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) override {
        auto best_predictor_type = PredictorType::MED;
        size_t smallest_size = std::numeric_limits<size_t>::max();


        for (const auto& type : predictor_types_) {
            trial_encoder_->setPredictor(getPredictorForType(type));

            std::vector<uint8_t> encoded_chunk = trial_encoder_->encode(chunk_content, width, height);

            if (encoded_chunk.size() < smallest_size) {
                smallest_size = encoded_chunk.size();
                best_predictor_type = type;
            }
        }

        return getPredictorForType(best_predictor_type);
    }

private:
    std::unique_ptr<LocoIEncoder> trial_encoder_;
    static std::unique_ptr<Predictor> getPredictorForType(const PredictorType type) {
        return PredictorFactory::create(type);
    }
};

#endif // DETERMINISTIC_BEST_LOGIC_HPP
