#ifndef NON_AI_BEST_LOGIC_HPP
#define NON_AI_BEST_LOGIC_HPP

#include "ai_logic.hpp"
#include "loco_i.hpp"
#include <vector>
#include <memory>
#include <limits>
#include "predictors/predictor_factory.hpp"

class NonAIBestLogic final : public AILogic {
private:
    std::vector<PredictorType> traditional_types_;
    std::unique_ptr<LocoIEncoder> trial_encoder_;

public:
    NonAIBestLogic() : trial_encoder_(std::make_unique<LocoIEncoder>()) {
        // Only include traditional predictors (0 to 19)
        // MLP starts at 20
        for (int type = 0; type < 20; ++type) {
            traditional_types_.push_back(static_cast<PredictorType>(type));
        }
    }

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) override {
        auto best_predictor_type = PredictorType::MED;
        size_t smallest_size = std::numeric_limits<size_t>::max();

        for (const auto& type : traditional_types_) {
            trial_encoder_->setPredictor(PredictorFactory::create(type));
            std::vector<uint8_t> encoded_chunk = trial_encoder_->encode(chunk_content, width, height);

            if (encoded_chunk.size() < smallest_size) {
                smallest_size = encoded_chunk.size();
                best_predictor_type = type;
            }
        }
        return PredictorFactory::create(best_predictor_type);
    }
};

#endif // NON_AI_BEST_LOGIC_HPP
