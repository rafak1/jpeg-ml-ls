#ifndef AI_LOGIC_HPP
#define AI_LOGIC_HPP

#include "predictor.hpp"
#include <vector>
#include <memory>

// Abstract base class for AI logic
class AILogic {
public:
    virtual ~AILogic() = default;
    virtual std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content) = 0;
};

#endif // AI_LOGIC_HPP
