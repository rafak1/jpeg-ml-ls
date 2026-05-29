#ifndef JPEGLS_AI_HPP
#define JPEGLS_AI_HPP

#include <vector>

enum class AIChooserType {
    None = 0,
    NeuralNet,
    Heuristic,
    Custom
};

class AIContext {
public:
    AIContext(AIChooserType type);
    ~AIContext();

    int encode(const std::vector<unsigned char>& image_data, int width, int height);

private:
    AIChooserType type_;
    int model_id_;
};

#endif // JPEGLS_AI_HPP
