#ifndef CNN_CHOOSER_WITH_ALL_AI_LOGIC_HPP
#define CNN_CHOOSER_WITH_ALL_AI_LOGIC_HPP

#include "../ai_logic.hpp"
#include "../predictors/predictor.hpp"
#include "../config.hpp"
#include <vector>
#include <memory>
#include <map>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using cnn_net_type_all = dlib::loss_multiclass_log<dlib::fc<static_cast<int>(PredictorType::COUNT),
                            dlib::relu<dlib::fc<84,
                            dlib::relu<dlib::fc<120,
                            dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<16,5,5,1,1,
                            dlib::max_pool<2,2,2,2,dlib::relu<dlib::con<6,5,5,1,1,
                            dlib::input<dlib::matrix<unsigned char>>
                            >>>>>>>>>>>>;

class CnnChooserWithAllAILogic : public AILogic {
private:
    void load_model();
    mutable std::map<PredictorType, int> choice_counts_;
public:
    CnnChooserWithAllAILogic(bool load_model);
    ~CnnChooserWithAllAILogic() override;

    std::unique_ptr<Predictor> getPredictor(const std::vector<unsigned char>& chunk_content, int width, int height) override;

    cnn_net_type_all model;
};

#endif // CNN_CHOOSER_WITH_ALL_AI_LOGIC_HPP
