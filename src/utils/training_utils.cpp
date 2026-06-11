#include "utils/training_utils.hpp"
#include "ai/2_layer_chooser_ai_logic.hpp"
#include "ai/3_layer_chooser_ai_logic.hpp"
#include "ai/cnn_chooser_ai_logic.hpp"
#include "ai/2_layer_chooser_with_all_ai_logic.hpp"
#include "ai/3_layer_chooser_with_all_ai_logic.hpp"
#include "ai/cnn_chooser_with_all_ai_logic.hpp"
#include "predictors/predictor_factory.hpp"
#include "config.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace fs = std::filesystem;

namespace {
    struct ImageData {
        std::vector<unsigned char> pixels;
        int width;
        int height;
    };

    ImageData load_image(const std::string& path) {
        int width, height, channels;
        unsigned char* img_data = stbi_load(path.c_str(), &width, &height, &channels, 1);

        if (img_data == nullptr) {
            std::cerr << "Failed to load PNG: " << path << std::endl;
            return {};
        }

        const std::vector<unsigned char> image(img_data, img_data + (width * height));

        return {image, width, height};
    }

    void get_training_data_fc(std::vector<dlib::matrix<float>>& samples, std::vector<unsigned long>& labels, bool all_predictors) {
        std::vector<std::unique_ptr<Predictor>> predictors;
        int num_predictors = all_predictors ? static_cast<int>(PredictorType::COUNT) : 20;
        for (int i = 0; i < num_predictors; ++i) {
            predictors.push_back(PredictorFactory::create(static_cast<PredictorType>(i)));
        }

        for (const auto& entry : fs::directory_iterator("../train_data")) {
            if (entry.is_regular_file()) {
                auto img_data = load_image(entry.path().string());
                if (img_data.pixels.empty()) continue;

                const int width = img_data.width;
                const int height = img_data.height;

                for (int y = 0; y <= height - CHUNK_SIZE; y += CHUNK_SIZE) {
                    for (int x = 0; x <= width - CHUNK_SIZE; x += CHUNK_SIZE) {
                        std::vector<unsigned char> chunk(CHUNK_SIZE * CHUNK_SIZE);
                        for (int i = 0; i < CHUNK_SIZE; ++i) {
                            for (int j = 0; j < CHUNK_SIZE; ++j) {
                                chunk[i * CHUNK_SIZE + j] = img_data.pixels[(y + i) * width + (x + j)];
                            }
                        }

                        long best_predictor_index = -1;
                        long min_error = -1;

                        for (int i = 0; i < num_predictors; ++i) {
                            long current_error = 0;

                            for (int cy = 0; cy < CHUNK_SIZE; ++cy) {
                                for (int cx = 0; cx < CHUNK_SIZE; ++cx) {
                                    if (cy == 0 && cx == 0) continue;

                                    const unsigned char current_pixel = chunk[cy * CHUNK_SIZE + cx];

                                    const unsigned char left = (cx > 0) ? chunk[cy * CHUNK_SIZE + (cx - 1)] : 0;
                                    const unsigned char up = (cy > 0) ? chunk[(cy - 1) * CHUNK_SIZE + cx] : 0;
                                    const unsigned char up_left = (cx > 0 && cy > 0) ? chunk[(cy - 1) * CHUNK_SIZE + (cx - 1)] : 0;

                                    const int prediction = predictors[i]->predict(left, up, up_left);
                                    current_error += std::abs(current_pixel - prediction);
                                }
                            }

                            if (min_error == -1 || current_error < min_error) {
                                min_error = current_error;
                                best_predictor_index = i;
                            }
                        }

                        dlib::matrix<float> input_mat(CHUNK_SIZE * CHUNK_SIZE, 1);
                        for (int k = 0; k < CHUNK_SIZE * CHUNK_SIZE; ++k) {
                            input_mat(k, 0) = static_cast<float>(chunk[k]);
                        }

                        samples.push_back(input_mat);
                        labels.push_back(best_predictor_index);
                    }
                }
            }
        }
    }

    void get_training_data_cnn(std::vector<dlib::matrix<unsigned char>>& samples, std::vector<unsigned long>& labels, bool all_predictors) {
        std::vector<std::unique_ptr<Predictor>> predictors;
        int num_predictors = all_predictors ? static_cast<int>(PredictorType::COUNT) : 20;
        for (int i = 0; i < num_predictors; ++i) {
            predictors.push_back(PredictorFactory::create(static_cast<PredictorType>(i)));
        }

         for (const auto& entry : fs::directory_iterator("../train_data")) {
            if (entry.is_regular_file()) {
                auto img_data = load_image(entry.path().string());
                if (img_data.pixels.empty()) continue;

                const int width = img_data.width;
                const int height = img_data.height;

                for (int y = 0; y <= height - CHUNK_SIZE; y += CHUNK_SIZE) {
                    for (int x = 0; x <= width - CHUNK_SIZE; x += CHUNK_SIZE) {
                        std::vector<unsigned char> chunk_vec(CHUNK_SIZE * CHUNK_SIZE);
                        dlib::matrix<unsigned char> chunk_mat(CHUNK_SIZE, CHUNK_SIZE);

                        for (int i = 0; i < CHUNK_SIZE; ++i) {
                            for (int j = 0; j < CHUNK_SIZE; ++j) {
                                const unsigned char val = img_data.pixels[(y + i) * width + (x + j)];
                                chunk_vec[i * CHUNK_SIZE + j] = val;
                                chunk_mat(i, j) = val;
                            }
                        }

                        long best_predictor_index = -1;
                        long min_error = -1;

                        for (int i = 0; i < num_predictors; ++i) {
                            long current_error = 0;

                            for (int cy = 0; cy < CHUNK_SIZE; ++cy) {
                                for (int cx = 0; cx < CHUNK_SIZE; ++cx) {
                                    if (cy == 0 && cx == 0) continue;

                                    const unsigned char current_pixel = chunk_vec[cy * CHUNK_SIZE + cx];

                                    const unsigned char left = (cx > 0) ? chunk_vec[cy * CHUNK_SIZE + (cx - 1)] : 0;
                                    const unsigned char up = (cy > 0) ? chunk_vec[(cy - 1) * CHUNK_SIZE + cx] : 0;
                                    const unsigned char up_left = (cx > 0 && cy > 0) ? chunk_vec[(cy - 1) * CHUNK_SIZE + (cx - 1)] : 0;

                                    const int prediction = predictors[i]->predict(left, up, up_left);
                                    current_error += std::abs(current_pixel - prediction);
                                }
                            }

                            if (min_error == -1 || current_error < min_error) {
                                min_error = current_error;
                                best_predictor_index = i;
                            }
                        }

                        samples.push_back(chunk_mat);
                        labels.push_back(best_predictor_index);
                    }
                }
            }
        }
    }

    template<typename Trainer>
    void setup_and_run_trainer(Trainer& trainer, const auto& samples, const auto& labels) {
        trainer.set_learning_rate(0.001);
        trainer.set_min_learning_rate(1e-5);
        trainer.set_mini_batch_size(64);
        trainer.set_max_num_epochs(20);
        trainer.be_verbose();
        trainer.train(samples, labels);
    }
}

void train_2_layer_model() {
    std::cout << "\n--- Training 2-Layer FC Model ---" << std::endl;
    std::vector<dlib::matrix<float>> samples;
    std::vector<unsigned long> labels;
    std::cout << "Loading FC training data..." << std::endl;
    get_training_data_fc(samples, labels, false);

    TwoLayerChooserAILogic logic(true);
    dlib::dnn_trainer<net_type, dlib::adam> trainer(logic.model);
    setup_and_run_trainer(trainer, samples, labels);
    logic.model.clean();
    dlib::serialize("2_layer_fc_logic_model.dat") << logic.model;
    std::cout << "2-Layer FC Model saved to 2_layer_fc_logic_model.dat" << std::endl;
}

void train_3_layer_model() {
    std::cout << "\n--- Training 3-Layer FC Model ---" << std::endl;
    std::vector<dlib::matrix<float>> samples;
    std::vector<unsigned long> labels;
    std::cout << "Loading FC training data..." << std::endl;
    get_training_data_fc(samples, labels, false);

    ThreeLayerChooserAILogic logic(true);
    dlib::dnn_trainer<net_type_3_layer, dlib::adam> trainer(logic.model);
    setup_and_run_trainer(trainer, samples, labels);
    logic.model.clean();
    dlib::serialize("3_layer_fc_logic_model.dat") << logic.model;
    std::cout << "3-Layer FC Model saved to 3_layer_fc_logic_model.dat" << std::endl;
}

void train_cnn_model() {
    std::cout << "\n--- Training CNN Model ---" << std::endl;
    std::vector<dlib::matrix<unsigned char>> samples;
    std::vector<unsigned long> labels;
    std::cout << "Loading CNN training data..." << std::endl;
    get_training_data_cnn(samples, labels, false);

    CnnChooserAILogic logic(true);
    dlib::dnn_trainer<cnn_net_type, dlib::adam> trainer(logic.model);
    setup_and_run_trainer(trainer, samples, labels);
    logic.model.clean();
    dlib::serialize("cnn_predictor_model.dat") << logic.model;
    std::cout << "CNN Model saved to cnn_predictor_model.dat" << std::endl;
}

void train_2_layer_model_all() {
    std::cout << "\n--- Training 2-Layer FC Model (All Predictors) ---" << std::endl;
    std::vector<dlib::matrix<float>> samples;
    std::vector<unsigned long> labels;
    std::cout << "Loading FC training data..." << std::endl;
    get_training_data_fc(samples, labels, true);

    TwoLayerChooserWithAllAILogic logic(true);
    dlib::dnn_trainer<net_type_all, dlib::adam> trainer(logic.model);
    setup_and_run_trainer(trainer, samples, labels);
    logic.model.clean();
    dlib::serialize("2_layer_fc_for_all_logic_model.dat") << logic.model;
    std::cout << "2-Layer FC Model saved to 2_layer_fc_for_all_logic_model.dat" << std::endl;
}

void train_3_layer_model_all() {
    std::cout << "\n--- Training 3-Layer FC Model (All Predictors) ---" << std::endl;
    std::vector<dlib::matrix<float>> samples;
    std::vector<unsigned long> labels;
    std::cout << "Loading FC training data..." << std::endl;
    get_training_data_fc(samples, labels, true);

    ThreeLayerChooserWithAllAILogic logic(true);
    dlib::dnn_trainer<three_layer_net_type_all, dlib::adam> trainer(logic.model);
    setup_and_run_trainer(trainer, samples, labels);
    logic.model.clean();
    dlib::serialize("3_layer_fc_for_all_logic_model.dat") << logic.model;
    std::cout << "3-Layer FC Model saved to 3_layer_fc_for_all_logic_model.dat" << std::endl;
}

void train_cnn_model_all() {
    std::cout << "\n--- Training CNN Model (All Predictors) ---" << std::endl;
    std::vector<dlib::matrix<unsigned char>> samples;
    std::vector<unsigned long> labels;
    std::cout << "Loading CNN training data..." << std::endl;
    get_training_data_cnn(samples, labels, true);

    CnnChooserWithAllAILogic logic(true);
    dlib::dnn_trainer<cnn_net_type_all, dlib::adam> trainer(logic.model);
    setup_and_run_trainer(trainer, samples, labels);
    logic.model.clean();
    dlib::serialize("cnn_for_all_logic_model.dat") << logic.model;
    std::cout << "CNN Model saved to cnn_for_all_logic_model.dat" << std::endl;
}
