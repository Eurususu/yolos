#pragma once

#include <string>
#include <memory>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include "trt_utils.h"

namespace trtinfer {

struct TensorBinding {
    int index;
    std::string name;
    nvinfer1::DataType dtype;
    std::vector<int64_t> shape;
    void* device_ptr;
    size_t size;
    bool is_input;
};

class TrtEngine {
public:
    TrtEngine(const std::string& engine_path, int max_batch_size = 1, int max_det = 300);
    ~TrtEngine();

    // Run inference on preprocessed image
    std::vector<void*> infer(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape);

    // Run inference on image file
    std::vector<Detection> inference(const std::string& img_path, const std::string& output_path = "result.jpg",
                                       float conf_threshold = 0.25,
                                       bool end2end = false, bool efficient_end2end = false,
                                       bool ultralytics = false, bool end2end_model = false);

    // Run inference on video
    void detect_video(const std::string& video_path, float conf_threshold = 0.25,
                      bool end2end = false, bool efficient_end2end = false,
                      bool ultralytics = false, bool end2end_model = false);

    // Get FPS
    float get_fps(int warmup_iterations = 5, int test_iterations = 100);

    // Getters
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    int get_num_classes() const { return num_classes_; }

private:
    void load_engine(const std::string& engine_path);
    void allocate_buffers();
    void cleanup();

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_;
    std::vector<TensorBinding> inputs_;
    std::vector<TensorBinding> outputs_;

    int input_width_;
    int input_height_;
    int num_classes_;
    int max_batch_size_;
    int max_det_;

    static const int NUM_CLASSES = 80;
    static const std::vector<std::string> CLASS_NAMES;
};

}  // namespace trtinfer
