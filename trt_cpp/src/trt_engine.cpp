#include "trt_engine.h"
#include "trt_utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>

// TensorRT logger implementation
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TrtLogger gLogger;

namespace trtinfer {

// COCO class names
const std::vector<std::string> TrtEngine::CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

TrtEngine::TrtEngine(const std::string& engine_path, int max_batch_size, int max_det, float iou_thres)
    : input_width_(0), input_height_(0),
      num_classes_(NUM_CLASSES), max_batch_size_(max_batch_size), max_det_(max_det), iou_thres_(iou_thres) {
    load_engine(engine_path);
    allocate_buffers();

    // Get input dimensions
    auto input_name = engine_->getIOTensorName(0);
    auto input_shape = engine_->getTensorShape(input_name);
    input_height_ = input_shape.d[2];
    input_width_ = input_shape.d[3];
}

TrtEngine::~TrtEngine() {
    cleanup();
}

void TrtEngine::load_engine(const std::string& engine_path) {
    // Load engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    // Create runtime and deserialize engine
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    initLibNvInferPlugins(&gLogger, "");
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(buffer.data(), size));

    if (!engine_) {
        throw std::runtime_error("Failed to deserialize engine");
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());

    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Create CUDA stream
    cudaStreamCreate(&stream_);
}

void TrtEngine::allocate_buffers() {
    int num_io_tensors = engine_->getNbIOTensors();

    for (int i = 0; i < num_io_tensors; i++) {
        const char* tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
        nvinfer1::DataType dtype = engine_->getTensorDataType(tensor_name);
        nvinfer1::Dims shape = engine_->getTensorShape(tensor_name);

        TensorBinding binding;
        binding.index = i;
        binding.name = tensor_name;
        binding.dtype = dtype;
        binding.is_input = (io_mode == nvinfer1::TensorIOMode::kINPUT);

        // Handle dynamic shapes
        std::vector<int64_t> fixed_shape;
        for (int j = 0; j < shape.nbDims; j++) {
            if (shape.d[j] == -1) {
                if (binding.is_input) {
                    fixed_shape.push_back(max_batch_size_);
                } else {
                    // Output tensor with -1 in first dimension (max_det)
                    if (j == 0) {
                        fixed_shape.push_back(max_det_);
                    } else {
                        fixed_shape.push_back(shape.d[j]);
                    }
                }
            } else {
                fixed_shape.push_back(shape.d[j]);
            }
        }
        binding.shape = fixed_shape;

        // Calculate size
        size_t element_size = 4;  // float32
        if (dtype == nvinfer1::DataType::kHALF) element_size = 2;
        else if (dtype == nvinfer1::DataType::kINT32) element_size = 4;
        else if (dtype == nvinfer1::DataType::kINT8) element_size = 1;

        size_t total_size = element_size;
        for (auto dim : binding.shape) {
            total_size *= dim;
        }
        binding.size = total_size;

        // Allocate GPU memory
        cudaMalloc(&binding.device_ptr, binding.size);

        // Set tensor address
        context_->setTensorAddress(tensor_name, binding.device_ptr);

        if (binding.is_input) {
            inputs_.push_back(binding);
        } else {
            outputs_.push_back(binding);
        }
    }
}

void TrtEngine::cleanup() {
    for (auto& binding : inputs_) {
        if (binding.device_ptr) {
            cudaFree(binding.device_ptr);
        }
    }
    for (auto& binding : outputs_) {
        if (binding.device_ptr) {
            cudaFree(binding.device_ptr);
        }
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

// Helper: Convert xywh to xyxy
static inline void xywh_to_xyxy(float x, float y, float w, float h, float& x1, float& y1, float& x2, float& y2) {
    x1 = x - w / 2;
    y1 = y - h / 2;
    x2 = x + w / 2;
    y2 = y + h / 2;
}

// Helper: Apply letterbox transform to bbox coordinates
static inline void apply_letterbox(float& x1, float& y1, float& x2, float& y2, float dw, float dh, float ratio) {
    x1 = (x1 - dw) / ratio;
    y1 = (y1 - dh) / ratio;
    x2 = (x2 - dw) / ratio;
    y2 = (y2 - dh) / ratio;
}

// Unified post-processing for all output formats
std::vector<Detection> TrtEngine::postprocess_output(const float* output_data, float ratio, float dw, float dh,
                                                       float conf_threshold, float iou_threshold,
                                                       bool end2end, bool efficient_end2end,
                                                       bool ultralytics, bool end2end_model) {
    std::vector<Detection> detections;

    if (end2end) {
        // end2end: [max_det, 7] (batch_idx, x1, y1, x2, y2, score, class_id)
        int num_dets = outputs_[0].shape[0];
        for (int i = 0; i < num_dets; i++) {
            float score = output_data[i * 7 + 5];
            if (score <= conf_threshold) continue;
            Detection det;
            det.bbox[0] = (output_data[i * 7 + 1] - dw) / ratio;
            det.bbox[1] = (output_data[i * 7 + 2] - dh) / ratio;
            det.bbox[2] = (output_data[i * 7 + 3] - dw) / ratio;
            det.bbox[3] = (output_data[i * 7 + 4] - dh) / ratio;
            det.score = score;
            det.class_id = static_cast<int>(output_data[i * 7 + 6]);
            detections.push_back(det);
        }
    } else if (efficient_end2end) {
        // efficient_end2end: output is a list [num, boxes, scores, class_ids]
        int num = static_cast<int>(output_data[0]);
        for (int i = 0; i < num && i < max_det_; i++) {
            int idx = 1 + i * 6;
            float score = output_data[idx + 4];
            if (score > conf_threshold) {
                Detection det;
                det.bbox[0] = (output_data[idx] - dw) / ratio;
                det.bbox[1] = (output_data[idx + 1] - dh) / ratio;
                det.bbox[2] = (output_data[idx + 2] - dw) / ratio;
                det.bbox[3] = (output_data[idx + 3] - dh) / ratio;
                det.score = score;
                det.class_id = static_cast<int>(output_data[idx + 5]);
                detections.push_back(det);
            }
        }
    } else if (end2end_model) {
        // end2end_model: [batch, num_boxes, 6] (x1, y1, x2, y2, score, class_id)
        int num_boxes = outputs_[0].shape[1];
        for (int i = 0; i < num_boxes; i++) {
            int base_idx = i * 6;
            float score = output_data[base_idx + 4];
            if (score > conf_threshold) {
                Detection det;
                det.bbox[0] = (output_data[base_idx] - dw) / ratio;
                det.bbox[1] = (output_data[base_idx + 1] - dh) / ratio;
                det.bbox[2] = (output_data[base_idx + 2] - dw) / ratio;
                det.bbox[3] = (output_data[base_idx + 3] - dh) / ratio;
                det.score = score;
                det.class_id = static_cast<int>(output_data[base_idx + 5]);
                detections.push_back(det);
            }
        }
    } else {
        // Standard or Ultralytics YOLO format
        bool is_3d = (outputs_[0].shape.size() == 3);

        if (ultralytics && is_3d) {
            // Ultralytics: [batch, num_classes+4, num_anchors] = [1, 84, 8400]
            // After Python transpose: [num_anchors, num_classes+4]
            // Transposed layout: predictions[n, c] = data[0, c, n]
            int num_anchors = outputs_[0].shape[2];  // 8400

            for (int n = 0; n < num_anchors; n++) {
                // Get box coordinates: output_data[c * num_anchors + n]
                float cx = output_data[0 * num_anchors + n];
                float cy = output_data[1 * num_anchors + n];
                float w = output_data[2 * num_anchors + n];
                float h = output_data[3 * num_anchors + n];

                // Skip invalid boxes
                if (w <= 0 || h <= 0 || cx <= 0 || cy <= 0) continue;

                float x1 = cx - w / 2;
                float y1 = cy - h / 2;
                float x2 = cx + w / 2;
                float y2 = cy + h / 2;

                x1 = (x1 - dw) / ratio;
                y1 = (y1 - dh) / ratio;
                x2 = (x2 - dw) / ratio;
                y2 = (y2 - dh) / ratio;

                for (int c = 0; c < num_classes_; c++) {
                    float score = output_data[(4 + c) * num_anchors + n];
                    if (score > conf_threshold) {
                        Detection det;
                        det.bbox[0] = x1;
                        det.bbox[1] = y1;
                        det.bbox[2] = x2;
                        det.bbox[3] = y2;
                        det.score = score;
                        det.class_id = c;
                        detections.push_back(det);
                    }
                }
            }
        } else {
            // Standard: [batch, num_anchors, 85] or [num_anchors, 85]
            int num_anchors = is_3d ? outputs_[0].shape[1] : outputs_[0].shape[0];
            int num_cols = is_3d ? outputs_[0].shape[2] : outputs_[0].shape[1];

            for (int i = 0; i < num_anchors; i++) {
                int base_idx = is_3d ? i * num_cols : i * (5 + num_classes_);
                float cx = output_data[base_idx];
                float cy = output_data[base_idx + 1];
                float w = output_data[base_idx + 2];
                float h = output_data[base_idx + 3];
                float obj_score = output_data[base_idx + 4];

                float x1 = cx - w / 2;
                float y1 = cy - h / 2;
                float x2 = cx + w / 2;
                float y2 = cy + h / 2;

                x1 = (x1 - dw) / ratio;
                y1 = (y1 - dh) / ratio;
                x2 = (x2 - dw) / ratio;
                y2 = (y2 - dh) / ratio;

                for (int c = 0; c < num_classes_; c++) {
                    float score = obj_score * output_data[base_idx + 5 + c];
                    if (score > conf_threshold) {
                        Detection det;
                        det.bbox[0] = x1;
                        det.bbox[1] = y1;
                        det.bbox[2] = x2;
                        det.bbox[3] = y2;
                        det.score = score;
                        det.class_id = c;
                        detections.push_back(det);
                    }
                }
            }
        }

        // Apply NMS
        detections = nms(detections, iou_threshold);
    }

    return detections;
}

std::vector<void*> TrtEngine::infer(const std::vector<float>& input_data,
                                      const std::vector<int64_t>& input_shape) {
    // Set input shape
    const char* input_name = inputs_[0].name.c_str();
    nvinfer1::Dims input_dims;
    input_dims.nbDims = input_shape.size();
    for (size_t i = 0; i < input_shape.size(); i++) {
        input_dims.d[i] = input_shape[i];
    }
    context_->setInputShape(input_name, input_dims);

    // Clear output buffers before inference to prevent residual data from previous frame
    for (auto& output : outputs_) {
        cudaMemsetAsync(output.device_ptr, 0, output.size, stream_);
    }

    // Copy input data to GPU
    cudaMemcpy(inputs_[0].device_ptr, input_data.data(),
                inputs_[0].size, cudaMemcpyHostToDevice);

    // Run inference
    context_->enqueueV3(stream_);

    // Copy output data from GPU
    std::vector<void*> output_ptrs;
    for (auto& output : outputs_) {
        output_ptrs.push_back(output.device_ptr);
    }

    cudaStreamSynchronize(stream_);

    return output_ptrs;
}

std::vector<Detection> TrtEngine::inference(const std::string& img_path, const std::string& output_path,
                                              float conf_threshold, float iou_threshold,
                                              bool end2end, bool efficient_end2end,
                                              bool ultralytics, bool end2end_model) {
    // Read image
    cv::Mat origin_img = cv::imread(img_path);
    if (origin_img.empty()) {
        throw std::runtime_error("Failed to read image: " + img_path);
    }

    // Preprocess
    cv::Mat img;
    float ratio;
    float dw, dh;
    letterbox(origin_img, img, ratio, dw, dh, input_width_, input_height_);

    // Prepare input data
    std::vector<float> input_data;
    input_data.reserve(3 * input_height_ * input_width_);

    // Convert HWC to CHW and normalize
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_height_; h++) {
            for (int w = 0; w < input_width_; w++) {
                input_data.push_back(img.at<cv::Vec3b>(h, w)[c] / 255.0f);
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    auto output_ptrs = infer(input_data, input_shape);

    // Copy output to host
    std::vector<float> output_data(outputs_[0].size / sizeof(float));
    cudaMemcpy(output_data.data(), outputs_[0].device_ptr,
                outputs_[0].size, cudaMemcpyDeviceToHost);

    // Process output using unified postprocess function
    auto detections = postprocess_output(output_data.data(), ratio, dw, dh,
                                         conf_threshold, iou_threshold,
                                         end2end, efficient_end2end,
                                         ultralytics, end2end_model);

    // Visualize results
    visualize_detection(origin_img, detections, conf_threshold, CLASS_NAMES);

    // Save result
    cv::imwrite(output_path, origin_img);

    return detections;
}

void TrtEngine::detect_video(const std::string& video_path, const std::string& output_path,
                               float conf_threshold, float iou_threshold,
                               bool end2end, bool efficient_end2end,
                               bool ultralytics, bool end2end_model) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + video_path);
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));

    cv::Mat frame;
    float curr_fps = 0;

    // Pre-allocate buffers for better performance
    std::vector<float> input_data(3 * input_height_ * input_width_);
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    std::vector<float> output_data(outputs_[0].size / sizeof(float));

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Preprocess
        cv::Mat img;
        float ratio;
        float dw, dh;
        letterbox(frame, img, ratio, dw, dh, input_width_, input_height_);

        // Prepare input data (use pre-allocated buffer)
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < input_height_; h++) {
                for (int w = 0; w < input_width_; w++) {
                    input_data[idx++] = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        infer(input_data, input_shape);
        auto end_time = std::chrono::high_resolution_clock::now();

        curr_fps = (curr_fps + 1.0f / std::chrono::duration<float>(end_time - start_time).count()) / 2;

        // Copy output to host (use pre-allocated buffer)
        cudaMemcpy(output_data.data(), outputs_[0].device_ptr,
                    outputs_[0].size, cudaMemcpyDeviceToHost);

        // Process output using unified postprocess function
        auto detections = postprocess_output(output_data.data(), ratio, dw, dh,
                                           conf_threshold, iou_threshold,
                                           end2end, efficient_end2end,
                                           ultralytics, end2end_model);

        // Visualize
        visualize_detection(frame, detections, conf_threshold, CLASS_NAMES);

        // Draw FPS
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(curr_fps));
        cv::putText(frame, fps_text, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Detection", frame);
        writer.write(frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
}

float TrtEngine::get_fps(int warmup_iterations, int test_iterations) {
    std::vector<float> input_data(3 * input_height_ * input_width_, 1.0f);
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};

    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        infer(input_data, input_shape);
    }

    // Test
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_iterations; i++) {
        infer(input_data, input_shape);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float avg_time = std::chrono::duration<float>(end - start).count() / test_iterations;
    return 1.0f / avg_time;
}

}  // namespace trtinfer
