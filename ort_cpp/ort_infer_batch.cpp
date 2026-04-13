#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <thread>
#include <filesystem>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

// 模拟 Python 的 argparse 参数结构
struct Args {
    std::string model = "../weights/yolov7-tiny.onnx";
    std::string source = "../data/";
    int batch_size = 1;
    std::string save_dir = "../results";
    float conf_thres = 0.25f;
    float iou_thres = 0.7f;
    int num_classes = 80;
    bool end2end = false;
    bool end2end_model = false;
    bool ultralytics = false;
    bool no_show = false;
    bool save = true;
};

// COCO 类别名称列表
static const std::vector<std::string> COCO_NAMES = {
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

// 图片扩展名列表
static const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};

// 保存单个 Batch 的结果
struct BatchResult {
    std::vector<cv::Vec4f> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
};

class YoloOnnxRunner {

private:
    float conf_thres;
    float iou_thres;
    int num_classes;
    std::vector<std::string> class_names;
    
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;

    int input_width;
    int input_height;

public:
    YoloOnnxRunner(const std::string& model_path, float confidence_thres = 0.4f, float iou_thres = 0.7f, int num_classes = 80, const std::vector<std::string>& class_names = COCO_NAMES)
        : conf_thres(confidence_thres), iou_thres(iou_thres), num_classes(num_classes), class_names(class_names),
          env(ORT_LOGGING_LEVEL_WARNING, "YOLO_ONNX"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (this->class_names.size() != static_cast<size_t>(this->num_classes)){
            std::cerr << "[警告] 传入的类别名称数量 (" << this->class_names.size() 
                      << ") 不等于 num_classes (" << this->num_classes << ")! 画框时可能会越界。" << std::endl;
        }

        // 尝试启用 CUDA
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "尝试使用 CUDA 设备..." << std::endl;
        } catch (const std::exception& e) {
            std::cout << "CUDA 不可用，回退到 CPU: " << e.what() << std::endl;
        }

        #ifdef _WIN32
            std::wstring w_model_path(model_path.begin(), model_path.end());
            session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
        #else
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        #endif

        getInputDetails();
        getOutputDetails();
    }

    void getInputDetails() {
        size_t num_input_nodes = session->GetInputCount();

        input_names_str.reserve(num_input_nodes);
        input_names.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(i, allocator);
            input_names_str.push_back(input_name.get());
            input_names.push_back(input_names_str.back().c_str());

            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape = tensor_info.GetShape();
            
            // 处理动态 batch size
            if (input_shape[0] == -1) input_shape[0] = 1;
            
            std::cout << "模型输入节点: " << input_names.back() << ", 形状: [";
            for (size_t j = 0; j < input_shape.size(); j++) std::cout << input_shape[j] << (j == input_shape.size() - 1 ? "" : ", ");
            std::cout << "]" << std::endl;

            input_height = input_shape[2];
            input_width = input_shape[3];
        }
    }

    void getOutputDetails() {
        size_t num_output_nodes = session->GetOutputCount();

        output_names_str.reserve(num_output_nodes);
        output_names.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(i, allocator);
            output_names_str.push_back(output_name.get());
            output_names.push_back(output_names_str.back().c_str());

            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape = tensor_info.GetShape();

            std::cout << "模型输出节点: " << output_names.back() << ", 形状: [";
            for (size_t j = 0; j < output_shape.size(); j++) std::cout << output_shape[j] << (j == output_shape.size() - 1 ? "" : ", ");
            std::cout << "]" << std::endl;
        }
    }

    // 多 Batch 预处理
    cv::Mat preprocess_batch(const std::vector<cv::Mat>& image_list, std::vector<float>& scales, std::vector<int>& dws, std::vector<int>& dhs) {
        int batch_size = image_list.size();
        scales.resize(batch_size);
        dws.resize(batch_size);
        dhs.resize(batch_size);

        std::vector<cv::Mat> padded_imgs;
        padded_imgs.reserve(batch_size);

        for (int b = 0; b < batch_size; ++b) {
            const cv::Mat& image_src = image_list[b];
            int img_h = image_src.rows;
            int img_w = image_src.cols;

            float scale = std::min((float)input_height / img_h, (float)input_width / img_w);
            int new_w = static_cast<int>(std::round(img_w * scale));
            int new_h = static_cast<int>(std::round(img_h * scale));

            int dw = (input_width - new_w) / 2;
            int dh = (input_height - new_h) / 2;

            scales[b] = scale;
            dws[b] = dw;
            dhs[b] = dh;

            cv::Mat image_resized;
            cv::resize(image_src, image_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

            cv::Mat image_padded;
            int top = dh;
            int bottom = input_height - new_h - dh;
            int left = dw;
            int right = input_width - new_w - dw;
            cv::copyMakeBorder(image_resized, image_padded, top, bottom, left, right,
                              cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            padded_imgs.push_back(image_padded);
        }

        // OpenCV blobFromImages 直接支持将多张图片转为 [N, C, H, W] 并附带均值方差处理和 RGB 转换
        cv::Mat blob = cv::dnn::blobFromImages(padded_imgs, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);

        // std::vector<float> batch_data((float*)blob.datastart, (float*)blob.dataend);
        return blob;
    }

    // NMS 处理单张图 (用于非 end2end 模型)
    void postprocess_single(const float* output_data, bool ultralytics, int num_anchors, int channels,
                            float scale, int dw, int dh, BatchResult& res) {
        
        std::vector<float> all_scores;
        std::vector<int> all_classes;
        std::vector<cv::Vec4f> all_boxes;
        
        all_scores.reserve(num_anchors);
        all_classes.reserve(num_anchors);
        all_boxes.reserve(num_anchors);

        if (ultralytics) {
            // 【性能优化核心】：解决内存跳跃访问问题
            // 原始 output_data 是 [channels, num_anchors] (例如 [84, 8400])
            // 直接用 cv::Mat 包装它 (零拷贝)
            cv::Mat out_mat(channels, num_anchors, CV_32F, (void*)output_data);
            cv::Mat transposed_mat;
            cv::transpose(out_mat, transposed_mat); // 转置后是 [num_anchors, channels]，访问更连续

            // 获取连续内存的指针
            const float* t_data = (const float*)transposed_mat.data;
            for (int i = 0; i < num_anchors; i++) {
                // 现在获取单个锚框的数据指针，就和 else 分支一样连续高效了
                const float* row = t_data + i * channels;
                
                float cx = row[0];
                float cy = row[1];
                float w  = row[2];
                float h  = row[3];

                float max_score = 0.0f;
                int max_class_id = -1;
                
                // 内存连续遍历，CPU Cache 命中率接近 100%
                for (int c = 0; c < num_classes; c++) {
                    float score = row[4 + c];
                    if (score > max_score) {
                        max_score = score;
                        max_class_id = c;
                    }
                }

                if (max_score >= conf_thres) {
                    all_boxes.push_back(cv::Vec4f(cx, cy, w, h));
                    all_scores.push_back(max_score);
                    all_classes.push_back(max_class_id);
                }
            }
            // for (int i = 0; i < num_anchors; i++) {
            //     float cx = output_data[0 * num_anchors + i];
            //     float cy = output_data[1 * num_anchors + i];
            //     float w  = output_data[2 * num_anchors + i];
            //     float h  = output_data[3 * num_anchors + i];

            //     float max_score = 0.0f;
            //     int max_class_id = -1;
            //     for (int c = 0; c < num_classes; c++) {
            //         float score = output_data[(4 + c) * num_anchors + i];
            //         if (score > max_score) {
            //             max_score = score;
            //             max_class_id = c;
            //         }
            //     }

            //     if (max_score >= conf_thres) {
            //         all_boxes.push_back(cv::Vec4f(cx, cy, w, h));
            //         all_scores.push_back(max_score);
            //         all_classes.push_back(max_class_id);
            //     }
            // }
        } else {
            for (int i = 0; i < num_anchors; i++) {
                const float* anchor_data = output_data + i * channels;
                float cx = anchor_data[0];
                float cy = anchor_data[1];
                float w  = anchor_data[2];
                float h  = anchor_data[3];
                float obj_conf = anchor_data[4];

                const float* class_probs = &anchor_data[5];
                auto max_iter = std::max_element(class_probs, class_probs + num_classes);
                float max_class_prob = *max_iter;
                int max_class_id = static_cast<int>(std::distance(class_probs, max_iter));

                float final_score = obj_conf * max_class_prob;

                if (final_score >= conf_thres) {
                    all_boxes.push_back(cv::Vec4f(cx, cy, w, h));
                    all_scores.push_back(final_score);
                    all_classes.push_back(max_class_id);
                }
            }
        }

        std::vector<std::vector<int>> class_indices(num_classes);
        for (size_t i = 0; i < all_classes.size(); i++) {
            class_indices[all_classes[i]].push_back(i);
        }

        float inv_scale = 1.f / scale;
        float neg_dw = -dw;
        float neg_dh = -dh;

        for (int cls_id = 0; cls_id < num_classes; cls_id++) {
            const auto& indices = class_indices[cls_id];
            if (indices.empty()) continue;

            std::vector<cv::Rect> cls_opencv_boxes;
            std::vector<float> cls_scores;
            std::vector<std::pair<int, cv::Vec4f>> cls_boxes_data;

            for (int idx : indices) {
                float cx = all_boxes[idx][0];
                float cy = all_boxes[idx][1];
                float w = all_boxes[idx][2];
                float h = all_boxes[idx][3];

                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;

                cls_opencv_boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                                              static_cast<int>(w), static_cast<int>(h));
                cls_scores.push_back(all_scores[idx]);
                cls_boxes_data.emplace_back(idx, cv::Vec4f(x1, y1, x2, y2));
            }

            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(cls_opencv_boxes, cls_scores, conf_thres, iou_thres, nms_indices);

            for (int nms_idx : nms_indices) {
                const auto& box_data = cls_boxes_data[nms_idx];
                float bx1 = (box_data.second[0] + neg_dw) * inv_scale;
                float by1 = (box_data.second[1] + neg_dh) * inv_scale;
                float bx2 = (box_data.second[2] + neg_dw) * inv_scale;
                float by2 = (box_data.second[3] + neg_dh) * inv_scale;

                res.boxes.emplace_back(bx1, by1, bx2, by2);
                res.scores.push_back(cls_scores[nms_idx]);
                res.classes.push_back(cls_id);
            }
        }
    }

    // 解析整体的输出
    std::vector<BatchResult> process_output(const float* output_data, const std::vector<int64_t>& current_output_shape,
                                            const std::vector<float>& scales, const std::vector<int>& dws, const std::vector<int>& dhs,
                                            const Args& args, int real_batch_size) {
        
        std::vector<BatchResult> batch_dets(real_batch_size);
        int ndim = current_output_shape.size();

        if (args.end2end) {
            // [num_dets, 7] 第一列为 batch_index
            int num_dets = current_output_shape[0];
            int dim = current_output_shape[1]; // 7
            for (int i = 0; i < num_dets; i++) {
                const float* row = output_data + i * dim;
                int b = static_cast<int>(row[0]);
                if (b >= real_batch_size) continue;
                
                float score = row[5];
                if (score > conf_thres) {
                    float inv_scale = 1.0f / scales[b];
                    float x1 = (row[1] - dws[b]) * inv_scale;
                    float y1 = (row[2] - dhs[b]) * inv_scale;
                    float x2 = (row[3] - dws[b]) * inv_scale;
                    float y2 = (row[4] - dhs[b]) * inv_scale;
                    int cls = static_cast<int>(row[6]);

                    batch_dets[b].boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                    batch_dets[b].scores.push_back(score);
                    batch_dets[b].classes.push_back(cls);
                }
            }
        } 
        else if (args.end2end_model) {
            // YOLOv10: [B, max_det, 6]
            int num_anchors = current_output_shape[ndim - 2];
            int dim = current_output_shape[ndim - 1]; // 6
            for (int b = 0; b < real_batch_size; b++) {
                const float* batch_ptr = output_data + b * num_anchors * dim;
                for (int i = 0; i < num_anchors; i++) {
                    const float* row = batch_ptr + i * dim;
                    float score = row[4];
                    if (score > conf_thres) {
                        float inv_scale = 1.0f / scales[b];
                        float x1 = (row[0] - dws[b]) * inv_scale;
                        float y1 = (row[1] - dhs[b]) * inv_scale;
                        float x2 = (row[2] - dws[b]) * inv_scale;
                        float y2 = (row[3] - dhs[b]) * inv_scale;
                        int cls = static_cast<int>(row[5]);

                        batch_dets[b].boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                        batch_dets[b].scores.push_back(score);
                        batch_dets[b].classes.push_back(cls);
                    }
                }
            }
        } 
        else {
            // Ultralytics: [B, 4+cls, num_anchors], Normal: [B, num_anchors, 5+cls]
            int num_anchors, channels;
            if (args.ultralytics) {
                channels = current_output_shape[ndim - 2];
                num_anchors = current_output_shape[ndim - 1];
                for (int b = 0; b < real_batch_size; b++) {
                    const float* batch_ptr = output_data + b * channels * num_anchors;
                    postprocess_single(batch_ptr, true, num_anchors, channels, scales[b], dws[b], dhs[b], batch_dets[b]);
                }
            } else {
                num_anchors = current_output_shape[ndim - 2];
                channels = current_output_shape[ndim - 1];
                for (int b = 0; b < real_batch_size; b++) {
                    const float* batch_ptr = output_data + b * num_anchors * channels;
                    postprocess_single(batch_ptr, false, num_anchors, channels, scales[b], dws[b], dhs[b], batch_dets[b]);
                }
            }
        }

        return batch_dets;
    }

    void draw_results(cv::Mat& img, const BatchResult& res) {
        for (size_t i = 0; i < res.boxes.size(); i++) {
            int x1 = std::round(res.boxes[i][0]);
            int y1 = std::round(res.boxes[i][1]);
            int x2 = std::round(res.boxes[i][2]);
            int y2 = std::round(res.boxes[i][3]);

            x1 = std::max(0, std::min(x1, img.cols));
            y1 = std::max(0, std::min(y1, img.rows));
            x2 = std::max(0, std::min(x2, img.cols));
            y2 = std::max(0, std::min(y2, img.rows));

            int cls_id = res.classes[i];
            float score = res.scores[i];

            cv::Scalar color( (cls_id * 50) % 255, (cls_id * 100) % 255, (cls_id * 150) % 255 );

            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

            char score_str[8];
            snprintf(score_str, sizeof(score_str), "%.2f", score);
            std::string label = (cls_id < (int)class_names.size() ? class_names[cls_id] : std::to_string(cls_id)) + ": " + score_str;
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(img, cv::Point(x1, y1 - labelSize.height - 3), cv::Point(x1 + labelSize.width, y1), color, cv::FILLED);
            cv::putText(img, label, cv::Point(x1, y1 - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    // 批量推理入口
    std::pair<std::vector<cv::Mat>, double> infer_batch(const std::vector<cv::Mat>& img_list, const Args& args) {
        int real_batch_size = img_list.size();
        
        std::vector<float> scales;
        std::vector<int> dws, dhs;
        cv::Mat blob = preprocess_batch(img_list, scales, dws, dhs);

        // 修改动态输入形状中的 Batch Size
        input_shape[0] = real_batch_size;

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), 
                                                                  blob.total(), 
                                                                  input_shape.data(), input_shape.size());

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{nullptr}, 
                                                              input_names.data(), &input_tensor, 1, 
                                                              output_names.data(), output_names.size());
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;

        const float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> current_output_shape = type_info.GetShape();

        // 剥离解析多 Batch 结果
        std::vector<BatchResult> batch_dets = process_output(output_data, current_output_shape, scales, dws, dhs, args, real_batch_size);

        std::vector<cv::Mat> result_imgs;
        for (int i = 0; i < real_batch_size; i++) {
            cv::Mat img = img_list[i].clone();
            draw_results(img, batch_dets[i]);
            result_imgs.push_back(img);
        }

        return {result_imgs, inference_time.count()};
    }

    void run(const Args& args) {
        std::string source = args.source;
        int batch_size = args.batch_size;
        std::string save_dir = args.save_dir;

        if (args.save) {
            fs::create_directories(save_dir);
        }

        if (fs::is_directory(source)) {
            // === 模式 1: 图片目录批量推理 ===
            std::vector<std::string> img_paths;
            for (const auto& entry : fs::directory_iterator(source)) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end()) {
                    img_paths.push_back(entry.path().string());
                }
            }
            std::sort(img_paths.begin(), img_paths.end());
            
            std::cout << "找到 " << img_paths.size() << " 张图片，开始目录批量推理 (Batch Size: " << batch_size << ")...\n";

            for (size_t i = 0; i < img_paths.size(); i += batch_size) {
                std::vector<cv::Mat> valid_imgs;
                std::vector<std::string> valid_names;

                for (size_t j = i; j < std::min(i + batch_size, img_paths.size()); ++j) {
                    cv::Mat img = cv::imread(img_paths[j]);
                    if (!img.empty()) {
                        valid_imgs.push_back(img);
                        valid_names.push_back(fs::path(img_paths[j]).filename().string());
                    }
                }

                if (valid_imgs.empty()) continue;

                auto [res_imgs, t] = infer_batch(valid_imgs, args);
                std::cout << "处理进度 " << i + valid_imgs.size() << "/" << img_paths.size() 
                          << " | Batch推理耗时: " << std::fixed << std::setprecision(2) << t << "ms\n";

                if (args.save) {
                    for (size_t k = 0; k < res_imgs.size(); ++k) {
                        cv::imwrite((fs::path(save_dir) / valid_names[k]).string(), res_imgs[k]);
                    }
                }
            }
            std::cout << "✅ 目录处理完成。\n";
        } 
        else {
            std::string ext = fs::path(source).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            bool is_image = std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end();

            if (is_image) {
                // === 模式 2: 单张图片推理 ===
                std::cout << "正在处理图片: " << source << "\n";
                cv::Mat img = cv::imread(source);
                if (img.empty()) {
                    std::cout << "无法读取图片: " << source << "\n";
                    return;
                }

                auto [res_imgs, t] = infer_batch({img}, args);
                if (args.save) {
                    cv::imwrite((fs::path(save_dir) / fs::path(source).filename()).string(), res_imgs[0]);
                }
                std::cout << "推理时间: " << t << "ms, 结果已保存\n";
            } 
            else {
                // === 模式 3: 视频/RTSP 攒 Batch 推理 ===
                std::cout << "正在尝试打开视频源: " << source << "\n";
                
                cv::VideoCapture cap;
                bool is_digit = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
                if (is_digit) cap.open(std::stoi(source)); 
                else cap.open(source);

                if (!cap.isOpened()) {
                    std::cout << "无法打开视频源: " << source << "\n";
                    return;
                }

                int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                double fps = cap.get(cv::CAP_PROP_FPS);
                if (fps == 0.0) fps = 25.0; 

                cv::VideoWriter out_writer;
                bool is_file = fs::exists(source);
                if (is_file && args.save) {
                    std::string save_path = (fs::path(save_dir) / fs::path(source).filename()).string();
                    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                    out_writer.open(save_path, fourcc, fps, cv::Size(width, height));
                    std::cout << "视频处理中 (Batch: " << batch_size << ")，结果将保存至: " << save_path << "\n";
                }

                std::vector<cv::Mat> batch_frames;
                int frame_count = 0;
                bool stop_flag = false;
                cv::Mat frame;

                while (cap.read(frame)) {
                    if (frame.empty()) break;
                    batch_frames.push_back(frame.clone());

                    if (batch_frames.size() == static_cast<size_t>(batch_size)) {
                        auto [res_imgs, t] = infer_batch(batch_frames, args);

                        for (size_t i = 0; i < res_imgs.size(); i++) {
                            char fps_text[64];
                            snprintf(fps_text, sizeof(fps_text), "FPS: %.1f (Batch Time: %.1fms)", 1000.0 / (t / batch_size), t);
                            cv::putText(res_imgs[i], fps_text, cv::Point(20, 40), 
                                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

                            if (out_writer.isOpened()) out_writer.write(res_imgs[i]);
                            
                            if (!args.no_show) {
                                cv::imshow("YOLO ONNX Runtime C++", res_imgs[i]);
                                if (cv::waitKey(1) == 'q') {
                                    stop_flag = true;
                                    break;
                                }
                            }
                        }

                        if (stop_flag) {
                            std::cout << "\n检测到手动退出按键 'q'，正在终止进程...\n";
                            break;
                        }

                        frame_count += batch_size;
                        if (frame_count % (batch_size * 5) == 0) {
                            std::cout << "已处理 " << frame_count << " 帧, 刚刚的Batch耗时: " << std::fixed << std::setprecision(2) << t << "ms\n";
                        }
                        batch_frames.clear();
                    }
                }

                // 处理结尾残留帧
                if (!batch_frames.empty() && !stop_flag) {
                    auto [res_imgs, t] = infer_batch(batch_frames, args);
                    for (size_t i = 0; i < res_imgs.size(); i++) {
                        if (out_writer.isOpened()) out_writer.write(res_imgs[i]);
                        if (!args.no_show) {
                            cv::imshow("YOLO ONNX Runtime C++", res_imgs[i]);
                            cv::waitKey(1);
                        }
                    }
                }

                cap.release();
                if (out_writer.isOpened()) out_writer.release();
                cv::destroyAllWindows();
                std::cout << "✅ 视频处理完成。\n";
            }
        }
    }
};

int main(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) args.model = argv[++i];
        else if (arg == "--source" && i + 1 < argc) args.source = argv[++i];
        else if (arg == "--batch_size" && i + 1 < argc) args.batch_size = std::stoi(argv[++i]);
        else if (arg == "--save_dir" && i + 1 < argc) args.save_dir = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) args.conf_thres = std::stof(argv[++i]);
        else if (arg == "--iou" && i + 1 < argc) args.iou_thres = std::stof(argv[++i]);
        else if (arg == "--classes" && i + 1 < argc) args.num_classes = std::stoi(argv[++i]);
        else if (arg == "--end2end") args.end2end = true;
        else if (arg == "--end2end_model") args.end2end_model = true;
        else if (arg == "--ultralytics") args.ultralytics = true;
        else if (arg == "--no-show") args.no_show = true;
        else if (arg == "--save") args.save = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "选项:\n"
                      << "  --model <path>      模型路径 (默认: weights/yolo11n.onnx)\n"
                      << "  --source <path>     输入图片/视频/目录或摄像头 (默认: data/)\n"
                      << "  --batch_size <int>  Batch推理大小 (默认: 8)\n"
                      << "  --save_dir <path>   保存结果目录 (默认: results)\n"
                      << "  --conf <float>      置信度阈值 (默认: 0.25)\n"
                      << "  --iou <float>       IOU阈值 (默认: 0.7)\n"
                      << "  --classes <int>     类别数 (默认: 80)\n"
                      << "  --end2end           使用end2end模式 (INMSlayer)\n"
                      << "  --end2end_model     使用end2end模型模式 (YOLOv10风格)\n"
                      << "  --ultralytics       使用ultralytics模式 (默认: false)\n"
                      << "  --no-show           不显示结果窗口\n"
                      << "  --save              保存结果\n"
                      << "  -h, --help          显示帮助信息\n";
            return 0;
        }
    }

    YoloOnnxRunner runner(args.model, args.conf_thres, args.iou_thres, args.num_classes);
    runner.run(args);

    return 0;
}