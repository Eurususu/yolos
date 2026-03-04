#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>  // 用于检查文件是否存在 (C++17)
#include <iomanip>     // 用于控制浮点数输出格式
#include <sstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 模拟 Python 的 argparse 参数结构
struct Args {
    std::string model = "weights/yolo11n.onnx";
    std::string source = "data/1.jpg";
    bool end2end = false;
    bool end2end_model = false;
    bool ultralytics = true; 
    bool no_show = false;
    bool save = false;
};

class YoloOnnxRunner {
private:
    float conf_thres;
    float iou_thres;
    int num_classes;
    
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    // 分配的内存需要保存生命周期
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;

    int input_width;
    int input_height;
    int img_w;
    int img_h;

public:
    YoloOnnxRunner(const std::string& model_path, float confidence_thres = 0.4f, float iou_thres = 0.7f, int num_classes = 80)
        : conf_thres(confidence_thres), iou_thres(iou_thres), num_classes(num_classes), env(ORT_LOGGING_LEVEL_WARNING, "YOLO_ONNX") {
        
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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

    // 预处理
    std::vector<float> preprocess(const cv::Mat& image_src, float& scale, int& dw, int& dh) {
        img_h = image_src.rows;
        img_w = image_src.cols;

        // 1. Letterbox Resize
        scale = std::min((float)input_height / img_h, (float)input_width / img_w);
        int new_h = std::round(img_h * scale);
        int new_w = std::round(img_w * scale);

        cv::Mat image_resized;
        cv::resize(image_src, image_resized, cv::Size(new_w, new_h));

        cv::Mat image_padded(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114));
        dw = (input_width - new_w) / 2;
        dh = (input_height - new_h) / 2;
        image_resized.copyTo(image_padded(cv::Rect(dw, dh, new_w, new_h)));

        // 2. 归一化 & 转换 (HWC -> CHW 是由 blobFromImage 自动完成的)
        cv::Mat blob = cv::dnn::blobFromImage(image_padded, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);
        
        // 将 blob 数据存入 vector
        std::vector<float> image_data(blob.ptr<float>(), blob.ptr<float>() + blob.total());
        return image_data;
    }

    // 后处理
    void postprocess(const float* output_data, bool ultralytics, 
                     std::vector<cv::Vec4f>& final_boxes, std::vector<float>& final_scores, std::vector<int>& final_classes) {
        
        std::vector<cv::Rect> opencv_boxes;  // 专门给 cv::dnn::NMSBoxes 用的 (x, y, w, h)
        std::vector<cv::Vec4f> nms_boxes;    // 存 x1, y1, x2, y2
        std::vector<float> scores;
        std::vector<int> class_ids;

        if (ultralytics) {
            // YOLOv8/v11 格式: 通常为 [1, 4 + num_classes, num_anchors] -> [1, 84, 8400]
            // 数据在内存中是按通道 (channel) 连续的
            int num_anchors = output_shape[2]; 

            for (int i = 0; i < num_anchors; i++) {
                float max_score = 0.0f;
                int max_class_id = -1;

                // 寻找最大类别分数
                for (int c = 0; c < num_classes; c++) {
                    float score = output_data[(4 + c) * num_anchors + i];
                    if (score > max_score) {
                        max_score = score;
                        max_class_id = c;
                    }
                }

                if (max_score >= conf_thres) {
                    float cx = output_data[0 * num_anchors + i];
                    float cy = output_data[1 * num_anchors + i];
                    float w  = output_data[2 * num_anchors + i];
                    float h  = output_data[3 * num_anchors + i];

                    // 转换为 x1, y1, x2, y2
                    float x1 = cx - w / 2.0f;
                    float y1 = cy - h / 2.0f;
                    float x2 = cx + w / 2.0f;
                    float y2 = cy + h / 2.0f;

                    nms_boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                    opencv_boxes.push_back(cv::Rect(int(x1), int(y1), int(w), int(h)));
                    scores.push_back(max_score);
                    class_ids.push_back(max_class_id);
                }
            }
        } else {
            // 非 Ultralytics 格式 (如 YOLOv5_u 之前版本): Reshape 为 [1, num_anchors, 5 + num_classes]
            // 数据在内存中是按锚框 (anchor) 连续的: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
            
            // 计算总元素推导 num_anchors (对应 Python 的 reshape(1, -1, 5+num_classes))
            int64_t total_elements = 1;
            for (auto s : output_shape) total_elements *= s;
            int dim = 5 + num_classes;
            int num_anchors = total_elements / dim;

            for (int i = 0; i < num_anchors; i++) {
                const float* anchor_data = output_data + i * dim; // 获取当前 anchor 的首地址
                
                float obj_conf = anchor_data[4]; // objectness confidence

                float max_class_prob = 0.0f;
                int max_class_id = -1;

                // 寻找最大类别分数
                for (int c = 0; c < num_classes; c++) {
                    float prob = anchor_data[5 + c];
                    if (prob > max_class_prob) {
                        max_class_prob = prob;
                        max_class_id = c;
                    }
                }

                // 计算最终得分: scores = prediction[:, 4:5] * prediction[:, 5:]
                float final_score = obj_conf * max_class_prob;

                if (final_score >= conf_thres) {
                    float cx = anchor_data[0];
                    float cy = anchor_data[1];
                    float w  = anchor_data[2];
                    float h  = anchor_data[3];

                    // 转换为 x1, y1, x2, y2
                    float x1 = cx - w / 2.0f;
                    float y1 = cy - h / 2.0f;
                    float x2 = cx + w / 2.0f;
                    float y2 = cy + h / 2.0f;

                    nms_boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                    opencv_boxes.push_back(cv::Rect(int(x1), int(y1), int(w), int(h)));
                    scores.push_back(final_score);
                    class_ids.push_back(max_class_id);
                }
            }
        }

        // 执行 NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(opencv_boxes, scores, conf_thres, iou_thres, indices);

        // 获取最终的检测框 (x1, y1, x2, y2 格式)
        for (int i : indices) {
            final_boxes.push_back(nms_boxes[i]);
            final_scores.push_back(scores[i]);
            final_classes.push_back(class_ids[i]);
        }
    }

    cv::Mat draw_results(cv::Mat& img, const std::vector<cv::Vec4f>& boxes, const std::vector<float>& scores, 
                         const std::vector<int>& classes, float scale, int dw, int dh) {
        
        std::vector<std::string> coco_names = {
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

        for (size_t i = 0; i < boxes.size(); i++) {
            // 完全对应 Python 代码的坐标还原：
            // boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
            // boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
            int x1 = std::round((boxes[i][0] - dw) / scale);
            int y1 = std::round((boxes[i][1] - dh) / scale);
            int x2 = std::round((boxes[i][2] - dw) / scale);
            int y2 = std::round((boxes[i][3] - dh) / scale);

            // 边界截断 clip(0, w) / clip(0, h)
            x1 = std::max(0, std::min(x1, img_w));
            y1 = std::max(0, std::min(y1, img_h));
            x2 = std::max(0, std::min(x2, img_w));
            y2 = std::max(0, std::min(y2, img_h));

            int cls_id = classes[i];
            float score = scores[i];

            // 颜色伪随机 (模仿 np.random.RandomState(cls_id))
            cv::Scalar color( (cls_id * 50) % 255, (cls_id * 100) % 255, (cls_id * 150) % 255 );

            // 画框
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

            // 写标签
            std::string label = (cls_id < coco_names.size() ? coco_names[cls_id] : std::to_string(cls_id)) + ": " + std::to_string(score).substr(0, 4);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(img, cv::Point(x1, y1 - labelSize.height - 3), cv::Point(x1 + labelSize.width, y1), color, cv::FILLED);
            cv::putText(img, label, cv::Point(x1, y1 - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        return img;
    }

    // 推理单帧
    std::pair<cv::Mat, double> infer_single_frame(cv::Mat& img, const Args& args) {
        float scale;
        int dw, dh;
        
        // 预处理
        std::vector<float> input_tensor_values = preprocess(img, scale, dw, dh);

        // 创建 Tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), 
                                                                  input_tensor_values.size(), 
                                                                  input_shape.data(), input_shape.size());

        // 推理
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{nullptr}, 
                                                              input_names.data(), &input_tensor, 1, 
                                                              output_names.data(), output_names.size());
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end_time - start_time;

        // 获取输出数据和当前的动态形状
        const float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> current_output_shape = type_info.GetShape();

        std::vector<cv::Vec4f> final_boxes;
        std::vector<float> final_scores;
        std::vector<int> final_classes;

        if (args.end2end) {
            // 对应 Python: det_boxes = outputs[:,1:5], det_scores = outputs[:, 5], det_classes = outputs[:, 6]
            // 假设 shape: [num_boxes, 7]
            int num_boxes = current_output_shape[0];
            int dim = (current_output_shape.size() > 1) ? current_output_shape[1] : 7; 

            for (int i = 0; i < num_boxes; i++) {
                int offset = i * dim;
                float x1 = output_data[offset + 1];
                float y1 = output_data[offset + 2];
                float x2 = output_data[offset + 3];
                float y2 = output_data[offset + 4];
                float score = output_data[offset + 5];
                int cls_id = static_cast<int>(output_data[offset + 6]);

                final_boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                final_scores.push_back(score);
                final_classes.push_back(cls_id);
            }
        } 
        else if (args.end2end_model) {
            // 对应 Python: outputs = outputs[0]; scores = outputs[:, 4]; mask = scores > conf_thres; ...
            // 假设 shape: [1, num_boxes, 6] (例如 YOLOv10)
            int num_boxes = current_output_shape[1]; // Python 取了 outputs[0]，所以维度下降一级
            int dim = (current_output_shape.size() > 2) ? current_output_shape[2] : 6;

            for (int i = 0; i < num_boxes; i++) {
                int offset = i * dim;
                float score = output_data[offset + 4];
                
                // mask = scores > self.conf_thres
                if (score > conf_thres) {
                    float x1 = output_data[offset + 0];
                    float y1 = output_data[offset + 1];
                    float x2 = output_data[offset + 2];
                    float y2 = output_data[offset + 3];
                    int cls_id = static_cast<int>(output_data[offset + 5]);

                    final_boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                    final_scores.push_back(score);
                    final_classes.push_back(cls_id);
                }
            }

            // 对应 Python: if len(outputs) == 0: return img, 0
            if (final_boxes.empty()) {
                return {img, 0.0};
            }
        } 
        else {
            // 对应 Python: det_boxes, det_scores, det_classes = self.postprocess(...)
            postprocess(output_data, args.ultralytics, final_boxes, final_scores, final_classes);
        }

        // 绘制
        cv::Mat result_img = draw_results(img, final_boxes, final_scores, final_classes, scale, dw, dh);

        return {result_img, inference_time.count()};
    }

    // 运行主循环
    void run(const Args& args) {
        std::string source = args.source;
        
        // 1. 转小写并判断是否为图片 (对齐 Python: ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
        std::string lower_source = source;
        std::transform(lower_source.begin(), lower_source.end(), lower_source.begin(), ::tolower);
        bool is_image = (lower_source.find(".jpg") != std::string::npos || 
                         lower_source.find(".jpeg") != std::string::npos || 
                         lower_source.find(".png") != std::string::npos || 
                         lower_source.find(".bmp") != std::string::npos || 
                         lower_source.find(".webp") != std::string::npos);

        if (is_image) {
            // === 图片模式 ===
            std::cout << "正在处理图片: " << source << std::endl;
            cv::Mat img = cv::imread(source);
            if (img.empty()) {
                std::cout << "无法读取图片: " << source << std::endl;
                return;
            }

            auto [result_img, t] = infer_single_frame(img, args);
            
            std::string output_path = "result.jpg";
            if (args.save) {
                cv::imwrite(output_path, result_img);
            }
            std::cout << "推理时间: " << std::fixed << std::setprecision(2) << t << "ms, 结果已保存至: " << output_path << std::endl;
        } else {
            // === 视频/RTSP 模式 ===
            std::cout << "正在尝试打开视频源: " << source << std::endl;
            
            cv::VideoCapture cap;
            // 2. 判断是否全是数字 (对齐 Python source.isdigit())
            bool is_digit = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
            if (is_digit) {
                cap.open(std::stoi(source)); // 打开摄像头
            } else {
                cap.open(source);            // 打开视频文件或 RTSP 流
            }

            if (!cap.isOpened()) {
                std::cout << "无法打开视频源: " << source << std::endl;
                return;
            }

            // 获取视频属性
            int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = cap.get(cv::CAP_PROP_FPS);
            if (fps == 0.0) fps = 25.0; // 3. 防止 RTSP 获取不到 FPS 导致报错

            // 4. 判断是否是本地文件并准备 VideoWriter
            cv::VideoWriter out_writer;
            bool is_file = std::filesystem::exists(source); // 需要 C++17
            
            if (is_file && args.save) {
                std::string save_path = "result_video.mp4";
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                out_writer.open(save_path, fourcc, fps, cv::Size(width, height));
                std::cout << "视频处理中，结果将保存至: " << save_path << std::endl;
            } else {
                std::cout << "正在处理实时流 (按 'q' 退出)..." << std::endl;
            }

            int frame_count = 0;
            cv::Mat frame;
            while (cap.read(frame)) {
                if (frame.empty()) break;

                // 推理
                auto [result_img, t] = infer_single_frame(frame, args);
                
                // 5. 格式化并显示 FPS
                std::ostringstream text_stream;
                text_stream << std::fixed << std::setprecision(1) 
                            << "FPS: " << (1000.0 / t) 
                            << " (Inference: " << t << "ms)";
                cv::putText(result_img, text_stream.str(), cv::Point(20, 40), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

                // 写入视频文件
                if (out_writer.isOpened()) {
                    out_writer.write(result_img);
                }
                
                // 显示画面
                if (!args.no_show) {
                    cv::imshow("YOLO ONNX Runtime C++", result_img);
                    if (cv::waitKey(1) == 'q') {
                        break;
                    }
                }
                
                frame_count++;
                if (frame_count % 30 == 0) {
                    std::cout << "已处理 " << frame_count << " 帧, 当前推理耗时: " 
                              << std::fixed << std::setprecision(2) << t << "ms" << std::endl;
                }
            }

            // 资源释放
            cap.release();
            if (out_writer.isOpened()) {
                out_writer.release();
            }
            cv::destroyAllWindows();
            
            if (is_file) {
                std::cout << "视频处理完成。" << std::endl;
            }
        }
    }
};

int main(int argc, char** argv) {
    // 简单的参数硬编码，替代 Python 的 argparse
    Args args;
    args.model = "weights/yolov7-tiny.onnx";
    args.source = "data/1.jpg";
    args.end2end = false;
    args.end2end_model = false;
    args.ultralytics = false;
    args.no_show = false;
    args.save = true;
    
    // 如果有命令行参数，可以这里解析赋值给 args 结构体

    YoloOnnxRunner runner(args.model, 0.4f, 0.7f, 80);
    runner.run(args);

    return 0;
}