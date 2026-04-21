#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <numeric>
#include "preprocess.h"
#include "NMSProcessor.h"

namespace fs = std::filesystem;


struct Args {
    std::string engine = "../weights/yolov7-tiny.engine";
    std::string source = "../data/";
    int opt_batch_size = 1;
    int max_batch_size = 32;
    bool save = false;
    std::string save_dir = "../results";
    float conf_thres = 0.25f;
    float iou_thres = 0.7f;
    int num_classes = 80;
    bool end2end = false;
    bool efficient_end2end = false;
    bool end2end_model = false;
    bool ultralytics = false;
    bool no_show = false;
    bool profile = false;
};


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

static const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};

struct BatchResult {
    std::vector<cv::Vec4f> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
};


// TRT 日志器
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING){
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TrtLogger gLogger;

// 为 TensorRT 对象专属定制的智能指针 Deleter
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) {
            delete obj; // TRT 8.0+ 标准销毁方式
        }
    }
};

// 存储张量信息的结构体
struct TensorInfo {
    std::string name;
    bool is_input;
    nvinfer1::DataType dtype;
    std::vector<int64_t> max_shape;     // 分配显存时用的最大形状
    std::vector<int64_t> actual_shape;  // 推理后得到的真实形状
    size_t size_bytes;                  // 显存分配大小
    void* dev_ptr = nullptr;            // GPU 指针
    std::vector<float> host_buffer;     // CPU 接收缓存 (仅输出需要)
};


class YoloTRTRunner {
    private:
        float conf_thres;
        float iou_thres;
        int num_classes;
        std::vector<std::string> class_names;
        int max_batch_size;
        int opt_batch_size;
        int max_det;

        std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine;
        std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> context;
        cudaStream_t stream;

        std::vector<TensorInfo> io_tensors;
        std::vector<uint8_t*> d_img_buffers;  // 指向每张图原数据显存的指针列表
        uint8_t** d_img_ptrs = nullptr;  // GPU 端的指针目录
        int max_src_bytes = 0; 
        int input_width;
        int input_height;

        static float box_iou(const cv::Vec4f& a, const cv::Vec4f& b){
            float x1 = std::max(a[0], b[0]);
            float y1 = std::max(a[1], b[1]);
            float x2 = std::min(a[2], b[2]);
            float y2 = std::min(a[3], b[3]);

            float w = std::max(0.0f, x2 - x1);
            float h = std::max(0.0f, y2 - y1);
            float inter = w * h;

            float area_a = (a[2] - a[0]) * (a[3] - a[1]);
            float area_b = (b[2] - b[0]) * (b[3] - b[1]);
            float uni = area_a + area_b - inter;

            if (uni <= 0.0f) return 0.0f;
            return inter / uni;
        }

        static std::vector<int> nms (const std::vector<cv::Vec4f>& boxes,
                                    const std::vector<float>& scores,
                                    const std::vector<int>& classes,
                                    float nms_threshold){
            if (scores.empty()) return {};
            
            // 初始化一个索引数组[0,1,2...,N-1]
            std::vector<int> indices(scores.size());
            std::iota(indices.begin(), indices.end(), 0);

            // 利用 Lambda 表达式根据类别和分数对“索引”进行排序
            std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                if (classes[a] != classes[b]) {
                    return classes[a] < classes[b];
                }
                return scores[a] > scores[b];
            });

            std::vector<int> keep;
            keep.reserve(scores.size() / 2);
            std::vector<uint8_t> suppressed(scores.size(), 0);

            for (size_t i = 0; i < indices.size(); i++) {
                int idx_i = indices[i];
                if (suppressed[idx_i]) continue;

                keep.push_back(idx_i);

                for (size_t j = i + 1; j < indices.size(); j++) {
                    int idx_j = indices[j];
                    
                    // 如果类别不同，直接跳出内层循环 (因为已经按类别排序过了)
                    if (classes[idx_i] != classes[idx_j]) break;

                    if (suppressed[idx_j]) continue;

                    // 使用对应索引的框计算 IoU
                    if (box_iou(boxes[idx_i], boxes[idx_j]) > nms_threshold) {
                        suppressed[idx_j] = 1;
                    }
                }
            }
            return keep;

        }

    public:
        YoloTRTRunner(const std::string& engine_path, int max_batch = 32, int opt_batch = 16, 
            int max_det = 300, float conf = 0.25f, float iou = 0.7f, int cls = 80, const std::vector<std::string>& class_names = COCO_NAMES)
            : conf_thres(conf), iou_thres(iou), num_classes(cls), class_names(class_names), max_batch_size(max_batch), max_det(max_det){
                opt_batch_size = (opt_batch > 0) ? opt_batch : max_batch_size;

                if (this->class_names.size() != static_cast<size_t>(this->num_classes)){
                    std::cerr << "[警告] 传入的类别名称数量 (" << this->class_names.size() 
                            << ") 不等于 num_classes (" << this->num_classes << ")! 画框时可能会越界。" << std::endl;
                }

                // 1. 加载 engine 二进制文件
                std::ifstream file(engine_path, std::ios::binary);
                if (!file.good()){
                    throw std::runtime_error("无法打开 Engine 文件: " + engine_path);
                }

                file.seekg(0, file.end);
                size_t size = file.tellg();
                file.seekg(0, file.beg);

                std::vector<char> engine_data(size);
                file.read(engine_data.data(), size);
                file.close();
                
                // 2. 实例化 TRT 对象，并使用 .reset() 交给智能指针接管
                runtime.reset(nvinfer1::createInferRuntime(gLogger));
                if (!runtime) throw std::runtime_error("创建 TRT Runtime 失败");

                initLibNvInferPlugins(&gLogger, "");

                engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
                if (!engine) throw std::runtime_error("反序列化 CudaEngine 失败");

                context.reset(engine->createExecutionContext());
                if (!context) throw std::runtime_error("创建 ExecutionContext 失败");

                cudaStreamCreate(&stream);

                // 3. 动态显存分配
                int num_io_tensors = engine->getNbIOTensors();
                for (int i = 0; i < num_io_tensors; ++i){
                    const char* name = engine->getIOTensorName(i);
                    bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
                    nvinfer1::Dims dims = engine->getTensorShape(name);

                    std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);

                    if (is_input){
                        if (shape[0] >= 1){
                            max_batch_size = shape[0];
                            if (opt_batch <= 0) opt_batch_size = shape[0];
                        }
                        
                        input_height = shape[2];
                        input_width = shape[3];

                        // 防止模型是全动态形状 [-1, 3, -1, -1] 导致宽高也是 -1
                        if (input_height == -1 || input_width == -1){
                            std::cout << "[警告] 检测到输入宽高为动态(-1)，强制使用默认 640x640！" << std::endl;
                            input_height = 640;
                            input_width = 640;
                        }

                    }
                    // IMSLayer output 显存分配
                    if (shape[0] == -1 && !is_input && shape.size() > 1 && shape[1] == 7){
                        shape[0] = max_det * max_batch_size;
                    } else if (shape[0] == -1) {
                        shape[0] = max_batch_size;
                    }

                    // 遍历其余维度，处理其他动态维度 (例如动态NMS插件的输出可能是 [batch, -1, 4])
                    for (size_t j = 1; j < shape.size(); ++j){
                        if (shape[j] == -1) shape[j] = max_det;
                    }

                    size_t vol = 1;
                    for (auto s : shape) vol *= s;
                    size_t bytes = vol * sizeof(float);

                    void* ptr = nullptr;
                    if (cudaMalloc(&ptr, bytes) != cudaSuccess){
                        throw std::runtime_error("CUDA Malloc 失败");
                    }

                    // TRT V3 API 绑定地址
                    context->setTensorAddress(name, ptr);

                    TensorInfo info;
                    info.name = name;
                    info.is_input = is_input;
                    info.dtype = engine->getTensorDataType(name);
                    info.max_shape = shape;
                    info.size_bytes = bytes;
                    info.dev_ptr = ptr;
                    if (!is_input){
                        info.host_buffer.resize(vol);
                    }
                    io_tensors.push_back(info);

                }
                cudaMalloc((void **)&d_img_ptrs, max_batch_size * sizeof(uint8_t*));
                d_img_buffers.resize(max_batch_size, nullptr);

        }
        
        ~YoloTRTRunner(){
            for (auto& t : io_tensors){
                if(t.dev_ptr) cudaFree(t.dev_ptr);
            }
            // 清理前处理显存池
            for (auto ptr: d_img_buffers) {if (ptr) cudaFree(ptr);}
            if (d_img_ptrs) cudaFree(d_img_ptrs);

            if (stream) cudaStreamDestroy(stream);
        }

        // 零拷贝预处理：返回 blob (cv::Mat 保证了底层的连续内存)
        cv::Mat preprocess_batch(const std::vector<cv::Mat>& image_list, std::vector<float>& scales, std::vector<int>& dws, std::vector<int>& dhs){
            int batch_size = image_list.size();
            scales.resize(batch_size);
            dws.resize(batch_size);
            dhs.resize(batch_size);

            std::vector<cv::Mat> padded_imgs;
            padded_imgs.reserve(batch_size);

            for (int b = 0; b < batch_size; ++b){
                const cv::Mat& image_src = image_list[b];
                int img_h = image_src.rows;
                int img_w = image_src.cols;

                float scale = std::min((float)input_height / img_h, (float) input_width / img_w);
                int new_w = static_cast<int>(std::round(img_w * scale));
                int new_h = static_cast<int>(std::round(img_h * scale));

                int dw = (input_width - new_w) / 2;
                int dh = (input_height - new_h) / 2;

                scales[b]  = scale;
                dws[b] = dw;
                dhs[b] = dh;

                cv::Mat image_resized, image_padded;
                cv::resize(image_src, image_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

                int top = dh;
                int bottom = input_height - new_h - dh;
                int left = dw;
                int right = input_width - new_w - dw;
                cv::copyMakeBorder(image_resized, image_padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
                padded_imgs.push_back(image_padded);
            }

            return cv::dnn::blobFromImages(padded_imgs, 1.0/255.0, cv::Size(input_width, input_height), cv::Scalar(), true, false);
        }
        void postprocess_single(const float* output_data, bool ultralytics, int num_anchors, int channels,
                                float scale, int dw, int dh, BatchResult& res) {
            std::vector<float> all_scores;
            std::vector<int> all_classes;
            std::vector<cv::Vec4f> all_boxes;

            all_scores.reserve(num_anchors);
            all_classes.reserve(num_anchors);
            all_boxes.reserve(num_anchors);

            if (ultralytics) {
                cv::Mat out_mat(channels, num_anchors, CV_32F, (void*)output_data);
                cv::Mat transposed_mat;
                cv::transpose(out_mat, transposed_mat);
                
                const float* t_data = (const float*)transposed_mat.data;
                for (int i = 0; i < num_anchors; i++) {
                    const float* row = t_data + i * channels;
                    float cx = row[0], cy = row[1], w = row[2], h = row[3];

                    float max_score = 0.0f;
                    int max_class_id = -1;
                    for(int c = 0; c < num_classes; c++) {
                        float score = row[4 + c];
                        if(score > max_score){
                            max_score = score;
                            max_class_id = c;
                        }
                    }

                    if (max_score > conf_thres) {
                        // 【优化】：直接将 cx, cy, w, h 转换为 x1, y1, x2, y2 存入
                        all_boxes.push_back(cv::Vec4f(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f));
                        all_scores.push_back(max_score);
                        all_classes.push_back(max_class_id);
                    }
                }
            } else {
                for (int i = 0; i < num_anchors; i++) {
                    const float* anchor_data = output_data + i * channels;
                    float cx = anchor_data[0], cy = anchor_data[1], w = anchor_data[2], h = anchor_data[3];
                    float obj_conf = anchor_data[4];

                    const float* class_probs = &anchor_data[5];
                    auto max_iter = std::max_element(class_probs, class_probs + num_classes);
                    float max_class_prob = *max_iter;
                    int max_class_id = static_cast<int>(std::distance(class_probs, max_iter));

                    float final_score = obj_conf * max_class_prob;
                    if (final_score >= conf_thres) {
                        // 【优化】：直接将 cx, cy, w, h 转换为 x1, y1, x2, y2 存入
                        all_boxes.push_back(cv::Vec4f(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f));
                        all_scores.push_back(final_score);
                        all_classes.push_back(max_class_id);
                    }
                }
            }

            // -------------------------------------------------------------
            // 这里彻底删除了原来冗余的 class_indices 和 cv::dnn 的 NMS 逻辑
            // -------------------------------------------------------------

            // 调用我们的静态 NMS，获取最终需要保留的框的索引
            std::vector<int> keep_indices = nms(all_boxes, all_scores, all_classes, iou_thres);

            // 计算缩放和平移还原系数
            float inv_scale = 1.f / scale;
            float neg_dw = -dw;
            float neg_dh = -dh;

            // 遍历保留的索引，映射回原图并存入结果中
            for (int idx : keep_indices) {
                const cv::Vec4f& box = all_boxes[idx];
                
                float bx1 = (box[0] + neg_dw) * inv_scale;
                float by1 = (box[1] + neg_dh) * inv_scale;
                float bx2 = (box[2] + neg_dw) * inv_scale;
                float by2 = (box[3] + neg_dh) * inv_scale;

                res.boxes.emplace_back(bx1, by1, bx2, by2);
                res.scores.push_back(all_scores[idx]);
                res.classes.push_back(all_classes[idx]);
            }
        }

        // 解析输出，拦截一切越界/幽灵数据
        std::vector<BatchResult> process_output(const Args& args, int real_batch_size, const std::vector<float>& scales, 
            const std::vector<int>& dws, const std::vector<int>& dhs){
            std::vector<BatchResult> batch_dets(real_batch_size);
            
            // 收集所有的输出 Tensor
            std::vector<TensorInfo*> outputs;
            for (auto& t : io_tensors){
                if (!t.is_input) outputs.push_back(&t);
            }
            if (outputs.empty()) return batch_dets;

            if (args.efficient_end2end){
                if (outputs.size() < 4){
                    std::cerr << "错误: efficient_end2end 需要模型有 4 个输出节点!" << std::endl;
                    return batch_dets;
                }

                // TRT 插件标准输出顺序：num_dets(0), boxes(1), scores(2), classes(3)
                TensorInfo* t_num = outputs[0];
                TensorInfo* t_box = outputs[1];
                TensorInfo* t_score = outputs[2];
                TensorInfo* t_cls = outputs[3];

                int max_det = t_box->actual_shape[1]; // [batch, max_det, 4]

                // 智能读取工具：因为 num_dets 和 classes 可能是 int32 或 float32
                auto get_int_val = [](TensorInfo* t, int index) -> int {
                    if (t->dtype == nvinfer1::DataType::kINT32) {
                        return reinterpret_cast<const int32_t*>(t->host_buffer.data())[index];
                    } else {
                        return static_cast<int>(t->host_buffer[index]);
                    }
                };
                
                const float* boxes_ptr = t_box->host_buffer.data();
                const float* scores_ptr = t_score->host_buffer.data();

                for (int b = 0; b < real_batch_size; b++){
                    // 读取当前帧有效框的数量
                    int valid_count = get_int_val(t_num, b);

                    for (int i =0; i < valid_count; i++){
                        float score = scores_ptr[b * max_det + i];
                        if (score > conf_thres){
                            float inv_scale = 1.0f / scales[b];
                            const float* box = boxes_ptr + (b * max_det + i) * 4;
                            float x1 = (box[0] - dws[b]) * inv_scale;
                            float y1 = (box[1] - dhs[b]) * inv_scale;
                            float x2 = (box[2] - dws[b]) * inv_scale;
                            float y2 = (box[3] - dhs[b]) * inv_scale;

                            int cls = get_int_val(t_cls, b * max_det + i);

                            batch_dets[b].boxes.push_back(cv::Vec4f(x1, y1, x2, y2));
                            batch_dets[b].scores.push_back(score);
                            batch_dets[b].classes.push_back(cls);
                        }
                    }
                }
            }

            else{
                TensorInfo* out_tensor = outputs[0];
                const float* output_data = out_tensor->host_buffer.data();
                const auto& actual_shape = out_tensor->actual_shape;
                int ndim = actual_shape.size();

                if (args.end2end){
                    int num_dets = actual_shape[0];
                    int dim = actual_shape[1]; // 7
                    for (int i = 0; i < num_dets; i++){
                        const float* row = output_data + i * dim;
                        int b = static_cast<int>(row[0]);
                        // batch 不能大于real_batch 不能小于0
                        if (b < 0 || b >= real_batch_size) continue;
                        float score = row[5];
                        if (score > conf_thres){
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
                else if (args.end2end_model){
                    int num_anchors = actual_shape[ndim - 2];
                    int dim = actual_shape[ndim - 1];
                    for (int b = 0; b < real_batch_size; b++){
                        const float* batch_ptr = output_data + b * num_anchors * dim;
                        for (int i = 0; i < num_anchors; i++){
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
                else{
                    int num_anchors, channels;
                    if (args.ultralytics){
                        channels = actual_shape[ndim - 2];
                        num_anchors = actual_shape[ndim - 1];
                        for (int b = 0; b < real_batch_size; b++) {
                            const float* batch_ptr = output_data + b * channels * num_anchors;
                            postprocess_single(batch_ptr, true, num_anchors, channels, scales[b], dws[b], dhs[b], batch_dets[b]);
                        }
                    } else {
                        num_anchors = actual_shape[ndim - 2];
                        channels = actual_shape[ndim - 1];
                        for (int b = 0; b < real_batch_size; b++) {
                            const float* batch_ptr = output_data + b * num_anchors * channels;
                            postprocess_single(batch_ptr, false, num_anchors, channels, scales[b], dws[b], dhs[b], batch_dets[b]);
                        }
                    }
                }
            }
            return batch_dets;
        }

        // draw rectangle
        void draw_results(cv::Mat& img, const BatchResult& res){
            for (size_t i = 0; i < res.boxes.size(); i++){
                // static_cast 杜绝隐式转换警告
                int x1 = static_cast<int>(std::round(res.boxes[i][0]));
                int y1 = static_cast<int>(std::round(res.boxes[i][1]));
                int x2 = static_cast<int>(std::round(res.boxes[i][2]));
                int y2 = static_cast<int>(std::round(res.boxes[i][3]));

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

        // 核心推理
        std::pair<std::vector<cv::Mat>, std::vector<float>> infer_batch(const std::vector<cv::Mat>& img_list, const Args& args){
            int real_batch_size = img_list.size();
            std::vector<float> scales;
            std::vector<int> dws, dhs;

            // 1. 预处理
            // cv::Mat blob = preprocess_batch(img_list, scales, dws, dhs);

            cudaEvent_t event_start, event_h2d_preprocess, event_comp, event_d2h;
            if (args.profile){
                cudaEventCreate(&event_start);
                cudaEventCreate(&event_h2d_preprocess);
                cudaEventCreate(&event_comp);
                cudaEventCreate(&event_d2h);
                cudaEventRecord(event_start, stream);
            }

            // 2. H2D
            TensorInfo* input_tensor = nullptr;
            for (auto& t : io_tensors){
                if (t.is_input){
                    input_tensor = &t;
                    // 设置输入维度
                    nvinfer1::Dims4 input_dims {real_batch_size, 3, input_height, input_width};
                    context->setInputShape(t.name.c_str(), input_dims);

                    // // 仅拷贝真实的 batch size 数据
                    // size_t input_bytes = real_batch_size * 3 * input_height * input_width * sizeof(float);
                    // cudaMemcpyAsync(t.dev_ptr, blob.ptr<float>(), input_bytes, cudaMemcpyHostToDevice, stream);
                } else {
                    // 防止INMSlayer的多余框
                    cudaMemsetAsync(t.dev_ptr, 0, t.size_bytes, stream);
                }
            }
            float* trt_input_ptr = static_cast<float*>(input_tensor->dev_ptr);

            // --- 2. 动态维护原图显存池 (懒加载机制) ---
            int max_current_bytes = 0;
            std::vector<cv::Mat> continuous_imgs;
            continuous_imgs.reserve(real_batch_size);

            for (int b = 0; b < real_batch_size; b++) {
                cv::Mat img = img_list[b];
                
                // 必须检查！打破 OpenCV 的内存 Padding 陷阱，强制内存连续
                if (!img.isContinuous()) {
                    img = img.clone();
                }
                continuous_imgs.push_back(img);
                
                int bytes = img.cols * img.rows * 3;
                if (bytes > max_current_bytes) {
                    max_current_bytes = bytes;
                }
            }

            // 如果遇到比以前更大的图，扩容 GPU 显存池
            // 如果是比之前小或者一样的图片，则不需要再次分配显存，因为现在的显存足够大了
            // 因为分配大块显存极其耗时，我们要尽可能白嫖已经开辟好的大空间。
            if (max_current_bytes > max_src_bytes) {
                for (auto ptr : d_img_buffers) { if(ptr) cudaFree(ptr); }
                for (int i = 0; i < max_batch_size; i++) {
                    cudaMalloc((void**)&d_img_buffers[i], max_current_bytes); 
                }
                max_src_bytes = max_current_bytes;
            }
            // d_img_buffers 存在显存复用， 做到零显存分配
            launch_preprocess_cuda(
                continuous_imgs,
                trt_input_ptr,
                input_width, input_height,
                d_img_buffers,
                d_img_ptrs,
                scales, dws, dhs,   // <--- 这里接收返回值
                stream
            );



            if (args.profile) cudaEventRecord(event_h2d_preprocess, stream);

            // 3. infer(V3接口)
            context->enqueueV3(stream);

            if (args.profile) cudaEventRecord(event_comp, stream);

            // 4. D2H & 获取真实维度 (零拷贝视图的等效实现)
            for (auto& t : io_tensors){
                if (!t.is_input){
                    nvinfer1::Dims actual_dims = context->getTensorShape(t.name.c_str());

                    bool has_dynamic = false;
                    size_t actual_vol = 1;
                    for (int j = 0; j < actual_dims.nbDims; ++j) {
                        if (actual_dims.d[j] < 0){
                            has_dynamic = true;
                            break;
                        }
                        actual_vol *= actual_dims.d[j];
                    }

                    size_t bytes_to_copy = has_dynamic ? t.size_bytes : actual_vol * sizeof(float);
                    t.actual_shape = has_dynamic ? t.max_shape : std::vector<int64_t>(actual_dims.d, actual_dims.d + actual_dims.nbDims);

                    // size_t bytes_to_copy = 0;
                    // if (has_dynamic){
                    //     bytes_to_copy = t.size_bytes;
                    //     t.actual_shape = t.max_shape;
                    // } else{
                    //     bytes_to_copy = actual_vol * sizeof(float);
                    //     t.actual_shape.assign(actual_dims.d, actual_dims.d + actual_dims.nbDims);
                    // }

                    // 仅拷回有效数据，剔除无用显存，极大幅度提升 D2H 速度 但是 INMSlayer 拷贝所有数据
                    cudaMemcpyAsync(t.host_buffer.data(), t.dev_ptr, bytes_to_copy, cudaMemcpyDeviceToHost, stream);
                }
            }

            if (args.profile) cudaEventRecord(event_d2h, stream);

            cudaStreamSynchronize(stream);

            std::vector<float> prof_times(5, 0.0f);
            if (args.profile) {
                cudaEventElapsedTime(&prof_times[0], event_start, event_h2d_preprocess);
                cudaEventElapsedTime(&prof_times[1], event_h2d_preprocess, event_comp);
                cudaEventElapsedTime(&prof_times[2], event_comp, event_d2h);
                cudaEventDestroy(event_start); cudaEventDestroy(event_h2d_preprocess);
                cudaEventDestroy(event_comp);  cudaEventDestroy(event_d2h);
            }

            // 5. 后处理
            auto t_post_start = std::chrono::high_resolution_clock::now();
            std::vector<BatchResult> batch_dets = process_output(args, real_batch_size, scales, dws, dhs);
            auto t_post_end = std::chrono::high_resolution_clock::now();

            if (args.profile) {
                // 计算后处理耗时存入 prof_times[3]
                prof_times[3] = std::chrono::duration<float, std::milli>(t_post_end - t_post_start).count();
            }

            // 6. 画框
            auto t_draw_start = std::chrono::high_resolution_clock::now();
            std::vector<cv::Mat> result_imgs;
            for (int i = 0; i < real_batch_size; i++) {
                cv::Mat img = img_list[i].clone();
                draw_results(img, batch_dets[i]);
                result_imgs.push_back(img);
            }
            auto t_draw_end = std::chrono::high_resolution_clock::now();
            if (args.profile) {
                prof_times[4] = std::chrono::duration<float, std::milli>(t_draw_end - t_draw_start).count();
            }

            return {result_imgs, prof_times};
        }

        void run(const Args& args){
            std::string source = args.source;
            int batch_size = opt_batch_size; // 按最优批次走
            std::string save_dir = args.save_dir;

            if (args.save) {
                fs::create_directories(save_dir);
            }

            if (fs::is_directory(source)){
                // === 模式 1: 目录图片多批次攒帧推理 ===
                std::vector<std::string> img_paths;
                for (const auto& entry : fs::directory_iterator(source)){
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end()){
                        img_paths.push_back(entry.path().string());
                    }
                }

                std::sort(img_paths.begin(), img_paths.end());

                std::cout << "找到 " << img_paths.size() << " 张图片，按照 opt_batch_size=" << batch_size << " 开始推理...\n";

                for (size_t i = 0; i < img_paths.size(); i += batch_size){
                    std::vector<cv::Mat> valid_imgs;
                    std::vector<std::string> valid_names;

                    for (size_t j = i; j < std::min(i + batch_size, img_paths.size()); ++j){
                        cv::Mat img = cv::imread(img_paths[j]);
                        if (!img.empty()){
                            valid_imgs.push_back(img);
                            valid_names.push_back(fs::path(img_paths[j]).filename().string());
                        }
                    }

                    if (valid_imgs.empty()) continue;

                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto [res_imgs, prof] = infer_batch(valid_imgs, args);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    double t = std::chrono::duration<double, std::milli>(t2 - t1).count();


                    // if (args.profile){
                    //     printf("[Profile] H2D: %.2fms | Compute: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
                    // }
                    if (args.profile){
                        printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | D2H: %.2fms | Postprocess: %.2fms | Draw: %.2fms\n", 
                        prof[0], prof[1], prof[2], prof[3], prof[4]);
                    }
                    std::cout << "已处理进度: " << std::min(i + batch_size, img_paths.size()) << "/" << img_paths.size()
                            << " | Batch总耗时: " << std::fixed << std::setprecision(2) << t << "ms\n";

                    
                    if (args.save){
                        for (size_t k = 0; k < res_imgs.size(); ++k){
                            cv::imwrite((fs::path(save_dir)/valid_names[k]).string(), res_imgs[k]);
                        }
                    }
                }
                std::cout << "✅ 目录处理完成。\n";
            }else{
                std::string ext = fs::path(source).extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                bool is_image = std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end();

                if (is_image){
                    // === 模式 2: 单张图片推理 ===
                    cv::Mat img = cv::imread(source);
                    if (img.empty()) return;

                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto [res_imgs, prof] = infer_batch({img}, args);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    double t = std::chrono::duration<double, std::milli>(t2 - t1).count();

                    // if (args.profile) printf("[Profile] H2D: %.2fms | Compute: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
                    if (args.profile){
                        printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | D2H: %.2fms | Postprocess: %.2fms | Draw: %.2fms\n", 
                        prof[0], prof[1], prof[2], prof[3], prof[4]);
                    }
                    if (args.save) cv::imwrite((fs::path(save_dir)/fs::path(source).filename()).string(), res_imgs[0]);
                    std::cout << "推理时间: " << t << "ms, 结果已保存\n";
                }
                else {
                    // === 模式 3: 视频/RTSP 攒帧加速推理 ===
                    cv::VideoCapture cap;
                    bool is_digit = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
                    if (is_digit) cap.open(std::stoi(source));
                    else cap.open(source);
                    if (!cap.isOpened()) return;

                    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                    double fps = cap.get(cv::CAP_PROP_FPS);
                    if (fps == 0.0) fps = 25.0;

                    cv::VideoWriter out_writer;
                    bool is_file = fs::exists(source);
                    if (is_file && args.save){
                        std::string save_path = (fs::path(save_dir)/fs::path(source).filename()).string();
                        out_writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
                        std::cout << "视频开始处理，按 opt_batch=" << batch_size << " 攒批...\n";
                    }

                    std::vector<cv::Mat> batch_frames;
                    int frame_count = 0;
                    bool stop_flag = false;
                    cv::Mat frame;

                    while (cap.read(frame)){
                        if (frame.empty()) break;
                        batch_frames.push_back(frame.clone());
                        
                        if (batch_frames.size() == static_cast<int>(batch_size)){
                            auto t1 = std::chrono::high_resolution_clock::now();
                            auto [res_imgs, prof] = infer_batch(batch_frames, args);
                            auto t2 = std::chrono::high_resolution_clock::now();
                            double batch_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
                            double fps_curr = 1000.0 / (batch_time / batch_size);

                            // if (args.profile) {
                            //     printf("[Profile] H2D: %.2fms | Comp: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
                            // }
                            if (args.profile){
                                printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | D2H: %.2fms | Postprocess: %.2fms | Draw: %.2fms\n", 
                                prof[0], prof[1], prof[2], prof[3], prof[4]);
                            }

                            for (size_t i = 0; i < res_imgs.size(); i++){
                                char fps_text[64];
                                snprintf(fps_text, sizeof(fps_text), "FPS: %.1f (Batch: %d)", fps_curr, batch_size);
                                cv::putText(res_imgs[i], fps_text, cv::Point(20, 40),
                                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

                                if (out_writer.isOpened()) out_writer.write(res_imgs[i]);

                                if (!args.no_show){
                                    cv::imshow("TRT C++ Inference", res_imgs[i]);
                                    if (cv::waitKey(1) == 'q') {stop_flag = true; break;}
                                }
                            }
                            if (stop_flag) break;

                            frame_count += batch_size;
                            if (frame_count % (batch_size * 5) == 0) {
                                std::cout << "已处理 " << frame_count << " 帧, 最近批次耗时: " << std::fixed << std::setprecision(1) << batch_time << "ms\n";
                            }

                            batch_frames.clear();
                        }
                    }

                    // 处理尾部帧
                    if (!batch_frames.empty() && !stop_flag){
                        auto [res_imgs, prof] = infer_batch(batch_frames, args);
                        for (size_t i = 0; i < res_imgs.size(); i++) {
                            if (out_writer.isOpened()) out_writer.write(res_imgs[i]);
                            if (!args.no_show) { cv::imshow("TRT C++ Inference", res_imgs[i]); cv::waitKey(1); }
                        }
                    }

                    cap.release();
                    if (out_writer.isOpened()) out_writer.release();
                    cv::destroyAllWindows();
                    std::cout << "✅ 视频检测完毕。\n";
                }
            }
        }
};



int main(int argc, char** argv){
    Args args;

    // 简单 CLI 解析
    for (int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) args.engine = argv[++i];
        else if (arg == "--source" && i + 1 < argc) args.source = argv[++i];
        else if (arg == "--opt_batch_size" && i + 1 < argc) args.opt_batch_size = std::stoi(argv[++i]);
        else if (arg == "--max_batch_size" && i + 1 < argc) args.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--save_dir" && i + 1 < argc) args.save_dir = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) args.conf_thres = std::stof(argv[++i]);
        else if (arg == "--iou" && i + 1 < argc) args.iou_thres = std::stof(argv[++i]);
        else if (arg == "--classes" && i + 1 < argc) args.num_classes = std::stoi(argv[++i]);
        else if (arg == "--save") args.save = true;
        else if (arg == "--efficient_end2end") args.efficient_end2end = true;
        else if (arg == "--end2end") args.end2end = true;
        else if (arg == "--end2end_model") args.end2end_model = true;
        else if (arg == "--ultralytics") args.ultralytics = true;
        else if (arg == "--no_show") args.no_show = true;
        else if (arg == "--profile") args.profile = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "  --engine <path>     引擎路径 (默认: weights/yolov7-tiny.engine)\n"
                      << "  --source <path>     输入文件/目录/摄像头\n"
                      << "  --opt_batch_size    推理批量大小 (默认: 1)\n"
                      << "  --max_batch_size    最大允许推理batch\n"
                      << "  --classes           类别数量\n"
                      << "  --save_dir <path>   保存结果目录\n"
                      << "  --save              是否保存结果\n"
                      << "  --profile           开启 CUDA 测速\n"
                      << "  --efficient_end2end 使用efficient_nms 插件\n"
                      << "  --end2end           使用 INMSlayer 插件\n"
                      << "  --end2end_model     使用端到端模型, 如yolo26\n"
                      << "  --ultralytics       使用 Ultralytics 模型\n"
                      << "  --no_show           是否显示画面,默认显示\n";
            return 0;
        }
    }

    try {
        YoloTRTRunner runner(args.engine, args.max_batch_size, args.opt_batch_size, 300, args.conf_thres, args.iou_thres, args.num_classes);
        runner.run(args);
    } catch (const std::exception& e){
        std::cerr << "致命错误: " << e.what() << std::endl;
    }
    return 0;
}
