#include "trt_utils.h"
#include "trt_engine.h"
#include <algorithm>
#include <cmath>

namespace trtinfer {

// Letterbox resize
void letterbox(const cv::Mat& img, cv::Mat& out, float& ratio, float& dw, float& dh,
               int target_width, int target_height, const cv::Scalar& color) {
    int orig_width = img.cols;
    int orig_height = img.rows;

    // Calculate scale ratio
    float r = std::min(static_cast<float>(target_width) / orig_width,
                       static_cast<float>(target_height) / orig_height);

    int new_unpad_width = static_cast<int>(orig_width * r);
    int new_unpad_height = static_cast<int>(orig_height * r);

    // Resize image
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_unpad_width, new_unpad_height), 0, 0, cv::INTER_LINEAR);

    // Calculate padding
    dw = (target_width - new_unpad_width) / 2.0f;
    dh = (target_height - new_unpad_height) / 2.0f;

    // Add border
    cv::copyMakeBorder(resized, out, static_cast<int>(dh), static_cast<int>(dh),
                       static_cast<int>(dw), static_cast<int>(dw),
                       cv::BORDER_CONSTANT, color);

    // Convert BGR to RGB
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);

    ratio = r;
}

// NMS implementation
float box_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.bbox[0], b.bbox[0]);
    float y1 = std::max(a.bbox[1], b.bbox[1]);
    float x2 = std::min(a.bbox[2], b.bbox[2]);
    float y2 = std::min(a.bbox[3], b.bbox[3]);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    float area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    float union_area = area_a + area_b - intersection;

    return intersection / union_area;
}

std::vector<Detection> nms(std::vector<Detection>& detections, float nms_threshold) {
    if (detections.empty()) return {};
    // Sort by score (descending)
    std::sort(detections.begin(), detections.end(),
               [](const Detection& a, const Detection& b){
                    if (a.class_id != b.class_id) {
                        return a.class_id < b.class_id;
                    }
                    return a.score > b.score;
                    });

    std::vector<Detection> result;
    result.reserve(detections.size() / 2);
    std::vector<uint8_t> suppressed(detections.size(), 0);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (detections[i].class_id != detections[j].class_id) {
                break;
            }

            if (suppressed[j]) continue;

            float iou = box_iou(detections[i], detections[j]);

            if (iou > nms_threshold){
                suppressed[j] = 1;
            }
        }
    }

    return result;
}

// Generate rainbow colors
std::vector<cv::Scalar> generate_colors(int num_classes) {
    std::vector<cv::Scalar> colors;
    colors.reserve(num_classes);

    for (int i = 0; i < num_classes; i++) {
        float hue = static_cast<float>(i) / num_classes;
        // HSV to BGR
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue * 180, 255, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
        colors.push_back(cv::Scalar(pixel[0], pixel[1], pixel[2]));
    }

    return colors;
}

// Visualization
void visualize_detection(cv::Mat& img, const std::vector<Detection>& detections,
                         float conf_threshold,
                         const std::vector<std::string>& class_names) {
    static auto colors = generate_colors(80);

    for (const auto& det : detections) {
        if (det.score < conf_threshold) continue;

        int class_id = det.class_id;
        float x1 = det.bbox[0];
        float y1 = det.bbox[1];
        float x2 = det.bbox[2];
        float y2 = det.bbox[3];

        // Draw box
        cv::Rect box(static_cast<int>(x1), static_cast<int>(y1),
                     static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        cv::rectangle(img, box, colors[class_id % colors.size()], 2);

        // Draw label
        std::string label;
        if (class_id < static_cast<int>(class_names.size())) {
            label = class_names[class_id] + ": " + std::to_string(static_cast<int>(det.score * 100)) + "%";
        } else {
            label = "Class " + std::to_string(class_id) + ": " + std::to_string(static_cast<int>(det.score * 100)) + "%";
        }

        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::rectangle(img,
                      cv::Point(x1, y1 - text_size.height - 10),
                      cv::Point(x1 + text_size.width, y1),
                      colors[class_id % colors.size()], -1);

        cv::putText(img, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

}  // namespace trtinfer
