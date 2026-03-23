#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace trtinfer {

// Detection structure
struct Detection {
    float bbox[4];  // x1, y1, x2, y2
    float score;
    int class_id;
};

// Letterbox resize with padding
void letterbox(const cv::Mat& img, cv::Mat& out, float& ratio, float& dw, float& dh,
               int target_width, int height, const cv::Scalar& color = cv::Scalar(114, 114, 114));

// NMS (Non-Maximum Suppression)
std::vector<Detection> nms(std::vector<Detection>& detections, float nms_threshold = 0.45f);

// Visualization
void visualize_detection(cv::Mat& img, const std::vector<Detection>& detections,
                         float conf_threshold = 0.5,
                         const std::vector<std::string>& class_names = {});

// Generate rainbow colors
std::vector<cv::Scalar> generate_colors(int num_classes);

}  // namespace trtinfer
