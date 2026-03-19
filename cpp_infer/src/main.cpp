#include "trt_engine.h"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -e, --engine <path>    TRT engine path (required)\n"
              << "  -i, --image <path>     Input image path\n"
              << "  -o, --output <path>    Output image/video path (default: result.jpg/result.mp4)\n"
              << "  -v, --video <path>     Input video path or camera index\n"
              << "  --conf <float>         Confidence threshold (default: 0.25)\n"
              << "  --iou <float>          NMS IoU threshold (default: 0.7)\n"
              << "  --end2end              Use end2end engine\n"
              << "  --efficient_end2end   Use efficient_end2end engine\n"
              << "  --ultralytics          Use ultralytics model\n"
              << "  --end2end_model        Use end2end model\n"
              << "  -h, --help             Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Default values
    std::string engine_path;
    std::string image_path;
    std::string output_path;
    std::string video_path;
    float conf_threshold = 0.25;
    float iou_threshold = 0.7;
    bool end2end = false;
    bool efficient_end2end = false;
    bool ultralytics = false;
    bool end2end_model = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-e" || arg == "--engine") {
            if (i + 1 < argc) {
                engine_path = argv[++i];
            }
        } else if (arg == "-i" || arg == "--image") {
            if (i + 1 < argc) {
                image_path = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_path = argv[++i];
            }
        } else if (arg == "-v" || arg == "--video") {
            if (i + 1 < argc) {
                video_path = argv[++i];
            }
        } else if (arg == "--conf") {
            if (i + 1 < argc) {
                conf_threshold = std::stof(argv[++i]);
            }
        } else if (arg == "--iou") {
            if (i + 1 < argc) {
                iou_threshold = std::stof(argv[++i]);
            }
        } else if (arg == "--end2end") {
            end2end = true;
        } else if (arg == "--efficient_end2end") {
            efficient_end2end = true;
        } else if (arg == "--ultralytics") {
            ultralytics = true;
        } else if (arg == "--end2end_model") {
            end2end_model = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate arguments
    if (engine_path.empty()) {
        std::cerr << "Error: Engine path is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (end2end && end2end_model) {
        std::cerr << "Error: end2end model is already End2End\n" << std::endl;
        return 1;
    }

    try {
        // Create predictor with iou_threshold
        std::cout << "Loading engine: " << engine_path << std::endl;
        trtinfer::TrtEngine predictor(engine_path, 1, 300, iou_threshold);
        std::cout << "Engine loaded successfully" << std::endl;

        // Get FPS
        float fps = predictor.get_fps();
        std::cout << "FPS: " << fps << std::endl;

        // Run inference on image
        if (!image_path.empty()) {
            if (output_path.empty()) {
                output_path = "result.jpg";
            }
            std::cout << "Processing image: " << image_path << std::endl;
            auto detections = predictor.inference(image_path, output_path, conf_threshold, iou_threshold,
                                                    end2end, efficient_end2end,
                                                    ultralytics, end2end_model);
            std::cout << "Detected " << detections.size() << " objects" << std::endl;
            std::cout << "Result saved to: " << output_path << std::endl;
        }

        // Run inference on video
        if (!video_path.empty()) {
            if (output_path.empty()) {
                output_path = "result.mp4";
            }
            std::cout << "Processing video: " << video_path << std::endl;
            predictor.detect_video(video_path, output_path, conf_threshold, iou_threshold,
                                    end2end, efficient_end2end,
                                    ultralytics, end2end_model);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
