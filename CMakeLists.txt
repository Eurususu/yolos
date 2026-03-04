cmake_minimum_required(VERSION 3.14)
project(YoloOnnxRunner CXX)

# 我们代码里使用了 auto [a, b] 结构化绑定和 std::filesystem，需要 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 1. 查找 OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "找到 OpenCV 版本: ${OpenCV_VERSION}")

# 2. 查找/配置 ONNX Runtime
# 如果你在命令行传入了 -DONNXRUNTIME_DIR，就使用你指定的路径
if (ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
    
    # 针对不同操作系统的库文件后缀名处理
    if(WIN32)
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    elseif(APPLE)
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    else()
        set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    endif()
else()
    # 尝试在系统默认路径中寻找
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}" OR NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "未找到 ONNX Runtime！请在 cmake 时指定路径，例如：\ncmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..")
endif()

message(STATUS "ONNX Runtime 头文件路径: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime 动态库文件: ${ONNXRUNTIME_LIBRARY}")

# 3. 创建可执行文件
add_executable(yolo_runner ort_infer.cpp)

# 4. 链接头文件和库
target_include_directories(yolo_runner PUBLIC ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(yolo_runner PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})