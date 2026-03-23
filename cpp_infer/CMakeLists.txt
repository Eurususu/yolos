cmake_minimum_required(VERSION 3.18)
project(trt_cpp_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT - manual path
set(TENSORRT_ROOT "/opt/TensorRT")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include" CACHE PATH "TensorRT include directory")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib" CACHE PATH "TensorRT library directory")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/trt_engine.cpp
    src/trt_utils.cpp
    src/main.cpp
)

add_executable(trt_infer ${SOURCES})

# Link libraries
target_link_libraries(trt_infer
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
)

# Compile options
target_compile_options(trt_infer PRIVATE -Wall -O3)

# Set output directory
set_target_properties(trt_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
