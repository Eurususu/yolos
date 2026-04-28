cmake_minimum_required(VERSION 3.18)
project(trt_cpp_infer LANGUAGES CXX CUDA)

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
# file(GLOB CU_SOURCES "src/*.cu")
# file(GLOB CUH_SOURCES "src/*.cuh")
set(SOURCES
    src/efficientNMSInference.cu
    # src/efficientNMSInference.cuh
    src/NMSProcessor.cu
    src/preprocess.cu
    src/trt_infer_batch_cuda.cpp
)

set(SOURCES_PIPELINE
    src/efficientNMSInference.cu
    # src/efficientNMSInference.cuh
    src/NMSProcessor.cu
    src/preprocess.cu
    src/trt_infer_cuda_pipeline.cpp
)

add_executable(trt_infer_batch_cuda ${SOURCES})
add_executable(trt_infer_cuda_pipeline ${SOURCES_PIPELINE})
add_executable(trt_infer_batch src/trt_infer_batch.cpp)
add_executable(trt_infer_batch_cvnms src/trt_infer_batch_cvnms.cpp)

# Link libraries
target_link_libraries(trt_infer_batch_cuda
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
)

target_link_libraries(trt_infer_cuda_pipeline
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
)

target_link_libraries(trt_infer_batch
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
)
target_link_libraries(trt_infer_batch_cvnms
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    nvinfer_plugin
)

# Compile options
target_compile_options(trt_infer_batch_cuda PRIVATE -Wall -O3)
target_compile_options(trt_infer_cuda_pipeline PRIVATE -Wall -O3 -pthread)
target_compile_options(trt_infer_batch PRIVATE -Wall -O3)
target_compile_options(trt_infer_batch_cvnms PRIVATE -Wall -O3)


# Set output directory
set_target_properties(trt_infer_batch_cuda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(trt_infer_cuda_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(trt_infer_batch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(trt_infer_batch_cvnms PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
