#!/bin/bash
# 开启报错即停止模式
set -e

# 设置你的 ONNX Runtime 路径
ORT_DIR="/media/jia/Data/project/yolos/onnxruntime-linux-x64-gpu-1.24.2"

echo "🚀 开始构建项目..."

# 智能判断：如果不存在 build 文件夹，才进行完整的 cmake 初始化
if [ ! -d "build" ]; then
    echo "📁 未检测到 build 文件夹，进行全量初始化 (Release 模式)..."
    mkdir build
    cd build
    # 强烈建议加上 Release 优化，速度直接起飞
    cmake -DONNXRUNTIME_DIR=$ORT_DIR -DCMAKE_BUILD_TYPE=Release ..
    cd ..
else
    echo "⚡ 检测到 build 文件夹，进行增量编译..."
fi

# 进入 build 目录进行编译
cd build
make -j8
cd ..

echo "✅ 编译成功！开始运行 YOLO 推理..."
echo "------------------------------------------------"

# 运行程序
./build/yolo_runner

echo "------------------------------------------------"
echo "🎉 运行结束！"