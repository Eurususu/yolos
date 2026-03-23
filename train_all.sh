#!/bin/bash

# ================= 配置区域 =================
# 定义你要训练的模型列表 (只需写文件名，不带路径和后缀，后续会自动拼接)
# 注意：确保 weights/ 目录下真的有这些 .pt 文件
MODELS=("yolov10s" "yolo11n" "yolov12n" "yolov13n")

# 基础参数配置
DATA_CFG="data/coco128.yaml"
EPOCHS=10
BATCH_SIZE=16
DEVICES="0"
START_PORT=9527  # 起始端口号

# ================= 脚本逻辑 =================

echo "=========================================="
echo "开始批量训练任务..."
echo "待训练模型: ${MODELS[*]}"
echo "=========================================="

# 循环遍历模型列表
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_PATH="weights/${MODEL_NAME}.pt"
    
    # 动态计算端口号：9527, 9528, 9529... 防止端口冲突
    CURRENT_PORT=$((START_PORT + i))
    
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在启动第 $((i+1)) 个任务: ${MODEL_NAME}"
    echo "使用端口: ${CURRENT_PORT}"
    echo "------------------------------------------------------------------"

    # 检查权重文件是否存在，不存在则跳过，防止报错中断整个脚本
    if [ ! -f "$MODEL_PATH" ]; then
        echo "警告: 文件 $MODEL_PATH 不存在，跳过该模型！"
        continue
    fi

    # 执行训练命令
    # 注意：添加了 --name 参数，这样结果会保存在 runs/train/yolov10s_bird 这样的文件夹里
    # torchrun --nproc_per_node 2 --master_port $CURRENT_PORT train.py \
    #     --data "$DATA_CFG" \
    #     --model "$MODEL_PATH" \
    #     --epochs $EPOCHS \
    #     --batch $BATCH_SIZE \
    #     --device $DEVICES \
    #     --name "${MODEL_NAME}_bird" \
    #     --plots
    python train.py \
        --data "$DATA_CFG" \
        --model "$MODEL_PATH" \
        --epochs $EPOCHS \
        --batch $BATCH_SIZE \
        --device $DEVICES \
        --name "${MODEL_NAME}_bird" \
        --plots

    # 检查上一个命令是否执行成功
    if [ $? -eq 0 ]; then
        echo ">>> 模型 ${MODEL_NAME} 训练完成。"
    else
        echo ">>> ❌ 模型 ${MODEL_NAME} 训练出错！"
        # 如果你希望出错就停止整个脚本，取消下面这行的注释
        # exit 1 
    fi

    # 休息 10 秒，等待显存释放和系统清理僵尸进程
    echo "等待 10 秒冷却..."
    sleep 10
done

echo "=========================================="
echo "所有任务已完成！"
echo "=========================================="