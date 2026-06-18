#!/bin/bash

# ================= 配置区域 =================
MODELS=("yolov8s" "yolov8m" "yolov10s" "yolov10m" "yolo11s" "yolo11m" "yolo12s" "yolo12m" "yolo26s" "yolo26m")
DATA_CFG="data/ball_human.yaml"
EPOCHS=100
BATCH_SIZE=32  # 如果 m 模型报 OOM，可以改为 16
DEVICES="0,1"

# ================= 脚本逻辑 =================

echo "=========================================="
echo "开始批量训练任务..."
echo "待训练模型: ${MODELS[*]}"
echo "=========================================="

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    MODEL_PATH="weights/${MODEL_NAME}.pt"
    
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在启动第 $((i+1)) 个任务: ${MODEL_NAME}"
    echo "------------------------------------------------------------------"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "警告: 权重文件 $MODEL_PATH 不存在，跳过该模型！"
        continue
    fi

    # 直接运行，输出直接打印在终端，你可以随时通过 tmux attach 回来查看漂亮的进度条
    python train.py \
        --data "$DATA_CFG" \
        --model "$MODEL_PATH" \
        --epochs $EPOCHS \
        --batch $BATCH_SIZE \
        --device $DEVICES \
        --name "${MODEL_NAME}_ball_human" \
        --plots

    if [ $? -eq 0 ]; then
        echo ">>> [成功] 模型 ${MODEL_NAME} 训练完成！"
    else
        echo ">>> [错误] 模型 ${MODEL_NAME} 训练出错！"
        # 如果希望某个模型出错就停止后续训练，可以取消下面这行的注释
        # exit 1 
    fi

    echo "等待 10 秒显存冷却释放..."
    sleep 10
done

echo "=========================================="
echo "所有批量训练任务已全部结束！"
echo "=========================================="