import argparse
import optuna
from ultralytics import YOLO
import torch
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train') # 建议调优时设小一点
    parser.add_argument('--trials', type=int, default=10, help='number of optuna trials')
    parser.add_argument('--device', default='0', help='cuda device')
    args = parser.parse_args()
    return args

def run_tune(args):
    # 1. 定义 Optuna 的目标函数
    def objective(trial):
        # --- 显存保护机制 ---
        # 在每次试验开始前，先清理一下（以防上一轮有残留）
        gc.collect()
        torch.cuda.empty_cache()

        # -----------------------------------------------------------
        # A. 定义搜索空间
        # -----------------------------------------------------------
        batch_size = trial.suggest_categorical("batch", [8, 16, 32])
        img_size = trial.suggest_categorical("imgsz", [320, 640, 960])
        model_type = trial.suggest_categorical("model", ["weights/yolov8n.pt", "weights/yolov8s.pt", "weights/yolov8m.pt"])
        
        # lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
        # momentum = trial.suggest_float("momentum", 0.8, 0.98)
        
        # -----------------------------------------------------------
        # B. 实例化模型
        # -----------------------------------------------------------
        # 建议使用 weights/yolov8n.pt 这种预训练权重，
        # 如果用 yaml 则是从零训练，收敛很难，对于调参来说太慢了。
        model = YOLO(model_type)
        
        # -----------------------------------------------------------
        # C. 运行训练 (加入异常捕获)
        # -----------------------------------------------------------
        try:
            results = model.train(
                data=args.data,
                epochs=args.epochs,
                batch=batch_size,
                imgsz=img_size,
                # lr0=lr0,
                # momentum=momentum,
                device=args.device,
                
                # --- 关键优化参数 ---
                workers=2,        # 降低 worker 数量，防止僵尸进程堆积 (默认是8)
                exist_ok=True,    # 覆盖旧的实验目录，防止磁盘塞满
                
                # 关闭非必要功能
                plots=False,
                save=False,       # 极其重要：Tuning 过程不要保存权重文件（特别是 .pt），否则硬盘会满
                val=True,
                verbose=False
            )
            
            # 获取指标
            metric = results.box.map50
            
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ [Trial {trial.number}] 显存不足 (OOM)！跳过此组合: {model_type}, Batch={batch_size}, Img={img_size}")
            # 遇到显存不够，清理后返回一个极低的分数，让 Optuna 知道这个路不通
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return 0.0
            
        except Exception as e:
            print(f"⚠️ [Trial {trial.number}] 发生未知错误: {e}")
            return 0.0

        # -----------------------------------------------------------
        # D. 试验结束后的清理 (最关键的一步)
        # -----------------------------------------------------------
        # 1. 删除模型对象
        del model
        # 2. 强制 Python 垃圾回收
        gc.collect()
        # 3. 强制 PyTorch 释放显存
        torch.cuda.empty_cache()

        return metric

    # 2. 创建 Study
    print(f"开始 Optuna 搜索，共运行 {args.trials} 次试验...")
    
    # 这里的 storage 是可选的，但建议加上，万一程序崩了还能断点续传
    # study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3", study_name="yolo_tune")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=args.trials)

    # 3. 输出结果
    print("\n" + "="*50)
    print("搜索结束！最佳超参数如下：")
    print(study.best_params)
    print("="*50 + "\n")


if __name__ == '__main__':
    args = parse_args()
    run_tune(args) 