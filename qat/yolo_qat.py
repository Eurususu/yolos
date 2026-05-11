import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import build_dataloader

import modelopt.torch.quantization as mtq
from modelopt.torch.utils import print_rank_0

import argparse

# 回调函数 1：在准备工作做完，但第 1 个 Epoch 还没开始时触发
def evaluate_ptq_baseline(trainer):
    print_rank_0("\n" + "="*60)
    print_rank_0("📊 [阶段 1] 正在评估 PTQ (校准后) 基线精度...")
    print_rank_0("="*60)
    
    if not hasattr(trainer, 'loss_items'):
        trainer.loss_items = torch.zeros(3, device=trainer.device)
    if getattr(trainer, 'loss', None) is None:
        # 【新增的最后一块拼图】：给 fitness 计算塞一个假的 loss 张量
        trainer.loss = torch.tensor(0.0, device=trainer.device)
        
    if not hasattr(trainer, 'epoch'):
        trainer.epoch = 0  # 强行塞一个当前 epoch 编号
    # 强制让 Trainer 跑一遍验证集
    trainer.validate()
    
    # 获取指标
    map50_95 = trainer.validator.metrics.box.map
    map50 = trainer.validator.metrics.box.map50
    
    print_rank_0("\n" + "*"*60)
    print_rank_0(f"🎯 PTQ 基线 mAP@0.5:0.95 = {map50_95:.4f}")
    print_rank_0(f"🎯 PTQ 基线 mAP@0.5      = {map50:.4f}")
    print_rank_0("*"*60 + "\n")
    
    # 💡 隐藏福利：由于在这里跑了验证，Ultralytics 会把这个 PTQ 精度
    # 作为初始的 best_fitness。这意味着后面的 QAT 只有跑出比 PTQ 更高的 mAP，
    # 才会去覆盖并保存新的 best.pt！极其安全。

# 回调函数 2：在所有的 Epoch 都跑完后触发
def evaluate_qat_final(trainer):
    print_rank_0("\n" + "="*60)
    print_rank_0("🎉 [阶段 2] QAT 训练已全部结束！")
    print_rank_0("="*60)

    if not hasattr(trainer, 'loss_items'):
        trainer.loss_items = torch.zeros(3, device=trainer.device)
    if getattr(trainer, 'loss', None) is None:
        # 【新增的最后一块拼图】：给 fitness 计算塞一个假的 loss 张量
        trainer.loss = torch.tensor(0.0, device=trainer.device)
    if not hasattr(trainer, 'epoch'):
        trainer.epoch = 0  # 强行塞一个当前 epoch 编号
    
    # 训练结束时，最新的指标存在 validator 中
    map50_95 = trainer.validator.metrics.box.map
    map50 = trainer.validator.metrics.box.map50
    
    print_rank_0("\n" + "🏆"*28)
    print_rank_0(f"🚀 QAT 最终 mAP@0.5:0.95 = {map50_95:.4f}")
    print_rank_0(f"🚀 QAT 最终 mAP@0.5      = {map50:.4f}")
    print_rank_0("🏆"*28 + "\n")

class YoloQATTrainer(DetectionTrainer):
    """
    继承 Ultralytics 的原生 Trainer，在模型初始化后注入 ModelOpt 的 QDQ 节点
    """
    
    def setup_model(self):
        """
        重写 setup_model 方法。这是 Ultralytics 初始化模型的关键生命周期。
        """
        # 1. 调用父类方法，正常加载原版 YOLO 模型 (FP32/FP16)
        super().setup_model()
        
        print_rank_0("\n" + "="*50)
        print_rank_0("🚀 准备注入 ModelOpt QDQ 节点...")
        print_rank_0("="*50 + "\n")

        # 2. 准备校准数据集 (Calibration Data)
        # 我们借用 Ultralytics 原生的 build_dataloader 来构建一个小批量的校准集
        # calib_loader = build_dataloader(
        #     dataset=self.data['val'], # 通常用验证集做校准
        #     batch=16,
        #     workers=4,
        #     rank=-1,
        #     mode='val'
        # )[0] # 返回的是 loader

        calib_batch_size = getattr(self.args, 'batch', 16)
        calib_loader = self.get_dataloader(
            dataset_path=self.data['val'], # 通常用验证集做校准
            batch_size=calib_batch_size,
            rank=-1,
            mode='val'
        )
        # 物理移除所有的 BatchNorm，将其参数吸收到 Conv 中
        if hasattr(self.model, 'fuse'):
            self.model = self.model.fuse()

        # 因为我们后续还要做 QAT 微调，必须把它强行掰回 train() 模式，并重新唤醒梯度！
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        # 3. 配置 ModelOpt 量化参数
        quant_cfg = mtq.INT8_DEFAULT_CFG
        
        # 🚨 极其重要：必须跳过 Detect 头！否则 mAP 直接雪崩。
        # Ultralytics 的检测头类名是 Detect，通过通配符过滤掉它
        # 🚨 极其重要：必须跳过 Detect 头！否则 mAP 直接雪崩。
        skip_modules_list = self.qat_config.get('skip_modules', ['*Detect*', '*head*'])
        print_rank_0(f"⚠️ 以下层将被跳过量化 (保留 FP16/FP32 精度): {skip_modules_list}")
        
        # 【修改点：新版 ModelOpt 禁用量化的标准写法】
        # 遍历你的跳过列表，逐个在 quant_cfg 的字典里把 enable 关掉
        for skip_layer in skip_modules_list:
            # 注意：MTQ 的默认配置字典里有一个同名的 'quant_cfg' 键，用于存放各层的规则
            quant_cfg["quant_cfg"][skip_layer] = {"enable": False}

        if self.qat_config.get('dq_only', False):
            print_rank_0("🛡️ 检测到 --dq_only 参数: 正在关闭全局权重量化 (Weights)，仅量化激活值 (Activations)！")
            # 利用 ModelOpt 的底层通配符机制，精准击杀所有包含 'weight_quantizer' 的节点
            quant_cfg["quant_cfg"]["*weight_quantizer"] = {"enable": False}

        # 4. 定义校准函数
        def calibrate(model: torch.nn.Module):
            model.to(self.device)
            model.eval()
            seen_batches = 0
            max_batches = max(1, 512 // calib_batch_size)
            
            print_rank_0(f"Starting PTQ Calibration with {max_batches} batches...")
            with torch.no_grad():
                for batch in calib_loader:
                    # Ultralytics 的 dataloader 产出的是一个字典，我们只需要提取图片
                    imgs = batch["img"].to(self.device, non_blocking=True)
                    
                    # 归一化 (YOLO 模型输入是 0-1)
                    imgs = imgs.float() / 255.0
                    
                    # 前向传播收集统计信息
                    model(imgs)
                    
                    seen_batches += 1
                    if seen_batches >= max_batches:
                        break
            print_rank_0("Calibration finished!")

        # 5. 执行量化魔法：将普通模型转换为带 QDQ 的模型
        # 注意：self.model 是 Ultralytics trainer 中挂载模型的变量
        self.model = mtq.quantize(self.model, quant_cfg, calibrate)
        self.ema = None
        self.model.to(self.device)
        
        print_rank_0("\n✅ QDQ 节点注入完成！把控制权交还给 Ultralytics 引擎进行 QAT 训练...\n")


    def save_model(self):
        """
        重写 Ultralytics 的保存逻辑，解决 ModelOpt 动态类导致的 PicklingError
        """
        import torch
        import modelopt.torch.opt as mto

        # 1. 获取当前无包装的模型 (原生 PyTorch 写法，完美兼容所有版本)
        model = self.model.module if hasattr(self.model, "module") else self.model

        # 2. 构造一个简化版的 ckpt 字典 (抛弃无法序列化的 model 对象，改存 state_dict)
        ckpt = {
            "epoch": getattr(self, "epoch", 0),
            "best_fitness": getattr(self, "best_fitness", 0.0),
            "model_state_dict": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # 3. 保存原版后缀的 pt 文件 (骗过 Ultralytics 引擎，防止它报错断开)
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)

        # 4. 【核心】使用 ModelOpt 官方的保存 API，额外保存一份专供导出的权重
        mto_last = str(self.last).replace(".pt", "_mto.pt")
        mto.save(model, mto_last)
        
        if self.best_fitness == self.fitness:
            mto_best = str(self.best).replace(".pt", "_mto.pt")
            mto.save(model, mto_best)
            # 注意：这里的 print_rank_0 是我们前面导出的工具函数
            from modelopt.torch.utils import print_rank_0
            print_rank_0(f"\n💎 成功保存 QAT 最佳权重: {mto_best}")
    def final_eval(self):
        """
        重写 Ultralytics 的 final_eval。
        原版会尝试从硬盘重新加载 best.pt，但这会导致 KeyError（因为我们剔除了 model 结构）。
        我们直接拦截并跳过这个步骤，把总结工作交给我们自己写的 evaluate_qat_final 回调即可。
        """
        from modelopt.torch.utils import print_rank_0
        print_rank_0("\n" + "="*60)
        print_rank_0("✅ 拦截原生 final_eval 成功 (已跳过重新加载 best.pt)")
        print_rank_0("💡 QAT 训练已彻底完成！你的最终部署权重在 best_mto.pt 中！")
        print_rank_0("="*60 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8/11 Quantization Aware Training (QAT)")
    
    # ==========================================
    # 1. 基础配置 (路径与环境)
    # ==========================================
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset path')
    parser.add_argument('--model', type=str, default='weights/yolo11s.pt', help='initial pre-trained weights path')
    parser.add_argument('--project', default='runs/qat', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='number of dataloader workers')
    
    # ==========================================
    # 2. QAT 核心训练超参数 (极小学习率，无正则)
    # ==========================================
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (QAT needs 10-20)')
    parser.add_argument('--batch', type=int, default=16, help='batch size for training')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    
    parser.add_argument('--optimizer', type=str, default='SGD', help='QAT requires pure SGD')
    parser.add_argument('--lr0', type=float, default=0.0001, help='initial learning rate (very small for QAT)')
    parser.add_argument('--lrf', type=float, default=0.1, help='final learning rate factor (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay set to 0 to preserve pretrained weights')
    
    # 彻底抛弃 Warmup
    parser.add_argument('--warmup_epochs', type=float, default=0.0, help='no warmup for QAT')
    
    # ==========================================
    # 3. 损失权重与层控制
    # ==========================================
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='class loss gain')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--freeze', type=int, nargs='+', help='e.g., --freeze 9 to freeze backbone during QAT')
    
    # ==========================================
    # 4. QAT 防坑专属配置 (禁用强增强与可能冲突的特性)
    # ==========================================
    parser.add_argument('--amp', action='store_true', default=False, help='disable AMP to avoid float16 overflow in FakeQuantize')
    # parser.add_argument('--ema', action='store_true', default=False, help='disable EMA to avoid weight desync with QDQ nodes')
    parser.add_argument('--cos_lr', action='store_true', default=True, help='use cosine annealing')
    
    parser.add_argument('--mosaic', type=float, default=0.0, help='disable mosaic to keep real INT8 statistics')
    parser.add_argument('--mixup', type=float, default=0.0, help='disable mixup')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='disable copy_paste')
    parser.add_argument('--degrees', type=float, default=0.0, help='disable severe geometric augmentation')
    parser.add_argument('--shear', type=float, default=0.0, help='disable shear')

    # 动态控制跳过量化的层 (支持通配符)
    parser.add_argument(
        '--skip_modules', 
        type=str, 
        nargs='+', 
        default=['*Detect*', '*head*'], 
        help='指定不进行量化的层名或通配符，例如: --skip_modules *Detect* model.22 *head*'
    )
    parser.add_argument('--dq_only', action='store_true', help='whether to perform DQ-only quantization (skip quantization of weights, only insert QDQ nodes for activations)')
    
    # ==========================================
    # 6. 其他功能选项
    # ==========================================
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--save', action='store_false', help='save init and last weights')
    parser.add_argument('--save_period', type=int, default=-1, help='save weights every x epochs')
    parser.add_argument('--cache', type=bool, default=False, help='cache images for faster training')
    parser.add_argument('--profile', action='store_true', help='profile model speed while training')
    parser.add_argument('--plots', action='store_true', help='save plots of training metrics')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ==========================================
    # 启动 QAT 训练
    # ==========================================
    opt = parse_args()
    
    # 这里的参数和你在终端里敲 `yolo train data=coco128.yaml ...` 是一模一样的
    overrides_dict = vars(opt)

    custom_qat_args = {
        'skip_modules': overrides_dict.pop('skip_modules', ['*Detect*', '*head*']),
        'dq_only': overrides_dict.pop('dq_only', False),
    }

    imgsz_val = overrides_dict.get('imgsz')
    if isinstance(imgsz_val, list):
        # 无论是 [640] 还是 [640, 640]，统统取最大值转成单 int
        overrides_dict['imgsz'] = max(imgsz_val)

    # 实例化我们魔改过的 Trainer
    trainer = YoloQATTrainer(overrides=overrides_dict)

    trainer.qat_config = custom_qat_args

    trainer.add_callback("on_pretrain_routine_end", evaluate_ptq_baseline)
    trainer.add_callback("on_train_end", evaluate_qat_final)
    
    # 一键启动！它会先经历你的 setup_model 注入 QDQ，然后无缝衔接原版训练大循环
    trainer.train()
