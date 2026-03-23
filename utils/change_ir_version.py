import onnx

# 1. 设置模型路径
input_model_path = "weights/yolo11s.onnx"  # 你的原模型
output_model_path = "weights/yolo11s_ir10.onnx"    # 保存的新模型

# 2. 加载模型
print(f"正在加载模型: {input_model_path}...")
model = onnx.load(input_model_path)

# 3. 检查当前的 IR Version
current_ir_version = model.ir_version
print(f"当前的 IR Version 是: {current_ir_version}")

# 4. 修改 IR Version
# 目标是将 12 降为 10 (对应 ONNX 1.15 左右)
# 如果你需要更低，比如对应 ONNX 1.12，通常设置为 8
target_ir_version = 10 
model.ir_version = target_ir_version
print(f"已将 IR Version 修改为: {model.ir_version}")

# 5. (可选) 检查模型格式是否依然合法
# 注意：这步只检查结构，不检查是否真的兼容旧 IR 的特性
try:
    onnx.checker.check_model(model)
    print("模型结构检查通过。")
except onnx.checker.ValidationError as e:
    print(f"警告：模型结构检查失败: {e}")

# 6. 保存模型
onnx.save(model, output_model_path)
print(f"新模型已保存至: {output_model_path}")