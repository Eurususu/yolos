import argparse
import os
import glob
import random
import numpy as np
import cv2  # 使用 OpenCV 以保持和你的推理代码一致

def preprocess_one_image(image_path, input_h, input_w):
    """
    完全复用你的预处理逻辑
    """
    # 1. 读取图片 (OpenCV 读取默认为 BGR)
    image_src = cv2.imread(image_path)
    if image_src is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # --- 开始你的逻辑 ---
    img_h, img_w = image_src.shape[:2]
    
    # 计算缩放比例
    scale = min(input_h / img_h, input_w / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    
    # Resize
    image_resized = cv2.resize(image_src, (new_w, new_h))

    # 创建画布并填充 (灰色 114)
    image_padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    
    # 计算居中偏移量
    dw = (input_w - new_w) // 2
    dh = (input_h - new_h) // 2
    
    # 将图片贴到画布中心
    image_padded[dh:dh+new_h, dw:dw+new_w, :] = image_resized
    
    # --- 颜色空间注意事项 (关键!) ---
    # OpenCV 默认读入是 BGR。
    # 如果你的模型训练时是 RGB (绝大多数模型都是 RGB)，你需要在这里转换。
    # 如果你的 inference 代码里没有 cv2.cvtColor，请确认模型输入是否真的期望 BGR。
    # 这里我先加上 RGB 转换，通常这是必须的，如果你的模型确实需要 BGR，请注释掉下面这行：
    image_padded = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)

    # 2. 归一化 & 转换
    image_data = image_padded.transpose(2, 0, 1) # HWC -> CHW
    
    # 注意：这里我们不加 expand_dims (axis=0)，因为我们在主循环里会统一堆叠
    image_data = image_data.astype(np.float32) / 255.0 # 0-255 -> 0.0-1.0
    
    return image_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="图片文件夹路径")
    parser.add_argument("--calibration_size", type=int, default=500, help="采样数量")
    parser.add_argument("--height", type=int, default=736, help="模型输入的 Height")
    parser.add_argument("--width", type=int, default=1280, help="模型输入的 Width")
    parser.add_argument("--output_path", type=str, default="calib_coco.npy", help="输出路径")
    args = parser.parse_args()

    # 1. 获取图片列表
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_paths = []
    for ext in extensions:
        all_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))

    if len(all_paths) == 0:
        raise ValueError("未找到任何图片，请检查路径。")
    
    # 2. 随机采样
    sample_size = min(len(all_paths), args.calibration_size)
    selected_paths = random.sample(all_paths, sample_size)
    print(f"从 {len(all_paths)} 张图片中随机选取了 {sample_size} 张。")

    # 3. 循环处理
    data_list = []
    for i, path in enumerate(selected_paths):
        try:
            img_tensor = preprocess_one_image(path, args.height, args.width)
            data_list.append(img_tensor)
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{sample_size} 张...")
        except Exception as e:
            print(f"跳过坏图 {path}: {e}")

    # 4. 堆叠成 Batch (N, C, H, W)
    if not data_list:
        raise RuntimeError("没有生成有效数据")
        
    calib_data = np.stack(data_list, axis=0)
    print(f"最终数据形状: {calib_data.shape}") # 应该是 (500, 3, 640, 640)

    # 5. 保存
    np.save(args.output_path, calib_data)
    print(f"保存成功: {args.output_path}")

if __name__ == "__main__":
    main()