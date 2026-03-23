import os
import json
import cv2
import argparse
from tqdm import tqdm

def yolo_to_coco(image_dir, label_dir, output_json, class_names=None):
    """
    将 YOLO 格式 (txt) 转换为 COCO 格式 (json)
    """
    # 1. 初始化 COCO 基本结构
    coco_data = {
        "info": {
            "description": "Converted YOLO Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "jia",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 2. 生成 Categories (如果有 class_names 文件则读取，否则自动生成)
    if class_names:
        if os.path.exists(class_names):
            with open(class_names, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            # 假设传入的是个列表或者是类数量的数字
            classes = [str(i) for i in range(int(class_names))]
    else:
        # 默认 80 类
        classes = [str(i) for i in range(80)]

    for idx, name in enumerate(classes):
        coco_data["categories"].append({
            "id": idx,  # YOLO 的 class 0 对应 COCO 的 id 0 (自定义数据集通常不需要映射到 91)
            "name": name,
            "supercategory": "none"
        })

    # 3. 遍历图片并匹配标签
    valid_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    annotation_id = 1
    
    print(f"Found {len(valid_images)} images. Converting...")

    for img_name in tqdm(valid_images):
        img_path = os.path.join(image_dir, img_name)
        
        # 读取图片获取宽高 (YOLO 还原坐标需要)
        # 优化：如果图片非常多，cv2.imread 可能会慢，只读 header 会快，但 cv2.imread 最稳
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        # 添加 Image 信息
        image_id = int(os.path.splitext(img_name)[0]) if os.path.splitext(img_name)[0].isdigit() else hash(img_name) % 100000000
        # 如果文件名不是数字，用 hash 做 ID，但要确保唯一。
        # 更好的方式是维护一个自增 ID map，这里为了简单演示：
        # image_id = len(coco_data["images"]) + 1 
        
        # 这里为了后续 val_trt.py 能通过 ID 找到文件名，我们采用自增 ID，并建立映射?
        # 不，pycocotools 通过 image_id 查找。为了最简单，我们重新定义 image_id 为自增
        img_id = len(coco_data["images"]) + 1
        
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_name,
            "height": h,
            "width": w
        })

        # 查找对应的 Label 文件
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    nw = float(parts[3])
                    nh = float(parts[4])

                    # YOLO (Normalized xywh) -> COCO (Absolute xywh - TopLeft)
                    # w_pixel = nw * w
                    # h_pixel = nh * h
                    # x_center = cx * w
                    # y_center = cy * h
                    # x_top_left = x_center - w_pixel / 2
                    # y_top_left = y_center - h_pixel / 2
                    
                    w_abs = nw * w
                    h_abs = nh * h
                    x_abs = (cx * w) - (w_abs / 2)
                    y_abs = (cy * h) - (h_abs / 2)

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": cls_id, # 直接对应，不做 +1 映射
                        "bbox": [x_abs, y_abs, w_abs, h_abs],
                        "area": w_abs * h_abs,
                        "iscrowd": 0
                    })
                    annotation_id += 1

    # 4. 保存 JSON
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"Conversion completed. Saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Path to images folder")
    parser.add_argument("--label_dir", required=True, help="Path to labels folder")
    parser.add_argument("--output", default="val_yolo_converted.json", help="Output json path")
    parser.add_argument("--classes", default=None, help="Path to classes.txt or number of classes (e.g. 80)")
    
    args = parser.parse_args()
    yolo_to_coco(args.img_dir, args.label_dir, args.output, args.classes)