import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
import os


class YOLO_ONNX_Runner:
    def __init__(self, model_path, confidence_thres=0.4, iou_thres=0.7, num_classes=80):
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thres
        self.num_classes = num_classes

        # 优先使用 CUDA, 其次 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"模型加载成功，使用设备: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)
        
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        print(f"模型输入节点: {self.input_name}, 形状: {self.input_shape}")

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name
        self.output_shape = model_outputs[0].shape
        print(f"模型输出节点: {self.output_name}, 形状: {self.output_shape}")

    def preprocess(self, image_src):
        self.img_h, self.img_w = image_src.shape[:2]
        # 1. Letterbox Resize (保持长宽比，填充灰色)
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        scale = min(self.input_height / self.img_h, self.input_width / self.img_w)
        new_h, new_w = int(self.img_h * scale), int(self.img_w * scale)
        
        image_resized = cv2.resize(image_src, (new_w, new_h))

        # 创建画布并填充
        image_padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        # 计算居中偏移量
        dw = (self.input_width - new_w) // 2
        dh = (self.input_height - new_h) // 2
        image_padded[dh:dh+new_h, dw:dw+new_w, :] = image_resized
        
        # 2. 归一化 & 转换
        image_data = image_padded.transpose(2, 0, 1) # HWC -> CHW
        image_data = np.expand_dims(image_data, axis=0) # Add Batch Dim
        image_data = image_data.astype(np.float32) / 255.0 # 0-255 -> 0.0-1.0
        return image_data, scale, (dw, dh)

    def postprocess(self, output, scale, pad, ultralytics):
        """
        后处理：解析 YOLO 输出, NMS, 坐标还原
        YOLOv8 输出形状通常为: [1, 4 + num_classes, num_anchors]
        例如: [1, 84, 8400] -> 4个坐标 + 80个类别
        """
        # if v8:
            # 1. Transpose: [1, 84, 8400] -> [1, 8400, 84]
        if ultralytics:
            output = np.transpose(output, (0, 2, 1))
        else:
            output = np.reshape(output, (1, -1, 5 + self.num_classes))

        
        # 去掉 Batch 维度 -> [8400, 84]
        prediction = output[0]
        
        # 2. 拆分 Box 和 Scores
        # cx, cy, w, h
        boxes = prediction[:, 0:4]
        # if v8:
            # classes scores
        if ultralytics:
            scores = prediction[:, 4:]
        else:
            scores = prediction[:, 4:5] * prediction[:, 5:]
        
        # 获取最大置信度的类别和分数
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        
        # 3. 初步过滤 (Confidence Threshold)
        mask = max_scores >= self.conf_thres
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]
        
        if len(boxes) == 0:
            return [], [], []

        # 4. 坐标转换: cx,cy,w,h -> x1,y1,x2,y2 (用于 NMS)
        # 这里的 boxes 还是基于 640x640 (input_size) 的
        nms_boxes = np.copy(boxes)
        nms_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        nms_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        nms_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        nms_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # 5. NMS (Non-Maximum Suppression)
        # cv2.dnn.NMSBoxes 需要 (x, y, w, h) 格式，或者我们可以用 x1,y1,x2,y2 手写
        # 这里简单起见，转换回 x,y,w,h 供 OpenCV 使用 (x,y 是左上角)
        opencv_boxes = []
        for box in nms_boxes:
            opencv_boxes.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])
            
        indices = cv2.dnn.NMSBoxes(opencv_boxes, max_scores.tolist(), self.conf_thres, self.iou_thres)
        
        final_boxes = []
        final_scores = []
        final_classes = []
        
        dw, dh = pad
        
        # 6. 还原坐标到原图尺寸
        if len(indices) > 0:
            # cv2.dnn.NMSBoxes 返回的是 list of list 或者 flat list，兼容处理
            indices = indices.flatten()
            
            for i in indices:
                box = nms_boxes[i] # x1, y1, x2, y2
                
                # # 移除 Padding
                # box[0] -= dw
                # box[1] -= dh
                # box[2] -= dw
                # box[3] -= dh
                
                # # 缩放回原图
                # box /= scale
                
                # # 边界截断
                # box[0] = max(0, box[0])
                # box[1] = max(0, box[1])
                # box[2] = min(self.img_w, box[2])
                # box[3] = min(self.img_h, box[3])
                
                final_boxes.append(box.astype(int))
                final_scores.append(max_scores[i])
                final_classes.append(class_ids[i])
                
        return np.array(final_boxes), np.array(final_scores), np.array(final_classes)

    def infer_single_frame(self, img, args):
        """
        对单帧图片进行推理的核心逻辑封装
        """
        # 预处理
        img_data, scale, pad = self.preprocess(img)

        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: img_data})
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        # 后处理
        if args.end2end:
            if isinstance(outputs, list): outputs = outputs[0]
            det_boxes = outputs[:,1:5]
            det_scores = outputs[:, 5]
            det_classes = outputs[:, 6]
        elif args.v10:
            if isinstance(outputs, list): outputs = outputs[0]
            outputs = outputs[0]
            scores = outputs[:, 4]
            mask = scores > self.conf_thres
            outputs = outputs[mask]
            if len(outputs) == 0:
                return img, 0
            det_boxes = outputs[:,:4]
            det_scores = outputs[:, 4]
            det_classes = outputs[:, 5]
        else:   
            det_boxes, det_scores, det_classes = self.postprocess(outputs[0], scale, pad, args.ultralytics)

        # 绘制结果
        img_res = self.draw_results(img, det_boxes, det_scores, det_classes, scale, pad)
        return img_res, inference_time

    
    def run(self, args):
        source = args.source
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        is_image = any(source.lower().endswith(ext) for ext in image_extensions)
        if is_image:
            # === 图片模式 ===
            print(f"正在处理图片: {source}")
            img = cv2.imread(source)
            if img is None:
                print(f"无法读取图片: {source}")
                return

            result_img, t = self.infer_single_frame(img, args)
            
            output_path = "result.jpg"
            if args.save:
                cv2.imwrite(output_path, result_img)
            print(f"推理时间: {t:.2f}ms, 结果已保存至: {output_path}")
        else:
            # === 视频/RTSP 模式 ===
            print(f"正在尝试打开视频源: {source}")
            
            # 如果是数字字符串（如 '0'），转换为整数以打开摄像头
            if source.isdigit():
                source = int(source)
                
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"无法打开视频源: {source}")
                return

            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 25 # 防止 RTSP 获取不到 FPS 导致报错

            # 如果不是实时流，准备写入文件
            out_writer = None
            is_file = isinstance(source, str) and os.path.exists(source)
            
            if is_file and args.save:
                save_path = "result_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                print(f"视频处理中，结果将保存至: {save_path}")
            else:
                print("正在处理实时流 (按 'q' 退出)...")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 推理
                result_img, t = self.infer_single_frame(frame, args)
                
                # 显示 FPS
                cv2.putText(result_img, f"FPS: {1000/t:.1f} (Inference: {t:.1f}ms)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 写入视频文件
                if out_writer:
                    out_writer.write(result_img)
                
                # 显示画面 (如果是 RTSP 或 摄像头)
                # 注意：在无头服务器上运行时请注释掉 imshow
                if not args.no_show:
                    cv2.imshow("YOLO ONNX Runtime", result_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧, 当前推理耗时: {t:.2f}ms")

            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            if is_file:
                print("视频处理完成。")

    
    def draw_results(self, img, boxes, scores, classes, scale, pad):
        # COCO 类别 (仅作示例，如果是自定义数据集需修改)
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        # coco_names = ['person','car', 'bicycle']
        h, w = img.shape[:2]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            boxes = np.round(boxes).astype(int)
            classes = classes.astype(int)
            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                
                # 随机颜色
                rng = np.random.RandomState(cls_id)
                color = tuple(rng.randint(0, 255, size=3).tolist())
                color = (int(color[0]), int(color[1]), int(color[2]))
                
                # 画框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 写标签
                label = f"{coco_names[cls_id] if cls_id < len(coco_names) else cls_id}: {score:.2f}"
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                cv2.rectangle(img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        return img
        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='weights/yolo11n.onnx', help="Path to ONNX model")
    parser.add_argument("--source", type=str, default='data/1.jpg', help="Path to input image, video file, or RTSP stream")
    parser.add_argument("--end2end", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--v10", action="store_true", help="Whether to use YOLOv10 model")
    parser.add_argument("--ultralytics", action="store_true", help="Whether to use Ultralytics model include yolov5u,yolov8,yolov10,yolo11,yolov12,yolov13")
    parser.add_argument("--no_show", action="store_true", help="Don't display window (useful for server/headless)")
    parser.add_argument("--save", action="store_true", help="Save output to file")
    args = parser.parse_args()
    if args.end2end and args.v10:
        raise NotImplementedError("YOLOv10 is already End2End.")
    runner = YOLO_ONNX_Runner(args.model)
    runner.run(args)