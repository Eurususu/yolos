import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
import os
import json
from tqdm import tqdm

class YOLO_ONNX_Runner:
    def __init__(self, model_path, confidence_thres=0.001, iou_thres=0.65, num_classes=80, device_id=0):
        # 注意：做精度验证时，建议将 conf_thres 设得很低（如 0.001），iou_thres 设为 0.65，这样计算的 mAP 才准确
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thres
        self.num_classes = num_classes
        self.device_id = device_id

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{'device_id': self.device_id}, {}]
        try:
            self.session = ort.InferenceSession(model_path, providers=providers, provider_options=provider_options)
            print(f"模型加载成功，使用设备: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)
        
        self.get_input_details()
        self.get_output_details()

        # YOLO 0~79 索引到 COCO 真实 Category ID 的映射字典
        self.coco_id_mapping = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

    # ... [保留原有的 get_input_details, get_output_details, preprocess, postprocess] ...
    # 为了节省篇幅，这里省略这几个未变动的函数，请将它们保留在类中
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
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        scale = min(self.input_height / self.img_h, self.input_width / self.img_w)
        new_unpad = int(round(self.img_w * scale)), int(round(self.img_h * scale))
        dw, dh = self.input_width - new_unpad[0], self.input_height - new_unpad[1]
        dw /= 2
        dh /= 2
        shape = image_src.shape[:2]

        if shape[::-1] != new_unpad:  # resize
            image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(image_src,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value=(114,114,114))  # add border
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.transpose(2,0,1)
        im = im[np.newaxis,:]
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.

        return im, scale, (dw, dh)

    # def postprocess(self, output, scale, pad, ultralytics):
    #     if ultralytics:
    #         output = np.transpose(output, (0, 2, 1))
    #     else:
    #         output = np.reshape(output, (1, -1, 5 + self.num_classes))
    #     prediction = output[0]
    #     boxes = prediction[:, 0:4]
    #     if ultralytics:
    #         scores = prediction[:, 4:]
    #     else:
    #         scores = prediction[:, 4:5] * prediction[:, 5:]
        
    #     class_ids = np.argmax(scores, axis=1)
    #     max_scores = np.max(scores, axis=1)
        
    #     mask = max_scores >= self.conf_thres
    #     boxes = boxes[mask]
    #     class_ids = class_ids[mask]
    #     max_scores = max_scores[mask]
        
    #     if len(boxes) == 0:
    #         return [], [], []

    #     nms_boxes = np.copy(boxes)
    #     nms_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  
    #     nms_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  
    #     nms_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  
    #     nms_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    #     # 1. 采用 YOLO 官方的按类别偏移策略 (Class-Aware NMS)
    #     max_wh = 7680 # 偏移常量
    #     c = class_ids * max_wh

    #     opencv_boxes = []
    #     # for box in nms_boxes:
    #     #     opencv_boxes.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])
    #     for i in range(len(nms_boxes)):
    #         box = nms_boxes[i]
    #         # 2. 去掉 int() 强转，保持浮点精度，并将类别偏移加到坐标上
    #         x1 = float(box[0]) + c[i]
    #         y1 = float(box[1]) + c[i]
    #         w = float(box[2] - box[0])
    #         h = float(box[3] - box[1])
    #         opencv_boxes.append([x1, y1, w, h])
            
    #     indices = cv2.dnn.NMSBoxes(opencv_boxes, max_scores.tolist(), self.conf_thres, self.iou_thres)
        
    #     final_boxes, final_scores, final_classes = [], [], []
    #     if len(indices) > 0:
    #         indices = indices.flatten()
    #         for i in indices:
    #             final_boxes.append(nms_boxes[i].astype(float))
    #             final_scores.append(max_scores[i])
    #             final_classes.append(class_ids[i])
    #     dets = np.concatenate([np.array(final_boxes), np.array(final_scores), np.array(final_classes)], axis=-1)
                
    #     # return np.array(final_boxes), np.array(final_scores), np.array(final_classes)
    #     return dets

    @staticmethod
    def postprocess(predictions, ratio, dwdh=None, ultralytics=False):
        boxes = predictions[:, :4]
        if ultralytics:
            scores = predictions[:, 4:]
        else:
            scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        if dwdh is not None:
            dw, dh = dwdh
            # x坐标减 dw, y坐标减 dh
            boxes_xyxy[:, 0] -= dw
            boxes_xyxy[:, 1] -= dh
            boxes_xyxy[:, 2] -= dw
            boxes_xyxy[:, 3] -= dh
        boxes_xyxy /= ratio
        dets = YOLO_ONNX_Runner.multiclass_nms(boxes_xyxy, scores, nms_thr=0.7, score_thr=0.001)
        return dets
    

    @staticmethod
    def multiclass_nms(boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = YOLO_ONNX_Runner.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)
    

    @staticmethod
    def nms(boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep


    # def rescale_boxes(self, boxes, scale, pad, img_w, img_h):
    #     """新增函数：将网络输出的坐标还原到原图尺寸"""
    #     if len(boxes) == 0:
    #         return boxes
    #     boxes = np.array(boxes, dtype=np.float32)
    #     boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    #     boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    #     boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    #     boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
    #     return boxes

    # def infer_single_frame(self, img, args):
    #     img_data, scale, pad = self.preprocess(img)

    #     io_binding = self.session.io_binding()
    #     ort_input = ort.OrtValue.ortvalue_from_numpy(img_data, "cuda", self.device_id)
    #     io_binding.bind_ortvalue_input(self.input_name, ort_input)
    #     io_binding.bind_output(self.output_name, "cuda", device_id=self.device_id)
    #     self.session.run_with_iobinding(io_binding)
    #     outputs = [out.numpy() for out in io_binding.get_outputs()]

    #     if args.end2end:
    #         if isinstance(outputs, list): outputs = outputs[0]
    #         det_boxes = outputs[:,1:5]
    #         det_scores = outputs[:, 5]
    #         det_classes = outputs[:, 6]
    #     elif args.end2end_model:
    #         if isinstance(outputs, list): outputs = outputs[0]
    #         outputs = outputs[0]
    #         scores = outputs[:, 4]
    #         mask = scores > self.conf_thres
    #         outputs = outputs[mask]
    #         if len(outputs) == 0:
    #             return [], [], []
    #         det_boxes = outputs[:,:4]
    #         det_scores = outputs[:, 4]
    #         det_classes = outputs[:, 5]
    #     else:   
    #         det_boxes, det_scores, det_classes = self.postprocess(outputs[0], scale, pad, args.ultralytics)

    #     # 还原到原图尺寸
    #     det_boxes = self.rescale_boxes(det_boxes, scale, pad, img.shape[1], img.shape[0])
    #     return det_boxes, det_scores, det_classes
    
    def infer_single_frame(self, img, args):
        img_data, scale, pad = self.preprocess(img)

        io_binding = self.session.io_binding()
        ort_input = ort.OrtValue.ortvalue_from_numpy(img_data, "cuda", self.device_id)
        io_binding.bind_ortvalue_input(self.input_name, ort_input)
        io_binding.bind_output(self.output_name, "cuda", device_id=self.device_id)
        self.session.run_with_iobinding(io_binding)
        data = [out.numpy() for out in io_binding.get_outputs()]

        if args.end2end:
            if isinstance(data, list):
                data = data[0]
            mask = data[:, 5] > self.conf_thres
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
            else:
                final_boxes = valid_predictions[:, 1:5]
                final_scores = valid_predictions[:, 5]
                final_cls_inds = valid_predictions[:, 6].astype(int)
                if pad is not None:
                    dw, dh = pad
                final_boxes[:, 0,2] -= dw
                final_boxes[:, 1,3] -= dh
                final_boxes /= scale
                final_scores = np.reshape(final_scores, (-1, 1))
                final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
                dets = np.concatenate([np.array(final_boxes), np.array(final_scores), np.array(final_cls_inds)], axis=-1)
        elif args.end2end_model:
            if isinstance(data, list):
                data = data[0]
            data = data[0] if data.ndim == 3 else data
            mask = data[:, 4] > args.conf
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
            elif pad is not None:
                dw, dh = pad 
                valid_predictions[:, 0,2] -= dw
                valid_predictions[:, 1,3] -= dh
            valid_predictions[:,:4] /= scale
            dets = valid_predictions
        else:
            if args.ultralytics:
                if isinstance(data, list):
                    data = data[0]
                predictions = data
                if predictions.ndim == 3:
                     predictions = predictions[0]
                predictions = predictions.transpose()
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,scale,pad,ultralytics=args.ultralytics)

        
        if dets is not None and len(dets) > 0:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            return final_boxes, final_scores, final_cls_inds
        else:
            return [], [], []



    def benchmark(self, img, batch_size=8, num_warmup=10, num_runs=50):
        """
        测试 GPU 在多 Batch 下的吞吐量 (Throughput)
        """
        print(f"\n--- 开始 Batch={batch_size} 性能压测 ---")
        
        # 1. 预处理单张图片
        single_img_data, scale, pad = self.preprocess(img)
        
        # 2. 复制拼装成 Batch 大小的 Tensor
        # single_img_data 形状是 (1, 3, 640, 640)
        # 沿 axis=0 复制 batch_size 次，变成 (batch_size, 3, 640, 640)
        batch_img_data = np.repeat(single_img_data, batch_size, axis=0)

        # 3. 准备 IOBinding
        io_binding = self.session.io_binding()
        ort_input = ort.OrtValue.ortvalue_from_numpy(batch_img_data, "cuda", self.device_id)
        io_binding.bind_ortvalue_input(self.input_name, ort_input)
        
        # 注意：这里 bind_output 依然有效，MacaEP 会自动根据输出 Shape 在显存分配空间
        io_binding.bind_output(self.output_name, "cuda", device_id=self.device_id)

        # ==================== 预热 ====================
        print("正在进行 GPU 预热...")
        for _ in range(num_warmup):
            self.session.run_with_iobinding(io_binding)

        # ==================== 计时 ====================
        print(f"开始正式计时 (共跑 {num_runs} 次)...")
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            self.session.run_with_iobinding(io_binding)
            _ = io_binding.get_outputs() # 模拟显存同步
            
        end_time = time.perf_counter()

        # ==================== 计算指标 ====================
        total_time_ms = (end_time - start_time) * 1000
        avg_batch_time_ms = total_time_ms / num_runs  # 跑完一个 Batch (N张图) 的平均时间
        
        # 核心指标：吞吐量 (每秒能处理多少张图片)
        # 一秒 (1000ms) / 平均一个batch的时间 * batch里面的图片数量
        fps = (1000.0 / avg_batch_time_ms) * batch_size 

        print(f"Batch Size: {batch_size}")
        print(f"跑完单个 Batch 平均耗时: {avg_batch_time_ms:.2f} ms")
        print(f"折合单张图片推理耗时: {avg_batch_time_ms / batch_size:.2f} ms")
        print(f"【极限吞吐量】: {fps:.2f} FPS (帧/秒)")
        print("-" * 40)

    def validate_coco(self, val_img_dir, val_anno_json, args):
        """核心评估函数：生成 COCO 格式的 JSON 并调用 pycocotools 评估"""
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            print("请先安装 pycocotools: pip install pycocotools")
            return

        print(f"正在加载真实标注文件: {val_anno_json} ...")
        coco_gt = COCO(val_anno_json)
        img_ids = coco_gt.getImgIds()
        
        results = []
        print(f"开始在验证集上推理，共 {len(img_ids)} 张图片...")
        
        for img_id in tqdm(img_ids):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(val_img_dir, img_info['file_name'])
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 推理获得检测结果
            det_boxes, det_scores, det_classes = self.infer_single_frame(img, args)
            
            if len(det_boxes) == 0:
                continue
                
            for box, score, cls_id in zip(det_boxes, det_scores, det_classes):
                x1, y1, x2, y2 = box
                # COCO 格式需要 [x_min, y_min, width, height]
                w, h = x2 - x1, y2 - y1
                
                # YOLO 内部 0~79 的类别 ID 转换为 COCO 标准 ID
                coco_cat_id = self.coco_id_mapping[int(cls_id)]
                
                results.append({
                    "image_id": img_id,
                    "category_id": coco_cat_id,
                    "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(w), 3), round(float(h), 3)],
                    "score": round(float(score), 5)
                })

        # 保存预测结果为 JSON
        res_json_path = "predictions.json"
        with open(res_json_path, 'w') as f:
            json.dump(results, f)
        print(f"\n预测结果已保存至 {res_json_path}，准备开始计算 mAP...")

        # 调用 pycocotools 进行评估
        coco_dt = coco_gt.loadRes(res_json_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='weights/yolo11s_noend_640.onnx', help="Path to ONNX model")
    parser.add_argument("--source", type=str, default='data/1.jpg', help="Path to input image")
    parser.add_argument("--end2end", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--end2end_model", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--ultralytics", action="store_true", help="Whether to use Ultralytics model")
    
    # 新增验证集专用参数
    parser.add_argument("--val", action="store_true", help="Run in validation mode to compute mAP")
    parser.add_argument("--val_dir", type=str, default='/home/jia/dataset/coco2017/images/val2017', help="Path to COCO val images directory")
    parser.add_argument("--val_json", type=str, default='/home/jia/dataset/coco2017/annotations/instances_val2017.json', help="Path to COCO val annotations json")

    # fps测试
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark to measure FPS")
    
    args = parser.parse_args()
    
    if args.end2end and args.end2end_model:
        raise NotImplementedError("end2end model is already End2End.")
    
    # 注意：验证模式下，强制降低置信度阈值，以便召回更多框让 mAP 计算准确
    conf_thres = 0.001 if args.val else 0.4
    runner = YOLO_ONNX_Runner(args.model, confidence_thres=conf_thres)
    
    if args.val:
        runner.validate_coco(args.val_dir, args.val_json, args)
    
    if args.benchmark:
        dummy_img = cv2.imread(args.source)
        if dummy_img is None:
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        runner.benchmark(dummy_img, batch_size=16, num_warmup=50, num_runs=200)