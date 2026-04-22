import os
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
import os
import json
from tqdm import tqdm
import logging
import warnings

# 关闭onnxruntime的logging
logging.getLogger('onnxruntime').setLevel(logging.ERROR)

# 过滤Python警告
warnings.filterwarnings('ignore')


class YOLO_ONNX_Runner:
    def __init__(self, model_path, confidence_thres=0.001, iou_thres=0.7, num_classes=80, device_id=0):
        # 注意：做精度验证时，建议将 conf_thres 设得很低（如 0.001），iou_thres 设为 0.65，这样计算的 mAP 才准确
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thres
        self.num_classes = num_classes
        self.device_id = device_id

        # 优先使用 CUDA, 其次 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{'device_id': self.device_id}, {}]

        # Session配置优化
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.log_severity_level = 3

        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers, provider_options=provider_options)
            print(f"模型加载成功，使用设备: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)

        self.get_input_details()
        self.get_output_details()

        # 缓存输入尺寸
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # YOLO 0~79 索引到 COCO 真实 Category ID 的映射字典
        self.coco_id_mapping = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

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
        img_h, img_w = image_src.shape[:2]
        scale = min(self.input_height / img_h, self.input_width / img_w)
        new_unpad = int(round(img_w * scale)), int(round(img_h * scale))
        dw, dh = self.input_width - new_unpad[0], self.input_height - new_unpad[1]
        dw /= 2
        dh /= 2
        shape = image_src.shape[:2]

        if shape[::-1] != new_unpad:
            image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(image_src, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.transpose(2, 0, 1)
        im = im[np.newaxis, :]
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.

        return im, scale, (dw, dh)

    def postprocess(self, predictions, ratio, dwdh=None, ultralytics=False):
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
            boxes_xyxy[:, 0] -= dw
            boxes_xyxy[:, 1] -= dh
            boxes_xyxy[:, 2] -= dw
            boxes_xyxy[:, 3] -= dh
        boxes_xyxy /= ratio
        dets = YOLO_ONNX_Runner.multiclass_nms(boxes_xyxy, scores, nms_thr=self.iou_thres, score_thr=self.conf_thres)
        return dets


    # @staticmethod
    # def multiclass_nms(boxes, scores, nms_thr, score_thr):
    #     """Multiclass NMS implemented in Numpy"""
    #     final_dets = []
    #     num_classes = scores.shape[1]
    #     for cls_ind in range(num_classes):
    #         cls_scores = scores[:, cls_ind]
    #         valid_score_mask = cls_scores > score_thr
    #         if valid_score_mask.sum() == 0:
    #             continue
    #         else:
    #             valid_scores = cls_scores[valid_score_mask]
    #             valid_boxes = boxes[valid_score_mask]
    #             keep = YOLO_ONNX_Runner.nms(valid_boxes, valid_scores, nms_thr)
    #             if len(keep) > 0:
    #                 cls_inds = np.ones((len(keep), 1)) * cls_ind
    #                 dets = np.concatenate(
    #                     [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
    #                 )
    #                 final_dets.append(dets)
    #     if len(final_dets) == 0:
    #         return None
    #     return np.concatenate(final_dets, 0)

    @staticmethod
    def multiclass_nms(boxes, scores, nms_thr, score_thr):
        """
        使用纯 Numpy 实现的高效 Offset NMS (偏移量NMS)
        将多类别 NMS 转化为单次 NMS 计算，消除 for 循环，速度提升 10 倍以上。
        """
        # 找到所有大于阈值的 (box索引, 类别索引)
        i, j = np.where(scores > score_thr)
        if len(i) == 0:
            return None
            
        valid_boxes = boxes[i]
        valid_scores = scores[i, j]
        valid_cls = j
        
        # 为了让不同类别的框不互相抑制，给框加上一个由类别ID决定的巨大偏移量
        max_coord = valid_boxes.max()
        offsets = valid_cls.astype(valid_boxes.dtype) * (max_coord + 1000)
        boxes_for_nms = valid_boxes + offsets[:, None]
        
        # 执行单类别 NMS
        keep = YOLO_ONNX_Runner.nms(boxes_for_nms, valid_scores, nms_thr)
        if len(keep) == 0:
            return None
            
        # 拼接结果 [x1, y1, x2, y2, score, class_id]
        dets = np.concatenate([
            valid_boxes[keep], 
            valid_scores[keep, None], 
            valid_cls[keep, None]
        ], axis=1)
        return dets


    @staticmethod
    def nms(boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep


    def infer_single_frame(self, img, args):
        img_data, scale, pad = self.preprocess(img)

        # IO binding 推理
        io_binding = self.session.io_binding()
        ort_input = ort.OrtValue.ortvalue_from_numpy(img_data, "cuda", self.device_id)
        io_binding.bind_ortvalue_input(self.input_name, ort_input)
        io_binding.bind_output(self.output_name, "cuda", device_id=self.device_id)

        start_time = time.time()
        self.session.run_with_iobinding(io_binding)
        data = [out.numpy() for out in io_binding.get_outputs()]
        inference_time = (time.time() - start_time) * 1000

        # 统一解包
        data = data[0]

        if args.end2end:
            mask = data[:, 5] > self.conf_thres
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
                dets = None
            else:
                dw, dh = pad
                final_boxes = valid_predictions[:, 1:5]
                final_boxes[:, [0, 2]] -= dw
                final_boxes[:, [1, 3]] -= dh
                final_boxes /= scale
                final_scores = valid_predictions[:, 5:6]
                final_cls_inds = valid_predictions[:, 6:7].astype(int)
                dets = np.concatenate([final_boxes, final_scores, final_cls_inds], axis=-1)

        elif args.end2end_model:
            data = data[0] if data.ndim == 3 else data
            mask = data[:, 4] > self.conf_thres
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
                dets = None
            else:
                if pad is not None:
                    dw, dh = pad
                    valid_predictions[:, [0, 2]] -= dw
                    valid_predictions[:, [1, 3]] -= dh
                valid_predictions[:, :4] /= scale
                dets = valid_predictions

        else:
            if args.ultralytics:
                predictions = data[0] if data.ndim == 3 else data
                predictions = predictions.transpose()
            else:
                predictions = data.reshape(1, -1, 5 + self.num_classes)[0]
            dets = self.postprocess(predictions, scale, pad,
                                    ultralytics=args.ultralytics)

        if dets is not None and len(dets) > 0:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            return final_boxes, final_scores, final_cls_inds
        else:
            return np.array([]), np.array([]), np.array([])


    def benchmark(self, img, batch_size=8, num_warmup=10, num_runs=50):
        """
        测试 GPU 在多 Batch 下的吞吐量 (Throughput)
        """
        # 检查模型是否支持动态batch
        model_batch = self.input_shape[0]
        if isinstance(model_batch, str) or model_batch == 'batch':
            actual_batch = batch_size
        elif model_batch == 1:
            actual_batch = 1
            print(f"模型仅支持 batch=1，将使用单帧推理测试")
        else:
            actual_batch = min(batch_size, model_batch)

        print(f"\n--- 开始 Batch={actual_batch} 性能压测 ---")

        # 1. 预处理单张图片
        single_img_data, scale, pad = self.preprocess(img)

        # 2. 复制拼装成 Batch 大小的 Tensor
        batch_img_data = np.repeat(single_img_data, actual_batch, axis=0)

        # 3. 准备 IOBinding
        io_binding = self.session.io_binding()
        ort_input = ort.OrtValue.ortvalue_from_numpy(batch_img_data, "cuda", self.device_id)
        io_binding.bind_ortvalue_input(self.input_name, ort_input)
        io_binding.bind_output(self.output_name, "cuda", device_id=self.device_id)

        # 预热
        print("正在进行 GPU 预热...")
        for _ in range(num_warmup):
            self.session.run_with_iobinding(io_binding)

        # 计时
        print(f"开始正式计时 (共跑 {num_runs} 次)...")
        start_time = time.perf_counter()

        for _ in range(num_runs):
            self.session.run_with_iobinding(io_binding)
            _ = io_binding.get_outputs()

        end_time = time.perf_counter()

        # 计算指标
        total_time_ms = (end_time - start_time) * 1000
        avg_batch_time_ms = total_time_ms / num_runs
        fps = (1000.0 / avg_batch_time_ms) * actual_batch

        print(f"Batch Size: {actual_batch}")
        print(f"跑完单个 Batch 平均耗时: {avg_batch_time_ms:.2f} ms")
        print(f"折合单张图片推理耗时: {avg_batch_time_ms / actual_batch:.2f} ms")
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

            det_boxes, det_scores, det_classes = self.infer_single_frame(img, args)

            if len(det_boxes) == 0:
                continue

            for box, score, cls_id in zip(det_boxes, det_scores, det_classes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if args.use_coco_map:
                    coco_cat_id = self.coco_id_mapping[int(cls_id)]
                else:
                    coco_cat_id = int(cls_id)

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
    parser.add_argument("--model", type=str, default='weights/yolo11s.onnx', help="Path to ONNX model")
    parser.add_argument("--source", type=str, default='data/1.jpg', help="Path to input image")
    parser.add_argument("--end2end", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--end2end_model", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--ultralytics", action="store_true", help="Whether to use Ultralytics model")

    # 验证集专用参数
    parser.add_argument("--val", action="store_true", help="Run in validation mode to compute mAP")
    parser.add_argument("--val_dir", type=str, default='/home/jia/dataset/coco2017/images/val2017', help="Path to COCO val images directory")
    parser.add_argument("--val_json", type=str, default='/home/jia/dataset/coco2017/annotations/instances_val2017.json', help="Path to COCO val annotations json")
    parser.add_argument("--use_coco_map", action="store_true", help="Map class 0-79 to COCO 1-90")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of classes in the model")

    # FPS测试
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark to measure FPS")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for benchmarking")

    args = parser.parse_args()

    if args.end2end and args.end2end_model:
        raise NotImplementedError("end2end model is already End2End.")

    # 注意：验证模式下，强制降低置信度阈值
    conf_thres = 0.001 if args.val else 0.4
    runner = YOLO_ONNX_Runner(args.model, confidence_thres=conf_thres, num_classes=args.num_classes)

    if args.val:
        runner.validate_coco(args.val_dir, args.val_json, args)

    if args.benchmark:
        dummy_img = cv2.imread(args.source)
        if dummy_img is None:
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        runner.benchmark(dummy_img, batch_size=args.batch_size, num_warmup=50, num_runs=200)