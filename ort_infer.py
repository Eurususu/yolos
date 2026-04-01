import os
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
import matplotlib.pyplot as plt
import logging
import warnings

# 关闭onnxruntime的logging
logging.getLogger('onnxruntime').setLevel(logging.ERROR)

# 过滤Python警告
warnings.filterwarnings('ignore')


class YOLO_ONNX_Runner:
    def __init__(self, model_path, confidence_thres=0.4, iou_thres=0.7, num_classes=80, device_id=0):
        self.conf_thres = confidence_thres
        self.iou_thres = iou_thres
        self.num_classes = num_classes
        self.device_id = device_id

        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        # 优先使用 CUDA, 其次 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{'device_id': self.device_id}, {}]

        # Session配置优化
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.log_severity_level = 3  # 3=ERROR, 2=WARNING, 1=INFO, 0=VERBOSE

        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers, provider_options=provider_options)
            print(f"模型加载成功，使用设备: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)
        
        self.get_input_details()
        self.get_output_details()

        # 缓存输入尺寸（预处理时重复使用）
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

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

    @staticmethod
    def postprocess(predictions, ratio, dwdh=None, ultralytics=False, conf_thres=0.25, iou_thres=0.7):
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
        dets = YOLO_ONNX_Runner.multiclass_nms(boxes_xyxy, scores, nms_thr=iou_thres, score_thr=conf_thres)
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

        # 统一解包 list
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
                final_boxes[:, [0,2]] -= dw
                final_boxes[:, [1,3]] -= dh
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
                    valid_predictions[:, [0,2]] -= dw
                    valid_predictions[:, [1,3]] -= dh
                valid_predictions[:, :4] /= scale
                dets = valid_predictions

        else:
            if args.ultralytics:
                predictions = data[0] if data.ndim == 3 else data
                predictions = predictions.transpose()
            else:
                predictions = data.reshape(1, -1, 5 + self.num_classes)[0]
            dets = self.postprocess(predictions, scale, pad,
                                    ultralytics=args.ultralytics,
                                    conf_thres=self.conf_thres,
                                    iou_thres=self.iou_thres)

        if dets is not None and len(dets) > 0:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            img = YOLO_ONNX_Runner.vis(img, final_boxes, final_scores, final_cls_inds,
                             conf=self.conf_thres, class_names=self.class_names)
        return img, inference_time


    
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

    @staticmethod
    def rainbow_fill(size=50):  # simpler way to generate rainbow color
        cmap = plt.get_cmap('jet')
        color_list = []

        for n in range(size):
            color = cmap(n/size)
            color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

        return np.array(color_list)

    @staticmethod
    def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        _COLORS = YOLO_ONNX_Runner.rainbow_fill(80).astype(np.float32).reshape(-1, 3)
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img
        
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='weights/yolo11s.onnx', help="Path to ONNX model")
    parser.add_argument("--source", type=str, default='data/1.jpg', help="Path to input image, video file, or RTSP stream")
    parser.add_argument("--end2end", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--end2end_model", action="store_true", help="Whether to use end2end model")
    parser.add_argument("--ultralytics", action="store_true", help="Whether to use Ultralytics model include yolov5u,yolov8,yolov10,yolo11,yolov12,yolo26")
    parser.add_argument("--no_show", action="store_true", help="Don't display window (useful for server/headless)")
    parser.add_argument("--save", action="store_true", help="Save output to file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for end2end model")
    args = parser.parse_args()
    if args.end2end and args.end2end_model:
        raise NotImplementedError("end2end model is already End2End.")
    runner = YOLO_ONNX_Runner(args.model, confidence_thres=args.conf)
    runner.run(args)