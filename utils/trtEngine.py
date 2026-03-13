import tensorrt as trt
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cuda import cudart
import time
import os

# 减少 TensorRT 日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BaseEngine(object):
    def __init__(self, engine_path, max_batch_size=1, max_det=300, conf_thres=0.25, iou_thres=0.7):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_batch_size = max_batch_size
        self.max_det = max_det

        # 1. 初始化 Logger
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        trt.init_libnvinfer_plugins(logger, '')

        # 2. 加载 engine
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        # 3. 获取输入图像尺寸
        input_tensor_name = self.engine.get_tensor_name(0)
        self.imgsz = self.engine.get_tensor_shape(input_tensor_name)[2:]

        # 4. 初始化cuda stream
        _, self.stream = cudart.cudaStreamCreate()

        # 5. 分配显存
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            raw_shape = self.engine.get_tensor_shape(name)

            is_input = False
            shape = list(raw_shape)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
                if shape[0] == -1:
                    shape[0] = self.max_batch_size
            else:
                if len(shape) == 2 and shape[0] == -1:
                    shape[0] = self.max_det
            shape = tuple(shape)

            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s if s > 0 else 1

            err, ptr = cudart.cudaMalloc(size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA Malloc failed for tensor {name}")

            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'ptr': ptr,
                'size': size
            }
            self.context.set_tensor_address(name, ptr)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            self.allocations.append(ptr)

    def __del__(self):
        if hasattr(self, 'allocations'):
            for ptr in self.allocations:
                cudart.cudaFree(ptr)
        if hasattr(self, 'stream'):
            cudart.cudaStreamDestroy(self.stream)

    def infer(self, img):
        outputs = []
        for out in self.outputs:
            outputs.append(np.zeros(out['shape'], out['dtype']))

        input_binding = self.inputs[0]
        input_name = input_binding['name']
        img = np.ascontiguousarray(img)
        self.context.set_input_shape(input_name, img.shape)

        cudart.cudaMemcpyAsync(
            input_binding['ptr'],
            img.ctypes.data,
            input_binding['size'],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream
        )

        self.context.execute_async_v3(stream_handle=self.stream)

        for i, out in enumerate(self.outputs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data,
                out['ptr'],
                out['size'],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )

        cudart.cudaStreamSynchronize(self.stream)
        return outputs

    def postprocess(self, predictions, ratio, dwdh=None, ultralytics=False):
        boxes = predictions[:, :4]
        if ultralytics:
            scores = predictions[:, 4:]
        else:
            scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        if dwdh is not None:
            dw, dh = dwdh
            boxes_xyxy[:, 0] -= dw
            boxes_xyxy[:, 1] -= dh
            boxes_xyxy[:, 2] -= dw
            boxes_xyxy[:, 3] -= dh

        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=self.iou_thres, score_thr=self.conf_thres)
        return dets

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = self.nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    @staticmethod
    def nms(boxes, scores, nms_thr):
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

    def process_output(self, data, ratio, dwdh, args):
        """统一处理模型输出"""
        # 统一解包 list
        if isinstance(data, list):
            data = data[0]

        dets = None

        if args.end2end:
            mask = data[:, 5] > self.conf_thres
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
            else:
                dw, dh = dwdh
                final_boxes = valid_predictions[:, 1:5]
                final_boxes[:, [0, 2]] -= dw
                final_boxes[:, [1, 3]] -= dh
                final_boxes /= ratio
                final_scores = valid_predictions[:, 5:6]
                final_cls_inds = valid_predictions[:, 6:7].astype(int)
                dets = np.concatenate([final_boxes, final_scores, final_cls_inds], axis=-1)

        elif args.efficient_end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = final_boxes.reshape(-1, 4)
            if dwdh is not None:
                dw, dh = dwdh
                final_boxes[:, [0, 2]] -= dw
                final_boxes[:, [1, 3]] -= dh
            final_boxes /= ratio
            valid_count = int(num[0])
            dets = np.concatenate([
                final_boxes[:valid_count],
                final_scores[:valid_count].reshape(-1, 1),
                final_cls_inds[:valid_count].reshape(-1, 1)
            ], axis=-1)

        elif args.end2end_model:
            data = data[0] if data.ndim == 3 else data
            mask = data[:, 4] > self.conf_thres
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
            else:
                if dwdh is not None:
                    dw, dh = dwdh
                    valid_predictions[:, [0, 2]] -= dw
                    valid_predictions[:, [1, 3]] -= dh
                valid_predictions[:, :4] /= ratio
                dets = valid_predictions

        else:
            if args.ultralytics:
                predictions = data[0] if data.ndim == 3 else data
                predictions = predictions.transpose()
            else:
                predictions = data.reshape(1, -1, 5 + self.n_classes)[0]
            dets = self.postprocess(predictions, ratio, dwdh=dwdh, ultralytics=args.ultralytics)

        return dets

    def detect_video(self, video_path, args):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_vid = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.mp4', fourcc, fps_vid, (width, height))

        curr_fps = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img, ratio, dwdh = letterbox(frame, self.imgsz)
            t1 = time.time()
            data = self.infer(img)
            t2 = time.time()
            curr_fps = (curr_fps + (1. / (t2 - t1))) / 2

            frame = cv2.putText(frame, "FPS:%d " % curr_fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

            dets = self.process_output(data, ratio, dwdh, args)

            if dets is not None and len(dets) > 0:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                           conf=self.conf_thres, class_names=self.class_names)

            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, args):
        origin_img = cv2.imread(img_path)
        img, ratio, dwdh = letterbox(origin_img, self.imgsz)
        data = self.infer(img)

        dets = self.process_output(data, ratio, dwdh, args)

        if dets is not None and len(dets) > 0:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=self.conf_thres, class_names=self.class_names)

        return origin_img

    def get_fps(self):
        img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):
            _ = self.infer(img)
        print(100 / (time.perf_counter() - t0), 'FPS')


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), swap=(2, 0, 1)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    target_h, target_w = new_shape
    r = min(target_w / shape[1], target_h / shape[0])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose(swap)
    im = im[np.newaxis, :]
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.
    return im, r, (dw, dh)


def rainbow_fill(size=50):
    cmap = plt.get_cmap('jet')
    color_list = []
    for n in range(size):
        color = cmap(n / size)
        color_list.append(color[:3])
    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1),
                      (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img