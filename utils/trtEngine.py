import tensorrt as trt
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cuda import cudart
import time

class BaseEngine(object):
    def __init__(self, engine_path,max_batch_size=1, max_det=300):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        self.max_batch_size = max_batch_size
        self.max_det = max_det
        # 1. 初始化 Logger
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
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
            # 处理动态 Shape: 如果是 -1，先给一个默认值用于计算显存，通常需要根据实际情况调整
            # 这里简单处理：如果有动态 batch，暂时按 1 或者最大值处理
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
            # 计算需要分配的字节数
            # 注意：如果 Shape 包含 -1，这里需要使用 Profile 的最大尺寸，这里简化处理
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s if s > 0 else 1 # 简单的安全措施
            # if is_input:
            #     self.batch_size = shape[0]

            # 分配GPU显存
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
            # 将GPU指针绑定到Context
            self.context.set_tensor_address(name, ptr)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            self.allocations.append(ptr)
    

    def __del__(self):
        # 析构函数：释放显存和 Stream
        if hasattr(self, 'allocations'):
            for ptr in self.allocations:
                cudart.cudaFree(ptr)
        if hasattr(self, 'stream'):
            cudart.cudaStreamDestroy(self.stream)

    def infer(self, img):
        # 1. 准备 Output Host Buffer
        outputs = []
        for out in self.outputs:
            outputs.append(np.zeros(out['shape'], out['dtype']))
        # 2. 设置输入 Shape (对于 V3 API，如果有动态 Shape 必须设置)
        # 假设只有一个输入
        input_binding = self.inputs[0]
        input_name = input_binding['name']
        # 检查输入数据类型并确保持续内存
        img = np.ascontiguousarray(img)
        # 如果 Engine 是动态 Shape，这里必须设置
        self.context.set_input_shape(input_name, img.shape)
        # 3. Host -> Device (异步 Copy)
        # input_binding['ptr'] 是 GPU 地址 (int)
        cudart.cudaMemcpyAsync(
            input_binding['ptr'], 
            img.ctypes.data, 
            input_binding['size'], 
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, 
            self.stream
        )
        # 4. 执行推理 (V3 API)
        self.context.execute_async_v3(stream_handle=self.stream)
        # 5. Device -> Host (异步 Copy)
        for i, out in enumerate(self.outputs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data, 
                out['ptr'], 
                out['size'], 
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, 
                self.stream
            )
        # 6. 同步 Stream (等待推理和拷贝完成)
        cudart.cudaStreamSynchronize(self.stream)
        return outputs

    def detect_video(self, video_path, args):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_vid = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.mp4',fourcc,fps_vid,(width,height))
        curr_fps = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            img, ratio, dwdh = letterbox(frame, self.imgsz)
            t1 = time.time()
            data = self.infer(img)
            t2 = time.time()
            curr_fps = (curr_fps + (1. / (t2 - t1))) / 2
            frame = cv2.putText(frame, "FPS:%d " %curr_fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            dets = None
            if args.end2end:
                if isinstance(data, list):
                    data = data[0]
                mask = data[:, 5] > args.conf
                valid_predictions = data[mask]
                if valid_predictions.shape[0] == 0:
                    print("没有检测到物体")
                else:
                    final_boxes = valid_predictions[:, 1:5]
                    final_scores = valid_predictions[:, 5]
                    final_cls_inds = valid_predictions[:, 6].astype(int)
                    if dwdh is not None:
                        dw, dh = dwdh
                    final_boxes[:, 0] -= dw
                    final_boxes[:, 1] -= dh
                    final_boxes[:, 2] -= dw
                    final_boxes[:, 3] -= dh
                    final_boxes /= ratio
                    final_scores = np.reshape(final_scores, (-1, 1))
                    final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
                    dets = np.concatenate([np.array(final_boxes), np.array(final_scores), np.array(final_cls_inds)], axis=-1)
            elif args.efficient_end2end:
                num, final_boxes, final_scores, final_cls_inds  = data
                # final_boxes, final_scores, final_cls_inds  = data
                final_boxes = np.reshape(final_boxes, (-1, 4))
                # 还原坐标：先减 padding，再除以 ratio
                if dwdh is not None:
                    dw, dh = dwdh
                    final_boxes[:, 0] -= dw
                    final_boxes[:, 1] -= dh
                    final_boxes[:, 2] -= dw
                    final_boxes[:, 3] -= dh
                final_boxes /= ratio
                final_scores = np.reshape(final_scores, (-1, 1))
                final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
                # 截取有效框
                valid_count = int(num[0])
                dets = np.concatenate([np.array(final_boxes)[:valid_count], np.array(final_scores)[:valid_count], np.array(final_cls_inds)[:valid_count]], axis=-1)
            else:
                if args.ultralytics:
                    if isinstance(data, list):
                        data = data[0]
                    predictions = data
                    # Ultralytics output通常是 (Batch, 4+cls, Num_Anchors) -> 需要 transpose
                    if predictions.ndim == 3: 
                        predictions = predictions[0]
                    predictions = predictions.transpose()
                else:
                    predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio,dwdh=dwdh,ultralytics=args.ultralytics)

            if dets is not None and len(dets) > 0:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=args.conf, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, args):
        origin_img = cv2.imread(img_path)
        # img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        img, ratio, dwdh = letterbox(origin_img, self.imgsz)
        data = self.infer(img)
        if args.end2end:
            if isinstance(data, list):
                data = data[0]
            mask = data[:, 5] > args.conf
            valid_predictions = data[mask]
            if valid_predictions.shape[0] == 0:
                print("没有检测到物体")
            else:
                final_boxes = valid_predictions[:, 1:5]
                final_scores = valid_predictions[:, 5]
                final_cls_inds = valid_predictions[:, 6].astype(int)
                if dwdh is not None:
                    dw, dh = dwdh
                final_boxes[:, 0] -= dw
                final_boxes[:, 1] -= dh
                final_boxes[:, 2] -= dw
                final_boxes[:, 3] -= dh
                final_boxes /= ratio
                final_scores = np.reshape(final_scores, (-1, 1))
                final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
                dets = np.concatenate([np.array(final_boxes), np.array(final_scores), np.array(final_cls_inds)], axis=-1)
        elif args.efficient_end2end:
            num, final_boxes, final_scores, final_cls_inds  = data
            # final_boxes, final_scores, final_cls_inds  = data
            final_boxes = np.reshape(final_boxes, (-1, 4))
            # 还原坐标：先减 padding，再除以 ratio
            if dwdh is not None:
                dw, dh = dwdh
                final_boxes[:, 0] -= dw
                final_boxes[:, 1] -= dh
                final_boxes[:, 2] -= dw
                final_boxes[:, 3] -= dh
            final_boxes /= ratio
            final_scores = np.reshape(final_scores, (-1, 1))
            final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
            # 截取有效框
            valid_count = int(num[0])
            dets = np.concatenate([np.array(final_boxes)[:valid_count], np.array(final_scores)[:valid_count], np.array(final_cls_inds)[:valid_count]], axis=-1)
        elif args.end2end_model:
            if isinstance(data, list):
                data = data[0]
            pred = data[0] if data.ndim == 3 else data
            if dwdh is not None:
                dw, dh = dwdh 
                data[:, 0] -= dw
                data[:, 1] -= dh
                data[:, 2] -= dw
                data[:, 3] -= dh
            data[:,:4] /= ratio
            dets = data
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
            dets = self.postprocess(predictions,ratio,dwdh=dwdh,ultralytics=args.ultralytics)

        if dets is not None and len(dets) > 0:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=args.conf, class_names=self.class_names)
        return origin_img

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
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


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
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def letterbox(im,
              new_shape = (640, 640),
              color = (114, 114, 114),
              swap=(2, 0, 1)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]
    target_h, target_w = new_shape
    # Scale ratio (new / old)
    r = min(target_w / shape[1], target_h / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #(w,h)
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose(swap)
    im = im[np.newaxis,:]
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.
    return im, r, (dw, dh)


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
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