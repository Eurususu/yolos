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
    def __init__(self, engine_path, max_batch_size=32, opt_batch_size=None, max_det=300, conf_thres=0.25, iou_thres=0.7):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size if opt_batch_size else max_batch_size
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

            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = list(raw_shape)
            # 0. 如果是输入且第0维大于等于1，记录这个维度作为 max_batch_size
            if is_input and shape[0] >= 1:
                self.max_batch_size = shape[0]
                # 同步更新 opt，防止静态模型导致逻辑不一致
                if opt_batch_size is None:
                    self.opt_batch_size = shape[0]
                
            # 1. 无论输入输出，如果第0维是-1，INMSlayer 设置为batch * max_det
            if shape[0] == -1 and not is_input and shape[1] == 7:
                shape[0] = self.max_batch_size * self.max_det
            # 其他情况第一个维度都设置为max_batch_size
            else:
                shape[0] = self.max_batch_size
                
                
            # 2. 遍历其余维度，处理其他动态维度 (例如动态NMS插件的输出可能是 [batch, -1, 4])
            for j in range(1, len(shape)):
                if shape[j] == -1:
                    shape[j] = self.max_det
                    
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
                binding['host_buffer'] = np.empty(shape, dtype=binding['dtype'])
                self.outputs.append(binding)
            self.allocations.append(ptr)

    def __del__(self):
        # 1. 首先释放 TensorRT 的执行上下文 (Context) 和引擎 (Engine)
        if hasattr(self, 'context') and self.context is not None:
            del self.context
            self.context = None
            
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
            self.engine = None
            
        # 2. 释放为输入/输出分配的 GPU 显存 (ptr)
        # 注意：这里要处理你 __init__ 里绑定的 inputs 和 outputs
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                if 'ptr' in inp and inp['ptr'] != 0:
                    try:
                        cudart.cudaFree(inp['ptr'])
                    except:
                        pass
        if hasattr(self, 'outputs'):
            for out in self.outputs:
                if 'ptr' in out and out['ptr'] != 0:
                    try:
                        cudart.cudaFree(out['ptr'])
                    except:
                        pass

        # 3. 最后销毁 CUDA Stream
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                cudart.cudaStreamDestroy(self.stream)
            except:
                pass
            self.stream = None

    def infer(self, img, profile=False):
        img = np.ascontiguousarray(img)
        input_binding = self.inputs[0]
        input_name = input_binding['name']

        self.context.set_input_shape(input_name, img.shape)
        input_bytes = img.nbytes

        if profile:
            _, event_start = cudart.cudaEventCreate()
            _, event_h2d_end = cudart.cudaEventCreate()
            _, event_compute_end = cudart.cudaEventCreate()
            _, event_d2h_end = cudart.cudaEventCreate()
            cudart.cudaEventRecord(event_start, self.stream)

        # 1. H2D
        cudart.cudaMemcpyAsync(
            input_binding['ptr'],
            img.ctypes.data,
            input_bytes,  
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream
        )

        if profile: cudart.cudaEventRecord(event_h2d_end, self.stream)

        # 2. 推理前洗地
        for out in self.outputs:
            cudart.cudaMemsetAsync(out['ptr'], 0, out['size'], self.stream)

        # 3. 异步执行推理
        self.context.execute_async_v3(stream_handle=self.stream)

        if profile: cudart.cudaEventRecord(event_compute_end, self.stream)

        # 4. 精准 D2H 拷贝 (直接写进预分配的 host_buffer)
        for out in self.outputs:
            actual_shape = tuple(self.context.get_tensor_shape(out['name']))

            # 动态计算真实拷贝大小
            if -1 in actual_shape:
                # 兜底：插件不回传真实大小，拷回整块最大显存
                actual_shape = tuple(out['shape'])
                copy_bytes = out['size']
            else:
                # 算出真实大小
                vol = 1
                for s in actual_shape: vol *= s
                copy_bytes = vol * out['dtype'].itemsize
            
            # 记录这帧实际的 shape，给下面的切片用
            out['actual_shape'] = actual_shape
            
            # 【极致提速】：直接拷入 __init__ 预分配好的 numpy 数组，0 内存分配开销！
            cudart.cudaMemcpyAsync(
                out['host_buffer'].ctypes.data,
                out['ptr'],
                copy_bytes,  
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )
        
        if profile: cudart.cudaEventRecord(event_d2h_end, self.stream)

        # 同步流
        cudart.cudaStreamSynchronize(self.stream)

        # 5. 零拷贝视图返回
        final_outputs = []
        for out in self.outputs:
            # 根据这帧的实际维度，切取预分配内存里的有效部分 (视图，不复制！)
            slices = tuple(slice(0, s) for s in out['actual_shape'])
            final_outputs.append(out['host_buffer'][slices])

        if profile:
            _, h2d_ms = cudart.cudaEventElapsedTime(event_start, event_h2d_end)
            _, compute_ms = cudart.cudaEventElapsedTime(event_h2d_end, event_compute_end)
            _, d2h_ms = cudart.cudaEventElapsedTime(event_compute_end, event_d2h_end)
            
            cudart.cudaEventDestroy(event_start)
            cudart.cudaEventDestroy(event_h2d_end)
            cudart.cudaEventDestroy(event_compute_end)
            cudart.cudaEventDestroy(event_d2h_end)
            
            return final_outputs, (h2d_ms, compute_ms, d2h_ms)
        
        return final_outputs

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
        dets = BaseEngine.multiclass_nms(boxes_xyxy, scores, nms_thr=self.iou_thres, score_thr=self.conf_thres)
        return dets

    # def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
    #     final_dets = []
    #     num_classes = scores.shape[1]
    #     for cls_ind in range(num_classes):
    #         cls_scores = scores[:, cls_ind]
    #         valid_score_mask = cls_scores > score_thr
    #         if valid_score_mask.sum() == 0:
    #             continue
    #         valid_scores = cls_scores[valid_score_mask]
    #         valid_boxes = boxes[valid_score_mask]
    #         keep = self.nms(valid_boxes, valid_scores, nms_thr)
    #         if len(keep) > 0:
    #             cls_inds = np.ones((len(keep), 1)) * cls_ind
    #             dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
    #             final_dets.append(dets)
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
        keep = BaseEngine.nms(boxes_for_nms, valid_scores, nms_thr)
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

    def process_output(self, data, ratio, dwdh, args):
        """统一处理模型输出"""
        # 1. 统一解包 list (针对非 efficient_end2end 模型)
        if not args.efficient_end2end and isinstance(data, list):
            data = data[0]

        # 2. 根据ratio列表来决定真实的batch大小
        if isinstance(ratio, (list, tuple, np.ndarray)):
            real_batch_size = len(ratio)
        else:
            real_batch_size = 1
            ratio = [ratio]  # 统一转为列表
        
        if dwdh is not None:
            if isinstance(dwdh[0], (int, float)): # 如果传的是单图的 (dw, dh)
                dwdh = [dwdh]
        else:
            dwdh = [None] * real_batch_size

        # # 2. 动态推断 batch size
        # if args.efficient_end2end:
        #     batch_size = data[0].shape[0]
        # elif args.end2end:
        #     # INMSlayer 形状为 [num_dets, 7]，第一列是 batch_id。
        #     if data.shape[0] > 0:
        #         batch_size = int(np.max(data[:, 0])) + 1
        #     else:
        #     # 兜底：如果整个 batch 都没有物体导致 data 为空，才使用外部列表长度
        #         batch_size = len(ratio) if isinstance(ratio, (list, tuple, np.ndarray)) else 1
        # elif args.end2end_model:
        #     # yolov10 这类模型形状为 [batch, max_det, 6]
        #     batch_size = data.shape[0]
        # else:
        #     # ultralytics 及其他常规模型
        #     batch_size = data.shape[0]

        # # 3. 将 ratio 和 dwdh 统一格式化为列表，兼容单图或多图传入
        # if not isinstance(ratio, (list, tuple, np.ndarray)):
        #     ratio = [ratio] * batch_size
        # if dwdh is not None:
        #     if isinstance(dwdh[0], (int, float)): # 如果传的是单图的 (dw, dh)
        #         dwdh = [dwdh] * batch_size
        # else:
        #     dwdh = [None] * batch_size

        batch_dets = []

        # 4. 根据不同的模型插件类型，遍历处理每一个 Batch

        # ---------------------------------------------------------
        # 类型 1: efficient_end2end (Nvidia Efficient NMS 插件: 4个输出头)
        # ---------------------------------------------------------
        if args.efficient_end2end:
            num_dets, boxes, scores, classes = data
            for b in range(real_batch_size):
                cur_ratio = ratio[b]
                cur_dwdh = dwdh[b]
                valid_count = int(num_dets[b][0])
                
                if valid_count == 0:
                    batch_dets.append(np.empty((0, 6)))
                    continue
                    
                cur_boxes = boxes[b][:valid_count].reshape(-1, 4)
                if cur_dwdh is not None:
                    dw, dh = cur_dwdh
                    cur_boxes[:, [0, 2]] -= dw
                    cur_boxes[:, [1, 3]] -= dh
                cur_boxes /= cur_ratio
                
                cur_scores = scores[b][:valid_count].reshape(-1, 1)
                cur_cls_inds = classes[b][:valid_count].reshape(-1, 1)
                
                dets = np.concatenate([cur_boxes, cur_scores, cur_cls_inds], axis=-1)
                batch_dets.append(dets)
        
        # ---------------------------------------------------------
        # 类型 2: end2end (INMSlayer 插件: [num_dets, 7])
        # ---------------------------------------------------------
        elif args.end2end:
            for b in range(real_batch_size):
                # 如果完全没检测到，直接放入空数组
                if data.shape[0] == 0:
                    batch_dets.append(np.empty((0, 6)))
                    continue
                    
                cur_ratio = ratio[b]
                cur_dwdh = dwdh[b]
                
                # 直接使用输出数据本身的第0列 (batch_id) 进行过滤
                batch_mask = data[:, 0] == b
                cur_data = data[batch_mask] 
                
                conf_mask = cur_data[:, 5] > self.conf_thres
                valid_predictions = cur_data[conf_mask]
                
                if valid_predictions.shape[0] == 0:
                    batch_dets.append(np.empty((0, 6)))
                    continue
                
                dw, dh = cur_dwdh if cur_dwdh is not None else (0, 0)
                
                # 提取数据: 1-4列为box，5列为score，6列为class
                final_boxes = valid_predictions[:, 1:5]
                final_boxes[:, [0, 2]] -= dw
                final_boxes[:, [1, 3]] -= dh
                final_boxes /= cur_ratio
                
                final_scores = valid_predictions[:, 5:6]
                final_cls_inds = valid_predictions[:, 6:7].astype(int)
                
                dets = np.concatenate([final_boxes, final_scores, final_cls_inds], axis=-1)
                batch_dets.append(dets)
        
        # ---------------------------------------------------------
        # 类型 3: end2end_model (如 YOLOv10: [batch, max_det, 6])
        # ---------------------------------------------------------
        elif args.end2end_model:
            for b in range(real_batch_size):
                cur_ratio = ratio[b]
                cur_dwdh = dwdh[b]
                
                cur_data = data[b] 
                mask = cur_data[:, 4] > self.conf_thres
                valid_predictions = cur_data[mask]
                
                if valid_predictions.shape[0] == 0:
                    batch_dets.append(np.empty((0, 6)))
                    continue
                    
                if cur_dwdh is not None:
                    dw, dh = cur_dwdh
                    valid_predictions[:, [0, 2]] -= dw
                    valid_predictions[:, [1, 3]] -= dh
                valid_predictions[:, :4] /= cur_ratio
                
                batch_dets.append(valid_predictions)
        
        # ---------------------------------------------------------
        # 类型 4 & 5: ultralytics 及 其他常规非端到端模型
        # ---------------------------------------------------------
        else:
            for b in range(real_batch_size):
                cur_ratio = ratio[b]
                cur_dwdh = dwdh[b]
                cur_data = data[b]
                
                if args.ultralytics:
                    # 原本的 [84, 8400] 转置为 [8400, 84]
                    predictions = cur_data.transpose()
                else:
                    predictions = cur_data.reshape(-1, 5 + self.n_classes)
                    
                dets = self.postprocess(predictions, cur_ratio, dwdh=cur_dwdh, ultralytics=args.ultralytics)
                batch_dets.append(dets if dets is not None else np.empty((0, 6)))

        return batch_dets



        # if args.end2end:
        #     mask = data[:, 5] > self.conf_thres
        #     valid_predictions = data[mask]
        #     if valid_predictions.shape[0] == 0:
        #         print("没有检测到物体")
        #         return None
        #     else:
        #         dw, dh = dwdh
        #         final_boxes = valid_predictions[:, 1:5]
        #         final_boxes[:, [0, 2]] -= dw
        #         final_boxes[:, [1, 3]] -= dh
        #         final_boxes /= ratio
        #         final_scores = valid_predictions[:, 5:6]
        #         final_cls_inds = valid_predictions[:, 6:7].astype(int)
        #         dets = np.concatenate([final_boxes, final_scores, final_cls_inds], axis=-1)

        # elif args.efficient_end2end:
        #     num, final_boxes, final_scores, final_cls_inds = data
        #     final_boxes = final_boxes.reshape(-1, 4)
        #     if dwdh is not None:
        #         dw, dh = dwdh
        #         final_boxes[:, [0, 2]] -= dw
        #         final_boxes[:, [1, 3]] -= dh
        #     final_boxes /= ratio
        #     valid_count = int(num[0])
        #     dets = np.concatenate([
        #         final_boxes[:valid_count],
        #         final_scores[:valid_count].reshape(-1, 1),
        #         final_cls_inds[:valid_count].reshape(-1, 1)
        #     ], axis=-1)

        # elif args.end2end_model:
        #     data = data[0] if data.ndim == 3 else data
        #     mask = data[:, 4] > self.conf_thres
        #     valid_predictions = data[mask]
        #     if valid_predictions.shape[0] == 0:
        #         print("没有检测到物体")
        #     else:
        #         if dwdh is not None:
        #             dw, dh = dwdh
        #             valid_predictions[:, [0, 2]] -= dw
        #             valid_predictions[:, [1, 3]] -= dh
        #         valid_predictions[:, :4] /= ratio
        #         dets = valid_predictions

        # else:
        #     if args.ultralytics:
        #         predictions = data[0] if data.ndim == 3 else data
        #         predictions = predictions.transpose()
        #     else:
        #         predictions = data.reshape(1, -1, 5 + self.n_classes)[0]
        #     dets = self.postprocess(predictions, ratio, dwdh=dwdh, ultralytics=args.ultralytics)

        # return dets

    def detect_video(self, video_path, args):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_vid = int(round(cap.get(cv2.CAP_PROP_FPS)))
        if fps_vid == 0: fps_vid = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        name = os.path.basename(video_path)
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        out = cv2.VideoWriter(os.path.join(save_dir, f"result_{name}"), fourcc, fps_vid, (width, height))

        print(f"开始视频推理，按 opt_batch_size={self.opt_batch_size} 攒批处理。按 'q' 退出...")
        batch_orig_frames = []
        batch_imgs = []
        batch_ratios = []
        batch_dwdhs = []

        frame_count = 0
        stop_flag = False
        is_profile = args.profile
        # curr_fps = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img, ratio, dwdh = letterbox(frame, self.imgsz)
            batch_orig_frames.append(frame)
            batch_imgs.append(img)
            batch_ratios.append(ratio)
            batch_dwdhs.append(dwdh)

            # 当攒够 opt_batch_size 时，执行一次批量推理
            if len(batch_imgs) == self.opt_batch_size:
                input_tensor = np.vstack(batch_imgs)
                input_tensor = np.ascontiguousarray(input_tensor)
                t1 = time.time()
                infer_result = self.infer(input_tensor, profile=is_profile)
                t2 = time.time()
                if is_profile:
                    # 分离出 推理数据 和 耗时数据
                    data, profile_times = infer_result
                    h2d_ms, compute_ms, d2h_ms = profile_times
                    
                    # 既然开启了 profile，顺便在终端打印一下极客视角的耗时细节
                    print(f"[Profile] H2D: {h2d_ms:.2f}ms | Compute: {compute_ms:.2f}ms | D2H: {d2h_ms:.2f}ms")
                else:
                    data = infer_result
                batch_time_ms = (t2 - t1) * 1000
                fps_curr = 1000 / (batch_time_ms / self.opt_batch_size) # 算出单帧等效 FPS

                # 处理当前批次
                batch_dets = self.process_output(data, batch_ratios, batch_dwdhs, args)
                if not isinstance(batch_dets, list):
                    batch_dets = [batch_dets]
                
                for j, dets in enumerate(batch_dets):
                    orig_img = batch_orig_frames[j]
                    if dets is not None and len(dets) > 0:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        orig_img = vis(orig_img, final_boxes, final_scores, final_cls_inds,
                                       conf=self.conf_thres, class_names=self.class_names)
                    
                    cv2.putText(orig_img, f"FPS: {fps_curr:.1f} (Batch: {self.opt_batch_size})", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    out.write(orig_img)
                    cv2.imshow('frame', orig_img)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_flag = True
                        break
                
                if stop_flag:
                    break
                    
                frame_count += self.opt_batch_size
                if frame_count % (self.opt_batch_size * 5) == 0:
                    print(f"已处理 {frame_count} 帧, 最近批次耗时: {batch_time_ms:.1f}ms")

                # 清空缓冲区，迎接下一批
                batch_orig_frames.clear()
                batch_imgs.clear()
                batch_ratios.clear()
                batch_dwdhs.clear()
                
        # 处理视频结尾不够一个 opt_batch_size 的尾巴数据
        if len(batch_imgs) > 0 and not stop_flag:
            input_tensor = np.vstack(batch_imgs)
            input_tensor = np.ascontiguousarray(input_tensor)
            infer_result = self.infer(input_tensor, profile=is_profile)
            if is_profile:
                # 分离出 推理数据 和 耗时数据
                data, profile_times = infer_result
                h2d_ms, compute_ms, d2h_ms = profile_times
                
                # 既然开启了 profile，顺便在终端打印一下极客视角的耗时细节
                print(f"[Profile] H2D: {h2d_ms:.2f}ms | Compute: {compute_ms:.2f}ms | D2H: {d2h_ms:.2f}ms")
            else:
                data = infer_result
            batch_dets = self.process_output(data, batch_ratios, batch_dwdhs, args)
            if not isinstance(batch_dets, list): batch_dets = [batch_dets]

            for j, dets in enumerate(batch_dets):
                orig_img = batch_orig_frames[j]
                if dets is not None and len(dets) > 0:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    orig_img = vis(orig_img, final_boxes, final_scores, final_cls_inds,
                                   conf=self.conf_thres, class_names=self.class_names)
                out.write(orig_img)
                cv2.imshow('frame', orig_img)
                cv2.waitKey(1)

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 视频检测完毕。")

    def inference(self, img_path, args):
        if os.path.isdir(img_path):
            valid_exts = {'.jpg', '.png', '.jpeg', '.bmp', '.webp'}
            img_paths = [os.path.join(img_path, f) for f in os.listdir(img_path) 
                         if os.path.splitext(f)[-1].lower() in valid_exts]
            img_paths.sort()
        else:
            img_paths = [img_path]
        
        total_imgs = len(img_paths)
        if total_imgs == 0:
            print(f"[警告] 未在 {img_path} 找到支持的图片格式。")
            return None
        
        # 如果是处理目录，或者想看保存结果，创建输出文件夹
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        is_profile = args.profile
        print(f"准备推理，共计 {total_imgs} 张图片。使用最优 Batch Size (opt_batch_size): {self.opt_batch_size}")
        # 2. 按照 opt_batch_size 切片，循环处理每个 Batch
        for i in range(0, total_imgs, self.opt_batch_size):
            batch_paths = img_paths[i : i + self.opt_batch_size]
            batch_imgs = []
            batch_origs = []
            batch_ratios = []
            batch_dwdhs = []
            batch_names = []
            # 2.1 逐图读取并预处理
            for p in batch_paths:
                orig_img = cv2.imread(p)
                if orig_img is None:
                    print(f"[警告] 图片读取失败: {p}")
                    continue
                img, ratio, dwdh = letterbox(orig_img, self.imgsz)
                batch_origs.append(orig_img)      # 保存原始图像用于画框
                batch_imgs.append(img)            # letterbox 返回的是 (1, C, H, W)
                batch_ratios.append(ratio)
                batch_dwdhs.append(dwdh)
                batch_names.append(os.path.basename(p))
            if not batch_imgs: continue

            # 2.2 将当前 batch 的图片在第 0 维拼接：多个 (1, C, H, W) -> (real_B, C, H, W)
            input_tensor = np.vstack(batch_imgs)
            input_tensor = np.ascontiguousarray(input_tensor)

            # 3. 执行推理 (TRT 的动态维度会自动接纳真实的 batch 数量)
            infer_result = self.infer(input_tensor, profile=is_profile)
            if is_profile:
                # 分离出 推理数据 和 耗时数据
                data, profile_times = infer_result
                h2d_ms, compute_ms, d2h_ms = profile_times
                
                # 既然开启了 profile，顺便在终端打印一下极客视角的耗时细节
                print(f"[Profile] H2D: {h2d_ms:.2f}ms | Compute: {compute_ms:.2f}ms | D2H: {d2h_ms:.2f}ms")
            else:
                data = infer_result
            # 4. 后处理 (利用我们之前改好的支持多 Batch 的 process_output)
            batch_dets = self.process_output(data, batch_ratios, batch_dwdhs, args)

            # 兼容性处理：如果 process_output 只有1张图返回的是数组，统一包成 list 方便遍历
            if not isinstance(batch_dets, list):
                batch_dets = [batch_dets]
            

            # 5. 遍历当前 Batch 的每一个结果，画框并保存
            for j, dets in enumerate(batch_dets):
                orig_img = batch_origs[j]
                
                # 画框
                if dets is not None and len(dets) > 0:
                    final_boxes = dets[:, :4]
                    final_scores = dets[:, 4]
                    final_cls_inds = dets[:, 5]
                    
                    orig_img = vis(orig_img, final_boxes, final_scores, final_cls_inds,
                                   conf=self.conf_thres, class_names=self.class_names)
                
                # 保存到输出目录
                save_path = os.path.join(save_dir, batch_names[j])
                cv2.imwrite(save_path, orig_img)
                

            # 打印进度条
            print(f"已处理进度: {min(i + self.opt_batch_size, total_imgs)} / {total_imgs}")

        print(f"✅ 所有图片处理完毕！已保存至目录: {save_dir}/")
        return None

        # origin_img = cv2.imread(img_path)
        # img, ratio, dwdh = letterbox(origin_img, self.imgsz)
        # data = self.infer(img)

        # dets = self.process_output(data, ratio, dwdh, args)

        # if dets is not None and len(dets) > 0:
        #     final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        #     origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
        #                     conf=self.conf_thres, class_names=self.class_names)

        # return origin_img

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