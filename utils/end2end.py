import torch
import torch.nn as nn
import random


class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class TRT8_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        batch, anchors, _ = x.shape
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf

        nms_box = box @ self.convert_matrix
        nms_score = score.transpose(1, 2).contiguous()

        selected_indices = ORT_NMS.apply(nms_box, nms_score, self.max_obj, self.iou_threshold, self.score_threshold)
        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)
        selected_score = nms_score[batch_inds, cls_inds, box_inds].unsqueeze(1)
        selected_box = nms_box[batch_inds, box_inds, ...]

        dets = torch.cat([selected_box, selected_score], dim=1)

        batched_dets = dets.unsqueeze(0).repeat(batch, 1, 1)
        batch_template = torch.arange(0, batch, dtype=batch_inds.dtype, device=batch_inds.device)
        batched_dets = batched_dets.where((batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),batched_dets.new_zeros(1))

        batched_labels = cls_inds.unsqueeze(0).repeat(batch, 1)
        batched_labels = batched_labels.where((batch_inds == batch_template.unsqueeze(1)),batched_labels.new_ones(1) * -1)

        N = batched_dets.shape[0]

        batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
        batched_labels = torch.cat((batched_labels, -batched_labels.new_ones((N, 1))), 1)

        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)

        topk_batch_inds = torch.arange(batch, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)
        batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
        det_classes = batched_labels[topk_batch_inds, topk_inds, ...]
        det_boxes, det_scores = batched_dets.split((4, 1), -1)
        det_scores = det_scores.squeeze(-1)
        num_det = (det_scores > 0).sum(1, keepdim=True)
        return num_det, det_boxes, det_scores, det_classes


class ONNX_TRT8(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = -1
        self.box_coding = 0
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        bboxes_cxcywh = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        scores *= conf

        bboxes_xyxy = torch.zeros_like(bboxes_cxcywh)
        dw = bboxes_cxcywh[:, :, 2] / 2.0
        dh = bboxes_cxcywh[:, :, 3] / 2.0
        
        bboxes_xyxy[:, :, 0] = bboxes_cxcywh[:, :, 0] - dw # x1
        bboxes_xyxy[:, :, 1] = bboxes_cxcywh[:, :, 1] - dh # y1
        bboxes_xyxy[:, :, 2] = bboxes_cxcywh[:, :, 0] + dw # x2
        bboxes_xyxy[:, :, 3] = bboxes_cxcywh[:, :, 1] + dh # y2

        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(
            bboxes_xyxy, 
            scores, 
            self.background_class, 
            self.box_coding,
            self.iou_threshold, 
            self.max_obj,
            self.plugin_version, 
            self.score_activation,
            self.score_threshold
        )
        return num_det, det_boxes, det_scores, det_classes
    
class ONNX_TRT8_U(nn.Module):
    '''onnx moudule with TensorRT NMS operation for ultralytics models'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = -1
        self.box_coding = 0
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    
    def forward(self, x):
        # 1. 处理输入维度
        # Ultralytics 输出可能是 list，也可能是 tensor。如果是 list 取第一个
        if isinstance(x, list):
            x = x[0]
        # 此时 x 的形状是 (Batch, 4+C, Anchors) -> (1, 84, 8400)
        # 我们需要把它变成 (Batch, Anchors, 4+C) -> (1, 8400, 84) 以便后续切片
        x = x.permute(0, 2, 1)
        # 2. 拆分 Box 和 Score
        # 此时 x 是 (Batch, 8400, 84)
        bboxes_cxcywh = x[:, :, :4]  # 前4列是 cx, cy, w, h
        scores = x[:, :, 4:]       # 后80列是类别概率
        # 3. 坐标转换: cx,cy,w,h -> x1,y1,x2,y2
        # EfficientNMS 最好 x1y1x2y2，这样输出也是 x1y1x2y2，方便画图
        bboxes_xyxy = torch.zeros_like(bboxes_cxcywh)
        dw = bboxes_cxcywh[:, :, 2] / 2.0
        dh = bboxes_cxcywh[:, :, 3] / 2.0
        
        bboxes_xyxy[:, :, 0] = bboxes_cxcywh[:, :, 0] - dw # x1
        bboxes_xyxy[:, :, 1] = bboxes_cxcywh[:, :, 1] - dh # y1
        bboxes_xyxy[:, :, 2] = bboxes_cxcywh[:, :, 0] + dw # x2
        bboxes_xyxy[:, :, 3] = bboxes_cxcywh[:, :, 1] + dh # y2

        # 4. 调用 TensorRT NMS 插件
        # 注意：这里假设 TRT8_NMS.apply 已经正确绑定了 EfficientNMS_TRT 插件
        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(
            bboxes_xyxy, 
            scores, 
            self.background_class, 
            self.box_coding,
            self.iou_threshold, 
            self.max_obj,
            self.plugin_version, 
            self.score_activation,
            self.score_threshold
        )
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, ultralytics=False, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None, ort=False, with_preprocess=False):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.with_preprocess = with_preprocess
        self.model = model.to(device)
        ORT = ONNX_TRT8_U if ultralytics else ONNX_TRT8
        self.patch_model = ONNX_ORT if ort else ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, device)
        self.end2end.eval()

    def forward(self, x):
        if self.with_preprocess:
            x = x[:,[2,1,0],...]
            x = x * (1/255)
        x = self.model(x)
        if isinstance(x, tuple):
            x = list(x)
            x = x[0]
        else:
            x = x
        x = self.end2end(x)
        return x

class TRT10_NMS_Op(torch.autograd.Function):
    """
    对应 TensorRT INMSLayer 的 ONNX 导出实现。
    生成标准的 onnx::NonMaxSuppression 节点。
    """
    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        """
        PyTorch 前向推理时的逻辑 (Dummy 实现，仅为了通过 shape 检查)
        TRT 部署时不会用到这里，而是用 symbolic 定义的算子。
        """
        # 假设这里只是简单返回一个空的索引，实际推理中 torch 无法直接模拟 TRT 的 NMS 行为
        # 返回形状: [N, 3] -> (batch_index, class_index, box_index)
        return torch.zeros((1, 3), dtype=torch.int64)

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        """
        定义 ONNX 算子，直接映射到 INMSLayer 的输入要求。
        """
        # 1. 将 Python 数值转换为 ONNX 的 Constant Tensor
        # INMSLayer Doc: MaxOutputBoxesPerClass is a scalar (0D tensor) of type int32
        max_output_boxes_t = g.op("Constant", value_t=torch.tensor([max_output_boxes_per_class], dtype=torch.int64))
        # INMSLayer Doc: IoUThreshold is a scalar float32
        iou_threshold_t = g.op("Constant", value_t=torch.tensor([iou_threshold], dtype=torch.float32))
        # INMSLayer Doc: ScoreThreshold is a scalar float32
        score_threshold_t = g.op("Constant", value_t=torch.tensor([score_threshold], dtype=torch.float32))

        # 2. 生成 Standard ONNX NonMaxSuppression 节点
        # 官方文档对应关系:
        # Boxes -> input[0]
        # Scores -> input[1]
        # MaxOutputBoxesPerClass -> input[2]
        # IoUThreshold -> input[3]
        # ScoreThreshold -> input[4]
        return g.op("NonMaxSuppression",
                    boxes,
                    scores,
                    max_output_boxes_t,
                    iou_threshold_t,
                    score_threshold_t,
                    center_point_box_i=0)
    

def apply_trt10_nms(boxes, scores, max_output_boxes, iou_thres, conf_thres):
    """
    Args:
        boxes: [Batch, N, 4] (x1y1x2y2)
        scores: [Batch, C, N] (注意维度顺序!)
    """
    # 1. 执行 NMS 算子，得到索引 [M, 3] -> (batch_idx, class_idx, box_idx)
    indices = TRT10_NMS_Op.apply(boxes, scores, max_output_boxes, iou_thres, conf_thres)
    
    # 2. 解析索引 (Gather)
    batch_ids = indices[:, 0]
    class_ids = indices[:, 1]
    box_ids   = indices[:, 2]

    # 3. 提取最终结果
    det_boxes = boxes[batch_ids, box_ids]      # [M, 4]
    # score 需要重新转置回去取值，或者直接用 batch/box/class 索引取
    # scores 是 [B, C, N]，这里有点绕，我们用原始形状取值更稳妥
    det_scores = scores[batch_ids, class_ids, box_ids] # [M]
    
    det_batch_ids = batch_ids.to(torch.float32)
    det_classes = class_ids.to(torch.float32)
    
    # 返回 [M, 7] 格式: batch_id, x1, y1, x2, y2, score, class
    # 这样 C++ 处理起来最方便，只需要读这一个 output tensor
    return torch.stack([det_batch_ids, det_boxes[:, 0], det_boxes[:, 1], det_boxes[:, 2], det_boxes[:, 3], det_scores, det_classes], dim=1)


class Ultralytics_TRT10_Wrapper(nn.Module):
    def __init__(self, model, max_det=100, iou_thres=0.45, conf_thres=0.25, seg=False):
        super().__init__()
        self.model = model
        self.max_det = max_det
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.seg = seg

    def forward(self, x):
        # 1. 处理输入维度
        # Ultralytics 输出可能是 list，也可能是 tensor。如果是 list 取第一个
        img_h, img_w = x.shape[2], x.shape[3]
        x = self.model(x)
        if isinstance(x, list):
            x = x[0]
        # 此时 x 的形状是 (Batch, 4+C, Anchors) -> (1, 84, 8400)
        # 我们需要把它变成 (Batch, Anchors, 4+C) -> (1, 8400, 84) 以便后续切片
        x = x.permute(0, 2, 1)
        # 2. 拆分 Box 和 Score
        # 此时 x 是 (Batch, 8400, 84)
        bboxes_cxcywh = x[:, :, :4]  # 前4列是 cx, cy, w, h
        if self.seg:
            scores = x[:, :, 4:-32]
        else:
            scores = x[:, :, 4:]       # 后80列是类别概率
        # 3. 坐标转换: cx,cy,w,h -> x1,y1,x2,y2
        # EfficientNMS 最好 x1y1x2y2，这样输出也是 x1y1x2y2，方便画图
        bboxes_xyxy = torch.zeros_like(bboxes_cxcywh)
        dw = bboxes_cxcywh[:, :, 2] / 2.0
        dh = bboxes_cxcywh[:, :, 3] / 2.0
        
        bboxes_xyxy[:, :, 0] = bboxes_cxcywh[:, :, 0] - dw # x1
        bboxes_xyxy[:, :, 1] = bboxes_cxcywh[:, :, 1] - dh # y1
        bboxes_xyxy[:, :, 2] = bboxes_cxcywh[:, :, 0] + dw # x2
        bboxes_xyxy[:, :, 3] = bboxes_cxcywh[:, :, 1] + dh # y2

        bboxes_xyxy[:, :, 0].clamp_(0, img_w)
        bboxes_xyxy[:, :, 1].clamp_(0, img_h)
        bboxes_xyxy[:, :, 2].clamp_(0, img_w)
        bboxes_xyxy[:, :, 3].clamp_(0, img_h)
        # 4. 格式调整适配 NMS
        # Standard NMS 要求 scores 格式为 [Batch, Classes, Boxes]
        final_scores_transposed = scores.transpose(1, 2)

        # 6. 调用 TRT10 NMS
        # 输出形状 [M, 7]: batch_id, x1, y1, x2, y2, score, class
        nms_out = apply_trt10_nms(
            bboxes_xyxy, 
            final_scores_transposed, 
            self.max_det, 
            self.iou_thres, 
            self.conf_thres
        )
        return nms_out