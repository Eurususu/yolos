import argparse
import os
import numpy as np
import json
import cv2
from tqdm import tqdm
import logging
import warnings

# 关闭警告
warnings.filterwarnings('ignore')

# 延迟导入，避免没有 TensorRT 时无法加载
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.trtEngine import BaseEngine
from utils.trtEngine import letterbox


class Validator(BaseEngine):
    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.65):
        super(Validator, self).__init__(engine_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.n_classes = 80

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
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = Validator.nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
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

    def inference(self, img_path, args):
        origin_img = cv2.imread(img_path)
        img, ratio, dwdh = letterbox(origin_img, self.imgsz)
        data = self.infer(img)

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
            # efficient_end2end 输出格式: (num, boxes, scores, classes)
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

        if dets is not None and len(dets) > 0:
            return dets
        else:
            return np.array([])

    def run_validate(self, args):
        if args.coco_json:
            coco = COCO(args.coco_json)
            if 'info' not in coco.dataset:
                coco.dataset['info'] = {'description': 'Converted YOLO Dataset'}
            if 'licenses' not in coco.dataset:
                coco.dataset['licenses'] = []
            img_ids = coco.getImgIds()
            results = []
            coco80_to_coco91 = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
            ]
            print(f"Starting validation on {len(img_ids)} images...")
            for img_id in tqdm(img_ids):
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(args.img_dir, img_info['file_name'])
                dets = self.inference(img_path, args)
                if len(dets) == 0:
                    continue
                boxes, scores, classes = dets[:, :4], dets[:, 4], dets[:, 5]
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    score = float(scores[i])
                    cls_idx = int(classes[i])

                    w = x2 - x1
                    h = y2 - y1

                    if args.use_coco_map:
                        if cls_idx < len(coco80_to_coco91):
                            category_id = coco80_to_coco91[cls_idx]
                        else:
                            continue
                    else:
                        category_id = cls_idx

                    results.append({
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": score
                    })

            if not results:
                print("No detections found!")
                return
            print("Evaluating...")
            cocoDt = coco.loadRes(results)
            cocoEval = COCOeval(coco, cocoDt, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            print(f"\nFinal mAP@0.5: {cocoEval.stats[1]:.3f}")
            print(f"Final mAP@0.5:0.95: {cocoEval.stats[0]:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("--img_dir", type=str, default='/home/jia/dataset/coco2017/images/val2017', help="Path to COCO val images directory")
    parser.add_argument("--coco_json", type=str, default='/home/jia/dataset/coco2017/annotations/instances_val2017.json', help="Path to COCO val annotations json")
    parser.add_argument("--use_coco_map", action="store_true", help="Map class 0-79 to COCO 1-90")

    parser.add_argument("--end2end", default=False, action="store_true", help="use end2end engine")
    parser.add_argument("--efficient_end2end", default=False, action="store_true", help='use efficient_end2end engine')
    parser.add_argument("--conf", type=float, default=0.001, help='object confidence threshold')
    parser.add_argument("--iou", type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--ultralytics', default=False, action="store_true", help='whether the model is from ultralytics')
    parser.add_argument('--end2end_model', action="store_true", help='whether the model is end2end')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    engine_path = args.engine
    val = Validator(engine_path, conf_thres=args.conf, iou_thres=args.iou)
    val.run_validate(args)