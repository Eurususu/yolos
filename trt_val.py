import argparse
import os
import numpy as np
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.trtEngine import BaseEngine
from utils.trtEngine import letterbox


class Validator(BaseEngine):
    def __init__(self, engine_path):
        super(Validator, self).__init__(engine_path)
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
    
    def inference(self, img_path, args):
        origin_img = cv2.imread(img_path)
        # img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        img, ratio, dwdh = letterbox(origin_img, self.imgsz)
        data = self.infer(img)
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
                # 1. 读取图片
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(args.img_dir, img_info['file_name'])
                dets = self.inference(img_path, args)
                if len(dets) == 0:
                    print(f"No detections for image {img_id}")
                    continue
                boxes, scores, classes = dets[:,:4], dets[:, 4], dets[:, 5]
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    score = float(scores[i])
                    cls_idx = int(classes[i])
                    
                    # 转换 xyxy -> xywh
                    w = x2 - x1
                    h = y2 - y1
                    
                    # 类别 ID 转换
                    if args.use_coco_map:
                        # 防止索引越界
                        if cls_idx < len(coco80_to_coco91):
                            category_id = coco80_to_coco91[cls_idx]
                        else:
                            continue 
                    else:
                        category_id = cls_idx
                    
                    # 添加到结果列表
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
            
            # 打印 mAP@0.5
            print(f"\nFinal mAP@0.5: {cocoEval.stats[1]:.3f}")
            print(f"Final mAP@0.5:0.95: {cocoEval.stats[0]:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("--img_dir", help="Directory containing validation images")
    parser.add_argument("--coco_json", help="COCO json file path")
    parser.add_argument("--use_coco_map", action="store_true", help="Map class 0-79 to COCO 1-90 (use this for standard COCO dataset)")

    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")
    parser.add_argument("--efficient_end2end", default=False, action="store_true", 
                        help='use efficient_end2end engine')
    parser.add_argument("--conf", type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--ultralytics', default=False, action="store_true",
                        help='whether the model is from ultralytics, only for not end2end model')
    parser.add_argument('--end2end_model', action="store_true", help='whether the model is end2end')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    engine_path = args.engine
    val = Validator(engine_path)
    val.run_validate(args)