import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.trtEngine import BaseEngine
import cv2


class Predictor(BaseEngine):
    def __init__(self, engine_path, opt_batch_size=16, max_batch_size=32, conf_thres=0.25, iou_thres=0.7):
        super(Predictor, self).__init__(engine_path, opt_batch_size=opt_batch_size, max_batch_size=max_batch_size, conf_thres=conf_thres, iou_thres=iou_thres)
        self.n_classes = 80
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-d", "--directory", help="directory path")
    parser.add_argument("-o", "--output_dir", default= "results", help="images output path")
    parser.add_argument("-v", "--video", help="video path or camera index")
    parser.add_argument("--end2end", default=False, action="store_true", help="use end2end engine")
    parser.add_argument("--efficient_end2end", default=False, action="store_true", help='use efficient_end2end engine')
    parser.add_argument("--conf", type=float, default=0.25, help='object confidence threshold')
    parser.add_argument("--iou", type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--ultralytics', default=False, action="store_true", help='whether the model is from ultralytics')
    parser.add_argument('--end2end_model', action="store_true", help='whether the model is end2end')
    parser.add_argument("--opt_batch_size", type=int, default=16, help="the batch size used for engine optimization")
    parser.add_argument("--max_batch_size", type=int, default=32, help="the max batch size supported by the engine")
    parser.add_argument("--profile", default=False, action="store_true", help="whether to profile the model")

    args = parser.parse_args()
    print(args)

    if args.end2end and args.end2end_model:
        raise NotImplementedError("end2end model is already End2End.")

    pred = Predictor(engine_path=args.engine, opt_batch_size=args.opt_batch_size, max_batch_size=args.max_batch_size, conf_thres=args.conf, iou_thres=args.iou)
    pred.get_fps()

    if args.image:
        pred.inference(args.image, args)
        
    if args.video:
        pred.detect_video(args.video, args)
    
    if args.directory:
        pred.inference(args.directory, args)