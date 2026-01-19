from ultralytics import YOLO
import argparse
import torch
def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model', type=str, default='', help='weights path')
    args_parser.add_argument('--source', type=str, default='data/1.jpg', help='image/video path')
    args_parser.add_argument('--batch', type=int, default=1, help='batch size')
    args_parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    args_parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    args_parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    args_parser.add_argument('--imgsz', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    args_parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    args_parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    args_parser.add_argument('--device', default='', help='device to run inference on')
    args_parser.add_argument('--save', action='store_true', help='save result image/video')
    args_parser.add_argument('--show_labels', action='store_true', help='show label on result image/video')
    args_parser.add_argument('--show_conf', action='store_true', help='show confidence score on result image/video')
    args_parser.add_argument('--line_width', type=int, default=1, help='bounding box line width')
    args_parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    args_parser.add_argument('--name', default='exp', help='save results to project/name')
    args_parser.add_argument('--visualize', action='store_true', help='visualize features')
    args = args_parser.parse_args()
    return args


def run_infer(args):
    model = YOLO(args.model)
    model.fuse()
    if args.half:
        model.to('cuda').half()
    else:
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        imgsz=args.imgsz,
        save=args.save,
        project=args.project,
        name=args.name,
        line_width=args.line_width,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        visualize=args.visualize
    )
    return results


if __name__ == '__main__':
    args = parse_args()
    results = run_infer(args)