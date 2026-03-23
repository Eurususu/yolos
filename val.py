from ultralytics import YOLO
import argparse
import torch

def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model', type=str, default='', help='weights path')
    args_parser.add_argument('--data', type=str, default='', help='image/video path')
    args_parser.add_argument('--conf', type=float, default=0.001, help='confidence threshold')
    args_parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
    args_parser.add_argument('--max_det', type=int, default=300, help='maximum detections per image')
    args_parser.add_argument('--save_json', action='store_true', help='save result json')
    args_parser.add_argument('--batch', type=int, default=16, help='batch size')
    args_parser.add_argument('--imgsz', type=int, default=640, help='height and width of the input image')
    args_parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    args_parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    args_parser.add_argument('--device', default='', help='device to run inference on')
    args_parser.add_argument('--save_txt', action='store_true', help='save label on result image/video')
    args_parser.add_argument('--save_conf', action='store_true', help='save confidence score on result image/video')
    args_parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    args_parser.add_argument('--name', default='exp', help='save results to project/name')
    args_parser.add_argument('--split', type=str, default='val', help='val or test or train')
    args_parser.add_argument('--plot', action='store_true', help='plot result curves')
    args = args_parser.parse_args()
    return args


def run_val(args):
    model = YOLO(args.model)
    model.fuse()
    if args.half:
        model.to('cuda').half()
    else:
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
        split=args.split,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )
    print(f'mAP50-90 is:{metrics.box.map}')  # mAP50-95
    print(f'mAP50 is:{metrics.box.map50}')  # mAP50
    print(f'mAP75 is:{metrics.box.map75}')  # mAP75


if __name__ == '__main__':
    args = parse_args()
    results = run_val(args)