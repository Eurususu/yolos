from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset path')
    parser.add_argument('--model', type=str, default='weights/yolo11n.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--iterations', type=int, default=10, help='number of tuning iterations')
    parser.add_argument('--batch', type=int, default=16, help='batch size for training')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    parser.add_argument('--project', default='runs/train', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--save', action='store_false', help='save init and last weights')
    parser.add_argument('--save_period', type=int, default=-1, help='save weights every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=bool, default=False, help='cache images for faster training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='number of dataloader workers')
    # parser.add_argument('--pretrained', type=str, default='', help='use pretrained model weights')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--freeze', type=int, nargs='+', help='freeze the first N layers of the model or the layers specified by index')
    parser.add_argument('--profile', action='store_true', help='profile model speed while training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--plots', action='store_true', help='save plots of training metrics')
    args = parser.parse_args()
    return args

def run_tune(args):
    model = YOLO(args.model)

    model.tune(
        data=args.data,model=args.model,epochs=args.epochs,
        iterations=args.iterations,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,name=args.name,save=args.save,save_period=args.save_period,
        cache=args.cache,device=args.device,workers=args.workers,
        single_cls=args.single_cls,classes=args.classes,
        freeze=args.freeze,profile=args.profile,resume=args.resume,plots=args.plots
    )


if __name__ == '__main__':
    args = parse_args()
    run_tune(args)