from ultralytics import YOLO
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset path')
    parser.add_argument('--model', type=str, default='weights/yolo11n.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='number of warmup epoches')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='warmup bias lr')
    parser.add_argument('--batch', type=int, default=16, help='batch size for training')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--project', default='runs/train', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--save', action='store_false', help='save init and last weights')
    parser.add_argument('--save_period', type=int, default=-1, help='save weights every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=bool, default=False, help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, default='auto', help='SGD, Adam, AdamW, NAdam, RAdam, RMSProp')
    parser.add_argument('--workers', type=int, default=8, help='number of dataloader workers')
    # parser.add_argument('--pretrained', type=str, default='', help='use pretrained model weights')
    parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='class loss gain')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--freeze', default='', help='freeze the first N layers of the model or the layers specified by index')
    parser.add_argument('--profile', action='store_true', help='profile model speed while training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--plots', action='store_true', help='save plots of training metrics')
    args = parser.parse_args()
    return args


def run_train(args):
    model = YOLO(args.model)

    model.train(
        data=args.data,model=args.model,epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,warmup_momentum=args.warmup_momentum,warmup_bias_lr=args.warmup_bias_lr,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,lrf=args.lrf,
        momentum=args.momentum,weight_decay=args.weight_decay,
        project=args.project,name=args.name,save=args.save,save_period=args.save_period,
        cache=args.cache,device=args.device,optimizer=args.optimizer,workers=args.workers,
        single_cls=args.single_cls,classes=args.classes,
        box=args.box,cls=args.cls,dfl=args.dfl,
        freeze=args.freeze,profile=args.profile,resume=args.resume,plots=args.plots
    )


if __name__ == '__main__':
    args = parse_args()
    run_train(args)