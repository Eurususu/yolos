from utils.trtEngine import BaseEngine
import cv2
import argparse

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 80  # your model classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use end2end engine")
    parser.add_argument("--efficient_end2end", default=False, action="store_true", 
                        help='use efficient_end2end engine')
    parser.add_argument("--conf", type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--ultralytics', default=False, action="store_true",
                        help='whether the model is from ultralytics, only for not end2end model')
    parser.add_argument('--v10', action="store_true", help='whether the model is yolov10')

    args = parser.parse_args()
    print(args)
    if args.end2end and args.v10:
        raise NotImplementedError("YOLOv10 is already End2End.")
    pred = Predictor(engine_path=args.engine)
    pred.get_fps()
    img_path = args.image
    video = args.video
    if img_path:
      origin_img = pred.inference(img_path, args)

      cv2.imwrite("%s" %args.output , origin_img)
    if video:
      pred.detect_video(video, args) # set 0 use a webcam
