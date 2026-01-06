from modelopt.onnx.quantization import quantize
import argparse
import numpy as np
import onnx

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default='', help="onnx path")
    parser.add_argument("--quantize_mode", type=str, default='', help="quantize mode support int8, int4, fp8 etc.")
    parser.add_argument("--calibration_data", type=str, default='', help="calibration data path")
    parser.add_argument("--calib_method", type=str, default='max', help="calibration method support max, entropy, awq_clip, rtn_dq etc.")
    parser.add_argument("--output_path", type=str, default='', help="output path")
    args = parser.parse_args()
    return args
def main(args):
    model = onnx.load(args.onnx_path)
    input_name = model.graph.input[0].name
    print(f"检测到模型输入名称为: {input_name}")

    calib_data = np.load(args.calibration_data)

    calibration_data = {
        input_name: calib_data
    }
    
    quantize(
        onnx_path=args.onnx_path,
        quantize_mode=args.quantize_mode,       # fp8, int8, int4 etc.
        calibration_data=calibration_data,
        calibration_method=args.calib_method,   # max, entropy, awq_clip, rtn_dq etc.
        output_path=args.output_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)