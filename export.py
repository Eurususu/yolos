# coding=utf-8
from utils.load_checkpoint import Wrapper_yolo
import torch
import onnx
import onnx_graphsurgeon as gs
from io import BytesIO
from utils.events import LOGGER
from utils.end2end import End2End, Ultralytics_TRT10_Wrapper
import argparse
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

import subprocess
import os

def convert_onnx_to_fp16_via_cli(input_path, output_path):
    """
    使用 Polygraphy 命令行将 FP32 ONNX 转换为 FP16
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入模型 {input_path}")
        return False

    # 1. 以列表形式构建命令行（推荐格式，防错率高）
    command = [
        "polygraphy", "convert", input_path,
        "--output", output_path,
        "--fp16"
    ]
    
    print(f"🚀 正在执行命令: {' '.join(command)}")
    
    try:
        # 2. 执行命令
        # check=True: 如果命令执行失败（返回码非0），会自动抛出异常
        # text=True, capture_output=True: 以文本形式捕获终端的打印日志
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        print(f"✅ 转换成功！FP16 模型已保存至: {output_path}")
        
        # 如果你想看 Polygraphy 内部打印的具体日志，可以取消下面这行的注释
        # print(result.stdout) 
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ 转换失败！")
        print("💡 错误原因 (stderr):")
        print(e.stderr)  # 打印具体的报错信息
        return False
    except FileNotFoundError:
        print("❌ 错误：找不到 polygraphy 命令，请确认是否已安装并在环境变量中。")
        return False

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='weights path')
    parser.add_argument("--qat_model", type=str, default='', help='qat weights path' )
    parser.add_argument('--end2end_model', action='store_true', help='whether the model is end2end')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--topk_all', type=int, default=300, help='max number of detections per image')
    parser.add_argument('--iou_thres', type=float, default=0.7, help='iou threshold for NMS')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold for NMS')
    parser.add_argument('--dynamic_batch', action='store_true', help='whether to export dynamic batch size')
    parser.add_argument('--end2end', action='store_true', help='whether to export end2end model')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640,640], help='height and width of the input image')
    parser.add_argument('--device', default='cpu', help='device to use for export')
    parser.add_argument('--opset', type=int, default=19, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='whether to simplify onnx model using onnxsim')
    parser.add_argument('--seg', action='store_true', help='whether to export segmentation model')
    parser.add_argument("--fp16", action='store_true', help="whether to convert the model to FP16 after export")
    opt = parser.parse_args()
    return opt

def run_export(opt):
    device = torch.device(opt.device)
    LOGGER.info("Loading model...")
    # model = load_checkpoint(opt.weights,ultralytics=opt.ultralytics, map_location=device)
    if opt.end2end and opt.end2end_model:
        raise NotImplementedError("End2End export for end2end model is not supported.")
    # model = YOLO(opt.model).model
    if opt.qat_model:
        LOGGER.info(f"🧱 1. Loading base architecture from: {opt.model}")
        # 先加载原版模型把骨架撑起来
        model = YOLO(opt.model).model
        model = model.fuse()
        
        LOGGER.info(f"👻 2. Restoring QAT weights and QDQ nodes from: {opt.qat_model}")
        # 导入 ModelOpt 恢复量化权重
        import modelopt.torch.opt as mto
        mto.restore(model, opt.qat_model)
    else:
        # 如果没有传 base_model，说明是导出普通的 YOLO 模型
        LOGGER.info(f"📦 Loading standard YOLO model from: {opt.model}")
        model = YOLO(opt.model).model

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True 
    model = Wrapper_yolo(model)
    model = model.to(device)
    model.eval()
    if len(opt.imgsz) == 1:
        opt.imgsz = [opt.imgsz[0], opt.imgsz[0]]
    img = torch.randn(opt.batch, 3, opt.imgsz[0], opt.imgsz[1]).to(device)
    dynamic_axes = None
    if opt.dynamic_batch:
        dynamic_axes = {
            'images': {
                0: 'batch',
            }, }
        output_axes = {
            'outputs': {0: 'batch'},
        }
        dynamic_axes.update(output_axes)
    if opt.end2end:
        LOGGER.info("Adding End2End (NMS) layers...")
        # model = End2End(model, ultralytics=opt.ultralytics, max_obj=opt.topk_all, iou_thres=opt.iou_thres, score_thres=opt.conf_thres,
        #                 device=device, ort=False, with_preprocess=False)
        model = Ultralytics_TRT10_Wrapper(model, opt.topk_all, opt.iou_thres, opt.conf_thres, opt.seg)
        if opt.dynamic_batch:
            current_dynamic_axes = {
                'images': {0: 'batch'},
                'detections': {0: 'num_dets'}  # 输出的第一维是动态的(检测框数量)
            }
        else:
            current_dynamic_axes = {'detections': {0: 'num_dets'}}
        dynamic_axes = current_dynamic_axes
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = opt.model.replace('.pt', '.onnx')  # filename
        output_names = ['detections'] if opt.end2end else ['outputs']
        with BytesIO() as f:
            torch.onnx.export(model, img, f, verbose=False, opset_version=opt.opset,
                            training=torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=True,
                            input_names=['images'],
                            dynamo=False,
                            output_names=output_names, 
                            dynamic_axes=dynamic_axes)
            f.seek(0)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            if opt.simplify:
                LOGGER.info("Simplifying with onnx-simplifier...")
                try:
                    import onnxsim
                    model_simp, check = onnxsim.simplify(onnx_model)
                    if check:
                        onnx_model = model_simp
                        LOGGER.info("Simplification successful.")
                    else:
                        LOGGER.warning("Simplification check failed. Saving unsimplified model.")
                except Exception as e:
                    LOGGER.warning(f"Simplification process error: {e}")

            LOGGER.info("Optimizing graph with onnx-graphsurgeon...")
            graph = gs.import_onnx(onnx_model)
            graph.cleanup().toposort()  #从图形中删除未使用的节点和张量，并对图形进行拓扑排序

            # Shape Estimation
            final_model = None
            try:
                # 即使是大模型，使用 export_onnx 生成 proto 也可能比较安全，但 infer_shapes 偶尔会失败
                final_model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
            except Exception as e:
                LOGGER.warning(f"Shape inference failed, saving without updated shapes: {e}")
                final_model = gs.export_onnx(graph)
            

            onnx.save(final_model, export_file)
            LOGGER.info(f'FP32 ONNX export success: {export_file}')

            if opt.fp16:
                LOGGER.info("Converting ONNX model to FP16 using Polygraphy CLI...")
                fp16_export_file = export_file.replace('.onnx', '_fp16.onnx')
                success = convert_onnx_to_fp16_via_cli(export_file, fp16_export_file)
                if success:
                    export_file = fp16_export_file  # 更新为 FP16 模型路径
                    LOGGER.info(f"FP16 ONNX export success: {export_file}")
                else:
                    LOGGER.warning("FP16 conversion failed, keeping original FP32 model.")
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')
        raise e



if __name__ == "__main__":
    opt = parse_opt()
    run_export(opt)
