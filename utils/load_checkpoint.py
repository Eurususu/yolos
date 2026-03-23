import torch
from utils.events import LOGGER
# from my_utils.torch_utils import fuse_model

# def fuse_model(model):
#     '''Fuse convolution and batchnorm layers of the model.'''
#     from models.common import Conv
#     for m in model.modules():
#         # if (type(m) is Conv or type(m) is SimConv or type(m) is Conv_C3) and hasattr(m, "bn"):
#         if (type(m) is Conv ) and hasattr(m, "bn"):
#             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#             delattr(m, "bn")  # remove batchnorm
#             m.forward = m.forward_fuse  # update forward
#     return model

class Wrapper_yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # YOLOv8 的 Detect head 返回 (pred, proto) 或 (pred, [feat1, feat2, feat3])
        # 我们只需要 out[0]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    '''Load model from checkpoint file. '''
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location, weights_only=False)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    # if fuse:
    #     LOGGER.info("\nFusing model...")
    #     model = fuse_model(model).eval()
    # else:
    model = model.eval()
    return model