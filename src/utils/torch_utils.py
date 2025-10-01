"""
PyTorch Utilities
=================

PyTorch-specific utilities for model operations and device management.
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Optional, Union, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def time_synchronized():
    """Pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def select_device(device='', batch_size=None):
    """Select device for inference"""
    s = f'YOLOv5 🚀 torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device count
            assert batch_size % n == 0, f'batch-size {batch_size} not divisible by GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024**3:.1f}GB)\n"  # bytes to GB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if hasattr(__import__('platform'), 'system') and __import__('platform').system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda' if cuda else 'cpu')

def model_info(model, verbose=False, img_size=640):
    """Print model information"""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print(f'{i:5g} {name:>40} {p.requires_grad:>9} {p.numel():>12} {str(list(p.shape)):>20} {p.mean():>10.3g} {p.std():>10.3g}')

    try:  # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except:
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a, optionally filtering by include/exclude lists"""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batch normalization layers"""
    # Fuse convolution and batch norm layers
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                   conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """Scale image"""
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = torch.nn.functional.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return torch.nn.functional.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

class ModelEMA:
    """Model Exponential Moving Average"""
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def check_anchor_order(m):
    """Check anchor order"""
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def check_img_size(img_size, s=32):
    """Verify img_size is a multiple of stride s"""
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def make_divisible(x, divisor):
    """Make x divisible by divisor"""
    return int(np.ceil(x / divisor) * divisor)

# Import required modules
import math
from copy import deepcopy
