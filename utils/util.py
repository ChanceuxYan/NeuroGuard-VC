# utils.py
import glob
import os
import torch
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm

def init_weights(m, mean=0.0, std=0.01):
    """初始化模型权重，除了卷积层外"""
    classname = m.__class__.__name__
    if classname.find("Conv")!= -1:
        m.weight.data.normal_(mean, std)

def apply_weight_norm(m):
    """递归应用 Weight Norm"""
    classname = m.__class__.__name__
    if classname.find("Conv")!= -1:
        weight_norm(m)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def load_checkpoint(filepath, device):
    """健壮的检查点加载函数"""
    assert os.path.isfile(filepath), f"Checkpoint path {filepath} does not exist!"
    print(f"Loading checkpoint '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    """自动查找目录下最新的 checkpoint"""
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

class AttrDict(dict):
    """允许通过 dot notation 访问字典属性"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self