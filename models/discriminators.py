import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

def get_padding(kernel_size, dilation=1):
    """计算 Padding 以保持卷积输出的时间维度不变 (Same Padding)"""
    return int((kernel_size*dilation - dilation)/2)

class DiscriminatorP(torch.nn.Module):
    """
    多周期判别器 (MPD) 子模块。
    原理：将 1D 音频重塑为 2D 矩阵，从而利用 2D 卷积捕获周期 P 下的纹理特征。
    """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        # 谱归一化能显著稳定 GAN 的训练，防止判别器过强
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # 卷积层堆叠
        # 注意：输入通道为 1，但在 reshape 后，第二维作为高度，第三维作为宽度
        # 这里的 Kernel (5, 1) 实际上是在时间轴上进行卷积，而在周期轴上保持独立
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))), 
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []  # Fix: initialize list

        # 1D -> 2D 重塑
        # 如果长度不能被周期整除，进行反射填充
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        # 关键步骤：视图变换
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1) # LRELU_SLOPE = 0.1
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    """
    MPD 容器：包含多个针对不同质数周期的 DiscriminatorP。
    """
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        # 质数周期集合，避免谐波重叠
        periods = [6, 1, 5, 2, 3]
        self.discriminators = nn.ModuleList()
        # Fix: initialize discriminators
        for period in periods:
            self.discriminators.append(DiscriminatorP(period, use_spectral_norm=use_spectral_norm))

    def forward(self, y, y_hat):
        y_d_rs = []  # Fix: initialize lists
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    """
    尺度判别器 (Scale Discriminator) 子模块。
    使用大感受野的 1D 卷积，直接分析波形结构。
    """
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # 卷积层配置
        # Grouped Convolution (groups>1) 用于减少参数量，同时保持大卷积核
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []  # Fix: initialize list
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiScaleDiscriminator(torch.nn.Module):
    """
    多尺度判别器 (MSD)。
    包含 3 个 DiscriminatorS，分别处理：
    1. 原始音频 (1x)
    2. 2倍下采样音频 (0.5x)
    3. 4倍下采样音频 (0.25x)
    
    之前版本省略了 meanpools 的具体实现，导致逻辑不完整。
    """
    def __init__(self, use_spectral_norm=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        # Fix: initialize discriminators (3 scales)
        for _ in range(3):
            self.discriminators.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))
        # 显式定义平均池化层用于下采样
        # Kernel=4, Stride=2, Padding=2 模拟了平滑的下采样过程
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []  # Fix: initialize lists
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            if i != 0:
                # 级联下采样：scale 2 是 scale 1 的下采样，scale 3 是 scale 2 的下采样
                # 这种级联结构构成了特征金字塔
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs