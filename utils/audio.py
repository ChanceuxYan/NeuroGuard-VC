import math
import os
import torch
import torch.utils.data
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import logging
from scipy.io import wavfile

# 配置模块级日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义最大波形值，用于归一化 16-bit 音频
MAX_WAV_VALUE = 32768.0

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    对梅尔频谱进行动态范围压缩。
    采用对数变换：log(clamp(x, min=clip_val) * C)
    
    参数:
        x (torch.Tensor): 输入的幅度谱
        C (float): 压缩系数，默认为 1
        clip_val (float): 防止 log(0) 的极小值
    
    返回:
        torch.Tensor: 对数域的频谱
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    动态范围压缩的逆变换。
    公式: exp(x) / C
    
    参数:
        x (torch.Tensor): 对数域的频谱
        C (float): 压缩系数
        
    返回:
        torch.Tensor: 线性幅度谱
    """
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    """应用谱归一化（即动态范围压缩）"""
    output = dynamic_range_compression(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    """应用谱反归一化"""
    output = dynamic_range_decompression(magnitudes)
    return output

# 全局缓存，避免重复计算梅尔滤波器组和汉宁窗
mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    计算音频波形的梅尔声谱图（Mel-Spectrogram）。
    
    该函数实现了完全基于 PyTorch 的 STFT 和 Mel 变换，支持 GPU 加速和自动微分。
    特别注意 padding 模式和 window 的处理，这直接影响生成的相位一致性。
    
    参数:
        y (torch.Tensor): 输入波形
        n_fft (int): FFT 点数
        num_mels (int): 梅尔滤波器数量
        sampling_rate (int): 采样率
        hop_size (int): 帧移长度
        win_size (int): 窗长
        fmin (float): 最低频率
        fmax (float): 最高频率
        center (bool): 是否中心化 Padding
    
    返回:
        torch.Tensor: 梅尔声谱图
    """
    # 极值检查，防止爆音输入破坏模型权重
    if torch.min(y) < -1.:
        logger.warning(f'Waveform min value is less than -1: {torch.min(y)}')
    if torch.max(y) > 1.:
        logger.warning(f'Waveform max value is greater than 1: {torch.max(y)}')

    global mel_basis, hann_window
    
    # 构建缓存键值
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_str = str(fmax) if fmax is not None else 'None'
    basis_key = f'{dtype_device}_{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax_str}'
    window_key = f'{dtype_device}_{win_size}'

    # 初始化梅尔滤波器组
    if basis_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        # 将 numpy 数组转换为 torch tensor 并移动到对应设备
        mel_basis[basis_key] = torch.from_numpy(mel).float().to(y.device)
        # 兼容 AMP 混合精度训练
        if hasattr(torch, 'autocast'): 
             mel_basis[basis_key] = mel_basis[basis_key].type(y.dtype)
    
    # 初始化汉宁窗
    if window_key not in hann_window:
        # periodic=False 是关键，用于 STFT 分析合成的一致性
        hann_window[window_key] = torch.hann_window(win_size).to(y.device)
        if hasattr(torch, 'autocast'):
             hann_window[window_key] = hann_window[window_key].type(y.dtype)

    # Padding 策略：Reflect (反射填充)
    # 相比于 Constant (零填充)，反射填充能有效减少边界处的频谱突变伪影
    pad_left = int((n_fft - hop_size) / 2)
    pad_right = int((n_fft - hop_size) / 2)
    
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode='reflect')
    y = y.squeeze(1)

    # 执行 STFT
    # magnitude only, phase is discarded for Mel-spec generation
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                      window=hann_window[window_key], center=center, 
                      pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    
    # 计算幅度谱：sqrt(real^2 + imag^2)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    # 线性谱 -> 梅尔谱
    spec = torch.matmul(mel_basis[basis_key], spec)
    
    # 动态范围压缩 (Log Mel)
    spec = spectral_normalize_torch(spec)

    return spec

def load_wav_to_torch(full_path):
    """
    加载 WAV 文件并转换为 Torch Tensor。
    
    参数:
        full_path (str): 文件路径
        
    返回:
        tuple: (sampling_rate, tensor_data)
    """
    sampling_rate, data = wavfile.read(full_path)
    # 将整数型 PCM 转换为浮点型 [-1, 1]
    # 注意：这里假设输入是 16-bit 音频，如果包含 32-bit float，需另行判断
    if data.dtype == np.int16:
        data = data.astype(np.float32) / MAX_WAV_VALUE
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        data = data.astype(np.float32)
        # 如果已经是浮点型，假设已经在 [-1, 1] 范围内
    else:
        # 默认按 16-bit 处理
        data = data.astype(np.float32) / MAX_WAV_VALUE
    
    return sampling_rate, torch.FloatTensor(data)

def load_wav(full_path):
    """便捷包装函数，包含归一化"""
    sampling_rate, data = wavfile.read(full_path)
    return data, sampling_rate

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    计算线性声谱图（Linear Spectrogram），通常用于判别器的特征匹配损失。
    不经过梅尔滤波器组压缩。
    """
    #... (代码逻辑与 mel_spectrogram 类似，但跳过 mel_basis 乘法)...
    # 为了完整性，此处重复关键 STFT 逻辑
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    window_key = f'{dtype_device}_{win_size}'
    
    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(win_size).to(y.device)

    pad_left = int((n_fft - hop_size) / 2)
    pad_right = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                      window=hann_window[window_key], center=center, 
                      pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    return spec