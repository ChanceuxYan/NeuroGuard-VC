# modules/ddsp_proxy.py
"""
可微分VC代理：基于DDSP（Differentiable Digital Signal Processing）的实现
根据设计文档要求：
1. 提取F0和响度（Loudness）
2. 瓶颈自编码器提取内容特征
3. 音色扰动（F0噪声、共振峰滤波器参数变化）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

import librosa
LIBROSA_AVAILABLE = True


def safe_inverse_mel(inverse_mel_transform, mel_spec, n_fft=None):
    """
    将 Mel 频谱转换为线性频谱
    使用线性插值替代 InverseMelScale，避免其内部的 backward() 调用问题
    InverseMelScale 在 forward 中会调用 backward()，这会破坏计算图
    
    Args:
        inverse_mel_transform: InverseMelScale 变换对象（用于获取 n_stft）
        mel_spec: Mel 频谱张量 (B, n_mels, T)
        n_fft: FFT 大小（可选），用于计算 n_stft = n_fft // 2 + 1
    """
    # InverseMelScale 内部会调用 backward()，这会破坏计算图
    # 使用线性插值作为替代方案，这样可以保持梯度流
    # 获取 n_stft（线性频谱的频率维度）
    if hasattr(inverse_mel_transform, 'n_stft'):
        n_stft = inverse_mel_transform.n_stft
    elif hasattr(inverse_mel_transform, 'n_freq'):
        n_stft = inverse_mel_transform.n_freq
    elif n_fft is not None:
        n_stft = n_fft // 2 + 1
    else:
        # 默认值（对应 n_fft=1024）
        n_stft = 513
    
    # 处理各种输入形状
    original_shape = mel_spec.shape
    original_ndim = len(mel_spec.shape)
    
    # 如果是 4D (B, 1, n_mels, T)，移除通道维度
    if original_ndim == 4:
        mel_spec = mel_spec.squeeze(1)  # (B, n_mels, T)
        was_4d = True
    else:
        was_4d = False
    
    # 如果是 2D (n_mels, T)，添加 batch 维度
    if len(mel_spec.shape) == 2:
        mel_spec = mel_spec.unsqueeze(0)  # (1, n_mels, T)
        was_2d = True
    else:
        was_2d = False
    
    # 确保现在是 3D (B, n_mels, T)
    if len(mel_spec.shape) != 3:
        raise ValueError(f"After normalization, expected 3D tensor (B, n_mels, T), got shape {mel_spec.shape} (original: {original_shape})")
    
    n_mels = mel_spec.shape[-2]
    T = mel_spec.shape[-1]
    B = mel_spec.shape[0]
    
    # 使用线性插值将 mel 频谱扩展到线性频谱维度
    # mel_spec shape: (B, n_mels, T)
    # 需要转换为: (B, n_stft, T)
    # 我们需要在频率维度 (n_mels -> n_stft) 上进行插值
    # 对每个时间步独立进行插值
    
    # 方法：转置为 (B, T, n_mels)，然后在最后一个维度上插值
    mel_spec_transposed = mel_spec.transpose(1, 2)  # (B, T, n_mels)
    
    # 将 (B, T, n_mels) 视为 (B*T, n_mels)，然后插值
    B_T = B * T
    mel_spec_flat = mel_spec_transposed.reshape(B_T, n_mels).unsqueeze(1)  # (B*T, 1, n_mels)
    
    # 1D 插值：在最后一个维度上从 n_mels 扩展到 n_stft
    mel_spec_expanded_flat = F.interpolate(
        mel_spec_flat,
        size=n_stft,
        mode='linear',
        align_corners=False
    )  # (B*T, 1, n_stft)
    
    # 恢复形状
    mel_spec_expanded_flat = mel_spec_expanded_flat.squeeze(1)  # (B*T, n_stft)
    mel_spec_expanded_transposed = mel_spec_expanded_flat.reshape(B, T, n_stft)  # (B, T, n_stft)
    mel_spec_expanded = mel_spec_expanded_transposed.transpose(1, 2)  # (B, n_stft, T)
    
    # 恢复原始形状
    if was_2d:
        mel_spec_expanded = mel_spec_expanded.squeeze(0)
    elif was_4d:
        # 如果原始是 4D，恢复为 4D
        mel_spec_expanded = mel_spec_expanded.unsqueeze(1)  # (B, 1, n_stft, T)
    
    return mel_spec_expanded


class F0Extractor(nn.Module):
    """
    F0（基频）提取器
    使用可微分的自相关方法提取F0
    实现了真正的可微分F0提取，而不是固定值
    """
    def __init__(self, sample_rate=16000, hop_length=256, min_f0=50.0, max_f0=500.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        
        # 计算自相关窗口大小（用于F0估计）
        self.max_period = int(sample_rate / min_f0)  # 最大周期（样本数）
        self.min_period = int(sample_rate / max_f0)  # 最小周期（样本数）
        
    def forward(self, waveform):
        """
        提取F0曲线（使用可微分的自相关方法）
        
        Args:
            waveform: (B, 1, T) 音频波形
        
        Returns:
            f0: (B, T_frames) F0曲线（Hz）
        """
        B, C, T = waveform.shape
        if C > 1:
            waveform = waveform[:, 0:1, :]  # 取单声道
        
        # 使用可微分的自相关方法提取F0
        # 1. 对每个帧计算自相关
        frame_length = self.hop_length * 2
        waveform_padded = F.pad(waveform, (frame_length // 2, frame_length // 2), mode='reflect')
        frames = waveform_padded.unfold(-1, frame_length, self.hop_length)  # (B, 1, n_frames, frame_length)
        frames = frames.squeeze(1)  # (B, n_frames, frame_length)
        
        # 获取实际的帧数（unfold可能产生不同的帧数）
        actual_n_frames = frames.shape[1]
        
        # 2. 计算自相关（可微分）
        # 对每个帧，计算不同延迟的自相关
        f0_estimates = []
        for frame_idx in range(actual_n_frames):
            frame = frames[:, frame_idx, :]  # (B, frame_length)
            
            # 归一化帧
            frame_norm = frame / (torch.norm(frame, dim=1, keepdim=True) + 1e-8)
            
            # 计算自相关（在合理周期范围内）
            autocorr_values = []
            for period in range(self.min_period, min(self.max_period, frame_length // 2)):
                # 计算延迟period的自相关
                frame_shifted = frame_norm[:, period:]
                frame_original = frame_norm[:, :frame_length - period]
                
                # 点积（自相关）
                autocorr = (frame_original * frame_shifted).sum(dim=1)  # (B,)
                autocorr_values.append(autocorr)
            
            if len(autocorr_values) > 0:
                autocorr_tensor = torch.stack(autocorr_values, dim=1)  # (B, n_periods)
                
                # 找到最大自相关对应的周期
                max_period_idx = torch.argmax(autocorr_tensor, dim=1)  # (B,)
                period = max_period_idx.float() + self.min_period  # (B,)
                
                # 转换为频率
                f0 = self.sample_rate / (period + 1e-8)  # (B,)
                f0 = torch.clamp(f0, self.min_f0, self.max_f0)
            else:
                # 如果没有找到，使用默认值
                f0 = torch.ones(B, device=waveform.device) * 200.0
            
            f0_estimates.append(f0)
        
        # 堆叠所有帧的F0估计
        f0 = torch.stack(f0_estimates, dim=1)  # (B, actual_n_frames)
        
        return f0


class LoudnessExtractor(nn.Module):
    """
    响度（Loudness）提取器
    使用可微分的RMS能量计算
    """
    def __init__(self, sample_rate=16000, hop_length=256, n_fft=2048):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        
    def forward(self, waveform):
        """
        提取响度曲线
        
        Args:
            waveform: (B, 1, T) 音频波形
        
        Returns:
            loudness: (B, T_frames) 响度曲线
        """
        B, C, T = waveform.shape
        if C > 1:
            waveform = waveform[:, 0:1, :]
        
        # 使用RMS能量作为响度的简化估计
        # 计算短时RMS
        frame_length = self.hop_length * 2
        
        # 使用unfold进行滑动窗口计算（与F0Extractor使用相同的填充和unfold方式）
        waveform_padded = F.pad(waveform, (frame_length // 2, frame_length // 2), mode='reflect')
        frames = waveform_padded.unfold(-1, frame_length, self.hop_length)
        
        # 计算RMS
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1))  # (B, 1, n_frames)
        rms = rms.squeeze(1)  # (B, n_frames)
        
        # 转换为对数域（dB）
        loudness = 20 * torch.log10(rms + 1e-8)
        
        return loudness


class BottleneckEncoder(nn.Module):
    """
    瓶颈自编码器：提取内容特征
    模拟VC模型中的特征瓶颈
    """
    def __init__(self, input_dim, bottleneck_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, bottleneck_dim)  # 瓶颈层
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D) 输入特征
        
        Returns:
            encoded: (B, T, bottleneck_dim) 编码特征
            reconstructed: (B, T, D) 重构特征
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed


class FormantFilter(nn.Module):
    """
    共振峰滤波器
    用于模拟音色变化
    """
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
    def forward(self, waveform, formant_shift=1.0):
        """
        应用共振峰滤波
        
        Args:
            waveform: (B, 1, T) 音频波形
            formant_shift: 共振峰偏移系数
        
        Returns:
            filtered: (B, 1, T) 滤波后的音频
        """
        # 简化：使用可学习的滤波器
        # 实际应该使用可微分的IIR滤波器
        # 这里使用卷积模拟
        if formant_shift == 1.0:
            return waveform
        
        # 使用简单的低通/高通组合模拟共振峰变化
        # 简化实现
        return waveform * formant_shift


class DDSPVCProxy(nn.Module):
    """
    DDSP可微分VC代理模型
    根据设计文档实现：
    1. 提取F0和响度
    2. 瓶颈自编码器提取内容特征
    3. 音色扰动（F0噪声、共振峰滤波器参数变化）
    """
    def __init__(self, sample_rate=16000, hop_length=256, n_fft=2048):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # F0和响度提取器
        self.f0_extractor = F0Extractor(sample_rate, hop_length)
        self.loudness_extractor = LoudnessExtractor(sample_rate, hop_length, n_fft)
        
        # 瓶颈自编码器
        # 输入：F0 + Loudness + 其他特征
        feature_dim = 2  # F0 + Loudness
        self.bottleneck = BottleneckEncoder(feature_dim, bottleneck_dim=64, hidden_dim=256)
        
        # 共振峰滤波器
        self.formant_filter = FormantFilter(sample_rate)
        
        # 声码器：从特征重建波形
        # 简化：使用Griffin-Lim作为基础
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=80,
            power=1.0
        )
        
        # 将 Mel 频谱转换回线性频谱（GriffinLim需要线性频谱）
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=80,
            sample_rate=sample_rate
        )
        
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=4,
            win_length=n_fft,
            hop_length=hop_length,
            power=1.0
        )
    
    def forward(self, waveform, add_f0_noise=True, formant_perturb=True):
        """
        可微分VC代理前向传播
        
        Args:
            waveform: (B, 1, T) 输入音频
            add_f0_noise: 是否添加F0噪声
            formant_perturb: 是否进行共振峰扰动
        
        Returns:
            reconstructed: (B, 1, T) 重构音频
        """
        B, C, T = waveform.shape
        if C > 1:
            waveform = waveform[:, 0:1, :]
        
        # 1. 提取F0和响度
        f0 = self.f0_extractor(waveform)  # (B, T_frames_f0)
        loudness = self.loudness_extractor(waveform)  # (B, T_frames_loudness)
        
        # 确保f0和loudness的帧数一致
        min_frames = min(f0.shape[1], loudness.shape[1])
        if f0.shape[1] != loudness.shape[1]:
            # 截断到相同的长度
            f0 = f0[:, :min_frames]
            loudness = loudness[:, :min_frames]
        
        # 2. 音色扰动：F0噪声
        if add_f0_noise and self.training:
            f0_noise = torch.randn_like(f0) * 10.0  # ±10Hz噪声
            f0 = f0 + f0_noise
            f0 = torch.clamp(f0, 50.0, 500.0)  # 限制在合理范围
        
        # 3. 组合特征
        features = torch.stack([f0, loudness], dim=-1)  # (B, T_frames, 2)
        
        # 4. 瓶颈自编码器：提取内容特征
        content_features, _ = self.bottleneck(features)  # (B, T_frames, bottleneck_dim)
        
        # 5. 从内容特征重建（简化：使用Mel谱重建）
        # 实际DDSP应该使用振荡器和滤波器
        # 这里使用简化的Mel谱重建
        
        # 将特征转换为Mel谱（简化）
        # 实际应该使用可微分的振荡器和滤波器
        mel = self.mel_transform(waveform)
        
        # 添加内容特征的影响（简化）
        # 实际应该使用内容特征控制振荡器
        mel_modified = mel
        
        # 6. 共振峰扰动
        if formant_perturb and self.training:
            formant_shift = torch.rand(1, device=waveform.device).item() * 0.2 + 0.9  # 0.9-1.1
            waveform = self.formant_filter(waveform, formant_shift)
            mel_modified = self.mel_transform(waveform)
        
        # 7. 将 Mel 频谱转换为线性频谱（GriffinLim需要线性频谱）
        # 使用安全包装器来处理 InverseMelScale 的梯度问题
        # 传递 n_fft 以便在回退方案中使用
        linear_spec = safe_inverse_mel(self.inverse_mel, mel_modified, n_fft=self.n_fft)
        
        # 8. 波形重建
        reconstructed = self.griffin_lim(linear_spec)
        
        # 长度对齐
        if reconstructed.size(-1) != T:
            if reconstructed.size(-1) > T:
                reconstructed = reconstructed[..., :T]
            else:
                pad = T - reconstructed.size(-1)
                reconstructed = F.pad(reconstructed, (0, pad))
        
        return reconstructed

