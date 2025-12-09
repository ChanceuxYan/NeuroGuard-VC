import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from modules.ddsp_proxy import DDSPVCProxy
from modules.vae_proxy import VAEVCProxy


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
    
    # 使用 1D 插值在频率维度上扩展
    # F.interpolate 对于 1D 插值需要 3D 输入 (B, C, L)
    mel_spec_transposed_3d = mel_spec_transposed.unsqueeze(1)  # (B, 1, T, n_mels) -> 不对，这是 4D
    
    # 重新思考：我们需要对每个时间步的频谱进行插值
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

class DifferentiableRVCProxy(nn.Module):
    """
    模拟 Voice Conversion 的信息丢失过程：
    1. 提取 Mel 频谱 (丢失相位)
    2. 加入特征噪声 (模拟 HuBERT 量化误差)
    3. 通过 Griffin-Lim 或 伪声码器 重建波形
    """
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.n_fft = 1024
        self.n_mels = n_mels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            win_length=1024,
            hop_length=256,
            n_mels=n_mels,
            power=1.0
        )
        # 将 Mel 频谱转换回线性频谱，以便 GriffinLim 使用
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        # 注意: 标准 GriffinLim 在高迭代下反向传播极慢且显存占用大
        # 这里使用低迭代次数的 GriffinLim 或 随机相位重建 作为近似
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            n_iter=4, # 仅需少量迭代模拟相位重建的失真
            win_length=1024,
            hop_length=256,
            power=1.0
        )

    def forward(self, waveform):
        # 1. Mel 变换 (Phase stripping)
        # waveform: -> mel:
        mel = self.mel_transform(waveform)
        
        # 2. 模拟语义提取的量化噪声 (Feature Bottleneck)
        # RVC 的 HuBERT 特征是高度压缩的，这里用加性高斯噪声和 Dropout 模拟信息的丢失
        noise = torch.randn_like(mel) * 0.1
        mel_noisy = mel + noise
        mel_noisy = F.dropout(mel_noisy, p=0.1, training=self.training)
        
        # 3. 将 Mel 频谱转换回线性频谱
        # GriffinLim 需要线性频谱（频率维度为 n_fft // 2 + 1）
        # 使用安全包装器来处理 InverseMelScale 的梯度问题
        # 传递 n_fft 以便在回退方案中使用
        linear_spec = safe_inverse_mel(self.inverse_mel, mel_noisy, n_fft=self.n_fft)
        
        # 4. 波形重构 (Re-synthesis)
        # GriffinLim 在低迭代次数（n_iter=4）下应该是可微分的
        # 但如果它不支持梯度，我们需要确保梯度流不被完全中断
        # 使用 Straight-Through Estimator (STE) 来保持梯度流
        if self.training and linear_spec.requires_grad:
            # 训练模式：使用 GriffinLim，如果它不支持梯度则使用 STE
            reconstructed = self.griffin_lim(linear_spec)
            
            # 检查梯度流：如果 GriffinLim 的输出没有梯度，使用 STE
            if not reconstructed.requires_grad:
                # GriffinLim 不支持梯度，使用 STE 传递梯度
                # 前向传播：使用 GriffinLim 的结果
                reconstructed_ste = reconstructed.detach()
                # 反向传播：使用线性频谱的 IFFT 作为可微分近似
                # 这是一个简化的相位重建，但可以传递梯度
                linear_magnitude = torch.abs(linear_spec)  # (B, n_stft, T)
                # 使用随机相位（可微分）重建波形
                phase = torch.randn_like(linear_magnitude) * 0.1  # 小的随机相位
                linear_spec_complex = torch.complex(linear_magnitude * torch.cos(phase), 
                                                   linear_magnitude * torch.sin(phase))
                reconstructed_grad = torch.fft.irfft(linear_spec_complex, n=self.n_fft, dim=-2)
                # 调整长度
                if reconstructed_grad.size(-1) > waveform.size(-1):
                    reconstructed_grad = reconstructed_grad[..., :waveform.size(-1)]
                elif reconstructed_grad.size(-1) < waveform.size(-1):
                    pad = waveform.size(-1) - reconstructed_grad.size(-1)
                    reconstructed_grad = F.pad(reconstructed_grad, (0, pad))
                # STE: 前向使用 GriffinLim，反向使用可微分近似
                reconstructed = reconstructed_ste + (reconstructed_grad - reconstructed_grad.detach())
        else:
            # 推理模式：直接使用 GriffinLim
            reconstructed = self.griffin_lim(linear_spec)
        
        # 截断或填充以匹配输入长度
        if reconstructed.size(-1) != waveform.size(-1):
            if reconstructed.size(-1) > waveform.size(-1):
                reconstructed = reconstructed[..., :waveform.size(-1)]
            else:
                pad = waveform.size(-1) - reconstructed.size(-1)
                reconstructed = torch.nn.functional.pad(reconstructed, (0, pad))
                
        return reconstructed

class AttackLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 旧的RVC代理（基于Griffin-Lim）
        self.rvc_proxy = DifferentiableRVCProxy()
        
        # 新的DDSP代理（根据设计文档）
        attack_config = config.get('attack', {})
        use_ddsp = attack_config.get('use_ddsp_proxy', True)
        if use_ddsp:
            self.ddsp_proxy = DDSPVCProxy(
                sample_rate=config.get('experiment', {}).get('sample_rate', 16000),
                hop_length=256,
                n_fft=2048
            )
            print("DDSP VC Proxy initialized")
        else:
            self.ddsp_proxy = None
        
        # VAE代理（根据设计文档）
        use_vae = attack_config.get('use_vae_proxy', True)
        if use_vae:
            self.vae_proxy = VAEVCProxy(
                sample_rate=config.get('experiment', {}).get('sample_rate', 16000),
                latent_dim=64
            )
            print("VAE VC Proxy initialized")
        else:
            self.vae_proxy = None
        
        # 攻击概率配置（用于课程学习）
        self.attack_probs = attack_config.get('attack_probs', {
            'no_attack': 0.2,
            'noise': 0.2,
            'scaling': 0.2,
            'masking': 0.2,
            'vc_proxy': 0.2
        })
        
    def forward(self, x, global_step=0, training_stage='stage3'):
        """
        随机应用多种攻击，增强鲁棒性
        
        Args:
            x: (B, 1, T) 或 (B, T) 输入音频
            global_step: 当前训练步数
            training_stage: 训练阶段 ('stage1', 'stage2', 'stage3')
        """
        B = x.shape[0]
        
        # 根据训练阶段调整攻击概率
        if training_stage == 'stage1':
            # 阶段I: 基础建立 - 无攻击或轻微噪声
            attack_prob = torch.rand(1).item()
            if attack_prob < 0.5:
                return x  # 无攻击
            else:
                # 轻微高斯噪声
                noise = torch.randn_like(x) * 0.01
                return x + noise
        
        elif training_stage == 'stage2':
            # 阶段II: 信号鲁棒 - 传统信号处理攻击
            attack_prob = torch.rand(1).item()
            probs = self.attack_probs
            
            if attack_prob < probs['no_attack']:
                return x
            elif attack_prob < probs['no_attack'] + probs['noise']:
                # 高斯噪声
                noise = torch.randn_like(x) * 0.02
                return x + noise
            elif attack_prob < probs['no_attack'] + probs['noise'] + probs['scaling']:
                # 幅度缩放
                if len(x.shape) == 3:
                    scaler = torch.rand(B, 1, 1, device=x.device) * 0.5 + 0.5
                else:
                    scaler = torch.rand(B, 1, device=x.device) * 0.5 + 0.5
                return x * scaler
            elif attack_prob < probs['no_attack'] + probs['noise'] + probs['scaling'] + probs['masking']:
                # 时间裁剪/遮挡
                mask_len = int(x.size(-1) * 0.1)
                start = torch.randint(0, max(1, x.size(-1) - mask_len), (1,)).item()
                # 使用clone()避免inplace操作
                x_masked = x.clone()
                # 使用mask而不是直接赋值，避免inplace操作
                mask = torch.ones_like(x_masked)
                mask[..., start:start+mask_len] = 0
                x_masked = x_masked * mask
                return x_masked
            else:
                # 简单RVC代理（概率较低）
                if global_step > 1000:
                    return self.rvc_proxy(x)
                else:
                    return x
        
        else:  # stage3
            # 阶段III: 攻坚重构 - 逐步引入DDSP Proxy和VAE Proxy
            attack_prob = torch.rand(1).item()
            probs = self.attack_probs
            
            # 计算DDSP/VAE Proxy的使用概率（从0.1逐步增加到0.5）
            proxy_prob_threshold = min(0.1 + (global_step / 10000) * 0.4, 0.5)
            
            if attack_prob < probs['no_attack']:
                return x
            elif attack_prob < probs['no_attack'] + probs['noise']:
                noise = torch.randn_like(x) * 0.02
                return x + noise
            elif attack_prob < probs['no_attack'] + probs['noise'] + probs['scaling']:
                if len(x.shape) == 3:
                    scaler = torch.rand(B, 1, 1, device=x.device) * 0.5 + 0.5
                else:
                    scaler = torch.rand(B, 1, device=x.device) * 0.5 + 0.5
                return x * scaler
            elif attack_prob < probs['no_attack'] + probs['noise'] + probs['scaling'] + probs['masking']:
                mask_len = int(x.size(-1) * 0.1)
                start = torch.randint(0, max(1, x.size(-1) - mask_len), (1,)).item()
                # 使用mask而不是直接赋值，避免inplace操作
                x_masked = x.clone()
                mask = torch.ones_like(x_masked)
                mask[..., start:start+mask_len] = 0
                x_masked = x_masked * mask
                return x_masked
            else:
                # VC代理攻击
                if global_step > 1000:
                    proxy_rand = torch.rand(1).item()
                    if proxy_rand < proxy_prob_threshold:
                        # 在DDSP和VAE之间随机选择
                        if torch.rand(1).item() < 0.5:
                            # 使用DDSP代理
                            if self.ddsp_proxy is not None:
                                return self.ddsp_proxy(x, add_f0_noise=True, formant_perturb=True)
                        else:
                            # 使用VAE代理
                            if self.vae_proxy is not None:
                                return self.vae_proxy(x, add_timbre_perturb=True)
                        # 如果代理都不可用，回退到RVC
                        return self.rvc_proxy(x)
                    else:
                        return self.rvc_proxy(x)
                else:
                    return x
