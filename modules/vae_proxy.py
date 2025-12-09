# modules/vae_proxy.py
"""
VAE可微分VC代理模型
根据设计文档要求，作为DDSP Proxy的补充，使用变分自编码器（VAE）模拟VC攻击
VAE通过编码-解码过程模拟VC模型的特征瓶颈和重构过程
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class VAEEncoder(nn.Module):
    """
    VAE编码器：将音频编码为潜在表示
    模拟VC模型中的Content Encoder
    """
    def __init__(self, input_dim=1, latent_dim=64, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器网络
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU()
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 均值和对数方差（用于VAE的潜在空间）
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, T) 输入音频
        
        Returns:
            mu: (B, latent_dim) 潜在空间均值
            logvar: (B, latent_dim) 潜在空间对数方差
            z: (B, latent_dim) 采样得到的潜在向量
        """
        # 编码
        h = self.encoder(x)  # (B, hidden_dim, T')
        # 全局平均池化
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # (B, hidden_dim)
        
        # 计算均值和方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z


class VAEDecoder(nn.Module):
    """
    VAE解码器：从潜在表示重构音频
    模拟VC模型中的Decoder + Vocoder
    """
    def __init__(self, latent_dim=64, output_dim=1, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # 初始投影
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 16)  # 假设初始长度为16
        
        # 解码器网络（转置卷积）
        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.ConvTranspose1d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ELU()
            ])
            in_dim = hidden_dim
        
        layers.append(nn.ConvTranspose1d(in_dim, output_dim, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z, target_length=None):
        """
        Args:
            z: (B, latent_dim) 潜在向量
            target_length: 目标输出长度
        
        Returns:
            reconstructed: (B, 1, T) 重构音频
        """
        B = z.size(0)
        
        # 投影到初始特征
        h = self.fc(z)  # (B, hidden_dim * 16)
        h = h.view(B, -1, 16)  # (B, hidden_dim, 16)
        
        # 解码
        reconstructed = self.decoder(h)  # (B, 1, T')
        
        # 调整长度
        if target_length is not None and reconstructed.size(-1) != target_length:
            if reconstructed.size(-1) < target_length:
                # 上采样
                reconstructed = F.interpolate(
                    reconstructed, 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                )
            else:
                # 裁剪
                reconstructed = reconstructed[..., :target_length]
        
        return reconstructed


class VAEVCProxy(nn.Module):
    """
    VAE可微分VC代理模型
    使用变分自编码器模拟VC攻击：
    1. 编码器提取内容特征（模拟Content Encoder）
    2. 潜在空间瓶颈（模拟特征压缩）
    3. 解码器重构音频（模拟Decoder + Vocoder）
    4. 添加音色扰动（模拟说话人转换）
    """
    def __init__(self, sample_rate=16000, latent_dim=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        
        # VAE编码器和解码器
        self.encoder = VAEEncoder(input_dim=1, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim, output_dim=1)
        
        # 音色扰动：在潜在空间中添加噪声，模拟说话人转换
        self.timbre_perturb_std = 0.1
    
    def forward(self, waveform, add_timbre_perturb=True):
        """
        可微分VAE VC代理前向传播
        
        Args:
            waveform: (B, 1, T) 输入音频
            add_timbre_perturb: 是否添加音色扰动
        
        Returns:
            reconstructed: (B, 1, T) 重构音频
        """
        B, C, T = waveform.shape
        if C > 1:
            waveform = waveform[:, 0:1, :]
        
        # 1. 编码：提取潜在表示（模拟Content Encoder）
        mu, logvar, z = self.encoder(waveform)  # (B, latent_dim)
        
        # 2. 音色扰动：在潜在空间中添加噪声，模拟说话人转换
        if add_timbre_perturb and self.training:
            timbre_noise = torch.randn_like(z) * self.timbre_perturb_std
            z = z + timbre_noise
        
        # 3. 解码：重构音频（模拟Decoder + Vocoder）
        reconstructed = self.decoder(z, target_length=T)
        
        # 4. 归一化到合理范围
        reconstructed = torch.tanh(reconstructed) * 0.95  # 限制在[-0.95, 0.95]
        
        return reconstructed
    
    def compute_kl_loss(self, mu, logvar):
        """
        计算VAE的KL散度损失（用于训练VAE本身，但在作为代理时通常冻结）
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

