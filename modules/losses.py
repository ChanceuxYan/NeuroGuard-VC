# modules/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[50, 120, 240], win_lengths=[240, 600, 1200]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft_size, hop_size, win_length):
        window = torch.hann_window(win_length).to(x.device)
        return torch.stft(x.squeeze(1), fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)

    def forward(self, x_fake, x_real):
        loss_sc = 0 # Spectral Convergence
        loss_mag = 0 # Log Magnitude
        
        for fs, hs, wl in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_fake_stft = self.stft(x_fake, fs, hs, wl)
            x_real_stft = self.stft(x_real, fs, hs, wl)
            
            x_fake_mag = torch.abs(x_fake_stft)
            x_real_mag = torch.abs(x_real_stft)
            
            # Spectral Convergence Loss
            loss_sc += torch.norm(x_real_mag - x_fake_mag, p="fro") / (torch.norm(x_real_mag, p="fro") + 1e-7)
            
            # Log Magnitude Loss
            loss_mag += torch.mean(torch.abs(torch.log(x_real_mag + 1e-7) - torch.log(x_fake_mag + 1e-7)))
            
        return loss_sc + loss_mag


class SemanticConsistencyLoss(nn.Module):
    """
    语义一致性损失
    L_sem = ||HuBERT(X) - HuBERT(X_w)||_2
    
    防止水印嵌入过度破坏语音的语义内容
    """
    def __init__(self, semantic_extractor):
        """
        Args:
            semantic_extractor: SemanticExtractor实例，用于提取语义特征
        """
        super().__init__()
        self.semantic_extractor = semantic_extractor
    
    def forward(self, x_original, x_watermarked):
        """
        Args:
            x_original: (B, 1, T) 原始音频
            x_watermarked: (B, 1, T) 水印音频
        
        Returns:
            loss: 语义一致性损失（标量）
        """
        if self.semantic_extractor is None or self.semantic_extractor.model is None:
            # 如果没有语义提取器，返回0（但需要梯度）
            return torch.tensor(0.0, device=x_original.device, requires_grad=True)
        
        # 提取语义特征
        # 注意：即使语义提取器被冻结，我们仍然需要梯度流
        # 因为我们需要计算 x_watermarked 对损失的梯度
        # 但语义提取器本身的参数不需要更新
        feat_original = self.semantic_extractor(x_original)
        feat_watermarked = self.semantic_extractor(x_watermarked)
        
        # 对齐时间维度（如果不同）
        if feat_original.size(-1) != feat_watermarked.size(-1):
            min_len = min(feat_original.size(-1), feat_watermarked.size(-1))
            feat_original = feat_original[..., :min_len]
            feat_watermarked = feat_watermarked[..., :min_len]
        
        # 计算L2距离
        # 使用detach()分离原始音频的梯度，只保留水印音频的梯度
        # 这样损失只对水印音频有梯度，不会更新原始音频
        loss = F.mse_loss(feat_original.detach(), feat_watermarked)
        
        return loss
