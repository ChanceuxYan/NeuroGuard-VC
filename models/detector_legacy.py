"""
Legacy Detector Implementation (Old Version)
用于兼容使用 ConvTranspose1d 的旧 checkpoint

注意：此文件仅用于加载旧 checkpoint，新训练应使用 detector.py 中的新版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from models.generator import EncoderBlock # 复用生成器的编码块
from models.components.temporal_aggregation import TemporalAggregationModule

class NeuroGuardDetectorLegacy(nn.Module):
    """
    旧版 Detector，使用 ConvTranspose1d 进行上采样
    此版本用于加载旧 checkpoint
    """
    def __init__(self, config):
        super().__init__()
        C = config['model']['generator']['base_channels']
        msg_dim = config['model']['generator']['message_bits']
        
        # 检测器配置
        detector_config = config['model'].get('detector', {})
        use_temporal_aggregation = detector_config.get('use_temporal_aggregation', True)
        aggregation_type = detector_config.get('aggregation_type', 'attention')  # 'attention' 或 'softmax'
        chunk_duration = detector_config.get('chunk_duration', 0.1)  # 0.1秒
        sample_rate = config.get('experiment', {}).get('sample_rate', 16000)
        use_time_freq_dual = detector_config.get('use_time_freq_dual', True)  # 时频双域输入
        
        self.use_time_freq_dual = use_time_freq_dual
        
        # 时频双域输入：梅尔频谱提取器
        if use_time_freq_dual:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                win_length=2048,
                hop_length=256,
                n_mels=80,
                power=1.0
            )
            # 梅尔频谱编码器（1D卷积处理）
            self.mel_encoder = nn.Sequential(
                nn.Conv1d(80, C, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv1d(C, C, kernel_size=3, padding=1),
                nn.ELU()
            )
        
        # 共享特征提取器 (与生成器结构类似，但权重不共享)
        # 如果使用时频双域，输入通道数需要调整
        input_channels = 1 if not use_time_freq_dual else 1  # 波形流仍然是1通道
        self.enc1 = EncoderBlock(input_channels, C, stride=1)
        self.enc2 = EncoderBlock(C, 2*C, stride=2)
        self.enc3 = EncoderBlock(2*C, 4*C, stride=2)
        self.enc4 = EncoderBlock(4*C, 8*C, stride=2)
        self.enc5 = EncoderBlock(8*C, 16*C, stride=2)  # 16x downsampling
        
        # 如果使用时频双域，需要在融合层融合梅尔频谱特征
        if use_time_freq_dual:
            # 融合层：将梅尔频谱特征与声学流特征融合
            self.fusion_layer = nn.Sequential(
                nn.Conv1d(16*C + C, 16*C, kernel_size=3, padding=1),  # 融合后维度
                nn.ELU(),
                nn.Conv1d(16*C, 16*C, kernel_size=3, padding=1),
                nn.ELU()
            )
        
        self.use_temporal_aggregation = use_temporal_aggregation
        
        if use_temporal_aggregation:
            # 使用时序聚合模块（局部解码策略）
            self.temporal_aggregation = TemporalAggregationModule(
                feature_dim=16*C,
                message_bits=msg_dim,
                chunk_duration=chunk_duration,
                sample_rate=sample_rate,
                aggregation_type=aggregation_type
            )
        else:
            # 传统的全局解码（向后兼容）
            self.msg_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16*C, 128),
            nn.ELU(),
                nn.Linear(128, msg_dim)  # 输出 Logits
        )
        
        # 定位头 (Localizing) - 旧版本使用 ConvTranspose1d
        # 需要将特征上采样回原始分辨率
        self.loc_upsample = nn.Sequential(
            nn.ConvTranspose1d(16*C, 8*C, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose1d(8*C, 4*C, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose1d(4*C, 2*C, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose1d(2*C, C, 4, 2, 1), nn.ELU(),
            nn.Conv1d(C, 1, 3, 1, 1)  # 输出 1 Channel 概率图
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, 1, T) 输入音频
            mask: (B, T) 可选掩码，用于忽略被破坏的片段（如静音段）
        
        Returns:
            loc_logits: (B, 1, T) 定位图
            msg_logits: (B, message_bits) 全局消息logits
            local_logits: (B, message_bits, T') 局部消息logits（如果使用时序聚合）
            attention_weights: (B, T') 注意力权重（如果使用attention聚合）
        """
        # 时频双域输入：提取梅尔频谱
        if self.use_time_freq_dual:
            # 提取梅尔频谱
            # x: (B, 1, T) -> mel: (B, n_mels, T_mel) 或 (B, 1, n_mels, T_mel)
            mel = self.mel_transform(x)  # 可能是 (B, n_mels, T_mel) 或 (B, 1, n_mels, T_mel)
            
            # 确保mel是3D格式 (B, n_mels, T_mel)
            if len(mel.shape) == 4:
                # 如果是4D (B, 1, n_mels, T_mel)，移除通道维度
                mel = mel.squeeze(1)  # (B, n_mels, T_mel)
            elif len(mel.shape) == 3 and mel.shape[1] == 1:
                # 如果是 (B, 1, T_mel)，需要转置或处理
                # 但实际上MelSpectrogram应该输出 (B, n_mels, T_mel)
                pass
            
            # 编码梅尔频谱特征（mel现在是(B, n_mels, T_mel)格式）
            mel_features = self.mel_encoder(mel)  # (B, C, T_mel)
            # 下采样梅尔特征到与声学流特征匹配的时间分辨率
            # 声学流经过5层下采样，总下采样率为16x
            # 梅尔频谱的hop_length是256，所以需要下采样到匹配
            target_length = x.size(-1) // 16  # 声学流的特征长度
            if mel_features.size(-1) != target_length:
                mel_features = F.interpolate(
                    mel_features, 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                )
        
        # 声学流：处理波形
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)  # (B, 16*C, T/16)
        
        # 时频双域融合
        if self.use_time_freq_dual:
            # 融合梅尔频谱特征和声学流特征
            fused_features = torch.cat([e5, mel_features], dim=1)  # (B, 16*C + C, T/16)
            e5 = self.fusion_layer(fused_features)  # (B, 16*C, T/16)
        
        # 如果使用掩码，需要下采样到特征分辨率
        feature_mask = None
        if mask is not None:
            # 下采样掩码到特征分辨率
            feature_mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=e5.size(-1),
                mode='nearest'
            ).squeeze(1)  # (B, T')
        
        # 消息解码
        if self.use_temporal_aggregation:
            # 使用时序聚合模块（局部解码策略）
            msg_logits, local_logits, attention_weights = self.temporal_aggregation(
                e5, feature_mask
            )
        else:
            # 传统全局解码
            msg_logits = self.msg_head(e5)
            local_logits = None
            attention_weights = None
        
        # 定位
        loc_logits = self.loc_upsample(e5)
        
        # 修正长度匹配 (由于卷积padding可能导致的微小差异)
        if loc_logits.size(-1) != x.size(-1):
            loc_logits = F.interpolate(loc_logits, size=x.size(-1), mode='linear', align_corners=False)
            
        return loc_logits, msg_logits, local_logits, attention_weights

