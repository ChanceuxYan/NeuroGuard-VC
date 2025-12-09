import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from models.generator import EncoderBlock # 复用生成器的编码块
from models.components.temporal_aggregation import TemporalAggregationModule

from models.components.semantic_extractor import SemanticExtractor

class NeuroGuardDetector(nn.Module):
    """
    语义增强型检测器 (Semantic-Aware Detector)
    
    架构变革：
    原架构：Waveform -> SEANet/CNN -> Bits
    新架构：Waveform -> Frozen HuBERT (Semantic) -> Lightweight Head -> Bits
    
    优势：与 Generator 在同一个语义空间对话，极大降低学习难度。
    """
    def __init__(self, config):
        super().__init__()
        
        # 1. 语义提取器 (与 Generator 保持一致的配置)
        # 必须 freeze=True，否则显存爆炸且难以训练
        semantic_config = config['model']['generator']['semantic']
        self.semantic_extractor = SemanticExtractor(
            model_type=semantic_config.get('model_type', 'hubert'),
            model_name=semantic_config.get('model_name', None),
            freeze=True 
        )
        
        # 获取语义特征维度 (例如 HuBERT-Large=1024, Base=768)
        sem_dim = self.semantic_extractor.get_feature_dim()
        msg_bits = config['model']['generator']['message_bits']
        
        # 2. 轻量级解码头 (Decoder Head)
        # 只需要简单的几层卷积就能从语义特征中提取 FSQ 的痕迹
        # 结构：降维 -> 时序聚合 -> 最终分类
        self.decoder_head = nn.Sequential(
            # Layer 1: 语义特征降维与初步整合
            nn.Conv1d(sem_dim, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            
            # Layer 2: 进一步特征提取
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            
            # Layer 3: 映射到消息空间
            nn.Conv1d(256, msg_bits, kernel_size=1)
        )
        
        # 3. 定位头 (Localization Head) - 可选，用于辅助任务
        # 输出 (B, 1, T) 的掩码，指示哪里有水印
        self.loc_head = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) 音频波形
        Returns:
            loc_logits: (B, 1, T_sem) 定位 logits
            msg_logits: (B, Bits) 全局消息 logits
        """
        # 1. 提取语义特征
        # 注意：虽然 extractor 是冻结的，但 x.requires_grad=True，
        # 所以梯度可以穿过 extractor 回传给 x，从而指导 Generator。
        # features: (B, sem_dim, T_sem)
        features = self.semantic_extractor(x)
        
        # 2. 解码过程
        # 中间特征
        mid_feats = self.decoder_head[0](features)
        mid_feats = self.decoder_head[1](mid_feats)
        mid_feats = self.decoder_head[2](mid_feats) # mid_feats 此时是 Layer 2 的输出
        
        # 为了复用 sequential，我们手动拆解一下或者重新定义 forward 逻辑
        # 这里为了简单，我们重新定义一下流向：
        
        h = features
        h = self.decoder_head[0](h)
        h = self.decoder_head[1](h) # (B, 256, T_sem)
        
        # 分支 A: 消息解码 (Message Decoding)
        # (B, 256, T) -> (B, Bits, T)
        token_logits = self.decoder_head[2](h)
        
        # 全局聚合：对时间维度取平均 (Global Average Pooling)
        # 假设水印是全局重复或全局分布的
        msg_logits = token_logits.mean(dim=-1) # (B, Bits)
        
        # 分支 B: 定位 (Localization)
        loc_logits = self.loc_head(h) # (B, 1, T_sem)
        
        # 兼容旧代码的返回值格式 (loc, msg, local, attention)
        # 我们这里不需要 local 和 attention，返回 None
        return loc_logits, msg_logits, None, None

# class NeuroGuardDetector(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         C = config['model']['generator']['base_channels']
#         msg_dim = config['model']['generator']['message_bits']
        
#         # 检测器配置
#         detector_config = config['model'].get('detector', {})
#         use_temporal_aggregation = detector_config.get('use_temporal_aggregation', True)
#         aggregation_type = detector_config.get('aggregation_type', 'attention')  # 'attention' 或 'softmax'
#         chunk_duration = detector_config.get('chunk_duration', 0.1)  # 0.1秒
#         sample_rate = config.get('experiment', {}).get('sample_rate', 16000)
#         use_time_freq_dual = detector_config.get('use_time_freq_dual', True)  # 时频双域输入
        
#         self.use_time_freq_dual = use_time_freq_dual
        
#         # 时频双域输入：梅尔频谱提取器
#         if use_time_freq_dual:
#             mel_bins = detector_config.get('n_mels', config.get('data', {}).get('n_mel_channels', 80))
#             self.mel_transform = torchaudio.transforms.MelSpectrogram(
#                 sample_rate=sample_rate,
#                 n_fft=2048,
#                 win_length=2048,
#                 hop_length=256,
#                 n_mels=mel_bins,
#                 power=1.0
#             )
#             # mel_transform 仅作为固定前处理，显式冻结
#             for p in self.mel_transform.parameters():
#                 p.requires_grad = False
#             # 梅尔频谱编码器（1D卷积处理）
#             self.mel_encoder = nn.Sequential(
#                 nn.Conv1d(mel_bins, C, kernel_size=3, padding=1),
#                 nn.ELU(),
#                 nn.Conv1d(C, C, kernel_size=3, padding=1),
#                 nn.ELU()
#             )
        
#         # 共享特征提取器 (与生成器结构类似，但权重不共享)
#         # 如果使用时频双域，输入通道数需要调整
#         input_channels = 1 if not use_time_freq_dual else 1  # 波形流仍然是1通道
#         self.enc1 = EncoderBlock(input_channels, C, stride=1)
#         self.enc2 = EncoderBlock(C, 2*C, stride=2)
#         self.enc3 = EncoderBlock(2*C, 4*C, stride=2)
#         self.enc4 = EncoderBlock(4*C, 8*C, stride=2)
#         self.enc5 = EncoderBlock(8*C, 16*C, stride=2)  # 16x downsampling
        
#         # 如果使用时频双域，需要在融合层融合梅尔频谱特征
#         if use_time_freq_dual:
#             # 融合层：将梅尔频谱特征与声学流特征融合
#             self.fusion_layer = nn.Sequential(
#                 nn.Conv1d(16*C + C, 16*C, kernel_size=3, padding=1),  # 融合后维度
#                 nn.ELU(),
#                 nn.Conv1d(16*C, 16*C, kernel_size=3, padding=1),
#                 nn.ELU()
#             )
        
#         self.use_temporal_aggregation = use_temporal_aggregation
        
#         if use_temporal_aggregation:
#             # 使用时序聚合模块（局部解码策略）
#             self.temporal_aggregation = TemporalAggregationModule(
#                 feature_dim=16*C,
#                 message_bits=msg_dim,
#                 chunk_duration=chunk_duration,
#                 sample_rate=sample_rate,
#                 aggregation_type=aggregation_type
#             )
#         else:
#             # 传统的全局解码（向后兼容）
#             self.msg_head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Linear(16*C, 128),
#             nn.ELU(),
#                 nn.Linear(128, msg_dim)  # 输出 Logits
#         )
        
#         # 定位头 (Localizing)
#         # 需要将特征上采样回原始分辨率
#         # 使用 UpSample + Conv 以减少棋盘效应
#         self.loc_upsample = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(16*C, 8*C, 3, 1, 1), nn.ELU(),
#             nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(8*C, 4*C, 3, 1, 1), nn.ELU(),
#             nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(4*C, 2*C, 3, 1, 1), nn.ELU(),
#             nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(2*C, C, 3, 1, 1), nn.ELU(),
#             nn.Conv1d(C, 1, 3, 1, 1)  # 输出 1 Channel 概率图
#         )

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: (B, 1, T) 输入音频
#             mask: (B, T) 可选掩码，用于忽略被破坏的片段（如静音段）
        
#         Returns:
#             loc_logits: (B, 1, T) 定位图
#             msg_logits: (B, message_bits) 全局消息logits
#             local_logits: (B, message_bits, T') 局部消息logits（如果使用时序聚合）
#             attention_weights: (B, T') 注意力权重（如果使用attention聚合）
#         """
#         # 时频双域输入：提取梅尔频谱
#         if self.use_time_freq_dual:
#             # 提取梅尔频谱
#             # x: (B, 1, T) -> mel: (B, n_mels, T_mel) 或 (B, 1, n_mels, T_mel)
#             mel = self.mel_transform(x)  # 可能是 (B, n_mels, T_mel) 或 (B, 1, n_mels, T_mel)
            
#             # 确保mel是3D格式 (B, n_mels, T_mel)
#             if len(mel.shape) == 4:
#                 # 如果是4D (B, 1, n_mels, T_mel)，移除通道维度
#                 mel = mel.squeeze(1)  # (B, n_mels, T_mel)
#             elif len(mel.shape) == 3 and mel.shape[1] == 1:
#                 # 如果是 (B, 1, T_mel)，需要转置或处理
#                 # 但实际上MelSpectrogram应该输出 (B, n_mels, T_mel)
#                 pass
            
#             # 编码梅尔频谱特征（mel现在是(B, n_mels, T_mel)格式）
#             mel_features = self.mel_encoder(mel)  # (B, C, T_mel)
#             # 下采样梅尔特征到与声学流特征匹配的时间分辨率
#             # 声学流经过5层下采样，总下采样率为16x
#             # 梅尔频谱的hop_length是256，所以需要下采样到匹配
#             target_length = x.size(-1) // 16  # 声学流的特征长度
#             if mel_features.size(-1) != target_length:
#                 mel_features = F.interpolate(
#                     mel_features, 
#                     size=target_length, 
#                     mode='linear', 
#                     align_corners=False
#                 )
        
#         # 声学流：处理波形
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)  # (B, 16*C, T/16)
        
#         # 时频双域融合
#         if self.use_time_freq_dual:
#             # 融合梅尔频谱特征和声学流特征
#             fused_features = torch.cat([e5, mel_features], dim=1)  # (B, 16*C + C, T/16)
#             e5 = self.fusion_layer(fused_features)  # (B, 16*C, T/16)
        
#         # 如果使用掩码，需要下采样到特征分辨率
#         feature_mask = None
#         if mask is not None:
#             # 下采样掩码到特征分辨率
#             feature_mask = F.interpolate(
#                 mask.unsqueeze(1).float(),
#                 size=e5.size(-1),
#                 mode='nearest'
#             ).squeeze(1)  # (B, T')
        
#         # 消息解码
#         if self.use_temporal_aggregation:
#             # 使用时序聚合模块（局部解码策略）
#             msg_logits, local_logits, attention_weights = self.temporal_aggregation(
#                 e5, feature_mask
#             )
#         else:
#             # 传统全局解码
#             msg_logits = self.msg_head(e5)
#             local_logits = None
#             attention_weights = None
        
#         # 定位
#         loc_logits = self.loc_upsample(e5)
        
#         # 修正长度匹配 (由于卷积padding可能导致的微小差异)
#         if loc_logits.size(-1) != x.size(-1):
#             loc_logits = F.interpolate(loc_logits, size=x.size(-1), mode='linear', align_corners=False)
            
#         return loc_logits, msg_logits, local_logits, attention_weights
