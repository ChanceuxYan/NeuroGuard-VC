# models/components/temporal_aggregation.py
"""
时序聚合模块：实现局部解码策略
根据设计文档：
- 输出形状为 (T, L) 的概率图，其中 T 是时间步，L 是水印比特长度
- 每0.1秒包含完整的水印信息（冗余编码）
- 使用 Softmax Pooling 或 Attention Pooling 聚合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxPooling(nn.Module):
    """
    Softmax Pooling：对时间维度进行加权聚合
    自动忽略被严重破坏的片段
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits, mask=None):
        """
        Args:
            logits: (B, T, L) 时间-消息概率图
            mask: (B, T) 可选掩码，1表示有效，0表示忽略
        
        Returns:
            aggregated: (B, L) 聚合后的消息logits
        """
        if mask is not None:
            # 应用掩码：被掩码的位置权重为0
            logits = logits * mask.unsqueeze(-1)
        
        # Softmax加权
        weights = F.softmax(logits.mean(dim=-1) / self.temperature, dim=-1)  # (B, T)
        
        # 加权聚合
        aggregated = torch.sum(weights.unsqueeze(-1) * logits, dim=1)  # (B, L)
        
        return aggregated


class AttentionPooling(nn.Module):
    """
    Attention Pooling：使用注意力机制聚合
    可以学习关注哪些时间片段
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = hidden_dim ** -0.5
    
    def forward(self, features, mask=None):
        """
        Args:
            features: (B, T, D) 特征序列
            mask: (B, T) 可选掩码
        
        Returns:
            aggregated: (B, D) 聚合后的特征
            attention_weights: (B, T) 注意力权重
        """
        B, T, D = features.shape
        
        # Self-attention
        Q = self.query(features)  # (B, T, hidden_dim)
        K = self.key(features)    # (B, T, hidden_dim)
        V = self.value(features)  # (B, T, D)
        
        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, T, T)
        
        # 应用掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(B, T, T)  # (B, T, T)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        
        # 聚合
        aggregated = torch.bmm(attention_weights, V)  # (B, T, D)
        
        # 对时间维度求平均
        if mask is not None:
            mask_sum = mask.sum(dim=1, keepdim=True) + 1e-8
            aggregated = (aggregated * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        else:
            aggregated = aggregated.mean(dim=1)  # (B, D)
        
        # 返回聚合特征和注意力权重（用于可视化）
        attention_weights_mean = attention_weights.mean(dim=1)  # (B, T)
        
        return aggregated, attention_weights_mean


class TemporalAggregationModule(nn.Module):
    """
    时序聚合模块：实现局部解码策略
    """
    def __init__(self, feature_dim, message_bits, chunk_duration=0.1, sample_rate=16000, 
                 aggregation_type='attention'):
        """
        Args:
            feature_dim: 输入特征维度
            message_bits: 水印比特长度
            chunk_duration: 每个chunk的时长（秒），默认0.1秒
            sample_rate: 采样率
            aggregation_type: 'softmax' 或 'attention'
        """
        super().__init__()
        self.message_bits = message_bits
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.aggregation_type = aggregation_type
        
        # 计算每个chunk的帧数
        # 假设特征已经下采样，需要根据实际下采样率计算
        # 这里假设特征时间分辨率是原始音频的1/16（与Generator的16x下采样一致）
        self.hop_length = 256  # 与Generator一致
        self.feature_hop = self.hop_length * 16  # 特征层的hop length
        self.chunk_frames = int(chunk_duration * sample_rate / self.feature_hop)
        
        # 局部解码头：每个时间步输出完整的消息logits
        self.local_decoder = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(feature_dim // 2, message_bits, kernel_size=1)
        )  # 输出: (B, message_bits, T)
        
        # 聚合模块
        if aggregation_type == 'attention':
            self.aggregator = AttentionPooling(message_bits)
        else:
            self.aggregator = SoftmaxPooling(temperature=1.0)
        
        # 更好的初始化：让最后一层的输出有更大的初始值
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，让输出logits有更大的初始值"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.out_channels == self.message_bits:
                    # 最后一层：使用更大的初始化，让输出logits有更大的初始值
                    # 使用Xavier初始化，让logits有合理的初始范围
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    # 或者使用更大的标准差
                    # nn.init.normal_(m.weight, mean=0.0, std=0.5)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    # 其他层：标准初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, features, mask=None):
        """
        Args:
            features: (B, C, T) 特征序列（来自编码器）
            mask: (B, T) 可选掩码，用于忽略被破坏的片段
        
        Returns:
            message_logits: (B, message_bits) 聚合后的消息logits
            local_logits: (B, message_bits, T) 局部消息logits（用于可视化）
            attention_weights: (B, T) 注意力权重（如果使用attention）
        """
        B, C, T = features.shape
        
        # 局部解码：每个时间步输出完整的消息logits
        local_logits = self.local_decoder(features)  # (B, message_bits, T)
        
        # 转换为 (B, T, message_bits) 用于聚合
        local_logits_t = local_logits.transpose(1, 2)  # (B, T, message_bits)
        
        # 聚合
        if self.aggregation_type == 'attention':
            message_logits, attention_weights = self.aggregator(local_logits_t, mask)
            return message_logits, local_logits, attention_weights
        else:
            message_logits = self.aggregator(local_logits_t, mask)
            return message_logits, local_logits, None

