# modules/hard_example_mining.py
"""
Hard Example Mining（困难样本挖掘）
根据设计文档要求：赋予那些导致高BER的攻击更高的权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardExampleMiner:
    """
    困难样本挖掘器
    根据BER（比特错误率）动态调整样本权重
    """
    def __init__(self, top_k_ratio=0.3, min_weight=0.5, max_weight=2.0, momentum=0.9):
        """
        Args:
            top_k_ratio: 选择前k%的高BER样本作为困难样本
            min_weight: 最小权重
            max_weight: 最大权重
            momentum: 权重更新的动量系数
        """
        self.top_k_ratio = top_k_ratio
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.momentum = momentum
        
        # 记录每个样本的历史BER
        self.ber_history = {}
        self.avg_ber = 0.0
    
    def compute_ber(self, predicted, target):
        """
        计算比特错误率（BER）
        
        Args:
            predicted: (B, L) 预测的logits
            target: (B, L) 目标二进制消息
        
        Returns:
            ber: (B,) 每个样本的BER
        """
        # 将logits转换为二进制预测
        pred_binary = (torch.sigmoid(predicted) > 0.5).float()
        
        # 计算错误比特数
        errors = (pred_binary != target).float()
        ber = errors.mean(dim=1)  # (B,)
        
        return ber
    
    def compute_weights(self, ber, sample_ids=None):
        """
        根据BER计算样本权重
        
        Args:
            ber: (B,) 每个样本的BER
            sample_ids: (B,) 可选的样本ID（用于跟踪历史）
        
        Returns:
            weights: (B,) 样本权重
        """
        B = ber.size(0)
        
        # 更新平均BER
        self.avg_ber = self.momentum * self.avg_ber + (1 - self.momentum) * ber.mean().item()
        
        # 计算相对BER（相对于平均BER）
        relative_ber = ber / (self.avg_ber + 1e-8)
        
        # 选择困难样本（高BER）
        k = max(1, int(B * self.top_k_ratio))
        _, top_k_indices = torch.topk(ber, k, dim=0)
        
        # 初始化权重
        weights = torch.ones(B, device=ber.device) * self.min_weight
        
        # 困难样本获得更高权重
        for idx in top_k_indices:
            # 权重与相对BER成正比，但限制在[min_weight, max_weight]范围内
            weight = self.min_weight + (self.max_weight - self.min_weight) * relative_ber[idx].item()
            weight = max(self.min_weight, min(self.max_weight, weight))
            weights[idx] = weight
        
        # 更新历史记录（如果提供了sample_ids）
        if sample_ids is not None:
            for i, sid in enumerate(sample_ids):
                if sid.item() not in self.ber_history:
                    self.ber_history[sid.item()] = []
                self.ber_history[sid.item()].append(ber[i].item())
                # 只保留最近10次记录
                if len(self.ber_history[sid.item()]) > 10:
                    self.ber_history[sid.item()].pop(0)
        
        return weights
    
    def apply_weights_to_loss(self, loss, ber, sample_ids=None):
        """
        将权重应用到损失上
        
        Args:
            loss: (B,) 每个样本的损失
            ber: (B,) 每个样本的BER
            sample_ids: (B,) 可选的样本ID
        
        Returns:
            weighted_loss: (B,) 加权后的损失
            weights: (B,) 应用的权重
        """
        weights = self.compute_weights(ber, sample_ids)
        weighted_loss = loss * weights
        return weighted_loss, weights
    
    def reset(self):
        """重置历史记录"""
        self.ber_history = {}
        self.avg_ber = 0.0

