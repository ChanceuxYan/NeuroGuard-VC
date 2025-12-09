import torch
import torch.nn as nn

class FiLMLayer1D(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for 1D Audio Features.
    Applies gamma(m) * x + beta(m)
    
    This implementation includes 'Aggressive Initialization' to overcome
    the 'Dead Zone' problem in FSQ/VQ quantization.
    """
    def __init__(self, in_channels, msg_dim):
        super().__init__()
        self.in_channels = in_channels
        
        # 两个全连接层分别生成 gamma (缩放) 和 beta (平移)
        self.gamma_fc = nn.Linear(msg_dim, in_channels)
        self.beta_fc = nn.Linear(msg_dim, in_channels)
        
        # 应用初始化策略
        self._init_weights()
        
    def _init_weights(self):
        """
        初始化权重。
        
        CRITICAL CHANGE:
        对于 FSQ/VQ 任务，如果 beta 初始化太小（如 std=0.02），产生的位移不足以跨越
        量化边界（Quantization Boundary），导致梯度消失，模型陷入“恒等映射”。
        
        此处采用 Aggressive Initialization：
        - Beta std: 0.5 (足以产生跨越 FSQ 格子的位移)
        - Gamma std: 0.1 (提供足够的缩放动力)
        """
        # --- Gamma (Scale) ---
        # 均值为 1.0 (保持特征幅度)，std 设为 0.1 以引入缩放扰动
        nn.init.normal_(self.gamma_fc.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.gamma_fc.bias, 1.0)
        
        # --- Beta (Shift) ---
        # 均值为 0.0，但 std 增大到 0.5。
        # 这意味着初始的偏移量有很大几率落在 [-0.5, 0.5] 之外，
        # 强行推动特征进入相邻的 FSQ 量化区间。
        nn.init.normal_(self.beta_fc.weight, mean=0.0, std=0.5)
        nn.init.constant_(self.beta_fc.bias, 0.0)

    def forward(self, x, msg):
        """
        Args:
            x: Input features. Shape [Batch, Time, Channels] (e.g., HuBERT features)
               OR [Batch, Channels] if global.
            msg: Watermark message. Shape [Batch, msg_dim]
            
        Returns:
            Modulated features with same shape as x.
        """
        # 1. 生成调制参数 [Batch, Channels]
        gamma = self.gamma_fc(msg)
        beta = self.beta_fc(msg)
        
        # 2. 维度对齐 (Broadcasting)
        if x.dim() == 3:
            if x.shape[1] == self.in_channels:
                # 形状为 [B, C, T]，在时间维上广播
                gamma = gamma.unsqueeze(2)  # [B, C, 1]
                beta = beta.unsqueeze(2)
            else:
                # 假定形状为 [B, T, C]，在时间维前广播
                gamma = gamma.unsqueeze(1)  # [B, 1, C]
                beta = beta.unsqueeze(1)

        # 3. 应用仿射变换
        return x * gamma + beta