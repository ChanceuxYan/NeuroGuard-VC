import torch
import torch.nn as nn

class FiLMLayer1D(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for 1D Audio Features.
    Applies gamma(m) * x + beta(m)
    
    [最终修正版] 
    移除针对 FSQ 的激进初始化，回归"恒等映射"初始化。
    这是打通 Generator 梯度的关键。
    """
    def __init__(self, in_channels, msg_dim):
        super().__init__()
        self.in_channels = in_channels
        
        # 两个全连接层分别生成 gamma (缩放) 和 beta (平移)
        self.gamma_fc = nn.Linear(msg_dim, in_channels)
        self.beta_fc = nn.Linear(msg_dim, in_channels)
        
        self._init_weights()
        
    def _init_weights(self):
        """
        初始化策略：接近恒等映射 (Identity Mapping)
        让初始输出 gamma ≈ 1, beta ≈ 0
        这样 Generator 就不会觉得语义特征是"噪音"而将其丢弃。
        """
        # Gamma (Scale): weight=0, bias=1 -> initial output = 1.0
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.constant_(self.gamma_fc.bias, 1.0)
        
        # Beta (Shift): weight=0, bias=0 -> initial output = 0.0
        # 给一个极小的扰动 (1e-4) 打破对称性，但这比之前的 0.5 小了 5000 倍！
        nn.init.normal_(self.beta_fc.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.beta_fc.bias, 0.0)

    def forward(self, x, msg):
        # 1. 生成调制参数
        gamma = self.gamma_fc(msg)
        beta = self.beta_fc(msg)
        
        # 2. 维度对齐
        if x.dim() == 3:
            if x.shape[1] == self.in_channels: # [B, C, T]
                gamma = gamma.unsqueeze(2)
                beta = beta.unsqueeze(2)
            else: # [B, T, C]
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # 3. 应用仿射变换
        return x * gamma + beta