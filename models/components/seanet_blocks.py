import torch
import torch.nn as nn
import torch.nn.functional as F

class SEANetResnetBlock1d(nn.Module):
    """
    SEANet 核心残差块：带有空洞卷积的残差连接
    """
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(dim, dim // 2, kernel_size, dilation=dilation, padding=(kernel_size * dilation - dilation) // 2),
            nn.ELU(),
            nn.Conv1d(dim // 2, dim, 1)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class SEANetEncoder(nn.Module):
    """
    SEANet 编码器结构
    """
    def __init__(self, input_channels=1, hidden_dim=32, n_filters=64, n_residual_layers=1, ratios=[1, 2, 3, 4]):
        super().__init__()
        self.ratios = ratios
        self.conv1 = nn.Conv1d(input_channels, n_filters, 7, padding=3)
        self.blocks = nn.ModuleList()
        
        current_dim = n_filters
        for ratio in ratios:
            # Downsample layer
            self.blocks.append(nn.Sequential(
                nn.ELU(),
                nn.Conv1d(current_dim, current_dim * 2, kernel_size=ratio * 2, stride=ratio, padding=ratio // 2)
            ))
            current_dim *= 2
            # Residual layers
            for i in range(n_residual_layers):
                self.blocks.append(SEANetResnetBlock1d(current_dim, dilation=3**i))
                
        self.final_conv = nn.Conv1d(current_dim, hidden_dim, 7, padding=3)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)

# 解码器逻辑是对称的，通常使用 ConvTranspose1d
class SEANetDecoder(nn.Module):
    def __init__(self, output_channels=1, hidden_dim=32, n_filters=64, n_residual_layers=1, ratios=[1, 2, 3, 4]):
        super().__init__()
        # Ratios 倒序
        ratios = ratios[::-1]
        self.initial_conv = nn.Conv1d(hidden_dim, n_filters * (2 ** len(ratios)), 7, padding=3)
        self.blocks = nn.ModuleList()
        
        current_dim = n_filters * (2 ** len(ratios))
        
        for ratio in ratios:
            # Residual layers first
            for i in range(n_residual_layers):
                self.blocks.append(SEANetResnetBlock1d(current_dim, dilation=3**i))
            
            # Upsample layer
            self.blocks.append(nn.Sequential(
                nn.ELU(),
                nn.ConvTranspose1d(current_dim, current_dim // 2, kernel_size=ratio * 2, stride=ratio, padding=ratio // 2)
            ))
            current_dim //= 2
            
        self.final_conv = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(current_dim, output_channels, 7, padding=3),
            nn.Tanh() # 输出限制在 [-1, 1]
        )

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)