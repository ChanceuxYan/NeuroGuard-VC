import torch
import torch.nn as nn
from models.components.semantic_extractor import SemanticExtractor

class NeuroGuardDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. 语义提取器
        semantic_config = config['model']['generator']['semantic']
        self.semantic_extractor = SemanticExtractor(
            model_type=semantic_config.get('model_type', 'hubert'),
            model_name=semantic_config.get('model_name', None),
            freeze=True,
            unfreeze_last_n_layers=2 
        )
        
        sem_dim = self.semantic_extractor.get_feature_dim()
        msg_bits = config['model']['generator']['message_bits']
        
        # 2. 解码头 - 分层定义，避免通道错误
        self.head_layer1 = nn.Sequential(
            nn.Conv1d(sem_dim, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU()
        )
        self.head_layer2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU()
        )
        self.head_out = nn.Conv1d(256, msg_bits, kernel_size=1)
        
        # 3. 定位头
        self.loc_head = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1)
        )
        
        # 兼容性保留
        self.decoder_head = nn.ModuleList([self.head_layer1, self.head_layer2, self.head_out])

    def forward(self, x):
        # 1. 提取语义 (Extractor 内部已处理归一化)
        features = self.semantic_extractor(x)
        
        # 2. 逐层传递 (修复了之前的通道不匹配Bug)
        x1 = self.head_layer1(features) # -> 512
        x2 = self.head_layer2(x1)       # -> 256
        
        # 3. 输出
        loc_logits = self.loc_head(x2)
        token_logits = self.head_out(x2)
        msg_logits = token_logits.mean(dim=-1)
        
        return loc_logits, msg_logits, None, None