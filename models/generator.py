# models/generator.py
"""
语义增强型编码器 (Semantic-Aware Encoder)
实现双流架构：声学流 + 语义流
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.film import FiLMLayer1D
from models.components.semantic_extractor import SemanticExtractor
from modules.fsq import FSQ

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        # 使用较大的卷积核和步长进行下采样
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.ELU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        # 转置卷积进行上采样
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, output_padding=stride-1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.ELU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class FusionBlock(nn.Module):
    """
    双流融合块：将语义特征与声学特征融合
    """
    def __init__(self, semantic_dim, acoustic_dim, out_dim):
        super().__init__()
        # 上采样语义特征到与声学特征相同的时间分辨率
        self.semantic_upsample = nn.ConvTranspose1d(
            semantic_dim, semantic_dim, 
            kernel_size=4, stride=2, padding=1
        )
        # 融合层：拼接后通过卷积融合
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(semantic_dim + acoustic_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_dim),
            nn.ELU()
        )
    
    def forward(self, semantic_feat, acoustic_feat):
        """
        Args:
            semantic_feat: (B, C_sem, T_sem) 语义特征
            acoustic_feat: (B, C_ac, T_ac) 声学特征
        
        Returns:
            fused: (B, out_dim, T_ac) 融合后的特征
        """
        # 上采样语义特征到声学特征的时间分辨率
        if semantic_feat.size(-1) != acoustic_feat.size(-1):
            semantic_feat = F.interpolate(
                semantic_feat, 
                size=acoustic_feat.size(-1), 
                mode='linear', 
                align_corners=False
            )
        
        # 如果还需要进一步上采样（语义特征可能比声学特征小很多）
        if semantic_feat.size(-1) < acoustic_feat.size(-1):
            # 多次上采样
            while semantic_feat.size(-1) < acoustic_feat.size(-1):
                semantic_feat = self.semantic_upsample(semantic_feat)
                if semantic_feat.size(-1) > acoustic_feat.size(-1):
                    # 裁剪到正确长度
                    semantic_feat = semantic_feat[..., :acoustic_feat.size(-1)]
                    break
        
        # 拼接
        fused = torch.cat([semantic_feat, acoustic_feat], dim=1)
        
        # 融合
        fused = self.fusion_conv(fused)
        
        return fused

class NeuroGuardGenerator(nn.Module):
    """
    语义增强型编码器
    双流架构：
    - 声学流：U-Net处理原始波形
    - 语义流：HuBERT/Wav2Vec提取语义特征，FiLM调制，FSQ离散化
    - 融合：量化后的语义特征与声学流特征拼接
    
    设计逻辑变更：
    原逻辑：HuBERT -> FiLM -> Upsample
    新逻辑：HuBERT -> FiLM -> FSQ (Bottleneck) -> Upsample
    """
    def __init__(self, config):
        super().__init__()
        C = config['model']['generator']['base_channels']
        msg_dim = config['model']['generator']['message_bits']
        
        # 强度控制系数
        self.alpha = config['model']['generator'].get('alpha', 0.1)
        
        # ========== 语义流 ==========
        semantic_config = config['model']['generator'].get('semantic', {})
        use_semantic = semantic_config.get('enabled', True)
        semantic_model_type = semantic_config.get('model_type', 'hubert')
        semantic_model_name = semantic_config.get('model_name', None)
        
        self.use_semantic = use_semantic
        if use_semantic:
            self.semantic_extractor = SemanticExtractor(
                model_type=semantic_model_type,
                model_name=semantic_model_name,
                freeze=True  # 冻结预训练模型
            )
            semantic_dim = self.semantic_extractor.get_feature_dim()
        else:
            self.semantic_extractor = None
            semantic_dim = 1024  # 默认维度（Large模型通常是1024维）
        
        # FiLM层：作用在语义特征上 [cite: 57]
        # 注意：FiLM 作用于 input_dim (768/1024)，在压缩前注入水印
        if use_semantic:
            self.semantic_film = FiLMLayer1D(semantic_dim, msg_dim)
        
        # FSQ 瓶颈层 (新增核心组件) [cite: FSQ论文]
        # dim=semantic_dim: 告诉 FSQ 输入是 768/1024 维
        # levels=[8,8,8,8,8]: 告诉 FSQ 内部压缩到 5 维 (len(levels))
        # FSQ 代码会自动创建 semantic_dim -> 5 的 project_in 和 5 -> semantic_dim 的 project_out
        if use_semantic:
            fsq_levels = semantic_config.get('fsq_levels', [8, 8, 8, 8, 8])  # 默认5维，每维8级
            self.fsq = FSQ(
                levels=fsq_levels,
                dim=semantic_dim,  # 输入维度：HuBERT的输出维度
                num_codebooks=1
            )
        else:
            self.fsq = None
        
        # ========== 声学流 ==========
        # 编码器路径: 降采样 16k -> latent
        self.enc1 = EncoderBlock(1, C, stride=1)
        self.enc2 = EncoderBlock(C, 2*C, stride=2)
        self.enc3 = EncoderBlock(2*C, 4*C, stride=2)
        self.enc4 = EncoderBlock(4*C, 8*C, stride=2)
        self.enc5 = EncoderBlock(8*C, 16*C, stride=2)  # 16x downsampling total
        
        # ========== 融合层 ==========
        # 在不同层级融合语义和声学特征
        if use_semantic:
            # 在瓶颈层融合
            self.fusion_bottleneck = FusionBlock(semantic_dim, 16*C, 16*C)
            # 在中间层也可以融合（可选）
            self.fusion_mid = FusionBlock(semantic_dim, 8*C, 8*C)
        else:
            # 如果没有语义流，不创建融合层
            self.fusion_bottleneck = None
            self.fusion_mid = None
        
        # LSTM（可选，用于时间连贯性）
        use_lstm = config['model']['generator'].get('use_lstm', False)
        if use_lstm:
            self.lstm = nn.LSTM(16*C, 16*C, num_layers=2, batch_first=True)
        else:
            self.lstm = None
        
        # ========== 解码器路径 ==========
        # 上采样 latent -> 16k
        self.dec5 = DecoderBlock(16*C, 8*C, stride=2)
        self.dec4 = DecoderBlock(8*C, 4*C, stride=2)
        self.dec3 = DecoderBlock(4*C, 2*C, stride=2)
        self.dec2 = DecoderBlock(2*C, C, stride=2)
        
        # 最终输出层: 生成水印残差 (Residual)
        self.final_conv = nn.Conv1d(C, 1, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()  # 限制幅度

    def forward(self, x, msg):
        """
        Args:
            x: (B, 1, T) 原始音频
            msg: (B, message_bits) 水印消息
        
        Returns:
            watermarked_audio: (B, 1, T) 水印音频
            watermark: (B, 1, T) 水印残差信号
            indices: (B, T_sem) FSQ量化索引（可选，用于监控）
        """
        # Helper: align two temporal tensors by center-cropping to the shorter length
        def _align_pair(a, b):
            ta, tb = a.size(-1), b.size(-1)
            if ta == tb:
                return a, b
            target = min(ta, tb)
            return a[..., :target], b[..., :target]

        # ========== 语义流 ==========
        if self.use_semantic and self.semantic_extractor is not None:
            # 1. 提取语义特征: F_sem = HuBERT(X) [cite: 56]
            with torch.no_grad():
                F_sem = self.semantic_extractor(x)  # (B, C_sem, T_sem)
            
            # 2. 注入水印 (FiLM): F̂_sem = γ(M) ⊙ F_sem + β(M) [cite: 58]
            # 这一步将水印 "画" 在高维语义空间中
            F_sem_modulated = self.semantic_film(F_sem, msg)  # (B, C_sem, T_sem)
            
            # 3. FSQ 离散化 (关键修改) [cite: FSQ论文]
            # F_modulated 被投影到低维 -> 量化 -> 投影回 semantic_dim
            # 注意：FSQ期望输入格式为 (B, T, C)，需要转换维度
            B, C_sem, T_sem = F_sem_modulated.shape
            F_sem_seq = F_sem_modulated.permute(0, 2, 1)  # (B, T_sem, C_sem)
            
            # FSQ量化：f_modulated -> f_quantized
            # indices: [B, T_sem]，如果你想监控离散码的使用率，可以打印它
            F_sem_quantized, indices = self.fsq(F_sem_seq)  # (B, T_sem, C_sem), (B, T_sem)
            
            # 转换回Conv1d格式: (B, T_sem, C_sem) -> (B, C_sem, T_sem)
            F_sem_quantized = F_sem_quantized.permute(0, 2, 1)  # (B, C_sem, T_sem)
            
            # 此时的 F_sem_quantized 已经经过了 "四舍五入" 的洗礼。
            # 如果水印信息还能留下来，说明它已经变成了 "语义坐标" 的一部分。
            F_sem_final = F_sem_quantized
        else:
            F_sem_final = None
            indices = None
        
        # ========== 声学流 ==========
        # Encoder
        e1 = self.enc1(x)      # (B, C, T)
        e2 = self.enc2(e1)     # (B, 2C, T/2)
        e3 = self.enc3(e2)     # (B, 4C, T/4)
        e4 = self.enc4(e3)     # (B, 8C, T/8)
        e5 = self.enc5(e4)     # (B, 16C, T/16)
        
        # ========== 双流融合 ==========
        # 注意：这里使用 F_sem_final (FSQ量化后的特征) 代替原来的 F_sem_modulated
        if F_sem_final is not None and self.fusion_bottleneck is not None:
            # 在瓶颈层融合量化后的语义特征与声学特征
            fused_bottleneck = self.fusion_bottleneck(F_sem_final, e5)  # (B, 16C, T/16)
            
            # 可选：在中间层也融合
            # 需要上采样语义特征到e4的分辨率
            if self.fusion_mid is not None:
                F_sem_mid = F.interpolate(
                    F_sem_final,
                    size=e4.size(-1),
                    mode='linear',
                    align_corners=False
                )
                fused_mid = self.fusion_mid(F_sem_mid, e4)  # (B, 8C, T/8)
            else:
                fused_mid = e4
            
            bottleneck_feat = fused_bottleneck
        else:
            # 如果没有语义流，直接使用声学特征
            bottleneck_feat = e5
            fused_mid = e4
        
        # ========== 时间连贯性处理（可选）==========
        if self.lstm is not None:
        # LSTM 处理 (需转换维度适配 LSTM: B, C, T -> B, T, C)
            B, C, T = bottleneck_feat.shape
            lstm_out, _ = self.lstm(bottleneck_feat.permute(0, 2, 1))
            lstm_out = lstm_out.permute(0, 2, 1)  # Back to B, C, T
            bottleneck_feat = lstm_out + bottleneck_feat  # 残差连接
        
        # ========== 解码器 ==========
        # Decoder (含跳跃连接 Skip Connections, 模仿U-Net)
        # 确保长度对齐，防止因上/下采样造成的±1长度误差
        bottleneck_feat_aligned, e5_aligned = _align_pair(bottleneck_feat, e5)
        d5 = self.dec5(bottleneck_feat_aligned + e5_aligned)  # 残差连接

        d5_aligned, fused_mid_aligned = _align_pair(d5, fused_mid)
        d4 = self.dec4(d5_aligned + fused_mid_aligned)  # 使用融合后的中间特征

        d4_aligned, e3_aligned = _align_pair(d4, e3)
        d3 = self.dec3(d4_aligned + e3_aligned)

        d3_aligned, e2_aligned = _align_pair(d3, e2)
        d2 = self.dec2(d3_aligned + e2_aligned)
        
        # Generate Watermark Signal
        d2_aligned, e1_aligned = _align_pair(d2, e1)
        watermark = self.tanh(self.final_conv(d2_aligned + e1_aligned))
        
        # Additive embedding with strength control: X_w = X + α · tanh(R)
        x_aligned, watermark_aligned = _align_pair(x, watermark)
        watermarked_audio = x_aligned + self.alpha * watermark_aligned
        
        return watermarked_audio, watermark_aligned, indices
