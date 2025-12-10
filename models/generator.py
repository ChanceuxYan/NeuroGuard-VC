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
        
        # FSQ 瓶颈层 [修改部分]
        if use_semantic:
            # 读取配置，默认为 True 以兼容旧配置，但现在我们在 yaml 里设为 False
            self.use_fsq = semantic_config.get('use_fsq', True) 
            
            if self.use_fsq:
                fsq_levels = semantic_config.get('fsq_levels', [8, 8, 8, 8, 8])
                self.fsq = FSQ(
                    levels=fsq_levels,
                    dim=semantic_dim,
                    num_codebooks=1
                )
                print(f"Build FSQ with levels: {fsq_levels}")
            else:
                self.fsq = None
                print("FSQ is disabled by config.")
        else:
            self.fsq = None
            self.use_fsq = False
        
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
            
            # 3. FSQ 离散化 [修改部分]
            if self.use_fsq and self.fsq is not None:
                # --- 原有 FSQ 逻辑 ---
                B, C_sem, T_sem = F_sem_modulated.shape
                F_sem_seq = F_sem_modulated.permute(0, 2, 1)
                F_sem_quantized, indices = self.fsq(F_sem_seq)
                F_sem_final = F_sem_quantized.permute(0, 2, 1)
            else:
                # --- [新增] 直通逻辑 ---
                # 不做量化，直接将含有水印的连续特征传给解码器
                # 这样梯度可以完美回传，微弱的水印信号也能被保留
                F_sem_final = F_sem_modulated
                indices = None  # 没有量化索引
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
        
        # 1. 原始水印信号 (保持不变)
        d2_aligned, e1_aligned = _align_pair(d2, e1)
        watermark = self.tanh(self.final_conv(d2_aligned + e1_aligned))
        
        # 2. 对齐 (保持不变)
        x_aligned, watermark_aligned = _align_pair(x, watermark)
        
        # 3. === 新增核心逻辑：能量掩蔽 (Energy Masking) ===
        # 计算音频的幅度包络 (模拟人耳对响度的感知)
        # kernel_size=400 在 16k 采样率下约为 25ms
        energy_mask = torch.nn.functional.avg_pool1d(
            x_aligned.abs(), 
            kernel_size=400, 
            stride=1, 
            padding=200
        )
        
        # 修正 padding 带来的尺寸误差
        if energy_mask.size(-1) > x_aligned.size(-1):
            energy_mask = energy_mask[..., :x_aligned.size(-1)]
        
        # 归一化并进行非线性映射
        # 逻辑：大音量处 mask -> 1.0 (允许最大alpha嵌入)
        #       小音量处 mask -> 0.0 (禁止嵌入)
        energy_mask = torch.clamp(energy_mask, min=0.0, max=1.0)
        
        # 静音门控 (Squelch)：彻底消除背景底噪
        # 如果局部音量低于 0.01，强制将水印置为 0
        # energy_mask[energy_mask < 0.01] = 0.0
        # energy_mask = energy_mask * 0.8 + 0.2
        energy_mask = energy_mask ** 2
        
        # 应用掩蔽：让水印信号跟随原始语音的起伏
        watermark_masked = watermark_aligned * energy_mask
        
        # 4. === 最终叠加 ===
        # 使用掩蔽后的水印进行叠加
        watermarked_audio = x_aligned + self.alpha * watermark_masked
        
        # 返回 masked 的水印以便后续 Loss 计算使用
        return watermarked_audio, watermark_masked, indices
