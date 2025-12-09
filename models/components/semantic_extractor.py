# models/components/semantic_extractor.py
"""
语义特征提取器：使用预训练的HuBERT或Wav2Vec 2.0模型提取语义特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import HubertModel
HUBERT_MODEL_AVAILABLE = True
TRANSFORMERS_AVAILABLE = True

# Fairseq is optional and may have compatibility issues with newer numpy
# We primarily use transformers library for HuBERT/Wav2Vec2
# Fairseq 在新版 numpy 中会报错（np.float 已被弃用），所以我们不导入它
# 如果需要 fairseq，应该使用旧版 numpy 或修复 fairseq 代码
FAIRSEQ_AVAILABLE = False


class SemanticExtractor(nn.Module):
    """
    语义特征提取器
    支持HuBERT和Wav2Vec 2.0模型
    """
    def __init__(self, model_type='hubert', model_name=None, freeze=True):
        """
        Args:
            model_type: 'hubert' 或 'wav2vec2'
            model_name: 模型名称或路径，None则使用默认
            freeze: 是否冻结模型参数
        """
        super().__init__()
        self.model_type = model_type
        self.freeze = freeze
        self.model = None
        self.processor = None
        
        if model_type == 'hubert':
            self._load_hubert(model_name)
        elif model_type == 'wav2vec2':
            self._load_wav2vec2(model_name)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        if self.model is not None and freeze:
            # 冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # 存储设备信息（用于后续移动）
        self._device = None
    
    def _load_hubert(self, model_name=None):
        """加载HuBERT模型"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available, using dummy semantic extractor")
            self.model = None
            self.processor = None
            return
        
        # 优先使用本地模型路径
        import os
        local_hubert_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "hubert")
        
        # 如果model_name是None，优先使用本地路径
        if model_name is None:
            if os.path.exists(local_hubert_path) and os.path.exists(os.path.join(local_hubert_path, "pytorch_model.bin")):
                model_name = local_hubert_path
                print(f"Using local HuBERT model from: {model_name}")
            else:
                # 回退到HuggingFace
                model_name = "facebook/hubert-large-ls960-ft"
                print(f"Local model not found, trying HuggingFace: {model_name}")
        elif os.path.isdir(model_name) or (os.path.exists(model_name) and os.path.isdir(os.path.dirname(model_name))):
            # 如果提供了路径，使用该路径
            print(f"Using provided HuBERT model path: {model_name}")
        else:
            # 假设是HuggingFace模型名称
            print(f"Using HuggingFace model: {model_name}")
        
        # 尝试使用HubertModel加载（如果可用），否则使用Wav2Vec2Model
        if HUBERT_MODEL_AVAILABLE:
            self.model = HubertModel.from_pretrained(model_name)
            print(f"✓ Loaded HuBERT model using HubertModel: {model_name}")
        else:
            # 使用Wav2Vec2Model加载HuBERT（这是正常的，因为HuBERT基于Wav2Vec2）
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        print(f"✓ Loaded HuBERT model: {model_name}")
    
    def _load_wav2vec2(self, model_name=None):
        """加载Wav2Vec 2.0模型"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available, using dummy semantic extractor")
            self.model = None
            self.processor = None
            return
        
        # 优先使用本地模型路径
        import os
        local_wav2vec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "wav2vec")
        
        # 如果model_name是None，优先使用本地路径
        if model_name is None:
            if os.path.exists(local_wav2vec_path) and os.path.exists(os.path.join(local_wav2vec_path, "pytorch_model.bin")):
                model_name = local_wav2vec_path
                print(f"Using local Wav2Vec2 model from: {model_name}")
            else:
                # 回退到HuggingFace
                model_name = "facebook/wav2vec2-large-960h"
                print(f"Local model not found, trying HuggingFace: {model_name}")
        elif os.path.isdir(model_name) or (os.path.exists(model_name) and os.path.isdir(os.path.dirname(model_name))):
            # 如果提供了路径，使用该路径
            print(f"Using provided Wav2Vec2 model path: {model_name}")
        else:
            # 假设是HuggingFace模型名称
            print(f"Using HuggingFace model: {model_name}")
        
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        print(f"✓ Loaded Wav2Vec2 model: {model_name}")
    
    def forward(self, waveform):
        """
        提取语义特征
        
        Args:
            waveform: (B, 1, T) 或 (B, T) 音频波形，范围应该在[-1, 1]
        
        Returns:
            features: (B, C, T') 语义特征，T'可能小于T（由于下采样）
        """
        # 确保模型在正确的设备上
        if self.model is not None:
            device = waveform.device
            # 如果模型不在正确的设备上，移动它
            if next(self.model.parameters()).device != device:
                self.model = self.model.to(device)
        
        if self.model is None:
            # 返回dummy特征（与输入形状匹配，但全零）
            feature_dim = self.get_feature_dim()  # 使用正确的特征维度
            if len(waveform.shape) == 3:
                B, C, T = waveform.shape
                return torch.zeros(B, feature_dim, T // 2, device=waveform.device, dtype=waveform.dtype)
            else:
                B, T = waveform.shape
                return torch.zeros(B, feature_dim, T // 2, device=waveform.device, dtype=waveform.dtype)
        
        # 确保输入格式正确
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)  # (B, 1, T) -> (B, T)
        
        # 归一化到[-1, 1]（如果还没有）
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # 使用processor预处理（如果需要）
        if self.processor is not None:
            # Wav2Vec2/HuBERT期望16kHz采样率
            # Processor期望numpy数组或list，需要转换
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.detach().cpu().numpy()
            else:
                waveform_np = waveform
            
            inputs = self.processor(waveform_np, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(waveform.device)
        else:
            # 如果没有processor，直接使用waveform
            # 注意：Wav2Vec2/HuBERT期望输入在[-1, 1]范围内
            input_values = waveform
        
        # 提取特征
        # 注意：即使模型被冻结，我们仍然需要梯度流（用于语义一致性损失）
        # 所以不使用torch.set_grad_enabled(False)，而是依赖模型的requires_grad=False
        outputs = self.model(input_values)
        # 使用最后一层的隐藏状态
        features = outputs.last_hidden_state  # (B, T', hidden_size)
        
        # 转换为 (B, C, T') 格式
        features = features.transpose(1, 2)  # (B, hidden_size, T')
        
        return features
    
    def get_feature_dim(self):
        """返回特征维度"""
        if self.model is None:
            return 768  # 默认维度
        # HuBERT/Wav2Vec2 Large通常是768维
        return self.model.config.hidden_size

