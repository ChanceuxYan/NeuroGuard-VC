# models/components/semantic_extractor.py
"""
语义特征提取器：使用预训练的HuBERT或Wav2Vec 2.0模型提取语义特征
最终修复版 (V3)：
1. [路径] 强制优先加载本地指定路径的 HuBERT 模型
2. [梯度] 移除 Processor，手动归一化，打通梯度流
3. [模式] 锁定 Eval 模式，解决 requires_grad 报错
4. [配置] 修改 Config 防止 detach
"""
import torch
import torch.nn as nn
import os
from transformers import Wav2Vec2Model, HubertModel

HUBERT_MODEL_AVAILABLE = True
TRANSFORMERS_AVAILABLE = True

class SemanticExtractor(nn.Module):
    """
    语义特征提取器
    支持HuBERT和Wav2Vec 2.0模型
    """
    def __init__(self, model_type='hubert', model_name=None, freeze=True):
        super().__init__()
        self.model_type = model_type
        
        # 1. 路径逻辑：强制优先使用您指定的本地路径
        # 您提供的文件路径
        target_bin_path = "/home/yanjunzhe/project/WM-V2/NeuroGuard-VC/hubert/pytorch_model.bin"
        target_dir = os.path.dirname(target_bin_path) # 获取目录: .../hubert/
        
        # 如果未指定名称，或指定的名称在本地存在，则优先使用本地
        final_model_path = None
        
        if os.path.exists(target_bin_path):
            print(f"✅ Found local model at: {target_bin_path}")
            final_model_path = target_dir
        elif model_name is None:
            # 回退：尝试项目根目录下的 hubert 文件夹
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_local = os.path.join(project_root, "hubert")
            if os.path.exists(os.path.join(default_local, "pytorch_model.bin")):
                final_model_path = default_local
            else:
                final_model_path = "facebook/hubert-large-ls960-ft" # 最后回退到 HF
        else:
            final_model_path = model_name

        print(f"Loading Semantic Model from: {final_model_path}")

        # 2. 加载模型
        if model_type == 'hubert':
            try:
                self.model = HubertModel.from_pretrained(final_model_path)
            except Exception as e:
                print(f"⚠ HubertModel load failed, trying Wav2Vec2Model: {e}")
                self.model = Wav2Vec2Model.from_pretrained(final_model_path)
        elif model_type == 'wav2vec2':
            self.model = Wav2Vec2Model.from_pretrained(final_model_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 3. [关键修复] 修改配置以允许梯度回传
        # 默认 freeze_feature_encoder=True 会导致 forward 时 detach()
        self.model.config.freeze_feature_encoder = False 
        
        # 关闭所有 Dropout，保证确定性
        self.model.config.feat_proj_dropout = 0.0
        self.model.config.attention_dropout = 0.0
        self.model.config.hidden_dropout = 0.0
        self.model.config.activation_dropout = 0.0
        self.model.config.mask_time_prob = 0.0
        self.model.config.layerdrop = 0.0

        # 4. 冻结参数 (物理冻结)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
    
    def train(self, mode=True):
        """
        [核心修复] 重写 train 方法
        拦截外部的 .train() 调用，强制保持 HuBERT 在 eval 模式
        防止 transformers 内部执行 requires_grad=True 导致 Crash
        """
        super().train(False) # 强制自身为 eval
        self.model.eval()    # 强制内部模型为 eval
        return self

    def forward(self, waveform):
        """
        提取语义特征
        Args:
            waveform: (B, 1, T) or (B, T)
        """
        # 1. 维度调整
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1) # (B, T)
            
        # 2. [核心修复] PyTorch 原生归一化 (替代 Processor)
        # 必须在 PyTorch 内部做，不能转 numpy，否则梯度断流！
        # 显式允许梯度计算 (即使在 eval 模式下)
        with torch.set_grad_enabled(True):
            mean = waveform.mean(dim=-1, keepdim=True)
            std = waveform.std(dim=-1, keepdim=True)
            # 加上 1e-7 防止除以零
            input_values = (waveform - mean) / (std + 1e-7)
        
        # 3. 提取特征
        # 此时 self.model 处于 eval 模式且 freeze_feature_encoder=False
        # 梯度可以从 hidden_states 穿过模型回传到 input_values
        outputs = self.model(input_values)
        features = outputs.last_hidden_state # (B, T_frame, D)
        
        # 4. 转置为 (B, D, T_frame) 以适配 Conv1d
        features = features.transpose(1, 2)
        
        return features

    def get_feature_dim(self):
        return self.model.config.hidden_size