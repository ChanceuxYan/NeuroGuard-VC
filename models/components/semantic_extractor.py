# models/components/semantic_extractor.py
import torch
import torch.nn as nn
import os
from transformers import Wav2Vec2Model, HubertModel

class SemanticExtractor(nn.Module):
    def __init__(self, model_type='hubert', model_name=None, freeze=True, unfreeze_last_n_layers=0):
        super().__init__()
        self.model_type = model_type
        
        # 1. 路径加载逻辑 (保持不变)
        target_bin_path = "/home/yanjunzhe/project/WM-V2/NeuroGuard-VC/hubert/pytorch_model.bin"
        target_dir = os.path.dirname(target_bin_path)
        
        if os.path.exists(target_bin_path):
            final_model_path = target_dir
            print(f"✅ Found local model at: {target_bin_path}")
        elif model_name is None:
            # 回退逻辑
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            default_local = os.path.join(project_root, "hubert")
            if os.path.exists(os.path.join(default_local, "pytorch_model.bin")):
                final_model_path = default_local
            else:
                final_model_path = "facebook/hubert-large-ls960-ft"
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

        # 3. [关键] 允许梯度回传配置
        self.model.config.freeze_feature_encoder = False 
        self.model.config.feat_proj_dropout = 0.0
        self.model.config.attention_dropout = 0.0
        self.model.config.hidden_dropout = 0.0
        self.model.config.activation_dropout = 0.0
        self.model.config.mask_time_prob = 0.0
        self.model.config.layerdrop = 0.0

        # 4. [关键] 冻结与策略性解冻
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            # === 新增逻辑：解冻最后 N 层 ===
            if unfreeze_last_n_layers > 0 and hasattr(self.model, 'encoder'):
                print(f"🔓 Unfreezing the last {unfreeze_last_n_layers} layers of Semantic Model...")
                layers = self.model.encoder.layers
                # 解冻最后 N 层 Transformer Encoder
                for i in range(1, unfreeze_last_n_layers + 1):
                    for param in layers[-i].parameters():
                        param.requires_grad = True

    def train(self, mode=True):
        # 始终保持底层模型为 eval 模式 (即使部分层解冻，BN/Dropout 也不要动)
        super().train(False) 
        self.model.eval()
        return self

    def forward(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        # PyTorch 原生归一化 (保留梯度)
        with torch.set_grad_enabled(True):
            mean = waveform.mean(dim=-1, keepdim=True)
            std = waveform.std(dim=-1, keepdim=True)
            input_values = (waveform - mean) / (std + 1e-7)
        
        outputs = self.model(input_values)
        features = outputs.last_hidden_state
        features = features.transpose(1, 2)
        return features

    def get_feature_dim(self):
        return self.model.config.hidden_size