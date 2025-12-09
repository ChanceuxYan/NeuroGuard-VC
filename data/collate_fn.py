import torch
from torch.nn.utils.rnn import pad_sequence

class CollateFn:
    """
    针对音频数据的动态批处理
    """
    def __init__(self, segment_length=None):
        self.segment_length = segment_length

    def __call__(self, batch):
        # batch is a list of tensors from Dataset
        # Filter out None/Errors
        batch = [x for x in batch if x is not None]
        
        if len(batch) == 0:
            return None

        # 如果指定了固定长度 (训练模式)
        if self.segment_length:
            # 已经在 Dataset 中做过裁剪，这里直接 Stack
            padded_audio = torch.stack(batch)
        else:
            # 推理/验证模式：Pad到最大长度
            # Input shape: (1, T) -> Permute to (T, 1) for pad_sequence
            batch_permuted = [x.squeeze(0) for x in batch] 
            padded_audio = pad_sequence(batch_permuted, batch_first=True, padding_value=0.0)
            # Restore channel dim: (B, T) -> (B, 1, T)
            padded_audio = padded_audio.unsqueeze(1)

        return padded_audio