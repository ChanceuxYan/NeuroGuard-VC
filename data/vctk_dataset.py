import torch
from torch.utils.data import Dataset
import torchaudio
import os
import random
import glob
import pandas as pd

class NeuroGuardVCTKDataset(Dataset):
    def __init__(self, root_dir=None, segment_length=32000, mode='train', train_split=0.9, 
                 train_csv=None, val_csv=None):
        """
        Args:
            root_dir: 指向预处理后的16k数据目录（如果使用目录模式）
        segment_length: 训练切片长度 (32000 samples = 2s @ 16kHz)
            mode: 'train' 或 'val'
            train_split: 训练集比例（如果使用目录模式）
            train_csv: 训练集CSV文件路径（如果使用CSV模式）
            val_csv: 验证集CSV文件路径（如果使用CSV模式）
        """
        self.segment_length = segment_length
        self.mode = mode
        
        # 优先使用CSV文件
        if train_csv is not None or val_csv is not None:
            # CSV模式
            if mode == 'train':
                csv_path = train_csv
            else:
                csv_path = val_csv
            
            if csv_path is None:
                raise ValueError(f"CSV file for {mode} mode is not provided")
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            if 'wav_path' not in df.columns:
                raise ValueError(f"CSV file must contain 'wav_path' column: {csv_path}")
            
            # 过滤掉无效路径
            self.files = df['wav_path'].dropna().tolist()
            # 过滤掉不存在的文件
            self.files = [f for f in self.files if os.path.exists(f)]
            
            print(f"[{mode.upper()}] Dataset loaded from CSV: {csv_path}")
            print(f"  - Total entries: {len(df)}")
            print(f"  - Valid files: {len(self.files)}")
            print(f"[{mode.upper()}] Dataset loaded with {len(self.files)} files.")
            
            # CSV模式加载完成，直接返回
            return
        
        # 目录模式（向后兼容，仅在CSV未提供时使用）
        if root_dir is None:
            raise ValueError("Either root_dir or CSV files must be provided")
        
        self.root_dir = root_dir
        # 递归搜索所有wav文件
        self.files = sorted(glob.glob(os.path.join(root_dir, "**/*.wav"), recursive=True))
        
        # 划分训练/验证集
        random.seed(42)
        random.shuffle(self.files)
        split_idx = int(len(self.files) * train_split)
        
        if mode == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
            
        print(f"[{mode.upper()}] Dataset loaded from directory: {root_dir}")
        print(f"[{mode.upper()}] Dataset loaded with {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # 加载音频
        wav, sr = torchaudio.load(path)
        
        # 随机裁剪逻辑
        if wav.size(1) >= self.segment_length:
            max_start = wav.size(1) - self.segment_length
            start = random.randint(0, max_start)
            wav = wav[:, start : start + self.segment_length]
        else:
            # 如果音频过短，进行填充
            pad_amount = self.segment_length - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad_amount), 'constant', 0)
        
        return wav
