import os
import torchaudio
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

def process_file(file_info):
    src_path, dst_path, target_sr = file_info
    
    # 检查是否已存在
    if os.path.exists(dst_path):
        return
        
    # 加载原始音频
    wav, sr = torchaudio.load(src_path)
    
    # 重采样
    if sr!= target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        
    # 混合为单声道 (VCTK通常是单声道，但为了鲁棒性)
    if len(wav.shape) > 1 and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    # 归一化 (峰值归一化，防止削波)
    max_val = torch.abs(wav).max()
    if max_val > 0.95:
        wav = wav / max_val * 0.95
        
    # 保存
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torchaudio.save(dst_path, wav, target_sr, bits_per_sample=16)

def main():
    parser = argparse.ArgumentParser(description="VCTK Preprocessor for NeuroGuard")
    parser.add_argument("--src", type=str, required=True, help="Path to original VCTK wav48")
    parser.add_argument("--dst", type=str, required=True, help="Output path for wav16")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    file_list = []  # Fix: initialize list
    for root, dirs, files in os.walk(args.src):
        for f in files:
            if f.endswith(".wav") or f.endswith(".flac"):
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, args.src)
                dst_path = os.path.join(args.dst, rel_path)
                file_list.append((src_path, dst_path, args.sr))

    print(f"Found {len(file_list)} files. Processing with {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_file, file_list), total=len(file_list)))

if __name__ == "__main__":
    main()
