#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：评估第一阶段训练的模型在 stage1（无攻击）下的表现

用法:
    python scripts/test_model.py --checkpoint checkpoints/best.pth --config configs/vctk_16k.yaml
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.generator import NeuroGuardGenerator
from models.detector import NeuroGuardDetector
from models.detector_legacy import NeuroGuardDetectorLegacy
from modules.losses import MultiResolutionSTFTLoss
from data.vctk_dataset import NeuroGuardVCTKDataset


def test_stage1(config_path, checkpoint_path, output_dir=None, num_samples=None, save_audio=True):
    """
    测试模型在 stage1（无攻击）下的表现
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径
        output_dir: 输出目录（用于保存音频样本和结果）
        num_samples: 测试样本数量（None表示使用全部验证集）
        save_audio: 是否保存音频样本
    """
    # 设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join('test_results', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # 加载checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化并加载 generator
    generator = NeuroGuardGenerator(config).to(device)
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
        print("✓ Loaded generator")
    else:
        generator.load_state_dict(checkpoint)
        print("✓ Loaded generator (legacy format)")
    
    # 尝试加载 detector（先尝试新版本，失败则使用旧版本）
    detector = None
    if 'detector' in checkpoint:
        # 先尝试新版本 detector
        try:
            detector = NeuroGuardDetector(config).to(device)
            detector.load_state_dict(checkpoint['detector'], strict=True)
            print("✓ Loaded detector (new version)")
        except RuntimeError as e:
            # 如果新版本失败，尝试旧版本
            print("⚠ Detector structure mismatch, trying legacy version...")
            try:
                detector = NeuroGuardDetectorLegacy(config).to(device)
                detector.load_state_dict(checkpoint['detector'], strict=True)
                print("✓ Loaded detector (legacy version with ConvTranspose1d)")
            except RuntimeError as e2:
                # 如果旧版本也失败，尝试部分加载
                print("⚠ Legacy version also failed, attempting partial load...")
                detector = NeuroGuardDetector(config).to(device)
                missing_keys, unexpected_keys = detector.load_state_dict(checkpoint['detector'], strict=False)
                if missing_keys:
                    print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                print("✓ Loaded detector (partial, using random initialization for mismatched layers)")
    else:
        print("⚠ Warning: Detector state not found in checkpoint, using random initialization")
        detector = NeuroGuardDetector(config).to(device)
    
    generator.eval()
    detector.eval()
    
    # 加载验证数据集
    data_config = config['data']
    val_dataset = NeuroGuardVCTKDataset(
        root_dir=data_config.get('root_path'),
        segment_length=data_config['segment_length'],
        mode='val',
        train_csv=data_config.get('train_csv'),
        val_csv=data_config.get('val_csv')
    )
    
    if num_samples is not None and num_samples < len(val_dataset):
        indices = list(range(min(num_samples, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    print(f"\nValidation dataset size: {len(val_dataset)}")
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['data'].get('batch_size', 8),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    # 损失函数
    stft_criterion = MultiResolutionSTFTLoss().to(device)
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    
    # 统计指标
    all_losses = {
        'stft': [],
        'loc': [],
        'msg': [],
        'sem': [],
        'total': []
    }
    all_accs = []
    all_bit_accs = []  # 逐位准确率
    all_bers = []  # 逐位误码率
    all_msg_logits = []
    all_msg_probs = []
    
    # 音频统计
    all_watermark_magnitudes = []
    all_audio_diffs = []
    
    # 测试循环
    print("\n" + "=" * 80)
    print("Testing on Stage1 (No Attack)")
    print("=" * 80)
    
    sample_count = 0
    with torch.no_grad():
        for step, audio_real in enumerate(tqdm(val_dataloader, desc="Testing")):
            audio_real = audio_real.to(device)
            
            # 确保格式正确 (B, 1, T)
            if len(audio_real.shape) == 2:
                audio_real = audio_real.unsqueeze(1)
            if audio_real.shape[1] != 1:
                audio_real = audio_real[:, 0:1, :]
            
            B = audio_real.shape[0]
            
            # 生成随机消息
            msg = torch.randint(0, 2, (B, config['model']['generator']['message_bits'])).float().to(device)
            
            # 生成水印音频（Stage1：无攻击）
            audio_wm, watermark_res, indices = generator(audio_real, msg)
            
            # Stage1：不使用攻击
            audio_attacked = audio_wm  # 无攻击
            
            # 检测器前向传播
            detector_output = detector(audio_attacked)
            if len(detector_output) == 4:
                loc_logits, msg_logits, local_logits, attention_weights = detector_output
            else:
                loc_logits, msg_logits = detector_output
            
            # 计算损失
            loss_stft = stft_criterion(audio_wm, audio_real)
            target_loc = torch.ones_like(loc_logits)
            loss_loc = bce_criterion(loc_logits, target_loc)
            loss_msg = bce_criterion(msg_logits, msg)
            
            # 计算准确率
            msg_pred = (torch.sigmoid(msg_logits) > 0.5).float()
            msg_acc = (msg_pred == msg).float().mean().item()
            
            # 逐位准确率
            bit_acc = (msg_pred == msg).float().mean(dim=0).cpu().numpy()  # (bits,)
            all_bit_accs.append(bit_acc)
            
            # 逐位误码率 (BER)
            bit_ber = (msg_pred != msg).float().mean(dim=0).cpu().numpy() * 100  # (bits,)
            all_bers.append(bit_ber)
            
            # 统计logits和probs
            all_msg_logits.append(msg_logits.cpu().numpy())
            msg_probs = torch.sigmoid(msg_logits).cpu().numpy()
            all_msg_probs.append(msg_probs)
            
            # 语义一致性损失（如果启用）
            loss_sem = 0
            if hasattr(generator, 'use_semantic') and generator.use_semantic:
                # 这里可以添加语义一致性损失的计算
                pass
            
            # 总损失（使用训练时的权重）
            total_loss = (
                config['training'].get('lambda_perceptual', 0.0) * loss_stft +
                config['training'].get('lambda_loc', 0.2) * loss_loc +
                config['training'].get('lambda_msg', 100.0) * loss_msg
            )
            if loss_sem > 0:
                total_loss += config['training'].get('lambda_sem', 0.05) * loss_sem
            
            # 记录损失
            all_losses['stft'].append(loss_stft.item())
            all_losses['loc'].append(loss_loc.item())
            all_losses['msg'].append(loss_msg.item())
            if loss_sem > 0:
                all_losses['sem'].append(loss_sem.item())
            all_losses['total'].append(total_loss.item())
            
            all_accs.append(msg_acc)
            
            # 音频统计
            watermark_magnitude = watermark_res.abs().mean().item()
            audio_diff = (audio_wm - audio_real).abs().mean().item()
            all_watermark_magnitudes.append(watermark_magnitude)
            all_audio_diffs.append(audio_diff)
            
            # 保存前几个样本的音频
            if save_audio and sample_count < 5:
                sr = config.get('experiment', {}).get('sample_rate', 16000)
                for i in range(min(B, 5 - sample_count)):
                    idx = sample_count + i
                    
                    # 保存原始音频
                    audio_clean_np = audio_real[i].squeeze().cpu().numpy()
                    audio_clean_np = np.clip(audio_clean_np, -1.0, 1.0)
                    sf.write(
                        os.path.join(output_dir, f'sample_{idx:03d}_clean.wav'),
                        audio_clean_np,
                        sr
                    )
                    
                    # 保存水印音频
                    audio_wm_np = audio_wm[i].squeeze().cpu().numpy()
                    audio_wm_np = np.clip(audio_wm_np, -1.0, 1.0)
                    sf.write(
                        os.path.join(output_dir, f'sample_{idx:03d}_watermarked.wav'),
                        audio_wm_np,
                        sr
                    )
                    
                    # 保存水印残差
                    watermark_np = watermark_res[i].squeeze().cpu().numpy()
                    watermark_np = np.clip(watermark_np, -1.0, 1.0)
                    sf.write(
                        os.path.join(output_dir, f'sample_{idx:03d}_watermark_residual.wav'),
                        watermark_np,
                        sr
                    )
                    
                    # 保存消息信息
                    msg_true = msg[i].cpu().numpy()
                    msg_pred_i = msg_pred[i].cpu().numpy()
                    msg_probs_i = msg_probs[i]
                    
                    with open(os.path.join(output_dir, f'sample_{idx:03d}_message.txt'), 'w') as f:
                        f.write(f"True Message:  {''.join(map(str, msg_true.astype(int)))}\n")
                        f.write(f"Pred Message:  {''.join(map(str, msg_pred_i.astype(int)))}\n")
                        f.write(f"Message Probs: {', '.join([f'{p:.4f}' for p in msg_probs_i])}\n")
                        f.write(f"Bit Accuracy:  {bit_acc.mean():.4f}\n")
                        f.write(f"Bit BER:       {bit_ber.mean():.2f}%\n")
            
            sample_count += B
    
    # 计算统计指标
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    
    # 损失统计
    print("\n[Losses]")
    for key, values in all_losses.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key.upper():10s}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    # 准确率统计
    print("\n[Accuracy]")
    acc_mean = np.mean(all_accs)
    acc_std = np.std(all_accs)
    print(f"  Overall ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    
    # 逐位准确率
    if all_bit_accs:
        bit_acc_array = np.array(all_bit_accs)  # (N_samples, bits)
        bit_acc_mean = bit_acc_array.mean(axis=0)  # (bits,)
        bit_acc_std = bit_acc_array.std(axis=0)
        bit_acc_min = bit_acc_array.min(axis=0)
        bit_acc_max = bit_acc_array.max(axis=0)
        
        print(f"\n  Bit-wise ACC:")
        print(f"    Mean: {bit_acc_mean.mean():.4f} ± {bit_acc_mean.std():.4f}")
        print(f"    Min:  {bit_acc_min.min():.4f} (bit {bit_acc_min.argmin()})")
        print(f"    Max:  {bit_acc_max.max():.4f} (bit {bit_acc_max.argmax()})")
        print(f"    Per-bit: {', '.join([f'{acc:.3f}' for acc in bit_acc_mean])}")
    
    # 误码率统计
    if all_bers:
        ber_array = np.array(all_bers)  # (N_samples, bits)
        ber_mean = ber_array.mean(axis=0)  # (bits,)
        print(f"\n  Bit-wise BER (%):")
        print(f"    Mean: {ber_mean.mean():.2f}% ± {ber_mean.std():.2f}%")
        print(f"    Per-bit: {', '.join([f'{b:.2f}' for b in ber_mean])}")
    
    # Logits和Probs统计
    if all_msg_logits:
        logits_array = np.concatenate(all_msg_logits, axis=0)  # (N_total, bits)
        probs_array = np.concatenate(all_msg_probs, axis=0)
        
        print(f"\n[Logits & Probabilities]")
        print(f"  Logits - Mean: {logits_array.mean():.4f}, Std: {logits_array.std():.4f}")
        print(f"  Logits - Min:  {logits_array.min():.4f}, Max: {logits_array.max():.4f}")
        print(f"  Probs  - Mean: {probs_array.mean():.4f}, Std: {probs_array.std():.4f}")
        print(f"  Probs  - Min:  {probs_array.min():.4f}, Max: {probs_array.max():.4f}")
    
    # 音频统计
    print(f"\n[Audio Statistics]")
    print(f"  Watermark Magnitude: {np.mean(all_watermark_magnitudes):.6f} ± {np.std(all_watermark_magnitudes):.6f}")
    print(f"  Audio Difference:    {np.mean(all_audio_diffs):.6f} ± {np.std(all_audio_diffs):.6f}")
    
    # 保存详细结果到文件
    results_file = os.path.join(output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage1 Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("[Losses]\n")
        for key, values in all_losses.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"  {key.upper():10s}: {mean_val:8.4f} ± {std_val:6.4f}\n")
        
        f.write(f"\n[Accuracy]\n")
        f.write(f"  Overall ACC: {acc_mean:.4f} ± {acc_std:.4f}\n")
        
        if all_bit_accs:
            f.write(f"\n  Bit-wise ACC:\n")
            f.write(f"    Mean: {bit_acc_mean.mean():.4f} ± {bit_acc_mean.std():.4f}\n")
            f.write(f"    Per-bit: {', '.join([f'{acc:.4f}' for acc in bit_acc_mean])}\n")
        
        if all_bers:
            f.write(f"\n  Bit-wise BER (%):\n")
            f.write(f"    Mean: {ber_mean.mean():.2f}% ± {ber_mean.std():.2f}%\n")
            f.write(f"    Per-bit: {', '.join([f'{b:.2f}' for b in ber_mean])}\n")
        
        f.write(f"\n[Audio Statistics]\n")
        f.write(f"  Watermark Magnitude: {np.mean(all_watermark_magnitudes):.6f} ± {np.std(all_watermark_magnitudes):.6f}\n")
        f.write(f"  Audio Difference:    {np.mean(all_audio_diffs):.6f} ± {np.std(all_audio_diffs):.6f}\n")
    
    print(f"\n✓ Detailed results saved to: {results_file}")
    print(f"✓ Audio samples saved to: {output_dir}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test Stage1 Model Performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., checkpoints/best.pth)')
    parser.add_argument('--config', type=str, default='configs/vctk_16k.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: test_results/TIMESTAMP)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to test (default: all validation set)')
    parser.add_argument('--no_audio', action='store_true',
                       help='Do not save audio samples')
    
    args = parser.parse_args()
    
    test_stage1(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        save_audio=not args.no_audio
    )


if __name__ == '__main__':
    main()

