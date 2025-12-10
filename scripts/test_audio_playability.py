#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：测试音频分解+水印嵌入后的音频是否可听

用法:
    # 单文件模式：使用随机初始化的模型（快速测试）
    python scripts/test_audio_playability.py --input_audio path/to/audio.wav --output_dir test_output
    
    # 单文件模式：使用训练好的checkpoint
    python scripts/test_audio_playability.py --input_audio path/to/audio.wav --output_dir test_output --checkpoint checkpoints/best.pth
    
    # 批量模式：从val.csv随机选择5条音频测试
    python scripts/test_audio_playability.py --csv /home/yanjunzhe/project/WM-V1/data/csvs/val.csv --output_dir test_output --num_samples 5
    
    # 批量模式：使用checkpoint
    python scripts/test_audio_playability.py --csv /home/yanjunzhe/project/WM-V1/data/csvs/val.csv --output_dir test_output --checkpoint checkpoints/best.pth --num_samples 5
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml
import soundfile as sf
import pandas as pd
import random
from pathlib import Path

# 尝试导入torchaudio（用于重采样）
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("警告: torchaudio未安装，如果音频采样率不匹配将无法重采样")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.generator import NeuroGuardGenerator
from utils.audio import load_wav_to_torch, MAX_WAV_VALUE


def test_audio_playability(
    input_audio_path,
    output_dir,
    config_path='configs/vctk_16k.yaml',
    checkpoint_path=None,
    message_bits=None,
    alpha=None
):
    """
    测试音频分解+水印嵌入后的音频是否可听
    
    Args:
        input_audio_path: 输入音频文件路径
        output_dir: 输出目录（保存原始音频和水印音频）
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径（可选，如果不提供则使用随机初始化的模型）
        message_bits: 水印消息位数（可选，默认从config读取）
        alpha: 水印强度系数（可选，默认从config读取）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 初始化Generator
    print("\n初始化Generator模型...")
    generator = NeuroGuardGenerator(config).to(device)
    
    # 加载checkpoint（如果提供）
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
            print("✓ 已加载generator权重")
        else:
            generator.load_state_dict(checkpoint)
            print("✓ 已加载generator权重（legacy格式）")
    else:
        print("⚠ 未提供checkpoint，使用随机初始化的模型（仅用于测试流程）")
    
    generator.eval()
    
    # 获取配置参数
    sample_rate = config.get('experiment', {}).get('sample_rate', 16000)
    msg_bits = message_bits or config['model']['generator']['message_bits']
    alpha_value = alpha or config['model']['generator'].get('alpha', 1.0)
    
    print(f"\n配置参数:")
    print(f"  - 采样率: {sample_rate} Hz")
    print(f"  - 水印位数: {msg_bits}")
    print(f"  - 强度系数 (alpha): {alpha_value}")
    
    # 加载音频
    print(f"\n加载音频: {input_audio_path}")
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"音频文件不存在: {input_audio_path}")
    
    sr, audio_tensor = load_wav_to_torch(input_audio_path)
    print(f"  - 原始采样率: {sr} Hz")
    print(f"  - 音频长度: {audio_tensor.shape[-1] / sr:.2f} 秒")
    
    # load_wav_to_torch已经返回归一化到[-1, 1]的浮点数，不需要再次归一化
    
    # 如果需要，重采样到目标采样率
    if sr != sample_rate:
        if TORCHAUDIO_AVAILABLE:
            print(f"  - 重采样到 {sample_rate} Hz...")
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio_tensor = resampler(audio_tensor)
            sr = sample_rate
        else:
            print(f"  ⚠ 警告: 采样率不匹配 ({sr} Hz vs {sample_rate} Hz)，但torchaudio未安装，无法重采样")
            print(f"    请安装torchaudio或使用采样率为{sample_rate} Hz的音频文件")
    
    # 转换为单声道
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # 添加channel维度
    elif audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor[:1]  # 取第一个声道
    
    # 添加batch维度: (1, 1, T)
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    print(f"  - 张量形状: {audio_tensor.shape}")
    
    # 生成随机水印消息
    print(f"\n生成随机水印消息 ({msg_bits} bits)...")
    msg = torch.randint(0, 2, (1, msg_bits), dtype=torch.float32).to(device)
    msg_str = ''.join([str(int(b)) for b in msg[0].cpu().numpy()])
    print(f"  - 消息: {msg_str}")
    
    # 保存消息到文件
    msg_file = os.path.join(output_dir, 'watermark_message.txt')
    with open(msg_file, 'w') as f:
        f.write(f"Watermark Message ({msg_bits} bits):\n")
        f.write(f"{msg_str}\n")
        f.write(f"\nBit values:\n")
        for i, bit in enumerate(msg[0].cpu().numpy()):
            f.write(f"  Bit {i:2d}: {int(bit)}\n")
    print(f"  - 消息已保存到: {msg_file}")
    
    # 进行水印嵌入
    print("\n进行水印嵌入...")
    with torch.no_grad():
        watermarked_audio, watermark_residual, indices = generator(audio_tensor, msg)
    
    print(f"  - 水印音频形状: {watermarked_audio.shape}")
    print(f"  - 水印残差形状: {watermark_residual.shape}")
    if indices is not None:
        print(f"  - FSQ量化索引形状: {indices.shape}")
    
    # 参考train.py的方式：直接使用torch tensor保存，保持(B, 1, T)格式
    # 裁剪到[-1, 1]范围
    audio_original_clipped = torch.clamp(audio_tensor, -1.0, 1.0)
    audio_watermarked_clipped = torch.clamp(watermarked_audio, -1.0, 1.0)
    watermark_residual_clipped = torch.clamp(watermark_residual, -1.0, 1.0)
    
    # 转换为numpy用于统计（保持tensor用于保存）
    audio_original_np = audio_original_clipped.squeeze().cpu().numpy()
    audio_watermarked_np = audio_watermarked_clipped.squeeze().cpu().numpy()
    watermark_residual_np = watermark_residual_clipped.squeeze().cpu().numpy()
    
    # 计算统计信息
    print("\n音频统计信息:")
    print(f"  原始音频:")
    print(f"    - 最小值: {audio_original_np.min():.6f}")
    print(f"    - 最大值: {audio_original_np.max():.6f}")
    print(f"    - 均值: {audio_original_np.mean():.6f}")
    print(f"    - 标准差: {audio_original_np.std():.6f}")
    print(f"  水印音频:")
    print(f"    - 最小值: {audio_watermarked_np.min():.6f}")
    print(f"    - 最大值: {audio_watermarked_np.max():.6f}")
    print(f"    - 均值: {audio_watermarked_np.mean():.6f}")
    print(f"    - 标准差: {audio_watermarked_np.std():.6f}")
    print(f"  差异:")
    diff = audio_watermarked_np - audio_original_np
    print(f"    - 平均绝对差异: {np.abs(diff).mean():.6f}")
    print(f"    - 最大绝对差异: {np.abs(diff).max():.6f}")
    print(f"    - SNR (估计): {20 * np.log10(np.abs(audio_original_np).mean() / (np.abs(diff).mean() + 1e-10)):.2f} dB")
    
    # 保存音频文件（参考train.py的方式，使用torchaudio）
    print("\n保存音频文件...")
    
    if not TORCHAUDIO_AVAILABLE:
        print("  ⚠ 警告: torchaudio未安装，使用soundfile保存")
        # 回退到soundfile方式
        audio_original_int16 = (audio_original_np * MAX_WAV_VALUE).astype(np.int16)
        audio_watermarked_int16 = (audio_watermarked_np * MAX_WAV_VALUE).astype(np.int16)
        residual_amplified = np.clip(watermark_residual_np * 10.0, -1.0, 1.0)
        residual_int16 = (residual_amplified * MAX_WAV_VALUE).astype(np.int16)
        diff_amplified = np.clip(diff * 10.0, -1.0, 1.0)
        diff_int16 = (diff_amplified * MAX_WAV_VALUE).astype(np.int16)
        
        original_output = os.path.join(output_dir, 'original.wav')
        sf.write(original_output, audio_original_int16, sample_rate, subtype='PCM_16')
        print(f"  ✓ 原始音频: {original_output}")
        
        watermarked_output = os.path.join(output_dir, 'watermarked.wav')
        sf.write(watermarked_output, audio_watermarked_int16, sample_rate, subtype='PCM_16')
        print(f"  ✓ 水印音频: {watermarked_output}")
        
        residual_output = os.path.join(output_dir, 'watermark_residual.wav')
        sf.write(residual_output, residual_int16, sample_rate, subtype='PCM_16')
        print(f"  ✓ 水印残差（放大10倍）: {residual_output}")
        
        diff_output = os.path.join(output_dir, 'difference.wav')
        sf.write(diff_output, diff_int16, sample_rate, subtype='PCM_16')
        print(f"  ✓ 差异音频（放大10倍）: {diff_output}")
    else:
        # 使用torchaudio保存（与train.py一致）
        # 取第一个样本，保持(1, T)格式
        audio_original_save = audio_original_clipped[0].detach().cpu()  # (1, T)
        audio_watermarked_save = audio_watermarked_clipped[0].detach().cpu()  # (1, T)
        watermark_residual_save = watermark_residual_clipped[0].detach().cpu()  # (1, T)
        
        # 放大残差和差异以便听
        watermark_residual_amplified = torch.clamp(watermark_residual_save * 10.0, -1.0, 1.0)
        diff_tensor = audio_watermarked_save - audio_original_save
        diff_amplified = torch.clamp(diff_tensor * 10.0, -1.0, 1.0)
        
        original_output = os.path.join(output_dir, 'original.wav')
        torchaudio.save(original_output, audio_original_save, sample_rate)
        print(f"  ✓ 原始音频: {original_output}")
        
        watermarked_output = os.path.join(output_dir, 'watermarked.wav')
        torchaudio.save(watermarked_output, audio_watermarked_save, sample_rate)
        print(f"  ✓ 水印音频: {watermarked_output}")
        
        residual_output = os.path.join(output_dir, 'watermark_residual.wav')
        torchaudio.save(residual_output, watermark_residual_amplified, sample_rate)
        print(f"  ✓ 水印残差（放大10倍）: {residual_output}")
        
        diff_output = os.path.join(output_dir, 'difference.wav')
        torchaudio.save(diff_output, diff_amplified, sample_rate)
        print(f"  ✓ 差异音频（放大10倍）: {diff_output}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print(f"\n请检查以下文件:")
    print(f"  1. {original_output} - 原始音频")
    print(f"  2. {watermarked_output} - 水印音频（请听这个，检查是否可听）")
    print(f"  3. {residual_output} - 水印残差（放大10倍）")
    print(f"  4. {diff_output} - 差异音频（放大10倍）")
    print(f"  5. {msg_file} - 水印消息")
    print("\n提示: 如果水印音频听起来正常，说明音频分解+水印嵌入流程工作正常。")
    print("=" * 80)


def test_batch_from_csv(
    csv_path,
    output_base_dir,
    config_path='configs/vctk_16k.yaml',
    checkpoint_path=None,
    num_samples=5,
    message_bits=None,
    alpha=None,
    seed=42
):
    """
    从CSV文件中随机选择几条音频进行批量测试
    
    Args:
        csv_path: CSV文件路径（包含wav_path列）
        output_base_dir: 输出基础目录
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径
        num_samples: 随机选择的音频数量
        message_bits: 水印消息位数
        alpha: 水印强度系数
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 读取CSV文件
    print(f"读取CSV文件: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if 'wav_path' not in df.columns:
        raise ValueError(f"CSV文件必须包含'wav_path'列")
    
    # 过滤存在的文件
    valid_files = []
    for idx, row in df.iterrows():
        wav_path = row['wav_path']
        if os.path.exists(wav_path):
            valid_files.append((idx, wav_path, row.get('speaker_id', 'unknown')))
    
    print(f"  - CSV总行数: {len(df)}")
    print(f"  - 有效音频文件: {len(valid_files)}")
    
    if len(valid_files) == 0:
        raise ValueError("没有找到有效的音频文件")
    
    # 随机选择
    num_samples = min(num_samples, len(valid_files))
    selected = random.sample(valid_files, num_samples)
    
    print(f"\n随机选择了 {num_samples} 条音频进行测试:")
    for i, (idx, wav_path, speaker_id) in enumerate(selected):
        print(f"  {i+1}. [{speaker_id}] {os.path.basename(wav_path)}")
    
    # 创建输出基础目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 批量测试
    print("\n" + "=" * 80)
    print("开始批量测试")
    print("=" * 80)
    
    results = []
    for i, (idx, wav_path, speaker_id) in enumerate(selected):
        print(f"\n{'='*80}")
        print(f"测试 {i+1}/{num_samples}: {os.path.basename(wav_path)}")
        print(f"{'='*80}")
        
        # 为每个音频创建单独的输出目录
        audio_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_dir = os.path.join(output_base_dir, f"{i+1:02d}_{audio_name}")
        
        try:
            test_audio_playability(
                input_audio_path=wav_path,
                output_dir=output_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                message_bits=message_bits,
                alpha=alpha
            )
            results.append({
                'index': i+1,
                'audio': wav_path,
                'speaker': speaker_id,
                'status': 'success',
                'output_dir': output_dir
            })
        except Exception as e:
            print(f"\n❌ 测试失败: {str(e)}")
            results.append({
                'index': i+1,
                'audio': wav_path,
                'speaker': speaker_id,
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存测试摘要
    summary_file = os.path.join(output_base_dir, 'test_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("批量测试摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"CSV文件: {csv_path}\n")
        f.write(f"测试数量: {num_samples}\n")
        f.write(f"成功: {sum(1 for r in results if r['status'] == 'success')}\n")
        f.write(f"失败: {sum(1 for r in results if r['status'] == 'failed')}\n\n")
        
        f.write("详细结果:\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"\n[{r['index']}] {os.path.basename(r['audio'])}\n")
            f.write(f"  说话人: {r['speaker']}\n")
            f.write(f"  状态: {r['status']}\n")
            if r['status'] == 'success':
                f.write(f"  输出目录: {r['output_dir']}\n")
            else:
                f.write(f"  错误: {r.get('error', 'Unknown')}\n")
    
    print("\n" + "=" * 80)
    print("批量测试完成！")
    print("=" * 80)
    print(f"\n测试摘要已保存到: {summary_file}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}/{num_samples}")
    print(f"失败: {sum(1 for r in results if r['status'] == 'failed')}/{num_samples}")


def main():
    parser = argparse.ArgumentParser(description='测试音频分解+水印嵌入后的音频是否可听')
    parser.add_argument('--input_audio', type=str, default=None,
                       help='输入音频文件路径（单文件模式）')
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV文件路径（批量模式，包含wav_path列）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录（保存原始音频和水印音频）')
    parser.add_argument('--config', type=str, default='configs/vctk_16k.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型checkpoint路径（可选，如果不提供则使用随机初始化的模型）')
    parser.add_argument('--message_bits', type=int, default=None,
                       help='水印消息位数（可选，默认从config读取）')
    parser.add_argument('--alpha', type=float, default=None,
                       help='水印强度系数（可选，默认从config读取）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='批量模式下的随机选择数量（默认：5）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认：42）')
    
    args = parser.parse_args()
    
    # 判断是单文件模式还是批量模式
    if args.csv:
        # 批量模式：从CSV读取
        test_batch_from_csv(
            csv_path=args.csv,
            output_base_dir=args.output_dir,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            message_bits=args.message_bits,
            alpha=args.alpha,
            seed=args.seed
        )
    elif args.input_audio:
        # 单文件模式
        test_audio_playability(
            input_audio_path=args.input_audio,
            output_dir=args.output_dir,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            message_bits=args.message_bits,
            alpha=args.alpha
        )
    else:
        parser.error("必须提供 --input_audio 或 --csv 参数之一")


if __name__ == '__main__':
    main()

