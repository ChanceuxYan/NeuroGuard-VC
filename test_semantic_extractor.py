#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义提取器测试脚本
测试 SemanticExtractor 的各种功能
使用真实的val.csv音频进行测试
"""
import torch
import numpy as np
import sys
import os
import pandas as pd
import random

# 导入torchaudio
import torchaudio
TORCHAUDIO_AVAILABLE = True
import librosa
import soundfile as sf
LIBROSA_AVAILABLE = True

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.components.semantic_extractor import SemanticExtractor

# 导入HubertForCTC用于CTC解码测试
from transformers import HubertForCTC, Wav2Vec2Processor
HUBERT_FOR_CTC_AVAILABLE = True

def test_semantic_extractor():
    """测试语义提取器"""
    print("=" * 60)
    print("语义提取器测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 测试1: 测试HuBERT模型加载（如果可用）
    print("测试1: HuBERT模型加载")
    print("-" * 60)
    extractor_hubert = SemanticExtractor(
        model_type='hubert',
        model_name=None,  # 使用默认
        freeze=True
    )
    print(f"✓ HuBERT提取器创建成功")
    print(f"  - 模型类型: {extractor_hubert.model_type}")
    print(f"  - 模型是否加载: {extractor_hubert.model is not None}")
    print(f"  - Processor是否加载: {extractor_hubert.processor is not None}")
    print(f"  - 特征维度: {extractor_hubert.get_feature_dim()}")
    
    print()
    
    # 测试2: 测试Wav2Vec2模型加载（如果可用）
    print("测试2: Wav2Vec2模型加载")
    print("-" * 60)
    extractor_wav2vec = SemanticExtractor(
        model_type='wav2vec2',
        model_name=None,  # 使用默认
        freeze=True
    )
    print(f"✓ Wav2Vec2提取器创建成功")
    print(f"  - 模型类型: {extractor_wav2vec.model_type}")
    print(f"  - 模型是否加载: {extractor_wav2vec.model is not None}")
    print(f"  - Processor是否加载: {extractor_wav2vec.processor is not None}")
    print(f"  - 特征维度: {extractor_wav2vec.get_feature_dim()}")
    
    print()
    
    # 选择可用的提取器进行测试
    extractor = extractor_hubert if extractor_hubert is not None else extractor_wav2vec
    # if extractor is None:
    #     print("⚠ 警告: 没有可用的语义提取器，将使用dummy模式测试")
    #     extractor = SemanticExtractor(model_type='hubert', freeze=True)
    
    # 测试3: 使用真实音频测试特征提取
    print("测试3: 使用真实音频测试特征提取")
    print("-" * 60)
    # 从val.csv加载真实音频
    val_csv_path = "/home/yanjunzhe/project/WM-V1/data/csvs/val.csv"
    if os.path.exists(val_csv_path):
        df = pd.read_csv(val_csv_path)
        valid_files = df[df['wav_path'].notna()]['wav_path'].tolist()
        valid_files = [f for f in valid_files if os.path.exists(f)]
        
        if len(valid_files) > 0:
            # 随机选择一个真实音频文件
            test_file = random.choice(valid_files)
            print(f"  使用真实音频: {test_file}")
            
            # 加载音频
            if TORCHAUDIO_AVAILABLE:
                wav, sr = torchaudio.load(test_file)
                # 重采样到16kHz（如果需要）
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    wav = resampler(wav)
            elif LIBROSA_AVAILABLE:
                wav, sr = librosa.load(test_file, sr=16000, mono=False)
                # librosa返回numpy数组，转换为torch tensor
                wav = torch.from_numpy(wav).float()
                if len(wav.shape) == 1:
                    wav = wav.unsqueeze(0)  # (1, T)
            else:
                raise ImportError("Neither torchaudio nor librosa is available for audio loading")
            
            # 转换为单声道
            if wav.shape[0] > 1:
                wav = wav[0:1, :]
            
            # 保存原始音频长度信息
            original_length = wav.shape[1]
            original_duration = original_length / 16000.0
            print(f"  原始音频长度: {original_length} samples ({original_duration:.2f} 秒)")
            
            # 对于ASR测试，使用完整音频（不裁剪）
            # 只对过短的音频进行填充，对过长的音频进行裁剪（但保留更多内容）
            # 使用更长的目标长度（例如3秒）以便更好地进行ASR识别
            target_length = 48000  # 3秒 @ 16kHz
            
            if wav.shape[1] > target_length:
                # 如果音频很长，只取前3秒（或者随机选择一段）
                print(f"  音频较长 ({original_duration:.2f}秒)，裁剪到 {target_length/16000:.2f} 秒")
                wav = wav[:, :target_length]
            elif wav.shape[1] < target_length:
                # 如果音频较短，进行填充
                pad_length = target_length - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad_length))
                print(f"  音频较短 ({original_duration:.2f}秒)，填充到 {target_length/16000:.2f} 秒")
            else:
                print(f"  音频长度正好: {original_duration:.2f} 秒")
            
            test_audio_1 = wav.unsqueeze(0).to(device)  # (1, 1, T)
            print(f"  输入形状: {test_audio_1.shape}")
            print(f"  实际音频长度: {test_audio_1.shape[2] / 16000:.2f} 秒")
            print(f"  音频范围: [{test_audio_1.min().item():.4f}, {test_audio_1.max().item():.4f}]")
        else:
            print("  ⚠ 警告: val.csv中没有有效文件，使用随机音频")
            test_audio_1 = torch.randn(1, 1, 16000).to(device)
    else:
        print(f"  ⚠ 警告: val.csv不存在 ({val_csv_path})，使用随机音频")
        test_audio_1 = torch.randn(1, 1, 16000).to(device)
    
    features_1 = extractor(test_audio_1)
    print(f"✓ 特征提取成功")
    print(f"  - 输出形状: {features_1.shape}")
    print(f"  - 输出类型: {features_1.dtype}")
    print(f"  - 输出设备: {features_1.device}")
    print(f"  - 特征统计: min={features_1.min().item():.4f}, max={features_1.max().item():.4f}, mean={features_1.mean().item():.4f}, std={features_1.std().item():.4f}")
    
    # 检查形状合理性
    if features_1.shape[0] == 1 and features_1.shape[1] == extractor.get_feature_dim():
        print(f"  ✓ 形状正确")
    else:
        print(f"  ✗ 形状错误: 期望 (1, {extractor.get_feature_dim()}, T'), 得到 {features_1.shape}")
    
    # 使用HuBERT进行ASR（自动语音识别）查看内容特征
    if HUBERT_FOR_CTC_AVAILABLE and extractor.model_type == 'hubert':
        print(f"\n  [ASR测试] 使用HuBERT进行语音识别，查看内容特征提取情况:")
        local_hubert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hubert")
        
        if os.path.exists(local_hubert_path) and os.path.exists(os.path.join(local_hubert_path, "pytorch_model.bin")):
            # 加载HuBERT CTC模型和processor
            processor_ctc = Wav2Vec2Processor.from_pretrained(local_hubert_path)
            model_ctc = HubertForCTC.from_pretrained(local_hubert_path)
            model_ctc = model_ctc.to(device)
            model_ctc.eval()
            
            # 将音频转换为numpy数组（processor需要）
            # test_audio_1是(B, 1, T)格式，需要转换为1D numpy数组
            audio_array = test_audio_1.squeeze().cpu().numpy()  # (T,)
            
            # 使用processor预处理
            input_values = processor_ctc(audio_array, return_tensors="pt", sampling_rate=16000).input_values
            input_values = input_values.to(device)
            
            # CTC解码
            with torch.no_grad():
                logits = model_ctc(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor_ctc.decode(predicted_ids[0])
            
            print(f"    ✓ ASR识别成功")
            print(f"    - 输入音频长度: {len(audio_array) / 16000:.2f} 秒")
            print(f"    - Logits形状: {logits.shape}")
            print(f"    - 预测ID数量: {predicted_ids.shape[1]}")
            print(f"    - 识别文本: \"{transcription}\"")
            print(f"    - 识别文本长度: {len(transcription)} 字符")
            
            # 显示一些预测ID的统计信息
            unique_ids = torch.unique(predicted_ids[0])
            print(f"    - 唯一token数量: {len(unique_ids)}")
            print(f"    - Token ID范围: [{predicted_ids[0].min().item()}, {predicted_ids[0].max().item()}]")
            
        else:
            print(f"    ⚠ 本地HuBERT模型不存在，跳过ASR测试")
    
    print()
    
    # 测试4: 使用真实音频测试批量特征提取
    print("测试4: 使用真实音频测试批量特征提取")
    print("-" * 60)
    # 从val.csv加载多个真实音频
    val_csv_path = "/home/yanjunzhe/project/WM-V1/data/csvs/val.csv"
    if os.path.exists(val_csv_path):
        df = pd.read_csv(val_csv_path)
        valid_files = df[df['wav_path'].notna()]['wav_path'].tolist()
        valid_files = [f for f in valid_files if os.path.exists(f)]
        
        if len(valid_files) >= 2:
            # 随机选择2个真实音频文件
            test_files = random.sample(valid_files, min(2, len(valid_files)))
            print(f"  使用真实音频: {len(test_files)} 个文件")
            
            audio_list = []
            for test_file in test_files:
                if TORCHAUDIO_AVAILABLE:
                    wav, sr = torchaudio.load(test_file)
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        wav = resampler(wav)
                elif LIBROSA_AVAILABLE:
                    wav, sr = librosa.load(test_file, sr=16000, mono=False)
                    wav = torch.from_numpy(wav).float()
                    if len(wav.shape) == 1:
                        wav = wav.unsqueeze(0)
                else:
                    raise ImportError("Neither torchaudio nor librosa is available")
                if wav.shape[0] > 1:
                    wav = wav[0:1, :]
                target_length = 16000
                if wav.shape[1] > target_length:
                    start = random.randint(0, wav.shape[1] - target_length)
                    wav = wav[:, start:start+target_length]
                elif wav.shape[1] < target_length:
                    pad_length = target_length - wav.shape[1]
                    wav = torch.nn.functional.pad(wav, (0, pad_length))
                audio_list.append(wav)
            
            test_audio_2 = torch.stack(audio_list, dim=0).to(device)  # (2, 1, 16000)
            print(f"  输入形状: {test_audio_2.shape}")
        else:
            print("  ⚠ 警告: val.csv中没有足够的有效文件，使用随机音频")
            test_audio_2 = torch.randn(2, 1, 16000).to(device)
    else:
        print(f"  ⚠ 警告: val.csv不存在 ({val_csv_path})，使用随机音频")
        test_audio_2 = torch.randn(2, 1, 16000).to(device)
    
    features_2 = extractor(test_audio_2)
    print(f"✓ 批量特征提取成功")
    print(f"  - 输出形状: {features_2.shape}")
    print(f"  - 批次大小: {features_2.shape[0]}")
    
    if features_2.shape[0] == test_audio_2.shape[0]:
        print(f"  ✓ 批次大小正确")
    else:
        print(f"  ✗ 批次大小错误: 期望 {test_audio_2.shape[0]}, 得到 {features_2.shape[0]}")
    
    print()
    
    # 测试5: 测试特征提取 - 无通道维度 (B, T)
    print("测试5: 特征提取 - 无通道维度格式 (B, T)")
    print("-" * 60)
    # 确保提取器在正确的设备上
    if extractor.model is not None:
        extractor.model = extractor.model.to(device)
    
    # 生成测试音频: (1, 16000) - 无通道维度
    test_audio_3 = torch.randn(1, 16000).to(device)
    print(f"输入形状: {test_audio_3.shape}")
    
    features_3 = extractor(test_audio_3)
    print(f"✓ 特征提取成功（无通道维度）")
    print(f"  - 输出形状: {features_3.shape}")
    
    if len(features_3.shape) == 3:
        print(f"  ✓ 输出维度正确")
    else:
        print(f"  ✗ 输出维度错误: 期望3维, 得到{len(features_3.shape)}维")
    
    print()
    
    # 测试6: 测试不同长度的音频
    print("测试6: 不同长度的音频")
    print("-" * 60)
    lengths = [8000, 16000, 32000, 48000]  # 0.5s, 1s, 2s, 3s
    for length in lengths:
        test_audio = torch.randn(1, 1, length).to(device)
        features = extractor(test_audio)
        print(f"  - 长度 {length:5d} samples -> 特征长度 {features.shape[2]:5d}")
    
    print(f"✓ 不同长度音频处理成功")
    
    print()
    
    # 测试7: 测试归一化处理
    print("测试7: 输入归一化处理")
    print("-" * 60)
    # 测试超出范围的输入
    test_audio_high = torch.randn(1, 1, 16000) * 2.0  # 超出[-1, 1]
    test_audio_low = torch.randn(1, 1, 16000) * -2.0  # 超出[-1, 1]
    
    features_high = extractor(test_audio_high.to(device))
    features_low = extractor(test_audio_low.to(device))
    
    print(f"✓ 超出范围输入处理成功")
    print(f"  - 高值输入特征范围: [{features_high.min().item():.4f}, {features_high.max().item():.4f}]")
    print(f"  - 低值输入特征范围: [{features_low.min().item():.4f}, {features_low.max().item():.4f}]")
    
    print()
    
    # 测试8: 测试梯度（如果模型加载成功）
    print("测试8: 梯度流测试")
    print("-" * 60)
    if extractor.model is not None:
            # 确保模型在正确的设备上
            extractor.model = extractor.model.to(device)
            
            test_audio = torch.randn(1, 1, 16000, requires_grad=True).to(device)
            
            # 检查模型参数是否被冻结
            model_params_require_grad = any(p.requires_grad for p in extractor.model.parameters())
            if not model_params_require_grad:
                print(f"  ✓ 模型参数已正确冻结（这是预期的）")
            
            # 即使模型被冻结，输入到输出的梯度流仍然应该存在
            # 但需要确保模型在eval模式下仍然允许梯度流
            extractor.model.eval()  # 确保在eval模式
            
            # 使用torch.enable_grad()确保梯度计算
            with torch.enable_grad():
                features = extractor(test_audio)
                
                # 检查features是否有梯度
                if features.requires_grad:
                    # 计算一个简单的损失
                    loss = features.mean()
                    loss.backward()
                    
                    # 检查梯度
                    if test_audio.grad is not None:
                        print(f"✓ 梯度流正常")
                        print(f"  - 输入梯度形状: {test_audio.grad.shape}")
                        print(f"  - 输入梯度统计: mean={test_audio.grad.mean().item():.6f}, std={test_audio.grad.std().item():.6f}")
                        print(f"  - 说明: 即使模型参数被冻结，输入到输出的梯度流仍然存在")
                    else:
                        print(f"⚠ 警告: 输入没有梯度")
                else:
                    print(f"⚠ 注意: 特征输出没有requires_grad（模型被冻结时这是正常的）")
                    print(f"  - 这是预期的行为，因为模型参数被冻结")
                    print(f"  - 在实际训练中，语义提取器参数不会被更新")
    else:
        print("⚠ 跳过梯度测试（模型未加载，使用dummy模式）")
    
    print()
    
    # 测试9: 使用真实音频进行性能测试
    print("测试9: 使用真实音频进行性能测试")
    print("-" * 60)
    import time
    
    # 从val.csv加载真实音频
    val_csv_path = "/home/yanjunzhe/project/WM-V1/data/csvs/val.csv"
    if os.path.exists(val_csv_path):
        df = pd.read_csv(val_csv_path)
        valid_files = df[df['wav_path'].notna()]['wav_path'].tolist()
        valid_files = [f for f in valid_files if os.path.exists(f)]
        
        if len(valid_files) > 0:
            test_file = random.choice(valid_files)
            print(f"  使用真实音频: {test_file}")
            if TORCHAUDIO_AVAILABLE:
                wav, sr = torchaudio.load(test_file)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    wav = resampler(wav)
            elif LIBROSA_AVAILABLE:
                wav, sr = librosa.load(test_file, sr=16000, mono=False)
                wav = torch.from_numpy(wav).float()
                if len(wav.shape) == 1:
                    wav = wav.unsqueeze(0)
            else:
                raise ImportError("Neither torchaudio nor librosa is available")
            if wav.shape[0] > 1:
                wav = wav[0:1, :]
            target_length = 16000
            if wav.shape[1] > target_length:
                wav = wav[:, :target_length]
            elif wav.shape[1] < target_length:
                pad_length = target_length - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad_length))
            test_audio = wav.unsqueeze(0).to(device)
        else:
            test_audio = torch.randn(1, 1, 16000).to(device)
    else:
        test_audio = torch.randn(1, 1, 16000).to(device)
    
    # 预热
    for _ in range(3):
        _ = extractor(test_audio)
    
    # 测试
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        _ = extractor(test_audio)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"✓ 性能测试完成")
    print(f"  - 平均处理时间: {avg_time*1000:.2f} ms")
    print(f"  - 吞吐量: {1.0/avg_time:.2f} samples/s")
    
    print()
    
    # 测试10: 与Generator集成测试
    print("测试10: 与Generator集成测试")
    print("-" * 60)
    from models.generator import NeuroGuardGenerator
    import yaml
    
    # 创建测试配置
    test_config = {
        'model': {
            'generator': {
                'base_channels': 32,
                'message_bits': 32,
                'alpha': 0.1,
                'semantic': {
                    'enabled': True,
                    'model_type': 'hubert',
                    'model_name': None
                },
                'use_lstm': False
            }
        }
    }
    
    generator = NeuroGuardGenerator(test_config).to(device)
    print(f"✓ Generator创建成功")
    print(f"  - 使用语义流: {generator.use_semantic}")
    print(f"  - 语义提取器: {generator.semantic_extractor is not None}")
    
    # 测试前向传播 - 使用真实音频
    val_csv_path = "/home/yanjunzhe/project/WM-V1/data/csvs/val.csv"
    if os.path.exists(val_csv_path):
        df = pd.read_csv(val_csv_path)
        valid_files = df[df['wav_path'].notna()]['wav_path'].tolist()
        valid_files = [f for f in valid_files if os.path.exists(f)]
        
        if len(valid_files) > 0:
            test_file = random.choice(valid_files)
            print(f"  使用真实音频: {test_file}")
            if TORCHAUDIO_AVAILABLE:
                wav, sr = torchaudio.load(test_file)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    wav = resampler(wav)
            elif LIBROSA_AVAILABLE:
                wav, sr = librosa.load(test_file, sr=16000, mono=False)
                wav = torch.from_numpy(wav).float()
                if len(wav.shape) == 1:
                    wav = wav.unsqueeze(0)
            else:
                raise ImportError("Neither torchaudio nor librosa is available")
            if wav.shape[0] > 1:
                wav = wav[0:1, :]
            target_length = 16000
            if wav.shape[1] > target_length:
                wav = wav[:, :target_length]
            elif wav.shape[1] < target_length:
                pad_length = target_length - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, pad_length))
            test_audio = wav.unsqueeze(0).to(device)
        else:
            test_audio = torch.randn(1, 1, 16000).to(device)
    else:
        test_audio = torch.randn(1, 1, 16000).to(device)
    
    test_msg = torch.randint(0, 2, (1, 32)).float().to(device)
    
    audio_wm, watermark, indices = generator(test_audio, test_msg)
    print(f"✓ Generator前向传播成功")
    print(f"  - 输入形状: {test_audio.shape}")
    print(f"  - 水印音频形状: {audio_wm.shape}")
    print(f"  - 水印信号形状: {watermark.shape}")
    print(f"  - 水印信号范围: [{watermark.min().item():.4f}, {watermark.max().item():.4f}]")
    if indices is not None:
        print(f"  - FSQ量化索引形状: {indices.shape}")
        print(f"  - FSQ量化索引范围: [{indices.min().item()}, {indices.max().item()}]")
    print(f"  - 强度系数α: {generator.alpha}")
    
    # 验证输出公式: X_w = X + α * tanh(R)
    expected = test_audio + generator.alpha * watermark
    if torch.allclose(audio_wm, expected, atol=1e-5):
        print(f"  ✓ 输出公式验证正确: X_w = X + α·tanh(R)")
    else:
        print(f"  ✗ 输出公式验证失败")
        print(f"    差异: {(audio_wm - expected).abs().max().item():.6f}")
    
    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_semantic_extractor()

