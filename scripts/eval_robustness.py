# eval_robustness.py
import os
import glob
import torch
import argparse
import numpy as np
import yaml
from scipy.io import wavfile
from pesq import pesq
from pystoi import stoi
import torchaudio
import logging
from tqdm import tqdm
import subprocess
import tempfile
import soundfile as sf

from models.generator import NeuroGuardGenerator
from models.detector import NeuroGuardDetector
from utils.audio import load_wav_to_torch, MAX_WAV_VALUE

# 配置日志记录到文件，以便保存实验结果
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('eval_robustness.log'),
        logging.StreamHandler()
    ]
)

def calculate_ber(original_msg, extracted_msg):
    """
    计算误码率 (Bit Error Rate)。
    用于评估水印鲁棒性。
    """
    if len(original_msg) != len(extracted_msg):
        # 处理可能的同步丢失
        min_len = min(len(original_msg), len(extracted_msg))
        original_msg = original_msg[:min_len]
        extracted_msg = extracted_msg[:min_len]
        
    original_bits = np.array(original_msg)
    extracted_bits = np.array(extracted_msg)
    
    errors = np.sum(original_bits != extracted_bits)
    if len(original_bits) == 0:
        return 0.0
    ber = errors / len(original_bits) * 100.0
    return ber

class AttackSimulator:
    """
    攻击模拟器：提供一组针对音频的可微或不可微攻击方法。
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def gaussian_noise(self, audio, snr_db=20):
        """
        添加高斯白噪声 (AWGN)。
        参数:
            audio (torch.Tensor): 输入音频
            snr_db (float): 信噪比 (dB)
        """
        audio_np = audio.cpu().numpy()
        sig_power = np.mean(audio_np ** 2)
        
        # 计算噪声功率: SNR = 10 * log10(Ps / Pn) => Pn = Ps / 10^(SNR/10)
        noise_power = sig_power / (10 ** (snr_db / 10))
        
        noise = np.random.normal(0, np.sqrt(noise_power), audio_np.shape)
        noisy_audio = audio_np + noise
        
        # 裁剪以防溢出
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        return torch.tensor(noisy_audio, dtype=torch.float32).to(audio.device)

    def resample(self, audio, target_sr=16000):
        """
        重采样攻击：降采样再升采样。
        这会消除原始音频中 target_sr/2 以上的所有频率成分（低通滤波效应）。
        """
        resampler_down = torchaudio.transforms.Resample(self.sample_rate, target_sr).to(audio.device)
        resampler_up = torchaudio.transforms.Resample(target_sr, self.sample_rate).to(audio.device)
        
        down = resampler_down(audio)
        up = resampler_up(down)
        
        # 处理重采样导致的长度舍入误差
        if up.shape[-1] != audio.shape[-1]:
            diff = audio.shape[-1] - up.shape[-1]
            if diff > 0:
                up = torch.nn.functional.pad(up, (0, diff))
            else:
                up = up[..., :audio.shape[-1]]
            
        return up

    def mp3_compression(self, audio, bitrate='64k'):
        """
        模拟 MP3 压缩攻击。
        由于 MP3 编码不可微，通过系统调用 ffmpeg 实现。
        """
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio_np = audio.cpu().numpy()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf_wav:
             wav_name = tf_wav.name
        
        mp3_name = wav_name.replace('.wav', '.mp3')
        out_wav_name = wav_name.replace('.wav', '_dec.wav')
        
        # 写入原始 WAV (int16)
        wavfile.write(wav_name, self.sample_rate, (audio_np * 32767).astype(np.int16))
         
        # 压缩: wav -> mp3
        subprocess.run(['ffmpeg', '-y', '-i', wav_name, '-b:a', bitrate, mp3_name], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 解压: mp3 -> wav
        subprocess.run(['ffmpeg', '-y', '-i', mp3_name, out_wav_name], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
         
        # 读取回 Tensor
        sr, deg_audio = wavfile.read(out_wav_name)
        deg_audio = deg_audio.astype(np.float32) / 32768.0
        
        # 长度对齐
        if len(deg_audio) != len(audio_np):
            min_len = min(len(deg_audio), len(audio_np))
            deg_audio = deg_audio[:min_len]
         
        # 清理临时文件
        for f in [wav_name, mp3_name, out_wav_name]:
            if os.path.exists(f):
                os.remove(f)
        
        return torch.tensor(deg_audio).to(audio.device)

def compute_metrics(gt_wav, gen_wav, sr, device='cpu'):
    """计算 PESQ 和 STOI 指标"""
    # PESQ 需要 16k 或 8k
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        gt_16k = resampler(torch.tensor(gt_wav, device=device)).cpu().numpy()
        gen_16k = resampler(torch.tensor(gen_wav, device=device)).cpu().numpy()
    else:
        gt_16k, gen_16k = gt_wav, gen_wav
    
    # Ensure same length
    min_len = min(len(gt_16k), len(gen_16k))
    gt_16k = gt_16k[:min_len]
    gen_16k = gen_16k[:min_len]
    
    p_score = pesq(16000, gt_16k, gen_16k, 'wb')  # Wideband PESQ
    s_score = stoi(gt_wav, gen_wav, sr, extended=False)
    
    return p_score, s_score

def run_evaluation(args):
    """
    执行完整的抗噪性评估循环。
    """
    print("--- Initializing NeuroGuard Robustness Evaluation ---")
    print(f"Metrics: PESQ (wb), STOI, BER")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack_sim = AttackSimulator(sample_rate=args.sample_rate)
    
    # Load models if checkpoints provided
    generator = None
    detector = None
    config = None
    
    if args.checkpoint_gen and args.checkpoint_det:
        # Load config
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Try to load from checkpoint directory
            checkpoint_dir = os.path.dirname(args.checkpoint_gen)
            config_path = os.path.join(checkpoint_dir, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config_path = 'configs/vctk_16k.yaml'
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
        
        # Load generator
        generator = NeuroGuardGenerator(config).to(device)
        checkpoint_g = torch.load(args.checkpoint_gen, map_location=device)
        if isinstance(checkpoint_g, dict) and 'generator' in checkpoint_g:
            generator.load_state_dict(checkpoint_g['generator'])
        else:
            generator.load_state_dict(checkpoint_g)
        generator.eval()
        print(f"Loaded generator from {args.checkpoint_gen}")
        
        # Load detector
        detector = NeuroGuardDetector(config).to(device)
        checkpoint_d = torch.load(args.checkpoint_det, map_location=device)
        if isinstance(checkpoint_d, dict) and 'detector' in checkpoint_d:
            detector.load_state_dict(checkpoint_d['detector'])
        else:
            detector.load_state_dict(checkpoint_d)
        detector.eval()
        print(f"Loaded detector from {args.checkpoint_det}")
    
    # 结果容器
    metrics = {
        'clean': {'pesq': [], 'stoi': [], 'ber': []},
        'noise_20db': {'pesq': [], 'stoi': [], 'ber': []},
        'noise_10db': {'pesq': [], 'stoi': [], 'ber': []},
        'mp3_64k': {'pesq': [], 'stoi': [], 'ber': []},
        'mp3_128k': {'pesq': [], 'stoi': [], 'ber': []},
        'resample_16k': {'pesq': [], 'stoi': [], 'ber': []}
    }
    
    # 获取测试文件
    if args.test_data_dir:
        # If test_data_dir is provided, use it for both ground truth and generated
        test_files = sorted(glob.glob(os.path.join(args.test_data_dir, '*.wav')))
        gt_files = test_files
        gen_files = test_files  # For now, assume same files
    else:
        # Use separate directories
        gt_files = sorted(glob.glob(os.path.join(args.ground_truth_dir, '*.wav')))
        gen_files = sorted(glob.glob(os.path.join(args.generated_dir, '*.wav')))
    
    if len(gt_files) == 0:
        logging.error("No test files found!")
        return
    
    if len(gt_files) != len(gen_files) and args.ground_truth_dir and args.generated_dir:
        logging.warning("File count mismatch! proceeding with min length.")
    
    num_files = min(len(gt_files), len(gen_files)) if gen_files else len(gt_files)
    print(f"Found {num_files} test files")
    
    # Generate random messages for watermarking
    message_bits = config['model']['generator']['message_bits'] if config else 32
    
    for i in tqdm(range(num_files), desc="Evaluating"):
        gt_path = gt_files[i]
        gen_path = gen_files[i] if gen_files else gt_path
        
        # Read audio
        sr, gt_wav = wavfile.read(gt_path)
        if gen_files and gen_path != gt_path:
            _, gen_wav = wavfile.read(gen_path)
        else:
            gen_wav = gt_wav.copy()
            
            # Normalize [-1, 1]
            gt_wav = gt_wav.astype(np.float32) / 32768.0
            gen_wav = gen_wav.astype(np.float32) / 32768.0
            
            # Length alignment
            min_len = min(len(gt_wav), len(gen_wav))
            gt_wav = gt_wav[:min_len]
            gen_wav = gen_wav[:min_len]
            
            # If generator is available, embed watermark
            original_msg = None
            watermarked_audio = None
            
            if generator is not None:
                # Generate random message
                original_msg = np.random.randint(0, 2, message_bits).astype(np.float32)
                msg_tensor = torch.tensor(original_msg).unsqueeze(0).to(device)
                
                # Embed watermark
                audio_tensor = torch.tensor(gen_wav).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    watermarked_audio_tensor, _, _ = generator(audio_tensor, msg_tensor)
                watermarked_audio = watermarked_audio_tensor.squeeze().cpu().numpy()
            else:
                # Use provided audio as-is
                watermarked_audio = gen_wav

            # 1. Clean audio metrics
            p_score, s_score = compute_metrics(gt_wav, watermarked_audio, sr, device)
            if p_score is not None:
                metrics['clean']['pesq'].append(p_score)
                metrics['clean']['stoi'].append(s_score)
            
            # Extract watermark from clean audio
            if detector is not None and original_msg is not None:
                audio_tensor = torch.tensor(watermarked_audio).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, msg_logits = detector(audio_tensor)
                    extracted_msg = (torch.sigmoid(msg_logits).cpu().numpy()[0] > 0.5).astype(int)
                    ber = calculate_ber(original_msg, extracted_msg)
                    metrics['clean']['ber'].append(ber)
            
            # 2. Attack: Gaussian Noise 20dB
            gen_tensor = torch.tensor(watermarked_audio).to(device)
            noisy_20db = attack_sim.gaussian_noise(gen_tensor, snr_db=20)
            noisy_20db_np = noisy_20db.cpu().numpy()
            
            p_score, s_score = compute_metrics(gt_wav, noisy_20db_np, sr, device)
            if p_score is not None:
                metrics['noise_20db']['pesq'].append(p_score)
                metrics['noise_20db']['stoi'].append(s_score)
            
            if detector is not None and original_msg is not None:
                audio_tensor = noisy_20db.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, msg_logits = detector(audio_tensor)
                    extracted_msg = (torch.sigmoid(msg_logits).cpu().numpy()[0] > 0.5).astype(int)
                    ber = calculate_ber(original_msg, extracted_msg)
                    metrics['noise_20db']['ber'].append(ber)
            
            # 3. Attack: Gaussian Noise 10dB
            noisy_10db = attack_sim.gaussian_noise(gen_tensor, snr_db=10)
            noisy_10db_np = noisy_10db.cpu().numpy()
            
            p_score, s_score = compute_metrics(gt_wav, noisy_10db_np, sr, device)
            if p_score is not None:
                metrics['noise_10db']['pesq'].append(p_score)
                metrics['noise_10db']['stoi'].append(s_score)
            
            if detector is not None and original_msg is not None:
                audio_tensor = noisy_10db.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, msg_logits = detector(audio_tensor)
                    extracted_msg = (torch.sigmoid(msg_logits).cpu().numpy()[0] > 0.5).astype(int)
                    ber = calculate_ber(original_msg, extracted_msg)
                    metrics['noise_10db']['ber'].append(ber)
            
            # 4. Attack: MP3 64k
            compressed_64k = attack_sim.mp3_compression(gen_tensor, bitrate='64k')
            compressed_64k_np = compressed_64k.cpu().numpy()
            
            p_score, s_score = compute_metrics(gt_wav, compressed_64k_np, sr, device)
            if p_score is not None:
                metrics['mp3_64k']['pesq'].append(p_score)
                metrics['mp3_64k']['stoi'].append(s_score)
            
            if detector is not None and original_msg is not None:
                audio_tensor = compressed_64k.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, msg_logits = detector(audio_tensor)
                    extracted_msg = (torch.sigmoid(msg_logits).cpu().numpy()[0] > 0.5).astype(int)
                    ber = calculate_ber(original_msg, extracted_msg)
                    metrics['mp3_64k']['ber'].append(ber)
            
            # 5. Attack: MP3 128k
            compressed_128k = attack_sim.mp3_compression(gen_tensor, bitrate='128k')
            compressed_128k_np = compressed_128k.cpu().numpy()
            
            p_score, s_score = compute_metrics(gt_wav, compressed_128k_np, sr, device)
            if p_score is not None:
                metrics['mp3_128k']['pesq'].append(p_score)
                metrics['mp3_128k']['stoi'].append(s_score)
            
            if detector is not None and original_msg is not None:
                audio_tensor = compressed_128k.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, msg_logits = detector(audio_tensor)
                    extracted_msg = (torch.sigmoid(msg_logits).cpu().numpy()[0] > 0.5).astype(int)
                    ber = calculate_ber(original_msg, extracted_msg)
                    metrics['mp3_128k']['ber'].append(ber)
            
            # 6. Attack: Resampling
            resampled = attack_sim.resample(gen_tensor, target_sr=16000)
            resampled_np = resampled.cpu().numpy()
            
            p_score, s_score = compute_metrics(gt_wav, resampled_np, sr, device)
            if p_score is not None:
                metrics['resample_16k']['pesq'].append(p_score)
                metrics['resample_16k']['stoi'].append(s_score)
                
                if detector is not None and original_msg is not None:
                    audio_tensor = resampled.unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        _, msg_logits = detector(audio_tensor)
                        extracted_msg = (torch.sigmoid(msg_logits).cpu().numpy()[0] > 0.5).astype(int)
                        ber = calculate_ber(original_msg, extracted_msg)
                        metrics['resample_16k']['ber'].append(ber)

    # 3. 输出统计报告
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Sample Count: {num_files}\n")
    
    for attack_name, attack_metrics in metrics.items():
        if len(attack_metrics['pesq']) > 0:
            pesq_mean = np.mean(attack_metrics['pesq'])
            stoi_mean = np.mean(attack_metrics['stoi'])
            print(f"{attack_name.upper():20s} | PESQ: {pesq_mean:.4f} | STOI: {stoi_mean:.4f}", end="")
            if len(attack_metrics['ber']) > 0:
                ber_mean = np.mean(attack_metrics['ber'])
                print(f" | BER: {ber_mean:.2f}%")
            else:
                print()
    
    print("=" * 60)
    
    # Save results
    if args.output_file:
        np.save(args.output_file, metrics)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Path to ground truth audio wavs')
    parser.add_argument('--generated_dir', type=str, default=None,
                       help='Path to generated/synthesized wavs')
    parser.add_argument('--test_data_dir', type=str, default=None,
                       help='Path to test data directory (alternative to ground_truth_dir/generated_dir)')
    parser.add_argument('--checkpoint_gen', type=str, default=None,
                       help='Path to generator checkpoint (optional, for watermark embedding)')
    parser.add_argument('--checkpoint_det', type=str, default=None,
                       help='Path to detector checkpoint (optional, for watermark extraction)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save evaluation results')
    args = parser.parse_args()
    
    if not args.test_data_dir and (not args.ground_truth_dir or not args.generated_dir):
        parser.error("Either --test_data_dir or both --ground_truth_dir and --generated_dir must be provided")
    
    run_evaluation(args)
