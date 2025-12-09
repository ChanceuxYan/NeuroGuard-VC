# inference.py
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import torch
import numpy as np
import yaml
from scipy.io import wavfile
import soundfile as sf

from models.generator import NeuroGuardGenerator
from models.detector import NeuroGuardDetector
from utils.audio import load_wav_to_torch, MAX_WAV_VALUE
from utils.util import AttrDict

def embed_watermark(audio_path, output_path, checkpoint_gen, checkpoint_det, message, config_path=None):
    """
    水印嵌入：将消息嵌入到音频中
    
    Args:
        audio_path: 输入音频路径
        output_path: 输出水印音频路径
        checkpoint_gen: Generator 检查点路径
        checkpoint_det: Detector 检查点路径（可选，用于验证）
        message: 要嵌入的消息（二进制数组或字符串）
        config_path: 配置文件路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    if config_path is None:
        # Try to load from checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_gen)
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if not os.path.exists(config_path):
            config_path = 'configs/vctk_16k.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize generator
    generator = NeuroGuardGenerator(config).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_gen):
        checkpoint = torch.load(checkpoint_gen, map_location=device)
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        else:
            generator.load_state_dict(checkpoint)
        print(f"Loaded generator from {checkpoint_gen}")
    else:
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_gen}")
    
    generator.eval()
    
    # Load audio
    sampling_rate, audio_tensor = load_wav_to_torch(audio_path)
    audio_tensor = audio_tensor / MAX_WAV_VALUE  # Normalize to [-1, 1]
    
    # Convert to tensor and add batch/channel dimensions
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dim
    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # Add batch dim: (1, 1, T)

    # Process message
    message_bits = config['model']['generator']['message_bits']
    if isinstance(message, str):
        # Convert string to binary
        message_binary = ''.join(format(ord(c), '08b') for c in message)
        message_binary = message_binary[:message_bits]  # Truncate if too long
        message_binary = message_binary.ljust(message_bits, '0')  # Pad if too short
        msg = torch.tensor([int(b) for b in message_binary], dtype=torch.float32).unsqueeze(0).to(device)
    elif isinstance(message, (list, np.ndarray)):
        # Convert list/array to tensor
        msg = torch.tensor(message, dtype=torch.float32).unsqueeze(0).to(device)
        if msg.shape[1] != message_bits:
            # Pad or truncate
            if msg.shape[1] < message_bits:
                padding = torch.zeros(1, message_bits - msg.shape[1], device=device)
                msg = torch.cat([msg, padding], dim=1)
            else:
                msg = msg[:, :message_bits]
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")
    
    # Embed watermark
    with torch.no_grad():
        watermarked_audio, watermark, indices = generator(audio_tensor, msg)
    
    # Convert back to numpy and denormalize
    watermarked_audio = watermarked_audio.squeeze().cpu().numpy()
    watermarked_audio = np.clip(watermarked_audio, -1.0, 1.0)
    watermarked_audio_int16 = (watermarked_audio * MAX_WAV_VALUE).astype(np.int16)
    
    # Save watermarked audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, watermarked_audio_int16, sampling_rate)
    print(f"Watermarked audio saved to {output_path}")
            
    # Optional: Verify with detector
    if checkpoint_det and os.path.exists(checkpoint_det):
        detector = NeuroGuardDetector(config).to(device)
        checkpoint_d = torch.load(checkpoint_det, map_location=device)
        if isinstance(checkpoint_d, dict) and 'detector' in checkpoint_d:
            detector.load_state_dict(checkpoint_d['detector'])
        else:
            detector.load_state_dict(checkpoint_d)
        detector.eval()
        
        with torch.no_grad():
            loc_logits, msg_logits = detector(watermarked_audio.unsqueeze(0).unsqueeze(0).to(device))
            extracted_msg = torch.sigmoid(msg_logits).cpu().numpy()[0]
            extracted_msg_binary = (extracted_msg > 0.5).astype(int)
            
            # Calculate BER
            original_msg_binary = msg.cpu().numpy()[0]
            ber = np.mean(original_msg_binary != extracted_msg_binary) * 100
            print(f"Verification - BER: {ber:.2f}%")
    
    return watermarked_audio_int16, sampling_rate

def extract_watermark(audio_path, checkpoint_det, config_path=None):
    """
    水印提取：从音频中提取消息和定位图
    
    Args:
        audio_path: 输入水印音频路径
        checkpoint_det: Detector 检查点路径
        config_path: 配置文件路径
    
    Returns:
        message: 提取的消息（二进制数组）
        localization_map: 定位图（概率图）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
            
    # Load config
    if config_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_det)
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if not os.path.exists(config_path):
            config_path = 'configs/vctk_16k.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize detector
    detector = NeuroGuardDetector(config).to(device)
    
    # Load checkpoint
    if os.path.exists(checkpoint_det):
        checkpoint = torch.load(checkpoint_det, map_location=device)
        if isinstance(checkpoint, dict) and 'detector' in checkpoint:
            detector.load_state_dict(checkpoint['detector'])
        else:
            detector.load_state_dict(checkpoint)
        print(f"Loaded detector from {checkpoint_det}")
    else:
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_det}")
    
    detector.eval()
            
    # Load audio
    sampling_rate, audio_tensor = load_wav_to_torch(audio_path)
    audio_tensor = audio_tensor / MAX_WAV_VALUE  # Normalize to [-1, 1]
    
    # Convert to tensor and add batch/channel dimensions
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dim
    audio_tensor = audio_tensor.unsqueeze(0).to(device)  # Add batch dim: (1, 1, T)

    # Extract watermark
    with torch.no_grad():
        loc_logits, msg_logits = detector(audio_tensor)
        
        # Convert to probabilities
        loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0, 0]  # (T,)
        msg_probs = torch.sigmoid(msg_logits).cpu().numpy()[0]  # (message_bits,)
        
        # Convert to binary message
        message_binary = (msg_probs > 0.5).astype(int)
    
    print(f"Extracted message: {message_binary}")
    print(f"Message as binary string: {''.join(map(str, message_binary))}")

    return message_binary, loc_probs

def main():
    parser = argparse.ArgumentParser(description='NeuroGuard-VC Watermark Inference')
    parser.add_argument('--mode', type=str, required=True, choices=['embed', 'extract'],
                       help='Mode: embed or extract watermark')
    parser.add_argument('--input_audio', type=str, required=True,
                       help='Input audio file path')
    parser.add_argument('--output_audio', type=str, default=None,
                       help='Output audio file path (for embed mode)')
    parser.add_argument('--checkpoint_gen', type=str, default=None,
                       help='Path to generator checkpoint (required for embed mode)')
    parser.add_argument('--checkpoint_det', type=str, required=True,
                       help='Path to detector checkpoint')
    parser.add_argument('--message', type=str, default=None,
                       help='Message to embed (string or binary, for embed mode)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--output_message', type=str, default=None,
                       help='Output file to save extracted message (for extract mode)')
    
    args = parser.parse_args()

    if args.mode == 'embed':
        if args.checkpoint_gen is None:
            raise ValueError("--checkpoint_gen is required for embed mode")
        if args.output_audio is None:
            raise ValueError("--output_audio is required for embed mode")
        if args.message is None:
            # Generate random message
            import yaml
            config_path = args.config or 'configs/vctk_16k.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            message_bits = config['model']['generator']['message_bits']
            args.message = ''.join([str(np.random.randint(0, 2)) for _ in range(message_bits)])
            print(f"Generated random message: {args.message}")
        
        embed_watermark(
            args.input_audio,
            args.output_audio,
            args.checkpoint_gen,
            args.checkpoint_det,
            args.message,
            args.config
        )
    
    elif args.mode == 'extract':
        message, loc_map = extract_watermark(
            args.input_audio,
            args.checkpoint_det,
            args.config
        )
        
        if args.output_message:
            np.save(args.output_message, {'message': message, 'localization': loc_map})
            print(f"Extracted message saved to {args.output_message}")

if __name__ == '__main__':
    main()
