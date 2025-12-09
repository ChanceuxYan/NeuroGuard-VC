import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.audio import dynamic_range_decompression, mel_spectrogram

def plot_spectrogram_to_numpy(spectrogram):
    """
    将频谱图转换为 numpy 数组，用于 TensorBoard 可视化。
    
    Args:
        spectrogram: (F, T) numpy array or torch tensor
    
    Returns:
        (H, W, 3) numpy array for image display
    """
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()
    
    # De-normalize if needed (assuming log mel spectrogram)
    # spectrogram = np.exp(spectrogram) - 1e-5
    
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return data.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)

class Logger(object):
    def __init__(self, log_dir):
        """
        初始化 TensorBoard SummaryWriter
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_training(self, loss_g, loss_d, mel_loss, step):
        """记录训练标量"""
        self.writer.add_scalar('train/loss_g', loss_g, step)
        self.writer.add_scalar('train/loss_d', loss_d, step)
        self.writer.add_scalar('train/mel_loss', mel_loss, step)

    def log_validation(self, reduced_loss, model, y, y_hat, step):
        """
        记录验证集指标及可视化。
        关键点：不仅记录 Loss，还要画出梅尔谱对比图，以便人工检查棋盘格效应。
        """
        self.writer.add_scalar('validation/loss', reduced_loss, step)
        
        # Extract Batch 中的第一个样本进行可视化
        # 重新计算梅尔谱以确保可视化参数一致
        # Default parameters (should match config)
        n_fft = 1024
        n_mels = 80
        sampling_rate = 16000
        hop_length = 256
        win_length = 1024
        mel_fmin = 0.0
        mel_fmax = 8000.0
        
        spec_fake = mel_spectrogram(
            y_hat.squeeze(1), n_fft, n_mels, sampling_rate, 
            hop_length, win_length, mel_fmin, mel_fmax
        )
        spec_real = mel_spectrogram(
            y.squeeze(1), n_fft, n_mels, sampling_rate, 
            hop_length, win_length, mel_fmin, mel_fmax
        )
        
        # 绘制并写入 TensorBoard
        self.writer.add_image(
            'validation/mel_fake', 
            plot_spectrogram_to_numpy(spec_fake.data.cpu().numpy()[0]), 
            step
        )
        self.writer.add_image(
            'validation/mel_real', 
            plot_spectrogram_to_numpy(spec_real.data.cpu().numpy()[0]), 
            step
        )
        
        # 写入音频，支持直接在 TensorBoard 网页端试听
        self.writer.add_audio('validation/audio_fake', y_hat[0], step, sample_rate=sampling_rate)
        self.writer.add_audio('validation/audio_real', y[0], step, sample_rate=sampling_rate)

    def close(self):
        self.writer.close()
