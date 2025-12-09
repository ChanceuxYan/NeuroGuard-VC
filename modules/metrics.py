import torch
import torch.nn.functional as F
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

class NeuroGuardMetrics:
    def __init__(self, sample_rate=16000, device='cuda'):
        self.device = device
        self.pesq = PerceptualEvaluationSpeechQuality(sample_rate, 'wb').to(device)
        self.si_snr = ScaleInvariantSignalNoiseRatio().to(device)

    def compute_ber(self, pred_logits, target_msg):
        """
        Bit Error Rate Calculation
        pred_logits: (B, N_bits) - raw output from detector
        target_msg: (B, N_bits) - binary ground truth
        """
        # Sigmoid -> Round -> Binary
        probs = torch.sigmoid(pred_logits)
        preds = torch.round(probs)
        
        # Correct bits
        correct = (preds == target_msg).float().sum()
        total = target_msg.numel()
        
        ber = 1.0 - (correct / total)
        return ber.item() * 100.0 # Return as percentage

    def compute_audio_metrics(self, original, watermarked):
        """
        Returns {'pesq': float, 'snr': float}
        """
        # Ensure lengths match for SI-SNR
        min_len = min(original.shape[-1], watermarked.shape[-1])
        orig = original[..., :min_len]
        wm = watermarked[..., :min_len]

        with torch.no_grad():
            pesq_score = self.pesq(wm, orig).item()
            snr_score = self.si_snr(wm, orig).item()
            
        return {'pesq': pesq_score, 'snr': snr_score}