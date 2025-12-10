import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [新增] 引入绘图和音频处理库 (用于离线监控)
import matplotlib
matplotlib.use('Agg') # 非交互模式，防止服务器报错
import matplotlib.pyplot as plt
import torchaudio

from models.generator import NeuroGuardGenerator
from models.detector import NeuroGuardDetector

from models.discriminators import MultiScaleDiscriminator, MultiPeriodDiscriminator
from modules.attack import AttackLayer
from modules.losses import MultiResolutionSTFTLoss, SemanticConsistencyLoss
from modules.hard_example_mining import HardExampleMiner
from data.vctk_dataset import NeuroGuardVCTKDataset
from utils.logger import Logger
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime

# [新增] 训练历史记录容器
history = {
    'steps': [],
    'loss_g': [],
    'loss_d': [],
    'acc': [],
    'fsq_usage': []
}

def save_training_plots(history, save_dir):
    """绘制并保存训练曲线 (离线监控核心)"""
    steps = history['steps']
    if len(steps) == 0: return

    plt.figure(figsize=(12, 8))
    
    # 1. Loss 曲线
    plt.subplot(2, 2, 1)
    plt.plot(steps, history['loss_g'], label='Loss G', alpha=0.7)
    plt.plot(steps, history['loss_d'], label='Loss D', alpha=0.7)
    plt.title('Loss Curves')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy 曲线
    plt.subplot(2, 2, 2)
    plt.plot(steps, history['acc'], color='green', label='Message Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. FSQ Usage 曲线 (如果有)
    # 过滤掉 None 值并对齐 step
    valid_fsq_data = [(s, v) for s, v in zip(steps, history['fsq_usage']) if v is not None]
    if valid_fsq_data:
        fsq_steps, fsq_vals = zip(*valid_fsq_data)
        plt.subplot(2, 2, 3)
        plt.plot(fsq_steps, fsq_vals, color='purple', label='Unique Codes')
        plt.title('FSQ Codebook Usage')
        plt.xlabel('Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(save_dir, 'training_status.png'))
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

def train(config_path, resume_checkpoint=None, debug_overfit=False, debug_single_batch_steps=None):
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Debug overfit: 使用极小子集、stage1-only、无攻击/HEM/判别器
    if debug_overfit:
        print("=" * 80)
        print("DEBUG OVERFIT MODE: stage1 only, no attack/HEM/discriminator, small subset")
        print("=" * 80)
        config.setdefault('training', {})
        config.setdefault('experiment', {})
        config['training']['use_discriminator'] = False
        config['training']['use_hard_example_mining'] = False
        config['training']['lambda_adv'] = 0.0
        config['training']['epochs'] = 1
        # 关闭 curriculum，强制 stage1
        config['training']['curriculum'] = {'enabled': False}
        # 确保 eval_interval 存在
        config['experiment']['eval_interval'] = 1
    
    # Create directories
    os.makedirs(config.get('experiment', {}).get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    os.makedirs(config.get('experiment', {}).get('tensorboard_dir', 'logs/tensorboard'), exist_ok=True)
    
    # [新增] 创建本地监控目录
    log_base = config.get('experiment', {}).get('tensorboard_dir', 'logs')
    plot_dir = os.path.join(log_base, 'plots')
    audio_dir = os.path.join(log_base, 'audio_samples')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Models
    generator = NeuroGuardGenerator(config).to(device)
    detector = NeuroGuardDetector(config).to(device)
    
    # [关键优化] 共享语义提取器以节省显存
    if hasattr(generator, 'semantic_extractor') and generator.semantic_extractor is not None:
        detector.semantic_extractor = generator.semantic_extractor
        print("Shared Semantic Extractor between Generator and Detector to save memory.")
    
    attack_layer = AttackLayer(config).to(device)

    # Discriminators (optional, for adversarial training)
    use_discriminator = config.get('training', {}).get('use_discriminator', False)
    if debug_overfit:
        use_discriminator = False
    if use_discriminator:
        msd = MultiScaleDiscriminator().to(device)
        mpd = MultiPeriodDiscriminator(use_spectral_norm=True).to(device)

    # Optimizers
    # FSQ Warm-up 策略（可选）：为FSQ投影层设置更高的学习率
    fsq_warmup_config = config.get('training', {}).get('fsq_warmup', {})
    use_fsq_warmup = fsq_warmup_config.get('enabled', False)
    
    if use_fsq_warmup and hasattr(generator, 'fsq') and generator.fsq is not None:
        # 分离FSQ投影层和其他参数
        fsq_params = []
        other_params = []
        for name, param in generator.named_parameters():
            if 'fsq.project_in' in name or 'fsq.project_out' in name:
                fsq_params.append(param)
            else:
                other_params.append(param)
        
        # 为FSQ投影层设置更高的学习率
        fsq_lr_multiplier = fsq_warmup_config.get('fsq_lr_multiplier', 2.0)
        base_lr = float(config['training']['lr_gen'])
        opt_G = optim.AdamW([
            {'params': other_params, 'lr': base_lr},
            {'params': fsq_params, 'lr': base_lr * fsq_lr_multiplier}
        ], betas=tuple(config['training']['betas']))
        print(f"FSQ Warm-up enabled: project_in/out LR = {base_lr * fsq_lr_multiplier:.2e} (×{fsq_lr_multiplier})")
    else:
        opt_G = optim.AdamW(generator.parameters(), 
                            lr=float(config['training']['lr_gen']),
                            betas=tuple(config['training']['betas']))
        if use_fsq_warmup:
            print("FSQ Warm-up requested but FSQ not found, using standard optimizer")
    
    opt_D = optim.AdamW(detector.parameters(), 
                        lr=float(config['training']['lr_det']),
                        betas=tuple(config['training']['betas']))
    
    if use_discriminator:
        opt_MSD = optim.AdamW(msd.parameters(), lr=float(config['training']['lr_det']), betas=tuple(config['training']['betas']))
        opt_MPD = optim.AdamW(mpd.parameters(), lr=float(config['training']['lr_det']), betas=tuple(config['training']['betas']))

    # Losses
    stft_criterion = MultiResolutionSTFTLoss().to(device)
    bce_criterion_none = torch.nn.BCEWithLogitsLoss(reduction='none')  # 使用none以便应用Hard Example Mining
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # 标准BCE损失（用于定位损失）
    mse_criterion = torch.nn.MSELoss()
    
    # 语义一致性损失（如果Generator使用语义流）
    semantic_loss_criterion = None
    if hasattr(generator, 'use_semantic') and generator.use_semantic and \
       hasattr(generator, 'semantic_extractor') and generator.semantic_extractor is not None:
        semantic_loss_criterion = SemanticConsistencyLoss(generator.semantic_extractor).to(device)
        print("Semantic consistency loss enabled")
    else:
        print("Semantic stream disabled, semantic consistency loss not used")
    
    # Hard Example Mining (可选)
    use_hard_example = config.get('training', {}).get('use_hard_example_mining', True)
    if debug_overfit:
        use_hard_example = False
    if use_hard_example:
        hard_example_miner = HardExampleMiner(
            top_k_ratio=0.3,
            min_weight=0.5,
            max_weight=2.0
        )
        print("Hard Example Mining enabled")
    else:
        hard_example_miner = None
    
    # Logger
    logger = Logger(config.get('experiment', {}).get('tensorboard_dir', 'logs/tensorboard'))
    
    # 文本日志文件
    log_dir = config.get('experiment', {}).get('tensorboard_dir', 'logs/tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 配置logging
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # 创建logger
    file_logger = logging.getLogger('training')
    file_logger.setLevel(logging.INFO)
    file_logger.addHandler(file_handler)
    file_logger.propagate = False  # 避免重复输出到控制台
    
    print(f"训练日志将保存到: {log_file_path}")
    file_logger.info("=" * 80)
    file_logger.info("NeuroGuard-VC 训练开始")
    file_logger.info(f"配置文件: {config_path}")
    file_logger.info(f"设备: {device}")
    file_logger.info("=" * 80)

    # 检查是否为测试模式
    test_mode = config.get('experiment', {}).get('test_mode', False)
    if test_mode:
        print("=" * 80)
        print("⚠️  测试模式已启用 - 使用小batch和限制样本数进行快速验证")
        print("⚠️  将测试三个阶段：stage1(1轮) + stage2(2轮) + stage3(2轮)")
        print("=" * 80)
        file_logger.info("=" * 80)
        file_logger.info("⚠️  测试模式已启用")
        file_logger.info("⚠️  将测试三个阶段：stage1(1轮) + stage2(2轮) + stage3(2轮)")
        file_logger.info("=" * 80)
        test_batch_size = config.get('experiment', {}).get('test_batch_size', 2)
        test_max_samples = config.get('experiment', {}).get('test_max_samples', 100)
        test_max_val_samples = config.get('experiment', {}).get('test_max_val_samples', 20)
        test_epochs = 5
        config['training']['epochs'] = test_epochs
        file_logger.info(f"测试模式配置: batch_size={test_batch_size}, max_samples={test_max_samples}, max_val_samples={test_max_val_samples}, epochs={test_epochs}")
    
    # Data
    data_config = config['data']
    dataset = NeuroGuardVCTKDataset(
        root_dir=data_config.get('root_path'),
        segment_length=data_config['segment_length'],
        mode='train',
        train_csv=data_config.get('train_csv'),
        val_csv=data_config.get('val_csv')
    )
    
    # 测试模式下限制训练集大小
    if test_mode:
        original_size = len(dataset)
        if len(dataset) > test_max_samples:
            indices = list(range(min(test_max_samples, len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)
            file_logger.info(f"训练集从 {original_size} 限制到 {len(dataset)} 个样本")
    
    file_logger.info(f"训练集大小: {len(dataset)}")
    
    val_dataset = NeuroGuardVCTKDataset(
        root_dir=data_config.get('root_path'),
        segment_length=data_config['segment_length'],
        mode='val',
        train_csv=data_config.get('train_csv'),
        val_csv=data_config.get('val_csv')
    )

    # Debug overfit: 使用极小子集/小batch/单线程
    if debug_overfit:
        debug_train_size = min(8, len(dataset))
        debug_val_size = min(4, len(val_dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(debug_train_size)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(debug_val_size)))
        data_config['batch_size'] = min(2, debug_train_size)
        data_config['num_workers'] = 0
    
    # 测试模式下限制验证集大小
    if test_mode:
        original_val_size = len(val_dataset)
        if len(val_dataset) > test_max_val_samples:
            indices = list(range(min(test_max_val_samples, len(val_dataset))))
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
            file_logger.info(f"验证集从 {original_val_size} 限制到 {len(val_dataset)} 个样本")
    
    file_logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 初始化best指标跟踪
    best_train_loss = float('inf')
    best_val_acc = 0.0
    checkpoint_dir = config.get('experiment', {}).get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_logger.info(f"Checkpoint目录: {checkpoint_dir}")
    
    # 根据测试模式选择batch size
    batch_size = test_batch_size if test_mode else config['data']['batch_size']
    num_workers = 0 if test_mode else config['data'].get('num_workers', 4)
    if debug_overfit:
        batch_size = data_config['batch_size']
        num_workers = 0
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        
        try:
            detector.load_state_dict(checkpoint['detector'], strict=True)
            print("✓ Loaded detector")
        except RuntimeError as e:
            print("⚠ Warning: Detector structure mismatch, attempting partial load...")
            missing_keys, unexpected_keys = detector.load_state_dict(checkpoint['detector'], strict=False)
            if missing_keys:
                print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("✓ Loaded detector (partial, mismatched layers will be retrained)")
            file_logger.warning(f"Detector partial load: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
        
        opt_G.load_state_dict(checkpoint['opt_G'])
        opt_D.load_state_dict(checkpoint['opt_D'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
        file_logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training Loop
    for epoch in range(start_epoch, config['training']['epochs']):
        generator.train()
        detector.train()
        if use_discriminator:
            msd.train()
            mpd.train()
        
        epoch_losses = {'total': 0, 'stft': 0, 'loc': 0, 'msg': 0, 'adv': 0, 'sem': 0, 'msg_acc': 0}
        
        if debug_overfit:
            current_training_stage = 'stage1'
        elif test_mode:
            if epoch == 0: current_training_stage = 'stage1'
            elif epoch <= 2: current_training_stage = 'stage2'
            else: current_training_stage = 'stage3'
        else:
            use_curriculum = config.get('training', {}).get('curriculum', {}).get('enabled', False)
            if use_curriculum:
                stage1_epochs = config.get('training', {}).get('curriculum', {}).get('stage1_epochs', 66)
                stage2_epochs = config.get('training', {}).get('curriculum', {}).get('stage2_epochs', 66)
                if epoch < stage1_epochs: current_training_stage = 'stage1'
                elif epoch < stage1_epochs + stage2_epochs: current_training_stage = 'stage2'
                else: current_training_stage = 'stage3'
            else:
                current_training_stage = 'stage3'
        
        stage_lambda_msg_config = config['training'].get('stage_lambda_msg', {})
        if stage_lambda_msg_config and current_training_stage in stage_lambda_msg_config:
            lambda_msg_current = stage_lambda_msg_config[current_training_stage]
            lambda_msg_info = f" (λ_msg={lambda_msg_current:.1f})"
        else:
            lambda_msg_current = config['training']['lambda_msg']
            lambda_msg_info = f" (λ_msg={lambda_msg_current:.1f})"
        
        epoch_info = f"Epoch {epoch+1}/{config['training']['epochs']} [Training Stage: {current_training_stage}{lambda_msg_info}]"
        print(f"\n{epoch_info}")
        file_logger.info("=" * 80)
        file_logger.info(epoch_info)
        file_logger.info("=" * 80)
        
        # Debug: reuse single batch for N steps
        if debug_single_batch_steps:
            if 'cached_debug_batch' not in locals():
                try:
                    cached_debug_batch = next(iter(dataloader))
                except StopIteration:
                    print("No data available for debug_single_batch_steps.")
                    break
            pbar_iter = tqdm(range(debug_single_batch_steps), desc=f"Epoch {epoch+1} [debug_single_batch]", leave=True, mininterval=0.1, maxinterval=1.0)
            def get_batch():
                return cached_debug_batch
        else:
            pbar_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [{current_training_stage}]", 
                             leave=True, mininterval=0.1, maxinterval=1.0)
            def get_batch():
                return next_batch
        
        for step, batch in enumerate(pbar_iter):
            if debug_single_batch_steps:
                audio_real = get_batch()
            else:
                next_batch = batch
                audio_real = get_batch()

            audio_real = audio_real.to(device)
            # Fix: extract batch size correctly
            if len(audio_real.shape) == 3:  # (B, C, T)
                B = audio_real.shape[0]
            else:  # (B, T)
                B = audio_real.shape[0]
                audio_real = audio_real.unsqueeze(1)
            
            if audio_real.shape[1] != 1:
                audio_real = audio_real[:, 0:1, :]
            
            # 1. 准备随机消息
            msg = torch.randint(0, 2, (B, config['model']['generator']['message_bits'])).float().to(device)
            
            # 2. 生成水印音频
            audio_wm, watermark_res, indices = generator(audio_real, msg)
            # audio_wm.retain_grad()
            
            # 3. 施加攻击
            if debug_overfit:
                training_stage = 'stage1'
            elif test_mode:
                if epoch == 0: training_stage = 'stage1'
                elif epoch <= 2: training_stage = 'stage2'
                else: training_stage = 'stage3'
            else:
                use_curriculum = config.get('training', {}).get('curriculum', {}).get('enabled', False)
                if use_curriculum:
                    stage1_epochs = config.get('training', {}).get('curriculum', {}).get('stage1_epochs', 66)
                    stage2_epochs = config.get('training', {}).get('curriculum', {}).get('stage2_epochs', 66)
                    if epoch < stage1_epochs: training_stage = 'stage1'
                    elif epoch < stage1_epochs + stage2_epochs: training_stage = 'stage2'
                    else: training_stage = 'stage3'
                else:
                    training_stage = 'stage3'
            
            if training_stage == 'stage1' or debug_overfit:
                audio_attacked = audio_wm
            else:
                audio_attacked = attack_layer(audio_wm, global_step=global_step, training_stage=training_stage)
            
            # [关键修改] FSQ 码本利用率监控 - 离线版
            fsq_log_interval = config.get('training', {}).get('fsq_log_interval', 500)
            current_fsq_usage = None # 初始化
            
            if indices is not None and (global_step % fsq_log_interval == 0):
                indices_detached = indices.detach().cpu()
                try:
                    unique_count = torch.unique(indices_detached).numel()
                    current_fsq_usage = unique_count
                    
                    # [修改] 显式打印到文本日志
                    fsq_info = f"[FSQ] Step {global_step}: Unique Codes = {unique_count} / {config['model']['generator']['semantic'].get('fsq_levels', 'Unknown')}"
                    print(f"\n{fsq_info}")
                    file_logger.info(fsq_info)
                    
                    # 尝试保留 TensorBoard 调用
                    logger.writer.add_scalar('fsq/unique_codes', unique_count, global_step)
                except Exception as e:
                    print(f"[Warning] FSQ logging failed: {e}")

            # [关键修改] 保存音频文件到本地 (离线监听)
            audio_log_interval = config.get('training', {}).get('audio_log_interval', 2000)
            if global_step % audio_log_interval == 0:
                sr = config.get('data', {}).get('target_sr', 16000)
                try:
                    # 保存 .wav 文件到 logs/audio_samples/
                    wm_wav = audio_wm[0].detach().cpu()
                    real_wav = audio_real[0].detach().cpu()
                    
                    torchaudio.save(os.path.join(audio_dir, f"step_{global_step}_wm.wav"), wm_wav, sr)
                    torchaudio.save(os.path.join(audio_dir, f"step_{global_step}_clean.wav"), real_wav, sr)
                    file_logger.info(f"Saved audio samples to {audio_dir} at step {global_step}")
                except Exception as e:
                    print(f"Error saving audio: {e}")

                logger.writer.add_audio('train/audio_wm', audio_wm[0].detach().cpu(), global_step, sample_rate=sr)
            
            # 4. 检测器前向传播
            detector.eval()
            detector_output = detector(audio_attacked)
            detector.train()
            
            if len(detector_output) == 4:
                loc_logits, msg_logits, local_logits, attention_weights = detector_output
            else:
                loc_logits, msg_logits = detector_output
            
            # --- Loss Calculation ---
            loss_stft = stft_criterion(audio_wm, audio_real)
            loss_res_energy = 0 # 暂时设为0，后续阶段可恢复
            
            # [维度对齐修正]
            target_loc = torch.ones_like(loc_logits)
            loss_loc = bce_criterion(loc_logits, target_loc)
            
            loss_msg_per_sample = bce_criterion_none(msg_logits, msg)
            loss_msg = loss_msg_per_sample.mean()
            
            if hard_example_miner is not None:
                ber = hard_example_miner.compute_ber(msg_logits, msg)
                loss_msg_weighted, weights = hard_example_miner.apply_weights_to_loss(
                    loss_msg_per_sample.mean(dim=1), ber
                )
                loss_msg = loss_msg_weighted.mean()
            else:
                bce_criterion_mean = torch.nn.BCEWithLogitsLoss()
                loss_msg = bce_criterion_mean(msg_logits, msg)
            
            loss_adv = 0
            if use_discriminator:
                # ... (Discriminator loss calculation)
                pass
            
            loss_sem = 0
            if semantic_loss_criterion is not None:
                loss_sem = semantic_loss_criterion(audio_real, audio_wm)
            
            lambda_perc = config['training']['lambda_perceptual']
            # if lambda_perc == 0.0: lambda_perc = 0.1
            
            total_loss_G = (lambda_perc * loss_stft) + \
                     (config['training']['lambda_loc'] * loss_loc) + \
                     (lambda_msg_current * loss_msg) + \
                     (100.0 * loss_res_energy)
                     
            if use_discriminator:
                total_loss_G += config['training'].get('lambda_adv', 0.5) * loss_adv
            if semantic_loss_criterion is not None:
                total_loss_G += config['training'].get('lambda_sem', 0.1) * loss_sem
            
            total_loss_D = loss_loc + loss_msg
            
            # Optimization - Generator
            # Optimization - Generator
            opt_G.zero_grad()
            total_loss_G.backward(retain_graph=False)
            
            # [新增] === 梯度探针 (Gradient Probe) ===
            # 只有在 debug 时打开，确认 Generator 是否收到了 Detector 的反馈
            # if step % 100 == 0:
            #     print("\n[Gradient Check]")
            #     if audio_wm.grad is not None:
            #         grad_norm = audio_wm.grad.norm().item()
            #         print(f"  Audio WM Grad Norm: {grad_norm:.6f} (Should be > 0)")
            #         if grad_norm == 0.0:
            #             print("  🔴 警告: 梯度断流！Generator 收不到反馈！")
            #     else:
            #         print("  🔴 严重警告: Audio WM 没有梯度 (None)！")
                    
            #     # 检查 Decoder Head 是否有梯度
            #     for name, param in detector.decoder_head.named_parameters():
            #         if param.grad is not None:
            #             print(f"  Detector Head Grad: {param.grad.norm().item():.6f}")
            #             break
            # ========================================

            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_G.step()
            
            # Optimization - Detector
            opt_D.zero_grad()
            detector_output_det = detector(audio_attacked.detach())
            if len(detector_output_det) == 4:
                loc_logits_det, msg_logits_det, _, _ = detector_output_det
            else:
                loc_logits_det, msg_logits_det = detector_output_det
            
            # [维度对齐修正]
            target_loc_det = torch.ones_like(loc_logits_det)
            loss_loc_det = bce_criterion(loc_logits_det, target_loc_det)
            loss_msg_det = bce_criterion(msg_logits_det, msg)
            total_loss_D = loss_loc_det + loss_msg_det
            
            if training_stage == 'stage1':
                total_loss_D = loss_loc_det + 2.0 * loss_msg_det
            
            total_loss_D.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            opt_D.step()
            
            # Optimization - Discriminators
            if use_discriminator:
                # ... (Discriminator optimization)
                pass
            
            # Update statistics
            epoch_losses['total'] += total_loss_G.item()
            epoch_losses['stft'] += loss_stft.item()
            epoch_losses['loc'] += loss_loc.item()
            epoch_losses['msg'] += loss_msg.item()
            if use_discriminator: epoch_losses['adv'] += loss_adv.item()
            if semantic_loss_criterion is not None: epoch_losses['sem'] += loss_sem.item()
            
            msg_pred = (torch.sigmoid(msg_logits) > 0.5).float()
            msg_acc = (msg_pred == msg).float().mean().item()
            epoch_losses['msg_acc'] += msg_acc
            
            global_step += 1
            
            avg_losses = {k: v / (step + 1) for k, v in epoch_losses.items()}
            postfix_dict = {
                'Loss': f"{avg_losses['total']:.4f}",
                'STFT': f"{avg_losses['stft']:.4f}",
                'Msg': f"{avg_losses['msg']:.4f}",
                'Acc': f"{avg_losses.get('msg_acc', 0.0):.3f}"
            }
            pbar_iter.set_postfix(postfix_dict)
            pbar_iter.update(1)
            
            # [新增] 收集历史数据
            if step % 100 == 0:
                history['steps'].append(global_step)
                history['loss_g'].append(avg_losses['total'])
                history['loss_d'].append(total_loss_D.item())
                history['acc'].append(avg_losses.get('msg_acc', 0.0))
                # 只有当 current_fsq_usage 不为 None 时才是有意义的数据，但为了对齐x轴，可以 append None 或填充
                history['fsq_usage'].append(current_fsq_usage)

            # [新增] 定期绘图 (每500步)
            if step % 500 == 0:
                try:
                    save_training_plots(history, plot_dir)
                    # print(f"Training plots saved to {plot_dir}/training_status.png")
                except Exception as e:
                    print(f"Plotting failed: {e}")

            # 日志写入
            if step % 200 == 0 and step > 0:
                file_logger.info(f"Epoch {epoch+1}, Step {step}")
                file_logger.info(f"  Loss_G: {avg_losses['total']:.4f}, Loss_D: {total_loss_D.item():.4f}")
                file_logger.info(f"  Acc: {avg_losses.get('msg_acc', 0.0):.3f}")
            
            if step % 500 == 0:
                logger.log_training(avg_losses['total'], total_loss_D.item(), avg_losses['stft'], global_step)
        
        # Validation
        if (epoch + 1) % config.get('experiment', {}).get('eval_interval', 1) == 0:
            generator.eval()
            detector.eval()
            
            if test_mode and (not debug_overfit):
                # 测试模式：验证所有三个阶段，确保所有模块都正常工作
                val_stages = ['stage1', 'stage2', 'stage3']
                stage_val_results = {}
                
                print("\n" + "=" * 80)
                print(f"测试模式：验证所有三个阶段（当前训练阶段: {training_stage}）")
                print("=" * 80)
                file_logger.info("=" * 80)
                file_logger.info(f"测试模式：验证所有三个阶段（当前训练阶段: {training_stage}）")
                file_logger.info("=" * 80)
                
                for val_stage in val_stages:
                    val_losses = {'total': 0, 'stft': 0, 'loc': 0, 'msg': 0, 'sem': 0, 'msg_acc': 0}
                    val_samples = 0
                    
                    with torch.no_grad():
                        # 测试模式下限制验证样本数
                        max_val_steps = 3
                        for val_step, audio_real in enumerate(val_dataloader):
                            if val_step >= max_val_steps:
                                break
                            
                            audio_real = audio_real.to(device)
                            if len(audio_real.shape) == 2:
                                audio_real = audio_real.unsqueeze(1)
                            if audio_real.shape[1] != 1:
                                audio_real = audio_real[:, 0:1, :]
                            
                            B = audio_real.shape[0]
                            msg = torch.randint(0, 2, (B, config['model']['generator']['message_bits'])).float().to(device)
                            
                            # 生成水印音频
                            audio_wm, _, _ = generator(audio_real, msg)
                            
                            # 使用当前阶段进行攻击
                            audio_attacked = attack_layer(audio_wm, global_step=global_step, training_stage=val_stage)
                            
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
                            
                            # 计算验证准确率
                            msg_pred = (torch.sigmoid(msg_logits) > 0.5).float()
                            msg_acc = (msg_pred == msg).float().mean().item()
                            
                            # 语义一致性损失（如果启用）
                            loss_sem = 0
                            if semantic_loss_criterion is not None:
                                loss_sem = semantic_loss_criterion(audio_real, audio_wm)
                            
                            total_loss = (config['training']['lambda_perceptual'] * loss_stft) + \
                                        (config['training']['lambda_loc'] * loss_loc) + \
                                        (config['training']['lambda_msg'] * loss_msg)
                            
                            if semantic_loss_criterion is not None:
                                total_loss += config['training'].get('lambda_sem', 0.1) * loss_sem
                            
                            val_losses['total'] += total_loss.item()
                            val_losses['stft'] += loss_stft.item()
                            val_losses['loc'] += loss_loc.item()
                            val_losses['msg'] += loss_msg.item()
                            val_losses['msg_acc'] += msg_acc
                            if semantic_loss_criterion is not None:
                                val_losses['sem'] += loss_sem.item()
                            val_samples += 1
                            
                            # 只在stage3的第一个样本时记录到TensorBoard
                            if val_stage == 'stage3' and val_step == 0:
                                logger.log_validation(
                                    total_loss.item(),
                                    generator,
                                    audio_real[:1],
                                    audio_wm[:1],
                                    global_step
                                )
                    
                    # 计算该阶段的平均指标
                    if val_samples > 0:
                        avg_stage_losses = {k: v / val_samples for k, v in val_losses.items()}
                        stage_val_results[val_stage] = avg_stage_losses
                        
                        # 打印该阶段的验证结果
                        stage_info = (f"Validation [{val_stage}] - Loss: {avg_stage_losses['total']:.4f}, "
                                     f"STFT: {avg_stage_losses['stft']:.4f}, "
                                     f"Loc: {avg_stage_losses['loc']:.4f}, "
                                     f"Msg: {avg_stage_losses['msg']:.4f}, "
                                     f"Acc: {avg_stage_losses.get('msg_acc', 0.0):.3f}")
                        if semantic_loss_criterion is not None and 'sem' in avg_stage_losses:
                            stage_info += f", Sem: {avg_stage_losses['sem']:.4f}"
                        print(f"\n{stage_info}")
                        file_logger.info(stage_info)
                
                # 使用stage3的结果作为主要指标（用于best模型判断）
                if stage_val_results and 'stage3' in stage_val_results:
                    avg_val_losses = stage_val_results['stage3']
                else:
                    # 如果没有stage3，使用最后一个阶段
                    avg_val_losses = stage_val_results[list(stage_val_results.keys())[-1]] if stage_val_results else {}
                
                print("\n" + "=" * 80)
                print("所有阶段验证完成 ✓")
                print("=" * 80)
                file_logger.info("=" * 80)
                file_logger.info("所有阶段验证完成 ✓")
                file_logger.info("=" * 80)
            elif debug_overfit:
                # Debug: 仅 stage1、无攻击
                val_stage = 'stage1'
                val_losses = {'total': 0, 'stft': 0, 'loc': 0, 'msg': 0, 'sem': 0, 'msg_acc': 0}
                val_samples = 0
                with torch.no_grad():
                    for val_step, audio_real in enumerate(val_dataloader):
                        audio_real = audio_real.to(device)
                        if len(audio_real.shape) == 2:
                            audio_real = audio_real.unsqueeze(1)
                        if audio_real.shape[1] != 1:
                            audio_real = audio_real[:, 0:1, :]
                        
                        B = audio_real.shape[0]
                        msg = torch.randint(0, 2, (B, config['model']['generator']['message_bits'])).float().to(device)
                        
                        audio_wm, _, _ = generator(audio_real, msg)
                        audio_attacked = audio_wm  # no attack
                        
                        detector_output = detector(audio_attacked)
                        if len(detector_output) == 4:
                            loc_logits, msg_logits, _, _ = detector_output
                        else:
                            loc_logits, msg_logits = detector_output
                        
                        loss_stft = stft_criterion(audio_wm, audio_real)
                        target_loc = torch.ones_like(loc_logits)
                        loss_loc = bce_criterion(loc_logits, target_loc)
                        loss_msg = bce_criterion(msg_logits, msg)
                        
                        msg_pred = (torch.sigmoid(msg_logits) > 0.5).float()
                        msg_acc = (msg_pred == msg).float().mean().item()
                        
                        loss_sem = 0
                        if semantic_loss_criterion is not None:
                            loss_sem = semantic_loss_criterion(audio_real, audio_wm)
                        
                        total_loss = (config['training']['lambda_perceptual'] * loss_stft) + \
                                    (config['training']['lambda_loc'] * loss_loc) + \
                                    (config['training']['lambda_msg'] * loss_msg)
                        if semantic_loss_criterion is not None:
                            total_loss += config['training'].get('lambda_sem', 0.1) * loss_sem
                        
                        val_losses['total'] += total_loss.item()
                        val_losses['stft'] += loss_stft.item()
                        val_losses['loc'] += loss_loc.item()
                        val_losses['msg'] += loss_msg.item()
                        val_losses['msg_acc'] += msg_acc
                        if semantic_loss_criterion is not None:
                            val_losses['sem'] += loss_sem.item()
                        val_samples += 1
                
                if val_samples > 0:
                    avg_val_losses = {k: v / val_samples for k, v in val_losses.items()}
                else:
                    avg_val_losses = {k: 0 for k in val_losses.keys()}
                
                val_info = (f"Validation [{val_stage}] - Loss: {avg_val_losses['total']:.4f}, "
                           f"STFT: {avg_val_losses['stft']:.4f}, "
                           f"Loc: {avg_val_losses['loc']:.4f}, "
                           f"Msg: {avg_val_losses['msg']:.4f}, "
                           f"Acc: {avg_val_losses.get('msg_acc', 0.0):.3f}")
                if semantic_loss_criterion is not None and 'sem' in avg_val_losses:
                    val_info += f", Sem: {avg_val_losses['sem']:.4f}"
                print(f"\n{val_info}")
                file_logger.info(val_info)
            
            else:
                # 正常模式：验证时使用与训练相同的阶段（保证一致性）
                val_losses = {'total': 0, 'stft': 0, 'loc': 0, 'msg': 0, 'sem': 0, 'msg_acc': 0}
                val_samples = 0
                
                # 确定验证阶段（与训练阶段保持一致）
                if debug_overfit:
                    val_stage = 'stage1'
                elif test_mode:
                    if epoch == 0:
                        val_stage = 'stage1'
                    elif epoch <= 2:
                        val_stage = 'stage2'
                    else:
                        val_stage = 'stage3'
                else:
                    use_curriculum = config.get('training', {}).get('curriculum', {}).get('enabled', False)
                    if use_curriculum:
                        stage1_epochs = config.get('training', {}).get('curriculum', {}).get('stage1_epochs', 66)
                        stage2_epochs = config.get('training', {}).get('curriculum', {}).get('stage2_epochs', 66)
                        if epoch < stage1_epochs:
                            val_stage = 'stage1'
                        elif epoch < stage1_epochs + stage2_epochs:
                            val_stage = 'stage2'
                        else:
                            val_stage = 'stage3'
                    else:
                        val_stage = 'stage3'  # 默认使用完整攻击
                
                with torch.no_grad():
                    max_val_steps = 10
                    for val_step, audio_real in enumerate(val_dataloader):
                        if val_step >= max_val_steps:
                            break
                        
                        audio_real = audio_real.to(device)
                        if len(audio_real.shape) == 2:
                            audio_real = audio_real.unsqueeze(1)
                        if audio_real.shape[1] != 1:
                            audio_real = audio_real[:, 0:1, :]
                        
                        B = audio_real.shape[0]
                        msg = torch.randint(0, 2, (B, config['model']['generator']['message_bits'])).float().to(device)
                        
                        audio_wm, _, _ = generator(audio_real, msg)
                        
                        # 验证时使用与训练相同的阶段（保证一致性）
                        # 阶段化攻击：在 stage1 关闭攻击，与训练保持一致
                        if val_stage == 'stage1':
                            audio_attacked = audio_wm
                        else:
                            audio_attacked = attack_layer(audio_wm, global_step=global_step, training_stage=val_stage)
                        
                        detector_output = detector(audio_attacked)
                        if len(detector_output) == 4:
                            loc_logits, msg_logits, _, _ = detector_output
                        else:
                            loc_logits, msg_logits = detector_output
                        
                        loss_stft = stft_criterion(audio_wm, audio_real)
                        target_loc = torch.ones_like(loc_logits)
                        loss_loc = bce_criterion(loc_logits, target_loc)
                        loss_msg = bce_criterion(msg_logits, msg)
                        
                        # 计算验证准确率
                        msg_pred = (torch.sigmoid(msg_logits) > 0.5).float()
                        msg_acc = (msg_pred == msg).float().mean().item()
                        
                        # 诊断信息：检查 logits 分布（仅在第一个验证样本时打印）
                        if val_step == 0:
                            msg_logits_mean = msg_logits.mean().item()
                            msg_logits_std = msg_logits.std().item()
                            msg_logits_min = msg_logits.min().item()
                            msg_logits_max = msg_logits.max().item()
                            msg_probs = torch.sigmoid(msg_logits)
                            msg_probs_mean = msg_probs.mean().item()
                            # 计算每个位的准确率
                            bit_acc = (msg_pred == msg).float().mean(dim=0)  # (bits,)
                            bit_acc_mean = bit_acc.mean().item()
                            bit_acc_min = bit_acc.min().item()
                            bit_acc_max = bit_acc.max().item()
                            
                            print(f"\n[验证诊断] Step {val_step}:")
                            print(f"  Logits: 均值={msg_logits_mean:.4f}, 标准差={msg_logits_std:.4f}, "
                                  f"范围=[{msg_logits_min:.2f}, {msg_logits_max:.2f}]")
                            print(f"  Probs:  均值={msg_probs_mean:.4f} (应该接近0.5)")
                            print(f"  ACC:    整体={msg_acc:.4f}, 逐位均值={bit_acc_mean:.4f}, "
                                  f"范围=[{bit_acc_min:.4f}, {bit_acc_max:.4f}]")
                            print(f"  Loss:   Msg={loss_msg.item():.4f} (未加权)")
                        
                        # 语义一致性损失（如果启用）
                        loss_sem = 0
                        if semantic_loss_criterion is not None:
                            loss_sem = semantic_loss_criterion(audio_real, audio_wm)
                        
                        total_loss = (config['training']['lambda_perceptual'] * loss_stft) + \
                                    (config['training']['lambda_loc'] * loss_loc) + \
                                    (config['training']['lambda_msg'] * loss_msg)
                        
                        if semantic_loss_criterion is not None:
                            total_loss += config['training'].get('lambda_sem', 0.1) * loss_sem
                        
                        val_losses['total'] += total_loss.item()
                        val_losses['stft'] += loss_stft.item()
                        val_losses['loc'] += loss_loc.item()
                        val_losses['msg'] += loss_msg.item()
                        val_losses['msg_acc'] += msg_acc
                        if semantic_loss_criterion is not None:
                            val_losses['sem'] += loss_sem.item()
                        val_samples += 1
                        
                        if val_step == 0:  # Log first validation sample
                            logger.log_validation(
                                total_loss.item(),
                                generator,
                                audio_real[:1],
                                audio_wm[:1],
                                global_step
                            )
                
                avg_val_losses = {k: v / val_samples for k, v in val_losses.items()}
                val_info = (f"Validation [{val_stage}] - Loss: {avg_val_losses['total']:.4f}, "
                           f"STFT: {avg_val_losses['stft']:.4f}, "
                           f"Loc: {avg_val_losses['loc']:.4f}, "
                           f"Msg: {avg_val_losses['msg']:.4f}, "
                           f"Acc: {avg_val_losses.get('msg_acc', 0.0):.3f}")
                if semantic_loss_criterion is not None and 'sem' in avg_val_losses:
                    val_info += f", Sem: {avg_val_losses['sem']:.4f}"
                print(f"\n{val_info}")
                file_logger.info(val_info)
            
            # 计算当前epoch的平均训练loss
            avg_train_loss = epoch_losses['total'] / len(dataloader)
            current_val_acc = avg_val_losses.get('msg_acc', 0.0)
            
            # 判断是否为best模型（loss更低或acc更高）
            is_best_loss = avg_train_loss < best_train_loss
            is_best_acc = current_val_acc > best_val_acc
            
            if is_best_loss:
                best_train_loss = avg_train_loss
                file_logger.info(f"✓ 新的最佳训练Loss: {best_train_loss:.4f}")
            
            if is_best_acc:
                best_val_acc = current_val_acc
                file_logger.info(f"✓ 新的最佳验证Acc: {best_val_acc:.3f}")
            
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'generator': generator.state_dict(),
                'detector': detector.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'train_loss': avg_train_loss,
                'val_acc': current_val_acc,
                'best_train_loss': best_train_loss,
                'best_val_acc': best_val_acc,
                'config': config
            }
            if use_discriminator:
                checkpoint['msd'] = msd.state_dict()
                checkpoint['mpd'] = mpd.state_dict()
                checkpoint['opt_MSD'] = opt_MSD.state_dict()
                checkpoint['opt_MPD'] = opt_MPD.state_dict()
            
            # 保存checkpoint（只保存latest和best，不保存中间checkpoint）
            # 保存latest checkpoint（每个epoch都更新）
            latest_path = os.path.join(checkpoint_dir, "latest.pth")
            torch.save(checkpoint, latest_path)
            file_logger.info(f"Saved latest checkpoint: {latest_path}")
            
            # 保存best checkpoint（如果loss更低或acc更高）
            if is_best_loss or is_best_acc:
                best_path = os.path.join(checkpoint_dir, "best.pth")
                torch.save(checkpoint, best_path)
                best_reason = []
                if is_best_loss:
                    best_reason.append(f"loss={best_train_loss:.4f}")
                if is_best_acc:
                    best_reason.append(f"acc={best_val_acc:.3f}")
                best_info = f"Saved best checkpoint: {best_path} ({', '.join(best_reason)})"
                print(best_info)
                file_logger.info(best_info)
    
    logger.close()
    file_logger.info("=" * 80)
    file_logger.info("训练完成！")
    file_logger.info("=" * 80)
    file_handler.close()
    print("Training completed!")
    print(f"训练日志已保存到: {log_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vctk_16k.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug_overfit', action='store_true', help='Debug: small subset, stage1 only, no attack/HEM/discriminator')
    parser.add_argument('--debug_single_batch_steps', type=int, default=None, help='Debug: reuse first batch for N steps (e.g., 2000)')
    args = parser.parse_args()
    
    train(args.config, args.resume, debug_overfit=args.debug_overfit, debug_single_batch_steps=args.debug_single_batch_steps)
