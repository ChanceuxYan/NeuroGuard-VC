import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def train(config_path, resume_checkpoint=None):
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.get('experiment', {}).get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    os.makedirs(config.get('experiment', {}).get('tensorboard_dir', 'logs/tensorboard'), exist_ok=True)

    # Models
    generator = NeuroGuardGenerator(config).to(device)
    detector = NeuroGuardDetector(config).to(device)
    detector.semantic_extractor = generator.semantic_extractor
    print("Shared Semantic Extractor between Generator and Detector to save memory.")
    attack_layer = AttackLayer(config).to(device)

    # Discriminators (optional, for adversarial training)
    use_discriminator = config.get('training', {}).get('use_discriminator', False)
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
    # 注意：需要在generator初始化后才能创建
    semantic_loss_criterion = None
    if hasattr(generator, 'use_semantic') and generator.use_semantic and \
       hasattr(generator, 'semantic_extractor') and generator.semantic_extractor is not None:
        semantic_loss_criterion = SemanticConsistencyLoss(generator.semantic_extractor).to(device)
        print("Semantic consistency loss enabled")
    else:
        print("Semantic stream disabled, semantic consistency loss not used")
    
    # Hard Example Mining (可选)
    use_hard_example = config.get('training', {}).get('use_hard_example_mining', True)
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
        # 测试模式：stage1(1个epoch) + stage2(2个epoch) + stage3(2个epoch) = 5个epoch
        test_epochs = 5
        # 覆盖训练epoch数
        config['training']['epochs'] = test_epochs
        file_logger.info(f"测试模式配置: batch_size={test_batch_size}, max_samples={test_max_samples}, max_val_samples={test_max_val_samples}, epochs={test_epochs}")
        file_logger.info(f"训练阶段分配: epoch 0=stage1, epoch 1-2=stage2, epoch 3-4=stage3")
    
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
            # 创建一个子集
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
    
    # 测试模式下限制验证集大小
    if test_mode:
        original_val_size = len(val_dataset)
        if len(val_dataset) > test_max_val_samples:
            # 创建一个子集
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
    num_workers = 0 if test_mode else config['data'].get('num_workers', 4)  # 测试模式下使用0个worker避免多进程问题
    
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
        
        # Detector加载：兼容旧checkpoint（ConvTranspose -> Upsample+Conv结构变化）
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
        # 恢复best指标
        best_train_loss = checkpoint.get('best_train_loss', float('inf'))
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
        print(f"Best train loss: {best_train_loss:.4f}, Best val acc: {best_val_acc:.3f}")
        file_logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
        file_logger.info(f"Best train loss: {best_train_loss:.4f}, Best val acc: {best_val_acc:.3f}")

    # Training Loop
    for epoch in range(start_epoch, config['training']['epochs']):
        generator.train()
        detector.train()
        if use_discriminator:
            msd.train()
            mpd.train()
        
        epoch_losses = {'total': 0, 'stft': 0, 'loc': 0, 'msg': 0, 'adv': 0, 'sem': 0, 'msg_acc': 0}
        
        # 确定当前epoch的训练阶段（用于日志显示）
        if test_mode:
            if epoch == 0:
                current_training_stage = 'stage1'
            elif epoch <= 2:
                current_training_stage = 'stage2'
            else:
                current_training_stage = 'stage3'
        else:
            use_curriculum = config.get('training', {}).get('curriculum', {}).get('enabled', False)
            if use_curriculum:
                stage1_epochs = config.get('training', {}).get('curriculum', {}).get('stage1_epochs', 66)
                stage2_epochs = config.get('training', {}).get('curriculum', {}).get('stage2_epochs', 66)
                if epoch < stage1_epochs:
                    current_training_stage = 'stage1'
                elif epoch < stage1_epochs + stage2_epochs:
                    current_training_stage = 'stage2'
                else:
                    current_training_stage = 'stage3'
            else:
                current_training_stage = 'stage3'
        
        # 计算当前阶段使用的消息解码损失权重
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
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [{current_training_stage}]", 
                    leave=True, mininterval=0.1, maxinterval=1.0)  # mininterval=0.1实现实时更新，不设置ncols自动撑满终端
        for step, audio_real in enumerate(pbar):
            audio_real = audio_real.to(device)
            # Fix: extract batch size correctly
            if len(audio_real.shape) == 3:  # (B, C, T)
                B = audio_real.shape[0]
            else:  # (B, T)
                B = audio_real.shape[0]
                audio_real = audio_real.unsqueeze(1)  # Add channel dimension
            
            # Ensure audio is in correct format (B, 1, T)
            if audio_real.shape[1] != 1:
                audio_real = audio_real[:, 0:1, :]  # Take first channel if multi-channel
            
            # 1. 准备随机消息
            msg = torch.randint(0, 2, (B, config['model']['generator']['message_bits'])).float().to(device)
            
            # 2. 生成水印音频
            audio_wm, watermark_res, indices = generator(audio_real, msg)
            
            # 3. 施加攻击 (Attack Layer) - 关键步骤
            # 根据课程学习策略或测试模式确定训练阶段
            if test_mode:
                # 测试模式：测试三个阶段，stage1(1轮) + stage2(2轮) + stage3(2轮)
                # epoch 0 -> stage1
                # epoch 1-2 -> stage2 (两轮)
                # epoch 3-4 -> stage3 (两轮)
                if epoch == 0:
                    training_stage = 'stage1'
                elif epoch <= 2:
                    training_stage = 'stage2'
                else:  # epoch >= 3
                    training_stage = 'stage3'
            else:
                # 正常模式：根据课程学习策略确定训练阶段
                use_curriculum = config.get('training', {}).get('curriculum', {}).get('enabled', False)
                if use_curriculum:
                    # 计算训练阶段
                    stage1_epochs = config.get('training', {}).get('curriculum', {}).get('stage1_epochs', 66)
                    stage2_epochs = config.get('training', {}).get('curriculum', {}).get('stage2_epochs', 66)
                    if epoch < stage1_epochs:
                        training_stage = 'stage1'
                    elif epoch < stage1_epochs + stage2_epochs:
                        training_stage = 'stage2'
                    else:
                        training_stage = 'stage3'
                else:
                    training_stage = 'stage3'  # 默认使用完整攻击
            
            # 阶段化攻击：在 stage1 可选择关闭攻击，先让解码器收敛
            if training_stage == 'stage1':
                audio_attacked = audio_wm
            else:
                audio_attacked = attack_layer(audio_wm, global_step=global_step, training_stage=training_stage)
            
            # FSQ 码本利用率监控
            fsq_log_interval = config.get('training', {}).get('fsq_log_interval', 500)
            
            if indices is not None and (global_step % fsq_log_interval == 0):
                # 1. 彻底剥离梯度并移至 CPU，防止计算图残留
                # 使用 .detach() 确保没有任何梯度关联
                indices_detached = indices.detach().cpu()
                
                # 2. 记录标量统计 (这些不容易报错)
                # 使用 .item() 将 Tensor 转换为纯 Python 数字
                try:
                    unique_count = torch.unique(indices_detached).numel()
                    logger.writer.add_scalar('fsq/unique_codes', unique_count, global_step)
                    logger.writer.add_scalar('fsq/min_code', indices_detached.min().item(), global_step)
                    logger.writer.add_scalar('fsq/max_code', indices_detached.max().item(), global_step)
                except Exception as e:
                    print(f"[Warning] FSQ Scalar logging failed: {e}")

                # 3. 记录直方图 (这是最容易报错的部分，加了特级保护)
                # try:
                #     # 转为 numpy
                #     flat_np = indices_detached.numpy().flatten()
                    
                #     # 【关键优化】采样！
                #     # 如果数据点太多(例如 > 10000)，TensorBoard 计算直方图会非常慢甚至报错
                #     # 我们只随机采样 10000 个点，足以看清分布了
                #     if flat_np.size > 10000:
                #         # 随机采样 10000 个
                #         flat_np = np.random.choice(flat_np, 10000, replace=False)
                    
                #     # 强制转为 float32，这是 TensorBoard 最喜欢的类型
                #     flat_np = flat_np.astype(np.float32)
                    
                #     logger.writer.add_histogram('fsq/indices', flat_np, global_step)
                    
                # except Exception as e:
                #     # 如果画图失败，只打印警告，不要让训练停下来！
                #     print(f"[Warning] FSQ Histogram logging skipped: {e}")

            # 监听水印音频（可选）：便于人工检查语义是否可懂
            audio_log_interval = config.get('training', {}).get('audio_log_interval', 2000)
            if global_step % audio_log_interval == 0:
                sr = config.get('data', {}).get('target_sr', 16000)
                logger.writer.add_audio('train/audio_wm', audio_wm[0].detach().cpu(), global_step, sample_rate=sr)
                logger.writer.add_audio('train/audio_clean', audio_real[0].detach().cpu(), global_step, sample_rate=sr)
            
            # 4. 检测器前向传播（用于Generator损失计算）
            # 关键修复：Generator的损失需要梯度流回generator，但不能更新detector参数
            # 解决方案：使用detector的eval模式（固定参数），但保持audio_attacked的梯度
            detector.eval()  # 临时设置为eval模式，固定detector参数
            detector_output = detector(audio_attacked)  # 保持梯度，让梯度可以流回generator
            detector.train()  # 恢复训练模式
            
            if len(detector_output) == 4:
                loc_logits, msg_logits, local_logits, attention_weights = detector_output
            else:
                # 向后兼容旧版本
                loc_logits, msg_logits = detector_output
                local_logits = None
                attention_weights = None
            
            # --- Loss Calculation ---
            
            # A. 感知损失
            loss_stft = stft_criterion(audio_wm, audio_real)
            
            # B. Generator的消息解码损失
            # [关键修改] 处理定位损失的尺寸不匹配问题
            # Detector 输出的 loc_logits 是 (B, 1, T_sem)，而 audio 是 (B, 1, T_wav)
            # 我们只需要生成一个全 1 的 target，尺寸与 loc_logits 一致即可
            target_loc = torch.ones_like(loc_logits) 
            loss_loc = bce_criterion(loc_logits, target_loc)
            
            # Message Target (计算每个样本的损失，用于Hard Example Mining)
            loss_msg_per_sample = bce_criterion_none(msg_logits, msg)  # (B, L)
            loss_msg = loss_msg_per_sample.mean()  # 平均损失（标量）
            
            # Hard Example Mining: 计算BER并应用权重
            if hard_example_miner is not None:
                ber = hard_example_miner.compute_ber(msg_logits, msg)  # (B,)
                # 对消息损失应用权重
                loss_msg_weighted, weights = hard_example_miner.apply_weights_to_loss(
                    loss_msg_per_sample.mean(dim=1), ber  # (B,)
                )
                loss_msg = loss_msg_weighted.mean()
            else:
                # 如果没有Hard Example Mining，使用标准BCE损失
                bce_criterion_mean = torch.nn.BCEWithLogitsLoss()
                loss_msg = bce_criterion_mean(msg_logits, msg)
                weights = None
            
            # C. Adversarial Loss (if using discriminator)
            loss_adv = 0
            if use_discriminator:
                # Generator wants to fool discriminators
                y_d_rs_msd, y_d_gs_msd, _, _ = msd(audio_real, audio_wm)
                y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(audio_real, audio_wm)
                
                # Generator adversarial loss (want discriminators to say "real")
                loss_adv_msd = sum([mse_criterion(y_d_g, torch.ones_like(y_d_g)) for y_d_g in y_d_gs_msd])
                loss_adv_mpd = sum([mse_criterion(y_d_g, torch.ones_like(y_d_g)) for y_d_g in y_d_gs_mpd])
                loss_adv = (loss_adv_msd + loss_adv_mpd) / (len(y_d_gs_msd) + len(y_d_gs_mpd))
            
            # D. Semantic Consistency Loss (if using semantic stream)
            loss_sem = 0
            if semantic_loss_criterion is not None:
                loss_sem = semantic_loss_criterion(audio_real, audio_wm)
            
            # 根据训练阶段动态调整消息解码损失权重
            stage_lambda_msg_config = config['training'].get('stage_lambda_msg', {})
            if stage_lambda_msg_config and training_stage in stage_lambda_msg_config:
                lambda_msg_current = stage_lambda_msg_config[training_stage]
            else:
                lambda_msg_current = config['training']['lambda_msg']
            
            # Combined Loss for Generator
            lambda_perc = config['training']['lambda_perceptual']
            
            # 安全检查：防止配置文件里还是 0
            if lambda_perc == 0.0:
                print("警告: 强制开启感知损失")
                lambda_perc = 0.1
            
            total_loss_G = (lambda_perc * loss_stft) + \
                     (config['training']['lambda_loc'] * loss_loc) + \
                     (lambda_msg_current * loss_msg) + \
                     (100.0 * loss_res_energy)  # 给一个大权重(100.0)压制水印幅度
                     
            if use_discriminator:
                total_loss_G += config['training'].get('lambda_adv', 0.5) * loss_adv
            
            if semantic_loss_criterion is not None:
                total_loss_G += config['training'].get('lambda_sem', 0.1) * loss_sem
            
            # Detector loss (separate, wants to correctly detect)
            total_loss_D = loss_loc + loss_msg
            
            # Optimization - Generator
            # 生成器始终训练，避免长时间“只训D”导致ACC停滞
            opt_G.zero_grad()
            total_loss_G.backward(retain_graph=False)  # 改为False，避免inplace操作问题
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_G.step()
            
            # Optimization - Detector
            # 重新计算detector loss（因为generator的backward可能修改了计算图）
            opt_D.zero_grad()
            # 重新前向传播detector（使用detached的audio_attacked）
            detector_output_det = detector(audio_attacked.detach())
            if len(detector_output_det) == 4:
                loc_logits_det, msg_logits_det, _, _ = detector_output_det
            else:
                loc_logits_det, msg_logits_det = detector_output_det
            
            # [关键修改] 同样在这里适配尺寸
            target_loc_det = torch.ones_like(loc_logits_det) # 这里的全1 target 匹配语义层长度
            loss_loc_det = bce_criterion(loc_logits_det, target_loc_det)
            loss_msg_det = bce_criterion(msg_logits_det, msg)
            total_loss_D = loss_loc_det + loss_msg_det
            
            # 在stage1阶段，增加消息解码损失的权重，让detector更快学习
            if training_stage == 'stage1':
                total_loss_D = loss_loc_det + 2.0 * loss_msg_det  # 增加消息损失权重
            
            total_loss_D.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            opt_D.step()
            
            # Optimization - Discriminators (if using)
            if use_discriminator:
                # Discriminator loss: distinguish real from fake
                opt_MSD.zero_grad()
                opt_MPD.zero_grad()
                
                y_d_rs_msd, y_d_gs_msd, _, _ = msd(audio_real.detach(), audio_wm.detach())
                y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(audio_real.detach(), audio_wm.detach())
                
                loss_d_msd = sum([mse_criterion(y_d_r, torch.ones_like(y_d_r)) + 
                                  mse_criterion(y_d_g, torch.zeros_like(y_d_g)) 
                                  for y_d_r, y_d_g in zip(y_d_rs_msd, y_d_gs_msd)])
                loss_d_mpd = sum([mse_criterion(y_d_r, torch.ones_like(y_d_r)) + 
                                  mse_criterion(y_d_g, torch.zeros_like(y_d_g)) 
                                  for y_d_r, y_d_g in zip(y_d_rs_mpd, y_d_gs_mpd)])
                
                loss_d_msd.backward()
                loss_d_mpd.backward()
                opt_MSD.step()
                opt_MPD.step()
            
            # Update statistics
            epoch_losses['total'] += total_loss_G.item()
            epoch_losses['stft'] += loss_stft.item()
            epoch_losses['loc'] += loss_loc.item()
            epoch_losses['msg'] += loss_msg.item()
            if use_discriminator:
                epoch_losses['adv'] += loss_adv.item()
            if semantic_loss_criterion is not None:
                epoch_losses['sem'] += loss_sem.item()
            
            # 计算准确率（消息解码准确率）
            msg_pred = (torch.sigmoid(msg_logits) > 0.5).float()
            msg_acc = (msg_pred == msg).float().mean().item()
            epoch_losses['msg_acc'] += msg_acc
            # 逐位 ACC/BER 监控，帮助定位是否整体未收敛
            bit_log_interval = config.get('training', {}).get('bit_log_interval', 1000)
            if global_step % bit_log_interval == 0:
                bit_acc = (msg_pred == msg).float().mean(dim=0)  # (bits,)
                logger.writer.add_histogram('msg/bit_acc', bit_acc.cpu(), global_step)
            
            # 诊断信息：每1000个batch打印一次详细诊断
            if step % 1000 == 0 and step > 0:
                # 检查水印信号幅度
                watermark_magnitude = watermark_res.abs().mean().item()
                # 检查水印音频与原始音频的差异
                audio_diff = (audio_wm - audio_real).abs().mean().item()
                # 检查消息logits的分布
                msg_logits_mean = msg_logits.mean().item()
                msg_logits_std = msg_logits.std().item()
                # 检查消息预测的分布（应该接近0.5如果随机）
                msg_pred_mean = msg_pred.mean().item()
                
                print(f"\n[诊断] Step {step}:")
                print(f"  水印信号幅度: {watermark_magnitude:.6f}")
                print(f"  音频差异: {audio_diff:.6f}")
                print(f"  消息logits均值: {msg_logits_mean:.4f}, 标准差: {msg_logits_std:.4f}")
                print(f"  消息预测均值: {msg_pred_mean:.4f} (应该接近0.5)")
                print(f"  消息准确率: {msg_acc:.4f}")
                print(f"  消息损失: {loss_msg.item():.4f}")
                # file_logger.info(f"[诊断] Step {step}: 水印幅度={watermark_magnitude:.6f}, 音频差异={audio_diff:.6f}, "
                #                f"logits均值={msg_logits_mean:.4f}, 准确率={msg_acc:.4f}, 损失={loss_msg.item():.4f}")
            
            # FSQ Warm-up 学习率调整：在 warm-up 阶段结束后，将 FSQ 投影层学习率恢复到正常值
            if use_fsq_warmup and hasattr(generator, 'fsq') and generator.fsq is not None:
                warmup_steps = fsq_warmup_config.get('warmup_steps', 2000)
                base_lr = float(config['training']['lr_gen'])
                if global_step == warmup_steps:
                    # Warm-up 结束，恢复 FSQ 投影层学习率到正常值
                    if len(opt_G.param_groups) > 1:
                        opt_G.param_groups[1]['lr'] = base_lr
                        print(f"\n[FSQ Warm-up] Step {global_step}: FSQ projection layers LR restored to {base_lr:.2e}")
                        file_logger.info(f"[FSQ Warm-up] Step {global_step}: FSQ projection layers LR restored to {base_lr:.2e}")
                elif global_step < warmup_steps:
                    # 可选：线性 warm-up（当前实现是固定高学习率）
                    # 如果需要线性 warm-up，可以在这里实现
                    pass
            
            global_step += 1
            
            # 实时更新进度条（每个batch）
            avg_losses = {k: v / (step + 1) for k, v in epoch_losses.items()}
            postfix_dict = {
                'Loss': f"{avg_losses['total']:.4f}",
                'STFT': f"{avg_losses['stft']:.4f}",
                'Loc': f"{avg_losses['loc']:.4f}",
                'Msg': f"{avg_losses['msg']:.4f}",
                'Acc': f"{avg_losses.get('msg_acc', 0.0):.3f}"
            }
            if use_discriminator:
                postfix_dict['Adv'] = f"{avg_losses.get('adv', 0.0):.4f}"
            if semantic_loss_criterion is not None:
                postfix_dict['Sem'] = f"{avg_losses.get('sem', 0.0):.4f}"
            pbar.set_postfix(postfix_dict)
            pbar.update(1)  # 手动更新进度条
            
            # 每200个batch写入日志文件
            if step % 200 == 0 and step > 0:
                file_logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Step {step}/{len(dataloader)}")
                file_logger.info(f"  Loss_G: {avg_losses['total']:.4f}, Loss_D: {total_loss_D.item():.4f}")
                file_logger.info(f"  STFT: {avg_losses['stft']:.4f}, Loc: {avg_losses['loc']:.4f}, Msg: {avg_losses['msg']:.4f}")
                file_logger.info(f"  Acc: {avg_losses.get('msg_acc', 0.0):.3f}")
                if use_discriminator:
                    file_logger.info(f"  Adv: {avg_losses.get('adv', 0.0):.4f}")
                if semantic_loss_criterion is not None:
                    file_logger.info(f"  Sem: {avg_losses.get('sem', 0.0):.4f}")
                file_logger.info("-" * 80)
            
            # TensorBoard日志（每500个batch）
            if step % 500 == 0:
                logger.log_training(
                    avg_losses['total'],
                    total_loss_D.item(),
                    avg_losses['stft'],
                    global_step
                )
        
        # Validation
        if (epoch + 1) % config.get('experiment', {}).get('eval_interval', 1) == 0:
            generator.eval()
            detector.eval()
            
            if test_mode:
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
            else:
                # 正常模式：验证时使用与训练相同的阶段（保证一致性）
                val_losses = {'total': 0, 'stft': 0, 'loc': 0, 'msg': 0, 'sem': 0, 'msg_acc': 0}
                val_samples = 0
                
                # 确定验证阶段（与训练阶段保持一致）
                if test_mode:
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
    args = parser.parse_args()
    
    train(args.config, args.resume)
