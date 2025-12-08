#!/usr/bin/env python
"""
测试主要模块是否能正常运行
"""
import sys
import os
import yaml
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """测试1: 导入所有主要模块"""
    print('=' * 60)
    print('测试1: 导入所有主要模块')
    print('=' * 60)
    
    modules_to_test = [
        ('models.generator', 'NeuroGuardGenerator'),
        ('models.detector', 'NeuroGuardDetector'),
        ('models.discriminators', 'MultiScaleDiscriminator'),
        ('models.discriminators', 'MultiPeriodDiscriminator'),
        ('modules.attack', 'AttackLayer'),
        ('modules.losses', 'MultiResolutionSTFTLoss'),
        ('modules.losses', 'SemanticConsistencyLoss'),
        ('modules.hard_example_mining', 'HardExampleMiner'),
        ('modules.ddsp_proxy', 'DDSPVCProxy'),
        ('modules.vae_proxy', 'VAEVCProxy'),
        ('data.vctk_dataset', 'NeuroGuardVCTKDataset'),
        ('utils.logger', 'Logger'),
    ]
    
    results = []
    for module_name, class_name in modules_to_test:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f'✓ {class_name} 导入成功')
        results.append((class_name, True, None))
    
    print()
    return results

def test_model_initialization():
    """测试2: 模型初始化"""
    print('=' * 60)
    print('测试2: 模型初始化')
    print('=' * 60)
    
    # 加载配置
    config_path = 'configs/vctk_16k.yaml'
    if not os.path.exists(config_path):
        print(f'✗ 配置文件不存在: {config_path}')
        return []
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print()
    
    results = []
    
    # 测试Generator
    from models.generator import NeuroGuardGenerator
    generator = NeuroGuardGenerator(config).to(device)
    param_count = sum(p.numel() for p in generator.parameters())
    print(f'✓ NeuroGuardGenerator 初始化成功')
    print(f'  - 参数量: {param_count:,}')
    results.append(('NeuroGuardGenerator', True, None))
    
    # 测试Detector
    from models.detector import NeuroGuardDetector
    detector = NeuroGuardDetector(config).to(device)
    param_count = sum(p.numel() for p in detector.parameters())
    print(f'✓ NeuroGuardDetector 初始化成功')
    print(f'  - 参数量: {param_count:,}')
    results.append(('NeuroGuardDetector', True, None))
    
    # 测试AttackLayer
    from modules.attack import AttackLayer
    attack_layer = AttackLayer(config).to(device)
    print(f'✓ AttackLayer 初始化成功')
    results.append(('AttackLayer', True, None))
    
    print()
    return results

def test_forward_pass():
    """测试3: 前向传播"""
    print('=' * 60)
    print('测试3: 前向传播测试')
    print('=' * 60)
    
    # 加载配置
    config_path = 'configs/vctk_16k.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    T = 32000  # 2秒 @ 16kHz
    msg_bits = config['model']['generator']['message_bits']
    
    # 创建测试数据
    audio_real = torch.randn(B, 1, T).to(device)
    msg = torch.randint(0, 2, (B, msg_bits)).float().to(device)
    
    results = []
    
    # 测试Generator
    from models.generator import NeuroGuardGenerator
    generator = NeuroGuardGenerator(config).to(device)
    generator.eval()
    
    with torch.no_grad():
        audio_wm, watermark_res, indices = generator(audio_real, msg)
        print('✓ Generator 前向传播成功')
        print(f'  - 输入形状: {audio_real.shape}')
        print(f'  - 输出形状: {audio_wm.shape}')
        print(f'  - 残差形状: {watermark_res.shape if watermark_res is not None else None}')
        if indices is not None:
            print(f'  - FSQ量化索引形状: {indices.shape}')
        results.append(('Generator Forward', True, None))
    
    # 测试AttackLayer
    from modules.attack import AttackLayer
    attack_layer = AttackLayer(config).to(device)
    attack_layer.eval()
    
    with torch.no_grad():
        audio_attacked = attack_layer(audio_wm, global_step=1000, training_stage='stage3')
        print('✓ AttackLayer 前向传播成功')
        print(f'  - 输入形状: {audio_wm.shape}')
        print(f'  - 输出形状: {audio_attacked.shape}')
        results.append(('AttackLayer Forward', True, None))
    
    # 测试Detector
    from models.detector import NeuroGuardDetector
    detector = NeuroGuardDetector(config).to(device)
    detector.eval()
    
    with torch.no_grad():
        detector_output = detector(audio_attacked)
        if len(detector_output) == 4:
            loc_logits, msg_logits, local_logits, attention_weights = detector_output
            print('✓ Detector 前向传播成功（4输出）')
        else:
            loc_logits, msg_logits = detector_output
            print('✓ Detector 前向传播成功（2输出）')
        print(f'  - 输入形状: {audio_attacked.shape}')
        print(f'  - 定位logits形状: {loc_logits.shape}')
        print(f'  - 消息logits形状: {msg_logits.shape}')
        results.append(('Detector Forward', True, None))
    
    print()
    return results

def test_losses():
    """测试4: 损失函数"""
    print('=' * 60)
    print('测试4: 损失函数测试')
    print('=' * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    T = 32000
    msg_bits = 32
    
    audio_real = torch.randn(B, 1, T).to(device)
    audio_wm = torch.randn(B, 1, T).to(device)
    msg = torch.randint(0, 2, (B, msg_bits)).float().to(device)
    
    results = []
    
    # 测试MultiResolutionSTFTLoss
    from modules.losses import MultiResolutionSTFTLoss
    stft_loss = MultiResolutionSTFTLoss().to(device)
    loss = stft_loss(audio_wm, audio_real)
    print(f'✓ MultiResolutionSTFTLoss 计算成功: {loss.item():.4f}')
    results.append(('MultiResolutionSTFTLoss', True, None))
    
    # 测试SemanticConsistencyLoss（需要generator）
    import yaml
    with open('configs/vctk_16k.yaml', 'r') as f:
        config = yaml.safe_load(f)
    from models.generator import NeuroGuardGenerator
    from modules.losses import SemanticConsistencyLoss
    
    generator = NeuroGuardGenerator(config).to(device)
    if hasattr(generator, 'semantic_extractor') and generator.semantic_extractor is not None:
        sem_loss = SemanticConsistencyLoss(generator.semantic_extractor).to(device)
        loss = sem_loss(audio_real, audio_wm)
        print(f'✓ SemanticConsistencyLoss 计算成功: {loss.item():.4f}')
        results.append(('SemanticConsistencyLoss', True, None))
    else:
        print('⚠ SemanticConsistencyLoss 跳过（语义流未启用）')
        results.append(('SemanticConsistencyLoss', True, 'Skipped'))
    
    print()
    return results

def test_hard_example_mining():
    """测试5: Hard Example Mining"""
    print('=' * 60)
    print('测试5: Hard Example Mining')
    print('=' * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 4
    msg_bits = 32
    
    msg_logits = torch.randn(B, msg_bits).to(device)
    msg = torch.randint(0, 2, (B, msg_bits)).float().to(device)
    
    results = []
    
    from modules.hard_example_mining import HardExampleMiner
    miner = HardExampleMiner(top_k_ratio=0.3, min_weight=0.5, max_weight=2.0)
    
    ber = miner.compute_ber(msg_logits, msg)
    print(f'✓ BER计算成功: {ber.shape}')
    
    loss_per_sample = torch.randn(B).to(device)
    weighted_loss, weights = miner.apply_weights_to_loss(loss_per_sample, ber)
    print(f'✓ 权重应用成功: {weights.shape}')
    print(f'  - 权重范围: [{weights.min().item():.2f}, {weights.max().item():.2f}]')
    
    results.append(('HardExampleMiner', True, None))
    
    print()
    return results

def main():
    """运行所有测试"""
    print('\n' + '=' * 60)
    print('NeuroGuard-VC 模块测试')
    print('=' * 60 + '\n')
    
    all_results = []
    
    # 测试1: 导入
    results1 = test_imports()
    all_results.extend(results1)
    
    # 测试2: 模型初始化
    results2 = test_model_initialization()
    all_results.extend(results2)
    
    # 测试3: 前向传播
    results3 = test_forward_pass()
    all_results.extend(results3)
    
    # 测试4: 损失函数
    results4 = test_losses()
    all_results.extend(results4)
    
    # 测试5: Hard Example Mining
    results5 = test_hard_example_mining()
    all_results.extend(results5)
    
    # 总结
    print('=' * 60)
    print('测试总结')
    print('=' * 60)
    
    total = len(all_results)
    passed = sum(1 for _, success, _ in all_results if success)
    failed = total - passed
    
    print(f'总测试数: {total}')
    print(f'通过: {passed}')
    print(f'失败: {failed}')
    
    if failed > 0:
        print('\n失败的测试:')
        for name, success, error in all_results:
            if not success:
                print(f'  - {name}: {error}')
    
    print()
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

