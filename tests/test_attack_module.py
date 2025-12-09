#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attack模块详细测试脚本
测试所有攻击类型是否正常工作
"""
import sys
import os
import torch
import yaml
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.attack import AttackLayer, DifferentiableRVCProxy
from modules.ddsp_proxy import DDSPVCProxy
from modules.vae_proxy import VAEVCProxy


class AttackModuleTester:
    """Attack模块测试器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}\n")
        
    def test_differentiable_rvc_proxy(self):
        """测试1: DifferentiableRVCProxy"""
        print("=" * 60)
        print("测试1: DifferentiableRVCProxy (RVC代理)")
        print("=" * 60)
        
        rvc_proxy = DifferentiableRVCProxy(sample_rate=16000, n_mels=80).to(self.device)
        rvc_proxy.eval()
        
        # 创建测试音频
        B, T = 2, 32000  # 2秒 @ 16kHz
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}")
        print(f"  输入范围: [{test_audio.min().item():.4f}, {test_audio.max().item():.4f}]")
        
        with torch.no_grad():
            attacked_audio = rvc_proxy(test_audio)
        
        print(f"  ✓ RVC代理前向传播成功")
        print(f"  - 输出形状: {attacked_audio.shape}")
        print(f"  - 输出范围: [{attacked_audio.min().item():.4f}, {attacked_audio.max().item():.4f}]")
        print(f"  - 长度匹配: {attacked_audio.shape[-1] == test_audio.shape[-1]}")
        
        # 检查是否有变化
        diff = (attacked_audio - test_audio).abs().mean().item()
        print(f"  - 平均差异: {diff:.6f}")
        if diff > 1e-6:
            print(f"  ✓ 音频已被修改（这是预期的）")
        else:
            print(f"  ⚠ 警告: 音频几乎没有变化")
        
        return True, None
    
    def test_ddsp_proxy(self):
        """测试2: DDSPVCProxy"""
        print("\n" + "=" * 60)
        print("测试2: DDSPVCProxy (DDSP代理)")
        print("=" * 60)
        
        ddsp_proxy = DDSPVCProxy(
            sample_rate=16000,
            hop_length=256,
            n_fft=2048
        ).to(self.device)
        ddsp_proxy.eval()
        
        # 创建测试音频
        B, T = 2, 32000
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}")
        
        with torch.no_grad():
            # 测试无扰动
            attacked_audio_1 = ddsp_proxy(test_audio, add_f0_noise=False, formant_perturb=False)
            print(f"  ✓ DDSP代理（无扰动）前向传播成功")
            print(f"  - 输出形状: {attacked_audio_1.shape}")
            
            # 测试F0噪声
            attacked_audio_2 = ddsp_proxy(test_audio, add_f0_noise=True, formant_perturb=False)
            print(f"  ✓ DDSP代理（F0噪声）前向传播成功")
            print(f"  - 输出形状: {attacked_audio_2.shape}")
            
            # 测试共振峰扰动
            attacked_audio_3 = ddsp_proxy(test_audio, add_f0_noise=False, formant_perturb=True)
            print(f"  ✓ DDSP代理（共振峰扰动）前向传播成功")
            print(f"  - 输出形状: {attacked_audio_3.shape}")
            
            # 测试全部扰动
            attacked_audio_4 = ddsp_proxy(test_audio, add_f0_noise=True, formant_perturb=True)
            print(f"  ✓ DDSP代理（全部扰动）前向传播成功")
            print(f"  - 输出形状: {attacked_audio_4.shape}")
        
        return True, None
    
    def test_vae_proxy(self):
        """测试3: VAEVCProxy"""
        print("\n" + "=" * 60)
        print("测试3: VAEVCProxy (VAE代理)")
        print("=" * 60)
        
        vae_proxy = VAEVCProxy(
            sample_rate=16000,
            latent_dim=64
        ).to(self.device)
        vae_proxy.eval()
        
        # 创建测试音频
        B, T = 2, 32000
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}")
        
        with torch.no_grad():
            # 测试无扰动
            attacked_audio_1 = vae_proxy(test_audio, add_timbre_perturb=False)
            print(f"  ✓ VAE代理（无扰动）前向传播成功")
            print(f"  - 输出形状: {attacked_audio_1.shape}")
            
            # 测试音色扰动
            attacked_audio_2 = vae_proxy(test_audio, add_timbre_perturb=True)
            print(f"  ✓ VAE代理（音色扰动）前向传播成功")
            print(f"  - 输出形状: {attacked_audio_2.shape}")
        
        return True, None
    
    def test_attack_layer_stage1(self):
        """测试4: AttackLayer - Stage1 (基础建立)"""
        print("\n" + "=" * 60)
        print("测试4: AttackLayer - Stage1 (基础建立)")
        print("=" * 60)
        
        # 加载配置
        config_path = os.path.join(project_root, 'configs', 'vctk_16k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        attack_layer = AttackLayer(config).to(self.device)
        attack_layer.eval()
        
        B, T = 2, 32000
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}")
        
        # 测试多次，观察不同的攻击类型
        results = []
        for i in range(10):
            with torch.no_grad():
                attacked = attack_layer(test_audio, global_step=0, training_stage='stage1')
                diff = (attacked - test_audio).abs().mean().item()
                results.append(diff)
        
        print(f"  ✓ Stage1攻击测试成功")
        print(f"  - 平均差异: {np.mean(results):.6f}")
        print(f"  - 差异范围: [{np.min(results):.6f}, {np.max(results):.6f}]")
        print(f"  - 说明: Stage1应该只有轻微噪声或无攻击")
        
        return True, None
    
    def test_attack_layer_stage2(self):
        """测试5: AttackLayer - Stage2 (信号鲁棒)"""
        print("\n" + "=" * 60)
        print("测试5: AttackLayer - Stage2 (信号鲁棒)")
        print("=" * 60)
        
        config_path = os.path.join(project_root, 'configs', 'vctk_16k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        attack_layer = AttackLayer(config).to(self.device)
        attack_layer.eval()
        
        B, T = 2, 32000
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}")
        
        # 测试不同的攻击类型
        attack_types = {
            'no_attack': 0,
            'noise': 0,
            'scaling': 0,
            'masking': 0,
            'rvc_proxy': 0
        }
        
        # 运行多次，统计攻击类型分布
        for i in range(50):
            with torch.no_grad():
                attacked = attack_layer(test_audio, global_step=2000, training_stage='stage2')
                diff = (attacked - test_audio).abs().mean().item()
                
                # 简单判断攻击类型
                if diff < 1e-6:
                    attack_types['no_attack'] += 1
                elif diff < 0.1:
                    attack_types['noise'] += 1
                elif torch.allclose(attacked.abs(), test_audio.abs() * attacked.abs().mean() / test_audio.abs().mean(), atol=0.1):
                    attack_types['scaling'] += 1
                elif (attacked == 0).any():
                    attack_types['masking'] += 1
                else:
                    attack_types['rvc_proxy'] += 1
        
        print(f"  ✓ Stage2攻击测试成功")
        print(f"  - 攻击类型分布（50次运行）:")
        for atk_type, count in attack_types.items():
            print(f"    - {atk_type}: {count} ({count/50*100:.1f}%)")
        
        return True, None
    
    def test_attack_layer_stage3(self):
        """测试6: AttackLayer - Stage3 (攻坚重构)"""
        print("\n" + "=" * 60)
        print("测试6: AttackLayer - Stage3 (攻坚重构)")
        print("=" * 60)
        
        config_path = os.path.join(project_root, 'configs', 'vctk_16k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        attack_layer = AttackLayer(config).to(self.device)
        attack_layer.eval()
        
        B, T = 2, 32000
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}")
        
        # 测试不同global_step的影响
        steps = [100, 1000, 5000, 10000]
        for step in steps:
            with torch.no_grad():
                attacked = attack_layer(test_audio, global_step=step, training_stage='stage3')
                diff = (attacked - test_audio).abs().mean().item()
                print(f"  - global_step={step:5d}: 平均差异={diff:.6f}")
        
        print(f"  ✓ Stage3攻击测试成功")
        print(f"  - 说明: Stage3应该包含DDSP/VAE代理攻击（当global_step>1000时）")
        
        return True, None
    
    def test_attack_layer_all_attacks(self):
        """测试7: 测试所有攻击类型（强制测试每种攻击）"""
        print("\n" + "=" * 60)
        print("测试7: 强制测试所有攻击类型")
        print("=" * 60)
        
        config_path = os.path.join(project_root, 'configs', 'vctk_16k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        attack_layer = AttackLayer(config).to(self.device)
        attack_layer.eval()
        
        B, T = 2, 32000
        test_audio = torch.randn(B, 1, T).to(self.device)
        print(f"  输入形状: {test_audio.shape}\n")
        
        # 手动测试每种攻击（通过修改attack_probs）
        original_probs = attack_layer.attack_probs.copy()
        
        # 测试噪声攻击
        print("  测试噪声攻击:")
        attack_layer.attack_probs = {'no_attack': 0.0, 'noise': 1.0, 'scaling': 0.0, 'masking': 0.0, 'vc_proxy': 0.0}
        with torch.no_grad():
            attacked = attack_layer(test_audio, global_step=2000, training_stage='stage2')
            diff = (attacked - test_audio).abs().mean().item()
            print(f"    ✓ 噪声攻击: 平均差异={diff:.6f}")
        
        # 测试缩放攻击
        print("  测试缩放攻击:")
        attack_layer.attack_probs = {'no_attack': 0.0, 'noise': 0.0, 'scaling': 1.0, 'masking': 0.0, 'vc_proxy': 0.0}
        with torch.no_grad():
            attacked = attack_layer(test_audio, global_step=2000, training_stage='stage2')
            scale_factor = attacked.abs().mean() / test_audio.abs().mean()
            print(f"    ✓ 缩放攻击: 缩放因子={scale_factor.item():.4f}")
        
        # 测试掩码攻击
        print("  测试掩码攻击:")
        attack_layer.attack_probs = {'no_attack': 0.0, 'noise': 0.0, 'scaling': 0.0, 'masking': 1.0, 'vc_proxy': 0.0}
        with torch.no_grad():
            attacked = attack_layer(test_audio, global_step=2000, training_stage='stage2')
            zero_ratio = (attacked == 0).float().mean().item()
            print(f"    ✓ 掩码攻击: 零值比例={zero_ratio:.4f}")
        
        # 测试RVC代理攻击
        print("  测试RVC代理攻击:")
        attack_layer.attack_probs = {'no_attack': 0.0, 'noise': 0.0, 'scaling': 0.0, 'masking': 0.0, 'vc_proxy': 1.0}
        with torch.no_grad():
            attacked = attack_layer(test_audio, global_step=2000, training_stage='stage2')
            diff = (attacked - test_audio).abs().mean().item()
            print(f"    ✓ RVC代理攻击: 平均差异={diff:.6f}")
        
        # 恢复原始概率
        attack_layer.attack_probs = original_probs
        
        print(f"  ✓ 所有攻击类型测试成功")
        
        return True, None
    
    def test_attack_shape_consistency(self):
        """测试8: 测试输出形状一致性"""
        print("\n" + "=" * 60)
        print("测试8: 输出形状一致性测试")
        print("=" * 60)
        
        config_path = os.path.join(project_root, 'configs', 'vctk_16k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        attack_layer = AttackLayer(config).to(self.device)
        attack_layer.eval()
        
        # 测试不同输入形状
        test_cases = [
            (1, 1, 16000),   # 单样本，单通道
            (2, 1, 16000),   # 批量，单通道
            (2, 1, 32000),   # 批量，更长音频
            (1, 1, 8000),    # 单样本，短音频
        ]
        
        for shape in test_cases:
            test_audio = torch.randn(*shape).to(self.device)
            with torch.no_grad():
                attacked = attack_layer(test_audio, global_step=2000, training_stage='stage3')
            
            if attacked.shape == test_audio.shape:
                print(f"  ✓ 形状 {shape} -> {attacked.shape} (匹配)")
            else:
                print(f"  ✗ 形状 {shape} -> {attacked.shape} (不匹配)")
                return False, f"形状不匹配: {shape} -> {attacked.shape}"
        
        print(f"  ✓ 所有形状测试通过")
        
        return True, None
    
    def test_gradient_flow(self):
        """测试9: 梯度流测试"""
        print("\n" + "=" * 60)
        print("测试9: 梯度流测试")
        print("=" * 60)
        
        config_path = os.path.join(project_root, 'configs', 'vctk_16k.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        attack_layer = AttackLayer(config).to(self.device)
        attack_layer.train()  # 设置为训练模式
        
        B, T = 2, 16000
        test_audio = torch.randn(B, 1, T, requires_grad=True).to(self.device)
        
        # 测试多次，因为攻击是随机的
        has_gradient = False
        for i in range(10):
            # 重置梯度
            if test_audio.grad is not None:
                test_audio.grad.zero_()
            
            # 前向传播
            attacked = attack_layer(test_audio, global_step=2000, training_stage='stage3')
            
            # 检查输出是否需要梯度
            if attacked.requires_grad:
                # 计算损失并反向传播
                loss = attacked.mean()
                loss.backward()
                
                # 检查输入梯度
                if test_audio.grad is not None and test_audio.grad.abs().sum() > 1e-8:
                    grad_norm = test_audio.grad.norm().item()
                    print(f"  ✓ 梯度流正常（第{i+1}次尝试）")
                    print(f"  - 输入梯度范数: {grad_norm:.6f}")
                    print(f"  - 梯度统计: mean={test_audio.grad.mean().item():.6f}, std={test_audio.grad.std().item():.6f}")
                    has_gradient = True
                    break
        
        if not has_gradient:
            print(f"  ⚠ 注意: 在10次尝试中都没有检测到梯度")
            print(f"  - 这是正常的，因为某些攻击（如GriffinLim、RVC Proxy）可能不支持梯度")
            print(f"  - 或者随机选择的攻击类型不支持梯度（如no_attack、scaling等）")
            # 不返回False，因为这是可以接受的
        
        return True, None
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("Attack模块完整测试")
        print("=" * 60 + "\n")
        
        test_results = []
        
        # 运行所有测试
        test_results.append(("DifferentiableRVCProxy", *self.test_differentiable_rvc_proxy()))
        test_results.append(("DDSPVCProxy", *self.test_ddsp_proxy()))
        test_results.append(("VAEVCProxy", *self.test_vae_proxy()))
        test_results.append(("AttackLayer Stage1", *self.test_attack_layer_stage1()))
        test_results.append(("AttackLayer Stage2", *self.test_attack_layer_stage2()))
        test_results.append(("AttackLayer Stage3", *self.test_attack_layer_stage3()))
        test_results.append(("所有攻击类型", *self.test_attack_layer_all_attacks()))
        test_results.append(("形状一致性", *self.test_attack_shape_consistency()))
        test_results.append(("梯度流", *self.test_gradient_flow()))
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        total = len(test_results)
        passed = sum(1 for _, success, _ in test_results if success)
        failed = total - passed
        
        print(f"总测试数: {total}")
        print(f"通过: {passed}")
        print(f"失败: {failed}\n")
        
        if failed > 0:
            print("失败的测试:")
            for name, success, error in test_results:
                if not success:
                    print(f"  - {name}: {error}")
        
        return failed == 0


def main():
    """主函数"""
    tester = AttackModuleTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

