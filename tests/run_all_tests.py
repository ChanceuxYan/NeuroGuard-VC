#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一运行所有模块测试的主脚本
"""
import sys
import os
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_attack_tests():
    """运行Attack模块测试"""
    print("\n" + "=" * 80)
    print("运行 Attack 模块测试")
    print("=" * 80)
    from tests.test_attack_module import AttackModuleTester
    tester = AttackModuleTester()
    return tester.run_all_tests()


def run_semantic_tests():
    """运行Semantic Extractor测试"""
    print("\n" + "=" * 80)
    print("运行 Semantic Extractor 测试")
    print("=" * 80)
    # 使用现有的test_semantic_extractor.py
    test_file = os.path.join(project_root, 'test_semantic_extractor.py')
    if os.path.exists(test_file):
        import subprocess
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              text=True)
        return result.returncode == 0
    else:
        print(f"⚠ 警告: {test_file} 不存在")
        return False


def run_module_tests():
    """运行基础模块测试"""
    print("\n" + "=" * 80)
    print("运行基础模块测试")
    print("=" * 80)
    # 使用现有的test_modules.py
    test_file = os.path.join(project_root, 'test_modules.py')
    if os.path.exists(test_file):
        import subprocess
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              text=True)
        return result.returncode == 0
    else:
        print(f"⚠ 警告: {test_file} 不存在")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行所有模块测试')
    parser.add_argument('--attack', action='store_true', help='只运行Attack模块测试')
    parser.add_argument('--semantic', action='store_true', help='只运行Semantic Extractor测试')
    parser.add_argument('--modules', action='store_true', help='只运行基础模块测试')
    parser.add_argument('--skip-attack', action='store_true', help='跳过Attack模块测试')
    parser.add_argument('--skip-semantic', action='store_true', help='跳过Semantic Extractor测试')
    parser.add_argument('--skip-modules', action='store_true', help='跳过基础模块测试')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("NeuroGuard-VC 完整测试套件")
    print("=" * 80)
    
    results = {}
    
    # 决定运行哪些测试
    run_all = not (args.attack or args.semantic or args.modules)
    
    # 运行测试
    if run_all or args.attack:
        if not args.skip_attack:
            results['Attack'] = run_attack_tests()
        else:
            print("\n跳过 Attack 模块测试")
    
    if run_all or args.semantic:
        if not args.skip_semantic:
            results['Semantic'] = run_semantic_tests()
        else:
            print("\n跳过 Semantic Extractor 测试")
    
    if run_all or args.modules:
        if not args.skip_modules:
            results['Modules'] = run_module_tests()
        else:
            print("\n跳过基础模块测试")
    
    # 总结
    print("\n" + "=" * 80)
    print("最终测试总结")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"\n总测试套件: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}\n")
    
    if results:
        print("详细结果:")
        for name, success in results.items():
            status = "✓ 通过" if success else "✗ 失败"
            print(f"  - {name}: {status}")
    
    print()
    
    # 返回状态码
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

