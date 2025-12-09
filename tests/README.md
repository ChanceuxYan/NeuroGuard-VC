# 模块测试说明

本目录包含了对NeuroGuard-VC系统各个模块的独立测试脚本。

## 测试结构

```
tests/
├── __init__.py              # 测试模块初始化
├── test_attack_module.py    # Attack模块详细测试
├── run_all_tests.py         # 统一运行所有测试的主脚本
└── README.md                # 本文件
```

## 测试模块

### 1. Attack模块测试 (`test_attack_module.py`)

测试Attack模块的所有功能，包括：

- **DifferentiableRVCProxy**: RVC代理测试
  - Mel频谱提取
  - Griffin-Lim重建
  - 输出形状一致性

- **DDSPVCProxy**: DDSP代理测试
  - F0提取
  - 响度提取
  - F0噪声扰动
  - 共振峰扰动

- **VAEVCProxy**: VAE代理测试
  - 编码-解码流程
  - 音色扰动

- **AttackLayer**: 攻击层测试
  - Stage1 (基础建立): 轻微噪声或无攻击
  - Stage2 (信号鲁棒): 传统信号处理攻击
  - Stage3 (攻坚重构): VC代理攻击
  - 所有攻击类型强制测试
  - 输出形状一致性
  - 梯度流测试

#### 运行方式：

```bash
# 直接运行
python tests/test_attack_module.py

# 或使用模块方式
python -m tests.test_attack_module
```

### 2. Semantic Extractor测试 (`test_semantic_extractor.py`)

测试语义特征提取器，位于项目根目录：

- HuBERT模型加载
- Wav2Vec2模型加载
- 特征提取功能
- 批量处理
- 不同输入格式
- 不同音频长度
- 归一化处理
- 梯度流
- 性能测试
- 与Generator集成测试

#### 运行方式：

```bash
python test_semantic_extractor.py
```

### 3. 基础模块测试 (`test_modules.py`)

测试所有主要模块的基础功能，位于项目根目录：

- 模块导入测试
- 模型初始化测试
- 前向传播测试
- 损失函数测试
- Hard Example Mining测试

#### 运行方式：

```bash
python test_modules.py
```

## 统一运行所有测试

使用 `run_all_tests.py` 可以统一运行所有测试：

```bash
# 运行所有测试
python tests/run_all_tests.py

# 只运行Attack模块测试
python tests/run_all_tests.py --attack

# 只运行Semantic Extractor测试
python tests/run_all_tests.py --semantic

# 只运行基础模块测试
python tests/run_all_tests.py --modules

# 跳过某些测试
python tests/run_all_tests.py --skip-attack  # 跳过Attack测试
python tests/run_all_tests.py --skip-semantic  # 跳过Semantic测试
python tests/run_all_tests.py --skip-modules  # 跳过基础模块测试
```

## 测试输出

每个测试脚本会输出详细的测试结果，包括：

- ✓ 通过的测试
- ✗ 失败的测试
- ⚠ 警告信息
- 测试统计信息（形状、范围、差异等）

## 注意事项

1. **设备要求**: 测试会自动检测并使用CUDA（如果可用），否则使用CPU
2. **配置文件**: 测试需要 `configs/vctk_16k.yaml` 配置文件
3. **数据文件**: Semantic Extractor测试可能需要 `val.csv` 文件（如果使用真实音频）
4. **模型文件**: 某些测试需要HuBERT/Wav2Vec2模型文件

## 故障排查

如果测试失败，请检查：

1. 所有依赖是否已安装
2. 配置文件是否存在
3. 模型文件是否正确加载
4. CUDA是否可用（如果使用GPU）

## 扩展测试

要添加新的测试：

1. 在 `tests/` 目录下创建新的测试文件
2. 在 `run_all_tests.py` 中添加对应的测试函数
3. 更新本README文档
