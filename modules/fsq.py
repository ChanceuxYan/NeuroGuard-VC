"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32

from einops import rearrange, pack, unpack


# helper functions
# 检查变量是否不为
def exists(v):
    return v is not None

# 返回第一个非None的参数
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# 使用einops的pack函数将单个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量按照指定模式解包
def unpack_one(t, ps, pattern):
    # [0]索引获取解包后的第一个（也是唯一的）张量
    return unpack(t, ps, pattern)[0]


# tensor helpers
# 实现带直通估计器(STE)的四舍五入
# 关键点：四舍五入操作本身是不可微的，但通过STE技巧，允许梯度"直通"量化操作
def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class

class FSQ(Module):
    def __init__(
            self,
            levels: List[int], # 每个维度的离散级别数量列表（核心参数）
            dim: Optional[int] = None, # 输入特征维度（可选）
            num_codebooks=1, # 代码簿数量（默认为1）
            keep_num_codebooks_dim: Optional[bool] = None, # 是否保留代码簿维度
            scale: Optional[float] = None # 缩放因子（可选）
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        # 作为缓冲区注册到模型中（persistent=False表示不会保存到模型状态字典）
        # levels=[8,5,5,5] → _levels=[8,5,5,5]
        self.register_buffer("_levels", _levels, persistent=False)

        # 计算基础向量_basis，用于将多维索引转换为一维索引
        # cumprod计算累积乘积
        # 例如：levels=[8,5,5,5] → _basis=[1,8,40,200]
        # 这是多维索引转一维索引的关键（类似多进制转换）
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        # 设置缩放因子（在论文中可能用于调整量化范围）
        self.scale = scale

        # 计算代码簿维度（即levels列表的长度）
        # 例如：levels=[8,5,5,5] → codebook_dim=4
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        # 计算有效的代码簿维度（代码簿维度×代码簿数量）
        # 例如：codebook_dim=4, num_codebooks=2 → effective_codebook_dim=8
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        # 确定是否保留代码簿维度：
        # 断言：当num_codebooks>1时，必须保留该维度
        # 例如：多个代码簿时，需要区分不同代码簿的索引
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # 设置输入维度：
        # 如果没有指定dim，默认为len(levels) * num_codebooks
        # 例如：levels=[8,5,5,5], num_codebooks=1 → dim=4
        self.dim = default(dim, len(_levels) * num_codebooks)

        # 检查是否需要投影层：
        # 如果输入维度不等于有效代码簿维度，则需要投影
        has_projections = self.dim != effective_codebook_dim
        # project_in：将输入投影到有效代码簿维度
        # project_out：将量化后的向量投影回原始维度
        # 如果不需要投影，使用Identity（恒等变换）
        # 例如：输入维度为128，但有效代码簿维度为4时需要投影
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        # 计算代码簿大小，即所有级别数量的乘积
        # 例如：levels=[8,5,5,5] → codebook_size=8×5×5×5=1000
        self.codebook_size = self._levels.prod().item()

        # 预先计算隐式代码簿，即所有可能的离散组合
        # indices_to_codes将索引转换为代码
        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        # 作为缓冲区注册，便于后续快速查找
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    # 关键函数：限制输入z的范围，使其适合量化
    # 目的：解决奇偶量化级别的根本差异问题
    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        # 计算量化范围的一半长度
        # eps避免数值边界问题
        half_l = (self._levels - 1) * (1 - eps) / 2
        # 偶数量化级别需要0.5偏移，奇数不需要
        # 例如：L=8（偶数）→ offset=0.5；L=7（奇数）→ offset=0.0
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        # 数学技巧，确保偶数量化级别也能正确对齐
        # 确保当z=0时，bound(z) = -0.5（对于偶数量化）
        shift = (offset / half_l).tan()

        # 使用tanh将无限范围输入压缩到有限范围
        # 调整范围以匹配量化级别
        # 减去offset确保正确对齐
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # 先通过bound函数限制z的范围
        # 然后使用round_ste进行四舍五入（带直通估计器）
        quantized = round_ste(self.bound(z))
        # 计算半宽（用于归一化）
        # Use floor division explicitly to avoid future torch deprecation warnings.
        half_width = torch.div(self._levels, 2, rounding_mode='floor')  # Renormalize to [-1, 1].
        # 归一化到[-1, 1]范围
        # 例如：L=8 → half_width=3 → 将[-3.5,3.5]映射到[-1,1]
        return quantized / half_width

    # 将归一化的量化值转换为原始量化级别
    # 例如：归一化值0.5 → 原始级别(0.5×3)+3=4.5（当L=8时）
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        # Use floor division explicitly to avoid future torch deprecation warnings.
        half_width = torch.div(self._levels, 2, rounding_mode='floor')
        return (zhat_normalized * half_width) + half_width

    # 将原始量化级别转换为归一化值
    # 例如：原始级别4.5 → 归一化值(4.5-3)/3=0.5（当L=8时）
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        # Use floor division explicitly to avoid future torch deprecation warnings.
        half_width = torch.div(self._levels, 2, rounding_mode='floor')
        return (zhat - half_width) / half_width

    # 将多维代码转换为一维索引
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # 确保最后一个维度等于代码簿维度
        assert zhat.shape[-1] == self.codebook_dim
        # 将归一化代码转换为原始量化级别
        zhat = self._scale_and_shift(zhat)
        # 使用_basis计算一维索引（多进制转换）
        # 例如：[3,2,1,0] with basis=[1,8,40,200] → 3×1 + 2×8 + 1×40 + 0×200 = 59
        return (zhat * self._basis).sum(dim=-1).to(int32)

    # 将一维索引转换为多维代码
    def indices_to_codes(
            self,
            indices: Tensor,
            project_out=True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""
        # 判断输入是否为图像或视频
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        # 将索引重塑为[..., 1]形状
        indices = rearrange(indices, '... -> ... 1')
        # 计算非中心化的代码（原始量化级别）
        # 例如：索引59 with basis=[1,8,40,200] → [3,2,1,0]
        # Use floor division explicitly to avoid future torch deprecation warnings.
        codes_non_centered = (torch.div(indices, self._basis, rounding_mode='floor')) % self._levels
        # 转换为归一化的代码
        codes = self._scale_and_shift_inverse(codes_non_centered)

        # 如果需要保留代码簿维度，重新排列形状
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        # 如果需要投影输出，应用project_out
        if project_out:
            codes = self.project_out(codes)

        # 如果是图像或视频，调整维度顺序
        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        # 判断输入是否为图像或视频（维度≥4）
        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)
        # 如果是，将维度标准化为(batch, seq, dimension)：
        if is_img_or_video:
            # 将通道维度移到最后
            z = rearrange(z, 'b d ... -> b ... d')
            # 将空间维度展平
            z, ps = pack_one(z, 'b * d')
        # 确保输入维度与预期匹配
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        # 应用输入投影（如果需要）
        z = self.project_in(z)

        # 重新排列维度，分离代码簿维度：
        # 将z从(b, n, c×d)重塑为(b, n, c, d)
        # 例如：z.shape=(1,256,4), num_codebooks=1 → (1,256,1,4)
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # 执行量化操作
        codes = self.quantize(z)
        # 将量化后的代码转换为索引
        indices = self.codes_to_indices(codes)

        # 重新排列维度，合并代码簿维度
        # 例如：(1,256,1,4) → (1,256,4)
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        # 应用输出投影（如果需要）
        out = self.project_out(codes)

        # reconstitute image or video dimensions
        # 如果是图像或视频，恢复原始维度：
        if is_img_or_video:
            # 恢复展平的空间维度
            out = unpack_one(out, ps, 'b * d')
            # 将通道维度移到前面
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # 如果不需要保留代码簿维度，去除该维度
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')
        # 返回量化后的输出和索引
        return out, indices


if __name__ == '__main__':
    levels = [8, 5, 5, 5]  # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels)

    x = torch.randn(1, 4, 16, 16)  # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    print(xhat.shape)  # (1, 1024, 4) - (batch, seq, dim)
    # print(indices.shape) # (1, 1024)    - (batch, seq)