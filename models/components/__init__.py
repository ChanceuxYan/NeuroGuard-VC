"""
模型组件模块
Model Components Module
"""

from .film import FiLMLayer1D

# SEANet blocks (可选，如果不需要可以不导入)
from .seanet_blocks import SEANetResnetBlock1d, SEANetEncoder, SEANetDecoder
__all__ = ['FiLMLayer1D', 'SEANetResnetBlock1d', 'SEANetEncoder', 'SEANetDecoder']


