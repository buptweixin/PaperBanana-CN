"""
Provider 包 - 管理不同 API 提供商的接口
"""

from .base import BaseProvider
from .api88996 import Api88996Provider
from .evolink import EvolinkProvider
from .ggboom import GgboomProvider
from .uniapi import UniapiProvider


def create_provider(provider_name: str, **kwargs) -> BaseProvider:
    """
    工厂函数：根据名称创建 provider 实例

    Args:
        provider_name: 提供商名称 ("evolink" 等)
        **kwargs: 传递给 provider 构造函数的参数

    Returns:
        BaseProvider 实例
    """
    providers = {
        "88996": Api88996Provider,
        "uniapi": UniapiProvider,
        "evolink": EvolinkProvider,
        "ggboom": GgboomProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"未知的 provider: {provider_name}。"
            f"可用的 provider: {list(providers.keys())}"
        )

    return providers[provider_name](**kwargs)
