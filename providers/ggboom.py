"""
GGboom 文本 Provider
基于 OpenAI 兼容文本接口，仅支持 /v1/chat/completions 和 /v1/responses
"""

from typing import List, Optional

from .api88996 import Api88996Provider


class GgboomProvider(Api88996Provider):
    """GGboom 文本专用 Provider。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ai.qaq.al",
    ):
        super().__init__(api_key=api_key, base_url=base_url)

    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "2K",
        image_urls: Optional[List[str]] = None,
        max_attempts: int = 3,
        retry_delay: float = 30,
        poll_interval: float = 0,
        max_polls: int = 1,
        error_context: str = "",
    ) -> List[str]:
        raise RuntimeError("GGboom Provider 仅支持文本接口，不支持图像生成。")

    async def edit_image(
        self,
        model_name: str,
        image_bytes: bytes,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "2K",
        max_attempts: int = 3,
        retry_delay: float = 30,
        error_context: str = "",
        image_filename: str = "input.jpg",
        media_type: str = "image/jpeg",
    ) -> List[str]:
        raise RuntimeError("GGboom Provider 仅支持文本接口，不支持图像编辑。")
