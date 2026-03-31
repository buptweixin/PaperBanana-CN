"""
UniAPI Provider
复用 OpenAI 兼容实现，并按 UniAPI 官方图片接口补充默认参数。
"""

import asyncio
import re
from typing import Dict, Any, List, Optional

from .api88996 import Api88996Provider, ClientError


class UniapiProvider(Api88996Provider):
    """UniAPI 专用 Provider。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.uniapi.io",
    ):
        super().__init__(api_key=api_key, base_url=base_url, provider_label="UniAPI")

    def _image_payload_extra(self) -> Dict[str, Any]:
        """对齐 UniAPI 官方 images/generations 示例参数。"""
        return {"n": 1}

    def _build_chat_image_payload(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """对齐 test_uniapi.py 中的文生图调用方式。"""
        return {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "extra_body": {
                "modalities": ["text", "image"],
            },
        }

    def _extract_base64_from_image_url(self, image_url: str) -> Optional[str]:
        """从 data URL 或 markdown 包裹的 data URL 中提取 base64。"""
        if not image_url:
            return None

        match = re.search(r"data:image/\w+;base64,([A-Za-z0-9+/=]+)", image_url)
        if match:
            return match.group(1)
        return None

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
        """使用 UniAPI 的 chat completions 图片生成方式。"""
        if image_urls:
            return await super().generate_image(
                model_name=model_name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                quality=quality,
                image_urls=image_urls,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                poll_interval=poll_interval,
                max_polls=max_polls,
                error_context=error_context,
            )

        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_chat_image_payload(model_name=model_name, prompt=prompt)
        tag = self._provider_tag()
        print(f"[DEBUG] [{tag} 图像] 请求: mode=chat_completions+modalities, model={model_name}")
        print(f"[DEBUG] [{tag} 图像]   prompt 长度={len(prompt)}, 前100字: {prompt[:100]}...")

        for attempt in range(max_attempts):
            try:
                response = await self._post_json(url, payload)
                choices = response.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    images = message.get("images", [])
                    if images:
                        image_url = images[0].get("image_url", {}).get("url", "")
                        image_b64 = self._extract_base64_from_image_url(image_url)
                        if image_b64:
                            print(f"[DEBUG] [{tag} 图像] ✓ 成功, base64 长度={len(image_b64)}")
                            return [image_b64]

                print(f"[{tag} 图像] 响应中未找到有效图片，{retry_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[{tag} 图像] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 60)
                print(
                    f"[{tag} 图像] 第 {attempt + 1} 次尝试失败{context_msg}: {e}。"
                    f"{current_delay}s 后重试..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[{tag} 图像] 全部 {max_attempts} 次尝试失败{context_msg}")

        return ["Error"]
