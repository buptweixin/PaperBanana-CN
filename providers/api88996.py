"""
88996 API Provider
支持文本生成（OpenAI 兼容接口）、文生图和图像编辑
"""

import asyncio
import base64
from typing import List, Dict, Any, Optional

import aiohttp

from .base import BaseProvider


class ClientError(Exception):
    """4xx 客户端错误，不应重试（如 400 Bad Request、401 Unauthorized）"""


class Api88996Provider(BaseProvider):
    """
    88996 API Provider

    文本模型: 通过 /v1/chat/completions
    图像模型: 通过 /v1/images/generations
    图像编辑: 通过 /v1/images/edits
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://88996.cloud",
        provider_label: str = "88996",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.provider_label = provider_label
        self._session: Optional[aiohttp.ClientSession] = None

    def _provider_tag(self) -> str:
        return self.provider_label

    def _image_payload_extra(self) -> Dict[str, Any]:
        """允许子类为图片接口补充额外参数。"""
        return {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取共享 aiohttp session，避免频繁创建连接。"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=30)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        """关闭共享 session。"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _convert_contents_to_messages(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """将项目内通用 content 转成 OpenAI 兼容 messages。"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_parts = []
        has_image = False

        for item in contents:
            item_type = item.get("type", "")
            if item_type == "text":
                user_parts.append({"type": "text", "text": item["text"]})
                continue

            if item_type != "image":
                continue

            has_image = True
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                user_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}",
                            "detail": "auto",
                        },
                    }
                )
            elif "image_base64" in item:
                user_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{item['image_base64']}",
                            "detail": "auto",
                        },
                    }
                )

        if not has_image and len(user_parts) == 1:
            messages.append({"role": "user", "content": user_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_parts})

        return messages

    def _build_text_payload(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        """构建文本生成请求体。"""
        return {
            "model": model_name,
            "messages": self._convert_contents_to_messages(contents, system_prompt),
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": False,
        }

    def _build_responses_input(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """构建 OpenAI Responses API 的 input。"""
        response_input = []

        if system_prompt:
            response_input.append({
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            })

        user_parts = []
        for item in contents:
            item_type = item.get("type", "")
            if item_type == "text":
                user_parts.append({"type": "input_text", "text": item["text"]})
                continue

            if item_type != "image":
                continue

            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                user_parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{data}",
                    }
                )
            elif "image_base64" in item:
                user_parts.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{item['image_base64']}",
                    }
                )

        response_input.append({"role": "user", "content": user_parts})
        return response_input

    def _build_responses_payload(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        """构建 Responses API 请求体。"""
        return {
            "model": model_name,
            "input": self._build_responses_input(contents, system_prompt),
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

    def _extract_text_from_responses(self, response: Dict[str, Any]) -> str:
        """从 Responses API 响应中提取文本。"""
        output_text = response.get("output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        text_parts = []
        for output_item in response.get("output", []):
            for content_item in output_item.get("content", []):
                text = content_item.get("text", "")
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    def _map_size(self, aspect_ratio: str, quality: str) -> str:
        """把项目里的宽高比/分辨率偏好映射成 88996 接口的 size 字符串。"""
        if "x" in aspect_ratio:
            return aspect_ratio

        size_map = {
            "2K": {
                "21:9": "2048x896",
                "16:9": "1920x1080",
                "3:2": "1536x1024",
                "1:1": "1024x1024",
            },
            "4K": {
                "21:9": "4096x1792",
                "16:9": "3840x2160",
                "3:2": "3072x2048",
                "1:1": "2048x2048",
            },
        }
        return size_map.get(quality, size_map["2K"]).get(aspect_ratio, "1024x1024")

    def _build_image_payload(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        quality: str,
    ) -> Dict[str, Any]:
        """构建文生图请求体。"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "size": self._map_size(aspect_ratio, quality),
        }
        payload.update(self._image_payload_extra())
        return payload

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送 POST JSON 请求并返回 JSON 响应。"""
        tag = self._provider_tag()
        print(f"[DEBUG] [{tag}] POST {url}")
        print(f"[DEBUG] [{tag}]   model={payload.get('model', 'N/A')}, payload keys={list(payload.keys())}")
        session = await self._get_session()
        async with session.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            status = resp.status
            body = await resp.json()
            print(f"[DEBUG] [{tag}]   响应 status={status}, keys={list(body.keys()) if isinstance(body, dict) else type(body)}")
            if status >= 400:
                error_msg = body.get("error", body) if isinstance(body, dict) else body
                print(f"[DEBUG] [{tag}]   ❌ 错误详情: {error_msg}")
                if 400 <= status < 500 and status != 429:
                    raise ClientError(f"HTTP {status}: {error_msg}")
            resp.raise_for_status()
            return body

    async def _post_form(
        self,
        url: str,
        form_data: aiohttp.FormData,
    ) -> Dict[str, Any]:
        """发送 multipart/form-data 请求并返回 JSON 响应。"""
        tag = self._provider_tag()
        print(f"[DEBUG] [{tag}] POST {url} (multipart/form-data)")
        session = await self._get_session()
        async with session.post(
            url,
            data=form_data,
            headers=self._get_auth_headers(),
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            status = resp.status
            body = await resp.json()
            print(f"[DEBUG] [{tag}]   响应 status={status}, keys={list(body.keys()) if isinstance(body, dict) else type(body)}")
            if status >= 400:
                error_msg = body.get("error", body) if isinstance(body, dict) else body
                print(f"[DEBUG] [{tag}]   ❌ 错误详情: {error_msg}")
                if 400 <= status < 500 and status != 429:
                    raise ClientError(f"HTTP {status}: {error_msg}")
            resp.raise_for_status()
            return body

    async def _download_image_as_base64(self, url: str) -> Optional[str]:
        """从 URL 下载图片并转成 base64。"""
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                image_data = await resp.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"[{self._provider_tag()}] 下载图片失败 ({url}): {e}")
            return None

    async def _download_image_bytes(self, url: str) -> Optional[bytes]:
        """下载远程图片，供 image edit 接口复用。"""
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            print(f"[{self._provider_tag()}] 下载参考图失败 ({url}): {e}")
            return None

    async def _extract_image_result(self, response: Dict[str, Any]) -> Optional[str]:
        """兼容返回 URL 或 b64_json 两种结果。"""
        data = response.get("data", [])
        if not data:
            return None

        first = data[0]
        if first.get("b64_json"):
            return first["b64_json"]

        image_url = first.get("url")
        if image_url:
            print(f"[{self._provider_tag()} 图像] 下载图片: {image_url[:80]}...")
            return await self._download_image_as_base64(image_url)

        return None

    async def generate_text(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
        temperature: float = 1.0,
        max_output_tokens: int = 50000,
        api_mode: str = "chat_completions",
        max_attempts: int = 3,
        retry_delay: float = 5,
        error_context: str = "",
    ) -> List[str]:
        """通过 /v1/chat/completions 或 /v1/responses 生成文本。"""
        if api_mode == "responses":
            url = f"{self.base_url}/v1/responses"
            payload = self._build_responses_payload(
                model_name=model_name,
                contents=contents,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload = self._build_text_payload(
                model_name=model_name,
                contents=contents,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        content_types = [item.get("type", "?") for item in contents]
        tag = self._provider_tag()
        print(f"[DEBUG] [{tag} 文本] 请求: mode={api_mode}, model={model_name}, temp={temperature}, max_tokens={max_output_tokens}")
        print(f"[DEBUG] [{tag} 文本]   内容: {content_types}, system_prompt 长度={len(system_prompt) if system_prompt else 0}")

        for attempt in range(max_attempts):
            try:
                response = await self._post_json(url, payload)
                text = ""
                if api_mode == "responses":
                    text = self._extract_text_from_responses(response)
                else:
                    choices = response.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")

                if text.strip():
                    usage = response.get("usage", {})
                    print(f"[DEBUG] [{tag} 文本] ✓ 成功, 响应长度={len(text)}, usage={usage}")
                    return [text]

                print(f"[{tag} 文本] 响应为空，{retry_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[{tag} 文本] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 30)
                print(
                    f"[{tag} 文本] 第 {attempt + 1} 次尝试失败{context_msg}: {e}。"
                    f"{current_delay}s 后重试..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[{tag} 文本] 全部 {max_attempts} 次尝试失败{context_msg}")

        return ["Error"]

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
        """
        通过 /v1/images/generations 生成图片。

        88996 接口通常为同步返回；若传入参考图 URL，则自动走 edits 接口。
        """
        if image_urls:
            ref_bytes = await self._download_image_bytes(image_urls[0])
            if not ref_bytes:
                return ["Error"]
            return await self.edit_image(
                model_name=model_name,
                image_bytes=ref_bytes,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                quality=quality,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )

        url = f"{self.base_url}/v1/images/generations"
        payload = self._build_image_payload(
            model_name=model_name,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            quality=quality,
        )
        tag = self._provider_tag()
        print(f"[DEBUG] [{tag} 图像] 请求: model={model_name}, size={payload['size']}")
        print(f"[DEBUG] [{tag} 图像]   prompt 长度={len(prompt)}, 前100字: {prompt[:100]}...")

        for attempt in range(max_attempts):
            try:
                response = await self._post_json(url, payload)
                image_b64 = await self._extract_image_result(response)
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
        """通过 /v1/images/edits 编辑图片。"""
        url = f"{self.base_url}/v1/images/edits"
        size = self._map_size(aspect_ratio, quality)
        tag = self._provider_tag()
        print(f"[DEBUG] [{tag} 编辑] 请求: model={model_name}, size={size}, prompt 长度={len(prompt)}")

        for attempt in range(max_attempts):
            try:
                form = aiohttp.FormData()
                form.add_field(
                    "image",
                    image_bytes,
                    filename=image_filename,
                    content_type=media_type,
                )
                form.add_field("prompt", prompt)
                form.add_field("model", model_name)
                form.add_field("size", size)

                response = await self._post_form(url, form)
                image_b64 = await self._extract_image_result(response)
                if image_b64:
                    print(f"[DEBUG] [{tag} 编辑] ✓ 成功, base64 长度={len(image_b64)}")
                    return [image_b64]

                print(f"[{tag} 编辑] 响应中未找到有效图片，{retry_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[{tag} 编辑] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 60)
                print(
                    f"[{tag} 编辑] 第 {attempt + 1} 次尝试失败{context_msg}: {e}。"
                    f"{current_delay}s 后重试..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[{tag} 编辑] 全部 {max_attempts} 次尝试失败{context_msg}")

        return ["Error"]
