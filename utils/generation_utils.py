# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
API 调用工具函数，支持 Evolink、Gemini、Claude、OpenAI 等多种 Provider。
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

from PIL import Image

import os
import yaml
from pathlib import Path

# ==================== 配置加载 ====================

config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# ==================== OpenAI 兼容 Provider 初始化 ====================

evolink_provider = None
api88996_provider = None
ggboom_provider = None

evolink_api_key = get_config_val("evolink", "api_key", "EVOLINK_API_KEY", "")
evolink_base_url = get_config_val("evolink", "base_url", "EVOLINK_BASE_URL", "https://api.evolink.ai")
api88996_api_key = get_config_val("api88996", "api_key", "API88996_API_KEY", "")
api88996_base_url = get_config_val("api88996", "base_url", "API88996_BASE_URL", "https://88996.cloud")
ggboom_api_key = get_config_val("ggboom", "api_key", "GGBOOM_API_KEY", "")
ggboom_base_url = get_config_val("ggboom", "base_url", "GGBOOM_BASE_URL", "https://ai.qaq.al")

if evolink_api_key:
    from providers.evolink import EvolinkProvider
    evolink_provider = EvolinkProvider(api_key=evolink_api_key, base_url=evolink_base_url)
    print(f"已初始化 Evolink Provider (base_url={evolink_base_url})")
else:
    print("警告：未配置 Evolink API Key，Evolink Provider 不可用。")

if api88996_api_key:
    from providers.api88996 import Api88996Provider
    api88996_provider = Api88996Provider(api_key=api88996_api_key, base_url=api88996_base_url)
    print(f"已初始化 88996 Provider (base_url={api88996_base_url})")
else:
    print("警告：未配置 88996 API Key，88996 Provider 不可用。")

if ggboom_api_key:
    from providers.ggboom import GgboomProvider
    ggboom_provider = GgboomProvider(api_key=ggboom_api_key, base_url=ggboom_base_url)
    print(f"已初始化 GGboom Provider (base_url={ggboom_base_url})")
else:
    print("警告：未配置 GGboom API Key，GGboom Provider 不可用。")


def init_evolink_provider(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 Evolink Provider（供界面动态传入）。"""
    global evolink_provider
    if not api_key:
        return
    url = base_url or evolink_base_url
    from providers.evolink import EvolinkProvider
    evolink_provider = EvolinkProvider(api_key=api_key, base_url=url)
    print(f"已通过界面初始化 Evolink Provider (base_url={url})")


def init_api88996_provider(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 88996 Provider（供界面动态传入）。"""
    global api88996_provider
    if not api_key:
        return
    url = base_url or api88996_base_url
    from providers.api88996 import Api88996Provider
    api88996_provider = Api88996Provider(api_key=api_key, base_url=url)
    print(f"已通过界面初始化 88996 Provider (base_url={url})")


def init_ggboom_provider(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 GGboom Provider（供界面动态传入）。"""
    global ggboom_provider
    if not api_key:
        return
    url = base_url or ggboom_base_url
    from providers.ggboom import GgboomProvider
    ggboom_provider = GgboomProvider(api_key=api_key, base_url=url)
    print(f"已通过界面初始化 GGboom Provider (base_url={url})")


def init_provider_client(provider_name: str, api_key: str, base_url: str = ""):
    """根据 provider 名称初始化对应客户端。"""
    if not api_key:
        return
    if provider_name == "evolink":
        init_evolink_provider(api_key, base_url)
    elif provider_name == "88996":
        init_api88996_provider(api_key, base_url)
    elif provider_name == "ggboom":
        init_ggboom_provider(api_key, base_url)
    elif provider_name == "gemini":
        init_gemini_client(api_key)


def is_openai_compatible_provider(provider_name: str) -> bool:
    """判断是否为项目内的 OpenAI 兼容 Provider。"""
    return provider_name in {"evolink", "88996", "ggboom"}


def get_openai_compatible_provider(provider_name: str):
    """获取已初始化的 OpenAI 兼容 Provider 实例。"""
    if provider_name == "evolink":
        return evolink_provider
    if provider_name == "88996":
        return api88996_provider
    if provider_name == "ggboom":
        return ggboom_provider
    return None


async def close_provider_client(provider_name: str):
    """关闭指定 provider 的共享连接。"""
    provider = get_openai_compatible_provider(provider_name)
    if provider and hasattr(provider, "close"):
        await provider.close()


def init_gemini_client(api_key: str):
    """用指定的 API Key 初始化或更新 Gemini Client（供界面动态传入）。"""
    global gemini_client
    if not api_key:
        return
    try:
        from google import genai
        gemini_client = genai.Client(api_key=api_key)
        print("已通过界面初始化 Gemini Client")
    except ImportError:
        print("警告：未安装 google-genai，Gemini Client 不可用。请运行 pip install google-genai")


# ==================== 原始 Provider 初始化（保留兼容性） ====================

gemini_client = None
anthropic_client = None
openai_client = None

api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(api_key=api_key)
        print("已初始化 Gemini Client")
    except ImportError:
        print("警告：未安装 google-genai，Gemini Client 不可用。")

anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    try:
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        print("已初始化 Anthropic Client")
    except ImportError:
        print("警告：未安装 anthropic，Anthropic Client 不可用。")

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        print("已初始化 OpenAI Client")
    except ImportError:
        print("警告：未安装 openai，OpenAI Client 不可用。")


# ==================== OpenAI 兼容 Provider 调用函数 ====================

def _extract_openai_compatible_text_config(config):
    """统一提取 OpenAI 兼容 provider 的文本配置。"""
    if hasattr(config, 'system_instruction'):
        return {
            "system_prompt": config.system_instruction or "",
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "api_mode": "chat_completions",
        }
    if isinstance(config, dict):
        return {
            "system_prompt": config.get("system_prompt", ""),
            "temperature": config.get("temperature", 1.0),
            "max_output_tokens": config.get("max_output_tokens", 50000),
            "api_mode": config.get("api_mode", "chat_completions"),
        }
    return {
        "system_prompt": "",
        "temperature": 1.0,
        "max_output_tokens": 50000,
        "api_mode": "chat_completions",
    }


async def call_openai_compatible_text_with_retry_async(
    provider_name,
    model_name,
    contents,
    config,
    max_attempts=5,
    retry_delay=5,
    error_context="",
):
    """通过项目内 OpenAI 兼容 Provider 进行文本生成。"""
    provider = get_openai_compatible_provider(provider_name)
    print(f"[DEBUG] call_openai_compatible_text: provider={provider_name}, model={model_name}, 已初始化={provider is not None}")
    if provider is None:
        raise RuntimeError(f"{provider_name} Provider 未初始化，请检查 API Key 配置。")

    text_config = _extract_openai_compatible_text_config(config)
    return await provider.generate_text(
        model_name=model_name,
        contents=contents,
        system_prompt=text_config["system_prompt"],
        temperature=text_config["temperature"],
        max_output_tokens=text_config["max_output_tokens"],
        api_mode=text_config["api_mode"],
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_openai_compatible_image_with_retry_async(
    provider_name,
    model_name,
    prompt,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """通过项目内 OpenAI 兼容 Provider 进行图像生成。"""
    provider = get_openai_compatible_provider(provider_name)
    print(f"[DEBUG] call_openai_compatible_image: provider={provider_name}, model={model_name}, config={config}, 已初始化={provider is not None}")
    if provider is None:
        raise RuntimeError(f"{provider_name} Provider 未初始化，请检查 API Key 配置。")

    aspect_ratio = config.get("aspect_ratio", "16:9")
    quality = config.get("quality", "2K")
    image_urls = config.get("image_urls", None)

    return await provider.generate_image(
        model_name=model_name,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        quality=quality,
        image_urls=image_urls,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def edit_openai_compatible_image_with_retry_async(
    provider_name,
    model_name,
    image_bytes,
    prompt,
    config,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """统一封装 image-to-image 能力，屏蔽各 provider 的差异。"""
    if provider_name == "evolink":
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        ref_image_url = await upload_image_to_evolink(
            image_b64,
            media_type=config.get("media_type", "image/jpeg"),
        )
        return await call_openai_compatible_image_with_retry_async(
            provider_name="evolink",
            model_name=model_name,
            prompt=prompt,
            config={
                "aspect_ratio": config.get("aspect_ratio", "16:9"),
                "quality": config.get("quality", "2K"),
                "image_urls": [ref_image_url],
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider_name == "88996":
        provider = get_openai_compatible_provider(provider_name)
        if provider is None:
            raise RuntimeError("88996 Provider 未初始化，请检查 API Key 配置。")
        return await provider.edit_image(
            model_name=model_name,
            image_bytes=image_bytes,
            prompt=prompt,
            aspect_ratio=config.get("aspect_ratio", "16:9"),
            quality=config.get("quality", "2K"),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            media_type=config.get("media_type", "image/jpeg"),
        )

    raise ValueError(f"不支持的 OpenAI 兼容 Provider: {provider_name}")

async def call_evolink_text_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """通过 Evolink Provider 进行文本生成。"""
    return await call_openai_compatible_text_with_retry_async(
        provider_name="evolink",
        model_name=model_name,
        contents=contents,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_api88996_text_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """通过 88996 Provider 进行文本生成。"""
    return await call_openai_compatible_text_with_retry_async(
        provider_name="88996",
        model_name=model_name,
        contents=contents,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def upload_image_to_evolink(image_b64: str, media_type: str = "image/jpeg") -> str:
    """
    将 base64 图片上传到 Evolink 文件服务，返回可访问的 URL。

    用于 image-to-image 场景（如 Polish Agent），需要先把本地 base64 图片
    上传为 URL，才能传给图像生成 API 的 image_urls 参数。
    """
    if evolink_provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")
    url = await evolink_provider.upload_image_base64(image_b64, media_type)
    if not url:
        raise RuntimeError("图片上传到 Evolink 文件服务失败")
    return url


async def call_evolink_image_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """通过 Evolink Provider 进行图像生成。"""
    return await call_openai_compatible_image_with_retry_async(
        provider_name="evolink",
        model_name=model_name,
        prompt=prompt,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def call_api88996_image_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """通过 88996 Provider 进行图像生成。"""
    return await call_openai_compatible_image_with_retry_async(
        provider_name="88996",
        model_name=model_name,
        prompt=prompt,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def edit_api88996_image_with_retry_async(
    model_name, image_bytes, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """通过 88996 Provider 进行图片编辑。"""
    return await edit_openai_compatible_image_with_retry_async(
        provider_name="88996",
        model_name=model_name,
        image_bytes=image_bytes,
        prompt=prompt,
        config=config,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


# ==================== 原始 Gemini 调用函数（保留兼容性） ====================

def _convert_to_gemini_parts(contents):
    """将通用内容列表转换为 Gemini 的 Part 对象列表"""
    from google.genai import types
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """原始 Gemini API 异步调用（保留兼容性）"""
    from google.genai import types

    result_list = []
    target_candidate_count = config.candidate_count
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            client = gemini_client
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            if "nanoviz" in model_name or "image" in model_name:
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 30)
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list


# ==================== 原始 Claude/OpenAI 调用函数（保留兼容性） ====================

def _convert_to_claude_format(contents):
    return contents

def _convert_to_openai_format(contents):
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 Claude API 异步调用（保留兼容性）"""
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        print(f"Error: All {max_attempts} attempts failed to validate the input. Returning errors.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": valid_claude_contents}],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 OpenAI API 异步调用（保留兼容性）"""
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        print(f"Error: All {max_attempts} attempts failed to validate the input. Returning errors.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 OpenAI 图像生成 API 异步调用（保留兼容性）"""
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
    }

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
