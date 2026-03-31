"""
UniAPI Provider 路由测试
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from providers import create_provider
from providers.uniapi import UniapiProvider
from utils import generation_utils


def test_create_provider_supports_uniapi():
    provider = create_provider(
        "uniapi",
        api_key="test-key",
        base_url="https://api.uniapi.io",
    )
    assert isinstance(provider, UniapiProvider)
    assert provider.base_url == "https://api.uniapi.io"


def test_uniapi_image_payload_matches_official_style():
    provider = UniapiProvider(api_key="test-key")
    payload = provider._build_image_payload(
        model_name="dall-e-3",
        prompt="a white siamese cat",
        aspect_ratio="1:1",
        quality="2K",
    )

    assert payload["model"] == "dall-e-3"
    assert payload["prompt"] == "a white siamese cat"
    assert payload["size"] == "1024x1024"
    assert payload["n"] == 1


def test_uniapi_chat_image_payload_matches_test_script():
    provider = UniapiProvider(api_key="test-key")
    payload = provider._build_chat_image_payload(
        model_name="gemini-3.1-flash-image-preview",
        prompt="请生成一张可爱的小海獭在海洋中游泳的图片",
    )

    assert payload["model"] == "gemini-3.1-flash-image-preview"
    assert payload["messages"] == [{"role": "user", "content": "请生成一张可爱的小海獭在海洋中游泳的图片"}]
    assert payload["stream"] is False
    assert payload["extra_body"] == {"modalities": ["text", "image"]}


def test_uniapi_generate_image_reads_image_from_chat_response():
    provider = UniapiProvider(api_key="test-key")
    image_b64 = "ZmFrZS1pbWFnZS1iYXNlNjQ="
    mock_response = {
        "choices": [
            {
                "message": {
                    "images": [
                        {
                            "image_url": {
                                "url": f"![image](data:image/png;base64,{image_b64})"
                            }
                        }
                    ]
                }
            }
        ]
    }

    with patch.object(provider, "_post_json", new_callable=AsyncMock, return_value=mock_response):
        result = asyncio.run(
            provider.generate_image(
                model_name="gemini-3.1-flash-image-preview",
                prompt="请生成一张可爱的小海獭在海洋中游泳的图片",
                max_attempts=1,
                retry_delay=0,
            )
        )

    assert result == [image_b64]


def test_generation_utils_routes_text_to_uniapi():
    mock_provider = AsyncMock()
    mock_provider.generate_text.return_value = ["ok"]

    with patch.object(generation_utils, "uniapi_provider", mock_provider):
        result = asyncio.run(
            generation_utils.call_openai_compatible_text_with_retry_async(
                provider_name="uniapi",
                model_name="gpt-5.4",
                contents=[{"type": "text", "text": "Hello"}],
                config={"system_prompt": "Be helpful", "temperature": 1.0, "max_output_tokens": 1024},
                max_attempts=1,
                retry_delay=0,
            )
        )

    assert result == ["ok"]
