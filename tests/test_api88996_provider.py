"""
88996 Provider 单元测试
"""

import base64
import asyncio
from unittest.mock import AsyncMock, patch
from io import BytesIO
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.api88996 import Api88996Provider
from utils import generation_utils


def make_png_base64():
    """创建一个最小 PNG 并返回 base64 字符串。"""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_provider(api_key="test-key", base_url="https://88996.cloud"):
    return Api88996Provider(api_key=api_key, base_url=base_url)


class TestApi88996ProviderInit:
    def test_init_with_params(self):
        p = make_provider(api_key="sk-abc", base_url="https://example.com")
        assert p.api_key == "sk-abc"
        assert p.base_url == "https://example.com"

    def test_init_default_base_url(self):
        p = Api88996Provider(api_key="sk-abc")
        assert p.base_url == "https://88996.cloud"

    def test_headers_contain_auth(self):
        p = make_provider(api_key="sk-test")
        headers = p._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"


class TestRequestBuilding:
    def test_text_request_payload(self):
        p = make_provider()
        payload = p._build_text_payload(
            model_name="gpt-5-mini",
            contents=[{"type": "text", "text": "Hello"}],
            system_prompt="Be helpful",
            temperature=0.5,
            max_output_tokens=4096,
        )

        assert payload["model"] == "gpt-5-mini"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 4096
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2

    def test_responses_request_payload(self):
        p = make_provider()
        payload = p._build_responses_payload(
            model_name="gpt-5-mini",
            contents=[{"type": "text", "text": "Hello"}],
            system_prompt="Be helpful",
            temperature=0.5,
            max_output_tokens=4096,
        )

        assert payload["model"] == "gpt-5-mini"
        assert payload["temperature"] == 0.5
        assert payload["max_output_tokens"] == 4096
        assert len(payload["input"]) == 2

    def test_image_request_payload(self):
        p = make_provider()
        payload = p._build_image_payload(
            model_name="gpt-image-1",
            prompt="A cat on grass",
            aspect_ratio="16:9",
            quality="2K",
        )

        assert payload["model"] == "gpt-image-1"
        assert payload["prompt"] == "A cat on grass"
        assert payload["size"] == "1920x1080"


class TestTextGeneration:
    def test_text_generation_success(self):
        p = make_provider()
        mock_response = {
            "choices": [{"message": {"content": "This is a test response"}}]
        }

        with patch.object(p, "_post_json", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(
                p.generate_text(
                    model_name="gpt-5-mini",
                    contents=[{"type": "text", "text": "Hello"}],
                    system_prompt="You are helpful",
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )

        assert result == ["This is a test response"]

    def test_text_generation_success_via_responses(self):
        p = make_provider()
        mock_response = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "This is a responses API result"}
                    ]
                }
            ]
        }

        with patch.object(p, "_post_json", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(
                p.generate_text(
                    model_name="gpt-5-mini",
                    contents=[{"type": "text", "text": "Hello"}],
                    system_prompt="You are helpful",
                    temperature=0.7,
                    max_output_tokens=1000,
                    api_mode="responses",
                )
            )

        assert result == ["This is a responses API result"]


class TestImageGeneration:
    def test_image_generation_reads_url_result(self):
        p = make_provider()
        mock_response = {
            "data": [{"url": "https://example.com/image.png"}]
        }

        with patch.object(p, "_post_json", new_callable=AsyncMock, return_value=mock_response), \
             patch.object(p, "_download_image_as_base64", new_callable=AsyncMock, return_value=make_png_base64()):
            result = asyncio.run(
                p.generate_image(
                    model_name="gpt-image-1",
                    prompt="A beautiful diagram",
                    aspect_ratio="16:9",
                    quality="2K",
                    max_attempts=1,
                    retry_delay=0,
                )
            )

        assert len(result) == 1
        assert len(result[0]) > 10

    def test_image_generation_reads_b64_result(self):
        p = make_provider()
        mock_response = {
            "data": [{"b64_json": make_png_base64()}]
        }

        with patch.object(p, "_post_json", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(
                p.generate_image(
                    model_name="gpt-image-1",
                    prompt="A beautiful diagram",
                    aspect_ratio="16:9",
                    quality="2K",
                    max_attempts=1,
                    retry_delay=0,
                )
            )

        assert len(result) == 1
        assert len(result[0]) > 10

    def test_image_generation_with_image_urls_routes_to_edit(self):
        p = make_provider()

        with patch.object(p, "_download_image_bytes", new_callable=AsyncMock, return_value=b"image-bytes"), \
             patch.object(p, "edit_image", new_callable=AsyncMock, return_value=[make_png_base64()]) as mock_edit:
            result = asyncio.run(
                p.generate_image(
                    model_name="gpt-image-1",
                    prompt="Edit this image",
                    aspect_ratio="16:9",
                    quality="2K",
                    image_urls=["https://example.com/ref.png"],
                    max_attempts=1,
                    retry_delay=0,
                )
            )

        assert len(result) == 1
        mock_edit.assert_awaited_once()


class TestImageEdit:
    def test_edit_image_success(self):
        p = make_provider()
        mock_response = {
            "data": [{"b64_json": make_png_base64()}]
        }

        with patch.object(p, "_post_form", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(
                p.edit_image(
                    model_name="gpt-image-1",
                    image_bytes=b"fake-image",
                    prompt="Make it cleaner",
                    aspect_ratio="3:2",
                    quality="2K",
                    max_attempts=1,
                    retry_delay=0,
                )
            )

        assert len(result) == 1
        assert len(result[0]) > 10


class TestGenerationUtilsRouting:
    def test_openai_compatible_text_routes_to_88996(self):
        mock_provider = AsyncMock()
        mock_provider.generate_text.return_value = ["ok"]

        with patch.object(generation_utils, "api88996_provider", mock_provider):
            result = asyncio.run(
                generation_utils.call_openai_compatible_text_with_retry_async(
                    provider_name="88996",
                    model_name="gpt-5-mini",
                    contents=[{"type": "text", "text": "Hello"}],
                    config={"system_prompt": "Be helpful", "temperature": 1.0, "max_output_tokens": 1024},
                    max_attempts=1,
                    retry_delay=0,
                )
            )

        assert result == ["ok"]
        mock_provider.generate_text.assert_awaited_once()
