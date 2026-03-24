"""
GGboom Provider 单元测试
"""

import asyncio
from unittest.mock import AsyncMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.ggboom import GgboomProvider
from utils import generation_utils


def make_provider(api_key="test-key", base_url="https://ai.qaq.al"):
    return GgboomProvider(api_key=api_key, base_url=base_url)


def test_init_default_base_url():
    p = GgboomProvider(api_key="sk-abc")
    assert p.base_url == "https://ai.qaq.al"


def test_generation_utils_routes_text_to_ggboom():
    mock_provider = AsyncMock()
    mock_provider.generate_text.return_value = ["ok"]

    with patch.object(generation_utils, "ggboom_provider", mock_provider):
        result = asyncio.run(
            generation_utils.call_openai_compatible_text_with_retry_async(
                provider_name="ggboom",
                model_name="gpt-5.4",
                contents=[{"type": "text", "text": "Hello"}],
                config={"system_prompt": "Be helpful", "temperature": 1.0, "max_output_tokens": 1024},
                max_attempts=1,
                retry_delay=0,
            )
        )

    assert result == ["ok"]
    mock_provider.generate_text.assert_awaited_once()
