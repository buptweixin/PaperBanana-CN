"""
双 Provider 配置测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import ExpConfig


def test_config_defaults_split_providers_from_provider():
    cfg = ExpConfig(
        dataset_name="Demo",
        provider="uniapi",
    )

    assert cfg.provider == "uniapi"
    assert cfg.text_provider == "uniapi"
    assert cfg.image_provider == "uniapi"
    assert cfg.text_api_mode == "chat_completions"


def test_config_supports_independent_text_and_image_providers():
    cfg = ExpConfig(
        dataset_name="Demo",
        provider="evolink",
        text_provider="88996",
        image_provider="evolink",
    )

    assert cfg.provider == "evolink"
    assert cfg.text_provider == "88996"
    assert cfg.image_provider == "evolink"


def test_config_supports_text_api_mode():
    cfg = ExpConfig(
        dataset_name="Demo",
        text_api_mode="responses",
    )

    assert cfg.text_api_mode == "responses"
