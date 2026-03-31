"""
Provider 槽位配置测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import generation_utils


class DummyProvider:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    async def close(self):
        self.closed = True


def test_slot_provider_uses_exact_alias_instance():
    original_clients = generation_utils.provider_clients.copy()
    original_evolink = generation_utils.evolink_provider

    try:
        text_provider = DummyProvider("text")
        image_provider = DummyProvider("image")

        generation_utils._register_provider_client("evolink#text", text_provider)
        generation_utils._register_provider_client("evolink#image", image_provider)

        assert generation_utils.is_openai_compatible_provider("evolink#text")
        assert generation_utils.is_openai_compatible_provider("evolink#image")
        assert generation_utils.get_openai_compatible_provider("evolink#text") is text_provider
        assert generation_utils.get_openai_compatible_provider("evolink#image") is image_provider
    finally:
        generation_utils.provider_clients.clear()
        generation_utils.provider_clients.update(original_clients)
        generation_utils.evolink_provider = original_evolink

