"""
PaperVizProcessor 容错测试
"""

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.modules.setdefault("json_repair", types.SimpleNamespace(loads=lambda value: {}))

from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor


class NoopAgent:
    async def process(self, data, **kwargs):
        return data


class FakeRetrieverAgent:
    async def process(self, data, retrieval_setting="auto"):
        data["top10_references"] = []
        return data


class FakePlannerAgent:
    async def process(self, data):
        data["target_diagram_desc0"] = f"draft-for-{data['candidate_id']}"
        return data


class FakeVisualizerAgent:
    async def process(self, data):
        if data["candidate_id"] == 1:
            raise RuntimeError("visualizer boom")
        data["target_diagram_desc0_base64_jpg"] = "fake-image"
        return data


async def _collect_results(processor, data_list):
    results = []
    async for item in processor.process_queries_batch(
        data_list, max_concurrent=2, do_eval=False
    ):
        results.append(item)
    return results


def test_process_queries_batch_keeps_partial_results_when_one_candidate_fails():
    exp_config = ExpConfig(
        dataset_name="Demo",
        exp_mode="dev_planner",
        retrieval_setting="none",
        work_dir=Path(__file__).parent.parent,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=NoopAgent(),
        planner_agent=FakePlannerAgent(),
        visualizer_agent=FakeVisualizerAgent(),
        stylist_agent=NoopAgent(),
        critic_agent=NoopAgent(),
        retriever_agent=FakeRetrieverAgent(),
        polish_agent=NoopAgent(),
    )

    data_list = [
        {"candidate_id": 0, "content": "a", "visual_intent": "b"},
        {"candidate_id": 1, "content": "x", "visual_intent": "y"},
    ]

    results = asyncio.run(_collect_results(processor, data_list))
    result_map = {item["candidate_id"]: item for item in results}

    assert len(results) == 2
    assert result_map[0]["processing_status"] == "success"
    assert result_map[0]["target_diagram_desc0_base64_jpg"] == "fake-image"

    assert result_map[1]["processing_status"] == "failed"
    assert result_map[1]["processing_error_type"] == "RuntimeError"
    assert "visualizer boom" in result_map[1]["processing_error"]
    assert result_map[1]["target_diagram_desc0"] == "draft-for-1"
