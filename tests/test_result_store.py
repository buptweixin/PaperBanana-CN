"""
结果持久化测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.result_store import dump_results_json, load_results_json


def test_dump_and_load_results_json_roundtrip(tmp_path):
    output_path = tmp_path / "results" / "checkpoint.json"
    payload = [
        {"candidate_id": 0, "processing_status": "success"},
        {"candidate_id": 1, "processing_status": "failed", "processing_error": "boom"},
    ]

    dump_results_json(output_path, payload)
    loaded = load_results_json(output_path)

    assert loaded == payload


def test_load_results_json_returns_empty_list_for_missing_or_invalid_file(tmp_path):
    missing_path = tmp_path / "missing.json"
    assert load_results_json(missing_path) == []

    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{invalid json", encoding="utf-8")
    assert load_results_json(broken_path) == []
