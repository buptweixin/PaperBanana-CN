"""
结果持久化工具。
"""

import json
import os
from pathlib import Path
from typing import Any


def dump_results_json(output_path: Path, results: list[dict[str, Any]]) -> None:
    """原子化写入结果 JSON，避免中途中断留下半截文件。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    with open(tmp_path, "w", encoding="utf-8", errors="surrogateescape") as f:
        json_string = json.dumps(results, ensure_ascii=False, indent=4)
        json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
        f.write(json_string)

    os.replace(tmp_path, output_path)


def load_results_json(input_path: Path) -> list[dict[str, Any]]:
    """读取结果 JSON；若文件不存在或损坏则返回空列表。"""
    input_path = Path(input_path)
    if not input_path.exists():
        return []

    try:
        with open(input_path, "r", encoding="utf-8", errors="surrogateescape") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []
