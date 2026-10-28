"""Dataset-scoped task and label mapping.

The classroom datasets use different label vocabularies.  This module keeps a
label's dataset namespace attached to it instead of silently merging similarly
named classes from unrelated datasets.
"""

from __future__ import annotations

import re
from typing import Any


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "daisee": {
        "display_name": "DAiSEE",
        "task": "投入度/情感状态",
        "label_namespace": "DAiSEE",
        "labels": {
            "boredom": "无聊",
            "confusion": "困惑",
            "frustration": "挫败",
            "engagement": "投入度",
            "very low": "很低",
            "low": "低",
            "high": "高",
            "very high": "很高",
        },
        "note": "DAiSEE 的状态/投入度标签，不作为微表情类别使用。",
    },
    "scb": {
        "display_name": "SCB-dataset",
        "task": "课堂行为",
        "label_namespace": "SCB-dataset",
        "labels": {
            "hand_up": "举手",
            "handup": "举手",
            "raise_hand": "举手",
            "reading": "阅读",
            "read": "阅读",
            "writing": "写字",
            "write": "写字",
            "listening": "听课",
            "listen": "听课",
            "using_phone": "使用手机",
            "phone": "使用手机",
            "sleeping": "睡觉",
            "talking": "交谈",
            "standing": "站立",
        },
        "note": "SCB 行为标签只在 SCB 命名空间内解释。",
    },
    "tianchi": {
        "display_name": "Tianchi/课堂行为数据",
        "task": "课堂行为",
        "label_namespace": "Tianchi",
        "labels": {},
        "note": "Tianchi/Roboflow 的原始类别以数据集自身 12 类标签为准，不自动映射到 SCB。",
    },
    "micro_expression": {
        "display_name": "微表情数据集",
        "task": "微表情",
        "label_namespace": "微表情数据集",
        "labels": {},
        "note": "仅当来源明确为微表情数据集时使用。",
    },
    "unknown": {
        "display_name": "未登记数据集",
        "task": "待指定",
        "label_namespace": "未登记",
        "labels": {},
        "note": "需要用户指定任务和标签含义。",
    },
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", " ", str(value or "").lower()).strip()


def identify_dataset(*values: Any) -> str:
    """Identify a dataset family from provenance text, conservatively."""
    normalized = [_norm(v) for v in values if v is not None]
    text = " ".join(normalized)
    # Prefer the dataset name, file path and archive member.  A generic source
    # description such as "DAiSEE/Tianchi data" must not classify every asset
    # under one dataset.
    primary = " ".join(normalized[index] for index in (0, 3, 4) if index < len(normalized))
    if any(token in primary for token in ("student behaviour detection", "student behavior detection", "tianchi", "roboflow")):
        return "tianchi"
    if any(token in primary for token in ("scb dataset", "scb", "whiffe")):
        return "scb"
    if any(token in primary for token in ("daisee", "engagement", "boredom", "frustration", "confusion")):
        return "daisee"
    if any(token in primary for token in ("micro expression", "microexpression", "微表情", "samm", "casme")):
        return "micro_expression"
    # Fall back to provenance text only when it identifies one family rather
    # than listing several unrelated sources together.
    candidates = []
    if "daisee" in text or any(token in text for token in ("engagement", "boredom", "frustration", "confusion")):
        candidates.append("daisee")
    if "scb" in text or "whiffe" in text:
        candidates.append("scb")
    if "tianchi" in text or "roboflow" in text:
        candidates.append("tianchi")
    if "micro expression" in text or "microexpression" in text or "微表情" in text:
        candidates.append("micro_expression")
    if len(set(candidates)) == 1:
        return candidates[0]
    return "unknown"


def dataset_info(dataset_id: str) -> dict[str, Any]:
    return DATASET_REGISTRY.get(dataset_id, DATASET_REGISTRY["unknown"])


def map_label(dataset_id: str, raw_label: Any) -> dict[str, str]:
    """Return a dataset-scoped mapping while preserving the original label."""
    info = dataset_info(dataset_id)
    source_label = str(raw_label or "未标注").strip() or "未标注"
    key = _norm(source_label)
    normalized_labels = {_norm(label): value for label, value in info.get("labels", {}).items()}
    canonical = normalized_labels.get(key, source_label)
    return {
        "source_label": source_label,
        "canonical_label": canonical,
        "label_namespace": str(info.get("label_namespace", dataset_id)),
        "qualified_label": f"{info.get('label_namespace', dataset_id)}::{canonical}",
    }


def annotate_asset(asset: dict[str, Any]) -> dict[str, Any]:
    """Add stable dataset/task/label fields without changing source labels."""
    dataset_id = identify_dataset(
        asset.get("dataset_name"),
        asset.get("origin_name"),
        asset.get("origin_url"),
        asset.get("source_path"),
        asset.get("member"),
    )
    info = dataset_info(dataset_id)
    raw_label = asset.get("label") or asset.get("annotation_label") or "未标注"
    mapping = map_label(dataset_id, raw_label)
    output = dict(asset)
    output.update(
        {
            "dataset_id": dataset_id,
            "dataset_display_name": info.get("display_name", dataset_id),
            "task_namespace": info.get("task", "待指定"),
            "label_namespace": mapping["label_namespace"],
            "source_label": mapping["source_label"],
            "canonical_label": mapping["canonical_label"],
            "qualified_label": mapping["qualified_label"],
            "label_mapping_note": info.get("note", ""),
        }
    )
    return output


def registry_rows() -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": dataset_id,
            "dataset_name": info["display_name"],
            "task": info["task"],
            "label_namespace": info["label_namespace"],
            "known_label_count": len(info.get("labels", {})),
            "note": info.get("note", ""),
        }
        for dataset_id, info in DATASET_REGISTRY.items()
    ]

