"""Optional model registry for classroom-analysis plugins.

The registry stores model metadata and weight provenance only.  Large model
weights are never bundled with the desktop application.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    task: str
    backend: str
    entrypoint: str = ""
    version: str = ""
    weights_path: str = ""
    weights_url: str = ""
    weights_sha256: str = ""
    license: str = ""
    input_schema: str = "model_input_manifest.csv"
    output_schema: str = "sample_id,condition,y_true,y_pred"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


DEFAULT_MODEL_SPECS = [
    ModelSpec(
        "yolo_detection",
        "YOLO 多人/课堂行为检测",
        "课堂行为检测",
        "ultralytics",
        entrypoint="yolo11n.pt",
        output_schema="sample_id,condition,boxes_json,scores_json,classes_json",
        notes="可选安装 ultralytics；权重需单独下载并记录 SHA-256。",
    ),
    ModelSpec(
        "yolo_pose",
        "YOLO 姿态估计",
        "姿态",
        "ultralytics",
        entrypoint="yolo11n-pose.pt",
        output_schema="sample_id,condition,keypoints_json,boxes_json,scores_json",
        notes="可选安装 ultralytics；AP/AR/PCK/OKS 需要真实关键点标注。",
    ),
    ModelSpec(
        "external_classification",
        "外部分类模型",
        "分类/微表情/课堂行为",
        "external_command",
        output_schema="sample_id,condition,task,y_true,y_pred,model_name,model_version",
        notes="通过自动模型评估命令模板运行。",
    ),
    ModelSpec(
        "external_regression",
        "外部投入度回归模型",
        "投入度",
        "external_command",
        output_schema="sample_id,condition,task,y_true,y_pred,model_name,model_version",
        notes="y_true/y_pred 必须是数值；程序计算 MAE/RMSE。",
    ),
]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(spec: ModelSpec) -> dict[str, Any]:
    path = Path(spec.weights_path) if spec.weights_path else None
    if path is None or not path.exists():
        return {"status": "missing", "path": str(path or ""), "sha256": ""}
    actual = sha256_file(path)
    expected = spec.weights_sha256.strip().lower()
    return {
        "status": "verified" if expected and actual.lower() == expected else "unverified",
        "path": str(path),
        "sha256": actual,
        "expected_sha256": expected,
    }


class ModelRegistry:
    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {spec.model_id: spec for spec in (specs or DEFAULT_MODEL_SPECS)}

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._specs:
            raise KeyError(f"未知模型：{model_id}")
        return self._specs[model_id]

    def list(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def rows(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in self.list()]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.rows(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        target = Path(path)
        if not target.exists():
            return cls()
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls([ModelSpec(**row) for row in data if isinstance(row, dict)])


__all__ = ["ModelSpec", "ModelRegistry", "DEFAULT_MODEL_SPECS", "sha256_file", "verify_weights"]
