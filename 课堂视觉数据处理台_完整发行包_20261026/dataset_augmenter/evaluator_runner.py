"""Run an external model on a processed manifest and score its CSV output."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping

import pandas as pd

from model_evaluation import compare_condition_impact, compute_evaluation_metrics, normalize_evaluation_csv

ALLOWED_PLACEHOLDERS = {"manifest_path", "output_csv", "output_dir", "task", "dataset_name", "model_name", "model_version"}


@dataclass
class ModelRunnerResult:
    status: str
    command: list[str]
    input_manifest: str
    output_csv: str
    report_dir: str
    exit_code: int | None
    started_at_utc: str
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_input_manifest(records: Iterable[Mapping[str, Any]], output_path: str | Path, include_missing: bool = False) -> pd.DataFrame:
    """Write one stable input row per processed image/frame."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if not include_missing and str(record.get("status", "完成")) != "完成":
            continue
        path = str(record.get("output_path", ""))
        if not include_missing and not path:
            continue
        rows.append({
            "sample_id": record.get("sample_id", ""),
            "condition": record.get("condition", ""),
            "task": record.get("task", record.get("task_namespace", "待指定")),
            "media_type": record.get("media_type", "image"),
            "input_path": path,
            "source_path": record.get("source_path", ""),
            "file_name": record.get("file_name", ""),
            "dataset_name": record.get("dataset_name", ""),
            "dataset_id": record.get("dataset_id", "unknown"),
            "label_namespace": record.get("label_namespace", ""),
            "source_label": record.get("source_label", record.get("label_from_annotation_or_folder", "")),
            "canonical_label": record.get("canonical_label", ""),
            "qualified_label": record.get("qualified_label", ""),
            "person_id": record.get("person_id", ""),
            "resolution_setting": record.get("resolution_setting", ""),
            "privacy_mode": record.get("privacy_mode", ""),
            "structured_occlusion_mode": record.get("structured_occlusion_mode", ""),
        })
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return frame


def validate_command_template(template: str) -> list[str]:
    if not str(template or "").strip():
        raise ValueError("模型命令不能为空")
    names = {field for _, field, _, _ in Formatter().parse(str(template)) if field}
    unknown = sorted(names - ALLOWED_PLACEHOLDERS)
    if unknown:
        raise ValueError(f"模型命令包含不支持的占位符：{', '.join(unknown)}")
    return shlex.split(str(template), posix=False)


def expand_command_template(template: str, values: Mapping[str, Any]) -> list[str]:
    argv = validate_command_template(template)
    substitutions = {key: str(values.get(key, "")) for key in ALLOWED_PLACEHOLDERS}
    expanded: list[str] = []
    for token in argv:
        if token.startswith('"') and token.endswith('"'):
            token = token[1:-1]
        expanded.append(token.format(**substitutions))
    return expanded


def run_external_model(records: Iterable[Mapping[str, Any]], command_template: str, report_dir: str | Path, *, task: str = "", model_name: str = "", model_version: str = "", dataset_name: str = "", timeout_seconds: float = 3600.0, env: Mapping[str, str] | None = None) -> tuple[ModelRunnerResult, pd.DataFrame]:
    """Run a user-owned model once; it must write the ``output_csv`` path."""
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_manifest = root / "model_input_manifest.csv"
    output_csv = root / "model_predictions.csv"
    build_model_input_manifest(records, input_manifest)
    values = {"manifest_path": input_manifest, "output_csv": output_csv, "output_dir": root, "task": task, "dataset_name": dataset_name, "model_name": model_name, "model_version": model_version}
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    try:
        argv = expand_command_template(command_template, values)
    except Exception as exc:
        return ModelRunnerResult("命令无效", [], str(input_manifest), str(output_csv), str(root), None, started, 0.0, error=str(exc)), pd.DataFrame()
    try:
        completed = subprocess.run(argv, cwd=str(root), env={**os.environ, **dict(env or {})}, capture_output=True, text=True, timeout=max(1.0, float(timeout_seconds)), shell=False, check=False)
        duration = time.perf_counter() - begin
        stdout, stderr = completed.stdout[-20000:], completed.stderr[-20000:]
        if completed.returncode != 0:
            result = ModelRunnerResult("模型命令失败", argv, str(input_manifest), str(output_csv), str(root), completed.returncode, started, duration, stdout, stderr, "外部模型返回非零退出码")
            return result, pd.DataFrame()
        if not output_csv.exists():
            result = ModelRunnerResult("未生成预测CSV", argv, str(input_manifest), str(output_csv), str(root), completed.returncode, started, duration, stdout, stderr, "外部模型没有写出output_csv")
            return result, pd.DataFrame()
        prediction = pd.read_csv(output_csv)
        return ModelRunnerResult("完成", argv, str(input_manifest), str(output_csv), str(root), completed.returncode, started, duration, stdout, stderr), prediction
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - begin
        return ModelRunnerResult("超时", argv, str(input_manifest), str(output_csv), str(root), None, started, duration, str(exc.stdout or "")[-20000:], str(exc.stderr or "")[-20000:], "超过设定的模型运行时间"), pd.DataFrame()
    except Exception as exc:
        duration = time.perf_counter() - begin
        return ModelRunnerResult("执行异常", argv, str(input_manifest), str(output_csv), str(root), None, started, duration, error=str(exc)), pd.DataFrame()


def evaluate_runner_output(prediction_df: pd.DataFrame, records: list[dict[str, Any]], baseline_condition: str = "原图") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    normalized, warnings = normalize_evaluation_csv(prediction_df, records)
    metrics = compute_evaluation_metrics(normalized)
    return metrics, compare_condition_impact(metrics, baseline_condition), warnings


def write_runner_report(result: ModelRunnerResult, report_dir: str | Path) -> Path:
    output = Path(report_dir) / "model_runner_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


__all__ = ["ALLOWED_PLACEHOLDERS", "ModelRunnerResult", "build_model_input_manifest", "validate_command_template", "expand_command_template", "run_external_model", "evaluate_runner_output", "write_runner_report"]
