"""Optional model plugin runners.

Plugins consume the stable ``model_input_manifest.csv`` contract and write a
prediction CSV.  The base application can therefore remain small while users
bring their own weights and task-specific dependencies.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from evaluator_runner import ModelRunnerResult, run_external_model
from model_registry import ModelSpec, verify_weights


def _write_prediction(rows: list[dict[str, Any]], output_csv: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return frame


def run_external_plugin(
    records: list[dict[str, Any]],
    command_template: str,
    report_dir: str | Path,
    **kwargs: Any,
) -> tuple[ModelRunnerResult, pd.DataFrame]:
    """Run a user-owned classifier/regressor/detector command."""
    return run_external_model(records, command_template, report_dir, **kwargs)


def run_local_python_plugin(
    records: list[dict[str, Any]],
    script_path: str | Path,
    output_csv: str | Path,
    *,
    plugin_function: str = "predict_batch",
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a local Python plugin exposing ``predict_batch(frame, config)``.

    The plugin is imported by file path and is responsible for loading its own
    model weights.  It may return a DataFrame, a list of dictionaries, or a
    scalar/list of predictions aligned with the input rows.
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(script)
    spec = importlib.util.spec_from_file_location("classroom_model_plugin", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载插件：{script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, plugin_function, None)
    if function is None:
        raise AttributeError(f"插件缺少函数 {plugin_function}(frame, config)")
    frame = pd.DataFrame(records)
    result = function(frame, dict(config or {}))
    if isinstance(result, pd.DataFrame):
        prediction = result
    elif isinstance(result, list) and all(isinstance(item, dict) for item in result):
        prediction = pd.DataFrame(result)
    else:
        values = list(result) if isinstance(result, (list, tuple)) else [result] * len(frame)
        if len(values) != len(frame):
            raise ValueError("插件预测数量必须与输入样本数一致")
        prediction = frame[["sample_id", "condition", "task"]].copy()
        prediction["y_pred"] = values
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output, index=False, encoding="utf-8-sig")
    return prediction


def run_ultralytics_plugin(
    records: list[dict[str, Any]],
    spec: ModelSpec,
    output_csv: str | Path,
    *,
    confidence: float = 0.25,
    device: str = "",
) -> pd.DataFrame:
    """Run optional Ultralytics detection or pose inference.

    This produces raw boxes/keypoints.  Detection/pose metrics require ground
    truth annotations and are calculated by the evaluation layer separately.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请运行 install_tracking_dependencies.bat 或模型插件依赖脚本") from exc
    weight_check = verify_weights(spec)
    model_path = spec.weights_path or spec.entrypoint
    model = YOLO(model_path)
    rows: list[dict[str, Any]] = []
    task = spec.task
    for record in records:
        input_path = str(record.get("output_path", ""))
        if not input_path:
            continue
        kwargs: dict[str, Any] = {"conf": float(confidence), "verbose": False}
        if device:
            kwargs["device"] = device
        results = model.predict(input_path, **kwargs)
        result = results[0] if results else None
        row = {
            "sample_id": record.get("sample_id", ""),
            "condition": record.get("condition", ""),
            "task": task,
            "model_id": spec.model_id,
            "model_version": spec.version,
            "weights_status": weight_check.get("status", ""),
        }
        if result is not None and getattr(result, "boxes", None) is not None:
            boxes = result.boxes
            row["boxes_json"] = json.dumps(boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else [], ensure_ascii=False)
            row["scores_json"] = json.dumps(boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else [], ensure_ascii=False)
            row["classes_json"] = json.dumps(boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else [], ensure_ascii=False)
        if "pose" in spec.model_id and result is not None and getattr(result, "keypoints", None) is not None:
            points = result.keypoints.xy
            row["keypoints_json"] = json.dumps(points.cpu().tolist() if hasattr(points, "cpu") else [], ensure_ascii=False)
        rows.append(row)
    return _write_prediction(rows, Path(output_csv))


def run_registered_plugin(
    records: list[dict[str, Any]],
    model_spec: ModelSpec,
    report_dir: str | Path,
    *,
    command_template: str = "",
    plugin_script: str = "",
    config: Mapping[str, Any] | None = None,
) -> tuple[ModelRunnerResult | None, pd.DataFrame]:
    """Dispatch a registered model without importing optional dependencies."""
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_csv = root / "model_predictions.csv"
    if model_spec.backend == "external_command":
        result, frame = run_external_plugin(records, command_template, root, model_name=model_spec.display_name, model_version=model_spec.version)
        return result, frame
    if model_spec.backend == "python_plugin":
        frame = run_local_python_plugin(records, plugin_script, output_csv, config=config)
        return None, frame
    if model_spec.backend == "ultralytics":
        frame = run_ultralytics_plugin(records, model_spec, output_csv, **dict(config or {}))
        return None, frame
    raise ValueError(f"不支持的插件后端：{model_spec.backend}")


__all__ = ["run_external_plugin", "run_local_python_plugin", "run_ultralytics_plugin", "run_registered_plugin"]
