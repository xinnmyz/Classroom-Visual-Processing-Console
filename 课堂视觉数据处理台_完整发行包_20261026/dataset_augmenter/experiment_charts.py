"""Experiment result charts for resolution and occlusion studies.

The report page already produces tabular metrics.  This module turns those
tables into portable PNG figures without making assumptions about a task
model.  It accepts the existing ``task, condition, metric, value`` table and
the condition-impact table produced by ``model_evaluation.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_condition(condition: Any) -> dict[str, Any]:
    """Parse a generated condition into sortable/plotting fields."""
    text = str(condition or "")
    lower = text.lower()
    if lower in {"原图", "clean", "original"} or lower.startswith("clean_"):
        resolution_rank = 1.0
        resolution_label = "原图"
    else:
        match = re.search(r"lowres[_-](\d+(?:\.\d+)?)", lower)
        divisor = float(match.group(1)) if match else 1.0
        resolution_rank = 1.0 / divisor if divisor else 1.0
        resolution_label = f"1/{int(divisor) if divisor.is_integer() else divisor:g}" if divisor else "原图"
    noise_match = re.search(r"noise[_-]([0-9]+(?:\.[0-9]+)?)pct", lower)
    occ_match = re.search(r"occ[_-]([0-9]+(?:\.[0-9]+)?)pct", lower)
    noise = float(noise_match.group(1)) if noise_match else 0.0
    occlusion = float(occ_match.group(1)) if occ_match else 0.0
    return {
        "condition": text,
        "resolution_rank": resolution_rank,
        "resolution_label": resolution_label,
        "noise_percent": noise,
        "occlusion_percent": occlusion,
        "condition_order": (-resolution_rank, noise, occlusion, text),
    }


def enrich_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add parsed condition fields while preserving the original columns."""
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    parsed = pd.DataFrame([parse_condition(value) for value in frame.get("condition", "")])
    for column in ("condition", "resolution_rank", "resolution_label", "noise_percent", "occlusion_percent", "condition_order"):
        if column in parsed and column not in frame:
            frame[column] = parsed[column].to_numpy()
    frame["value"] = pd.to_numeric(frame.get("value", np.nan), errors="coerce")
    return frame


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z一-龥_.-]+", "_", str(value or "unknown"))
    return text.strip("_.")[:80] or "chart"


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("生成图表需要 matplotlib，请运行 pip install matplotlib。") from exc


def plot_condition_curves(metrics: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate one condition curve per task/metric pair."""
    frame = enrich_metric_table(metrics)
    if frame.empty:
        return []
    plt = _mpl()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (task, metric), group in frame.groupby(["task", "metric"], dropna=False):
        group = group.sort_values("condition_order")
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
        ax.plot(group["condition"].astype(str), group["value"], marker="o", linewidth=2)
        ax.set_title(f"{task} - {metric}")
        ax.set_xlabel("处理条件")
        ax.set_ylabel(str(metric))
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = root / f"condition_curve_{_safe_name(task)}_{_safe_name(metric)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_resolution_curves(metrics: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Plot metrics against simulated resolution, grouped by task/metric."""
    frame = enrich_metric_table(metrics)
    if frame.empty:
        return []
    plt = _mpl()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    grouped = frame.groupby(["task", "metric", "resolution_label"], dropna=False)["value"].mean().reset_index()
    for (task, metric), group in grouped.groupby(["task", "metric"], dropna=False):
        order = {label: parse_condition(label)["resolution_rank"] for label in group["resolution_label"]}
        group = group.assign(_rank=group["resolution_label"].map(order)).sort_values("_rank", ascending=False)
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        ax.plot(group["resolution_label"], group["value"], marker="o", linewidth=2)
        ax.set_title(f"分辨率影响：{task} - {metric}")
        ax.set_xlabel("分辨率条件")
        ax.set_ylabel(str(metric))
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = root / f"resolution_curve_{_safe_name(task)}_{_safe_name(metric)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_impact_bars(comparison: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate relative-drop and impact-share bar charts."""
    if comparison is None or comparison.empty:
        return []
    frame = comparison.copy()
    for column in ("relative_drop_pct", "impact_share_pct"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["condition"])
    if frame.empty:
        return []
    plt = _mpl()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for column, title, name in (
        ("relative_drop_pct", "相对基准性能变化", "relative_drop"),
        ("impact_share_pct", "条件影响占比", "impact_share"),
    ):
        if column not in frame:
            continue
        grouped = frame.groupby(["task", "metric", "condition"], dropna=False)[column].mean().reset_index()
        grouped["label"] = grouped.apply(lambda row: f"{row['task']}|{row['metric']}|{row['condition']}", axis=1)
        grouped = grouped.sort_values(column, ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(11, max(4.5, 0.28 * len(grouped))), dpi=150)
        ax.barh(grouped["label"].astype(str)[::-1], grouped[column].to_numpy()[::-1])
        ax.set_title(title)
        ax.set_xlabel("百分比")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = root / f"{name}_bar.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_experiment_charts(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate all available charts and write a portable chart index."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    error = ""
    try:
        paths = plot_condition_curves(metrics, root)
        paths.extend(plot_resolution_curves(metrics, root))
        paths.extend(plot_impact_bars(comparison, root))
    except RuntimeError as exc:
        # Keep the report pipeline usable on the lightweight installation.
        # Installing matplotlib enables PNG output later without changing the
        # metrics tables.
        paths = []
        error = str(exc)
    index = {
        "chart_count": len(paths),
        "charts": [str(path) for path in paths],
        "metrics_rows": int(len(metrics)) if metrics is not None else 0,
        "comparison_rows": int(len(comparison)) if comparison is not None else 0,
    }
    if error:
        index["status"] = "matplotlib_unavailable"
        index["error"] = error
    import json

    (root / "chart_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


__all__ = [
    "parse_condition",
    "enrich_metric_table",
    "plot_condition_curves",
    "plot_resolution_curves",
    "plot_impact_bars",
    "write_experiment_charts",
]
