"""Model evaluation import and condition-comparison helpers.

The desktop data-preprocessing app intentionally does not run a task model. This
module consumes model evaluation CSV files produced by an external detector,
classifier, engagement regressor, or pose estimator and computes comparable
condition-level metrics.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd

CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "uar", "macro_f1")
REGRESSION_METRICS = ("mae", "rmse")
DETECTION_METRICS = ("precision", "recall", "map", "macro_f1")
POSE_METRICS = ("ap", "ar", "pck", "oks")


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lowered = {str(c).lower().strip(): c for c in frame.columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _as_text(series: pd.Series, default: str = "") -> pd.Series:
    return series.fillna(default).astype(str).str.strip()


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _classification_metrics(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    valid = ~(true.isna() | pred.isna())
    true = true[valid].astype(str)
    pred = pred[valid].astype(str)
    n = len(true)
    if n == 0:
        return {key: float("nan") for key in CLASSIFICATION_METRICS} | {"sample_count": 0}
    labels = sorted(set(true.tolist()) | set(pred.tolist()))
    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = int(((true == label) & (pred == label)).sum())
        fn = int(((true == label) & (pred != label)).sum())
        fp = int(((true != label) & (pred == label)).sum())
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        p, r = precisions[-1], recalls[-1]
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    accuracy = float((true == pred).mean())
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "sample_count": n,
    }


def _regression_metrics(true: pd.Series, pred: pd.Series) -> dict[str, float]:
    true = pd.to_numeric(true, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    valid = ~(true.isna() | pred.isna())
    if not valid.any():
        return {key: float("nan") for key in REGRESSION_METRICS} | {"sample_count": 0}
    delta = pred[valid].to_numpy(dtype=float) - true[valid].to_numpy(dtype=float)
    return {"mae": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(delta ** 2))), "sample_count": int(valid.sum())}


def _binary_detection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate detection rows when each row carries tp/fp/fn or gt/pred counts."""
    def number(names: tuple[str, ...]) -> float:
        col = _first_column(frame, names)
        return float(pd.to_numeric(frame[col], errors="coerce").fillna(0).sum()) if col else 0.0
    tp, fp, fn = number(("tp", "true_positive", "true_positives")), number(("fp", "false_positive", "false_positives")), number(("fn", "false_negative", "false_negatives"))
    if tp + fp + fn == 0:
        # Some exporters only contain is_correct and positive flags.
        corr = _first_column(frame, ("is_correct", "correct", "match"))
        if corr:
            values = frame[corr].astype(str).str.lower().isin({"1", "true", "yes", "correct"})
            tp = float(values.sum())
            fp = float((~values).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and precision + recall else float("nan")
    return {"precision": precision, "recall": recall, "macro_f1": f1, "map": precision, "sample_count": int(len(frame))}


def normalize_evaluation_csv(uploaded: pd.DataFrame, records: list[dict[str, Any]] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Normalize common external evaluation CSV layouts.

    Supported layouts:
    1. Metric summary: ``task,condition,metric,value``.
    2. Per-sample classification/regression: ``sample_id,condition,y_true,y_pred``.
    3. Legacy app layout: ``sample_id,baseline_score,condition_score``.
    Missing condition values are joined from the latest processed manifest when
    the sample IDs are available.
    """
    warnings: list[str] = []
    if uploaded is None or uploaded.empty:
        return pd.DataFrame(), ["评估CSV为空"]
    raw = uploaded.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    metric_col = _first_column(raw, ("metric", "metric_name", "指标"))
    value_col = _first_column(raw, ("value", "metric_value", "score", "指标值"))
    task_col = _first_column(raw, ("task", "任务", "task_name"))
    condition_col = _first_column(raw, ("condition", "条件", "condition_name"))
    sample_col = _first_column(raw, ("sample_id", "id", "sample"))
    out = pd.DataFrame(index=raw.index)
    out["task"] = _as_text(raw[task_col], "待指定") if task_col else "待指定"
    out["condition"] = _as_text(raw[condition_col], "原图") if condition_col else "原图"
    if sample_col:
        out["sample_id"] = _as_text(raw[sample_col])
    else:
        out["sample_id"] = [f"row_{i}" for i in range(len(raw))]
    if metric_col and value_col:
        out["metric"] = _as_text(raw[metric_col], "custom")
        out["value"] = pd.to_numeric(raw[value_col], errors="coerce")
        out["row_type"] = "metric_summary"
        out["sample_count"] = pd.to_numeric(raw[_first_column(raw, ("sample_count", "n", "count"))], errors="coerce") if _first_column(raw, ("sample_count", "n", "count")) else np.nan
    else:
        true_col = _first_column(raw, ("y_true", "true_label", "ground_truth", "label", "target", "真实标签"))
        pred_col = _first_column(raw, ("y_pred", "pred_label", "prediction", "predicted_label", "预测标签"))
        base_col = _first_column(raw, ("baseline_score", "base_score"))
        cond_score_col = _first_column(raw, ("condition_score", "processed_score"))
        if base_col and cond_score_col:
            # Preserve legacy score comparison as a metric-summary compatible table.
            out["metric"] = _as_text(raw[metric_col], "score") if metric_col else "score"
            out["baseline_score"] = pd.to_numeric(raw[base_col], errors="coerce")
            out["condition_score"] = pd.to_numeric(raw[cond_score_col], errors="coerce")
            out["value"] = out["condition_score"]
            out["baseline_value"] = out["baseline_score"]
            out["condition"] = _as_text(raw[condition_col], "处理条件") if condition_col else "处理条件"
            out["row_type"] = "paired_score"
            warnings.append("已识别旧版baseline_score/condition_score格式；建议改用按condition分行的y_true/y_pred格式。")
        elif true_col and pred_col:
            out["y_true"] = raw[true_col]
            out["y_pred"] = raw[pred_col]
            out["row_type"] = "sample"
            task_text = out["task"].str.lower()
            out["metric"] = np.where(task_text.str.contains("投入|engagement|regress|回归"), "rmse", "macro_f1")
        elif any(_first_column(raw, names) for names in (("tp", "true_positive"), ("fp", "false_positive"), ("fn", "false_negative"), ("iou",), ("pck",), ("oks",), ("ap",), ("ar",), ("map",))):
            # Detection/pose exporters often provide counts or one metric per row.
            direct_metric = _first_column(raw, ("metric", "metric_name", "指标"))
            direct_value = _first_column(raw, ("value", "metric_value", "指标值", "ap", "ar", "map", "pck", "oks", "iou"))
            out["row_type"] = "detection_pose"
            out["metric"] = _as_text(raw[direct_metric], "custom") if direct_metric else (str(direct_value) if direct_value else "custom")
            out["value"] = pd.to_numeric(raw[direct_value], errors="coerce") if direct_value else np.nan
            for alias in ("tp", "true_positive", "fp", "false_positive", "fn", "false_negative", "iou", "pck", "oks", "ap", "ar", "map"):
                col = _first_column(raw, (alias,))
                if col and alias not in out.columns:
                    out[alias] = pd.to_numeric(raw[col], errors="coerce")
        else:
            warnings.append("未识别评估CSV格式，需要metric/value、y_true/y_pred，或检测/姿态指标列。")
            return pd.DataFrame(), warnings
    # Join conditions from the current manifest only for per-sample rows.
    # Paired baseline/condition files intentionally carry a synthetic condition
    # when they do not have one, so do not overwrite it with the first manifest row.
    can_join_manifest = not out.get("row_type", pd.Series(dtype=str)).eq("paired_score").any()
    if records and can_join_manifest and (condition_col is None or out["condition"].eq("原图").all()):
        manifest = pd.DataFrame(records)
        if "sample_id" in manifest.columns:
            manifest = manifest[["sample_id", "condition"]].drop_duplicates("sample_id")
            out = out.drop(columns=["condition"]).merge(manifest, on="sample_id", how="left")
            out["condition"] = out["condition"].fillna("原图")
            out = out[["sample_id", "task", "condition"] + [c for c in out.columns if c not in {"sample_id", "task", "condition"}]]
    return out.reset_index(drop=True), warnings


def compute_evaluation_metrics(normalized: pd.DataFrame) -> pd.DataFrame:
    """Compute metric rows grouped by task, condition and requested metric."""
    if normalized is None or normalized.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (task, condition), frame in normalized.groupby(["task", "condition"], dropna=False):
        task_text = str(task).lower()
        # A metric summary is already computed by the external evaluator.
        summary = frame[frame["row_type"].eq("metric_summary")]
        if not summary.empty:
            for metric, group in summary.groupby("metric", dropna=False):
                values = pd.to_numeric(group["value"], errors="coerce").dropna()
                if values.empty:
                    continue
                row = {"task": task, "condition": condition, "metric": str(metric), "value": float(values.mean()), "sample_count": int(pd.to_numeric(group.get("sample_count", pd.Series(dtype=float)), errors="coerce").sum() or len(group)), "source": "外部模型汇总"}
                if "baseline_value" in group.columns:
                    base = pd.to_numeric(group["baseline_value"], errors="coerce").dropna()
                    if not base.empty:
                        row["baseline_value"] = float(base.mean())
                rows.append(row)
            continue
        paired = frame[frame.get("row_type", pd.Series(index=frame.index, dtype=str)).eq("paired_score")]
        if not paired.empty:
            for metric, group in paired.groupby("metric", dropna=False):
                values = pd.to_numeric(group["value"], errors="coerce").dropna()
                bases = pd.to_numeric(group.get("baseline_value"), errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append({
                    "task": task, "condition": condition, "metric": str(metric),
                    "value": float(values.mean()),
                    "baseline_value": float(bases.mean()) if not bases.empty else np.nan,
                    "sample_count": int(len(values)), "source": "外部模型配对结果",
                })
            continue
        if "row_type" in frame.columns and frame["row_type"].eq("detection_pose").any():
            # Direct AP/AR/PCK/OKS values take precedence when present.
            direct_values = pd.to_numeric(frame.get("value", pd.Series(index=frame.index)), errors="coerce")
            direct_metrics = frame.get("metric", pd.Series(["custom"] * len(frame), index=frame.index)).astype(str)
            for metric, group in frame.assign(_direct_value=direct_values).groupby(direct_metrics, dropna=False):
                values = pd.to_numeric(group["_direct_value"], errors="coerce").dropna()
                if not values.empty:
                    rows.append({"task": task, "condition": condition, "metric": str(metric), "value": float(values.mean()), "sample_count": int(len(values)), "source": "检测/姿态逐条指标"})
            # A row may carry several pose metrics at once (AP, AR, PCK, OKS).
            for alias in ("ap", "ar", "map", "pck", "oks", "iou"):
                if alias not in frame.columns:
                    continue
                values = pd.to_numeric(frame[alias], errors="coerce").dropna()
                if not values.empty and not (direct_metrics.str.lower().eq(alias).any()):
                    rows.append({"task": task, "condition": condition, "metric": alias, "value": float(values.mean()), "sample_count": int(len(values)), "source": "检测/姿态逐条指标"})
            # Also support rows containing tp/fp/fn counts.
            count_metrics = _binary_detection_metrics(frame)
            has_counts = any(_first_column(frame, names) for names in (("tp", "true_positive"), ("fp", "false_positive"), ("fn", "false_negative")))
            if has_counts:
                for metric, value in count_metrics.items():
                    if metric == "sample_count" or value != value:
                        continue
                    rows.append({"task": task, "condition": condition, "metric": metric, "value": float(value), "sample_count": int(len(frame)), "source": "检测逐条计数"})
            continue
        if "y_true" in frame.columns and "y_pred" in frame.columns:
            if any(x in task_text for x in ("投入", "engagement", "回归", "regress")):
                metrics = _regression_metrics(frame["y_true"], frame["y_pred"])
            elif any(x in task_text for x in ("检测", "detection", "行为检测", "object")) and not frame["y_true"].astype(str).str.len().eq(0).all():
                metrics = _binary_detection_metrics(frame)
            else:
                metrics = _classification_metrics(frame["y_true"], frame["y_pred"])
            for metric, value in metrics.items():
                if metric == "sample_count" or value != value:
                    continue
                rows.append({"task": task, "condition": condition, "metric": metric, "value": float(value), "sample_count": int(metrics.get("sample_count", len(frame))), "source": "逐样本计算"})
    return pd.DataFrame(rows)


def compare_condition_impact(metrics: pd.DataFrame, baseline_condition: str = "原图") -> pd.DataFrame:
    """Compare every condition against baseline; positive drop means degradation."""
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    keys = ["task", "metric"]
    base = metrics[metrics["condition"].astype(str).eq(baseline_condition)][keys + ["value"]].rename(columns={"value": "baseline_value"})
    out = metrics.merge(base, on=keys, how="left", suffixes=("", "_from_baseline"))
    if "baseline_value" in metrics.columns:
        out["baseline_value"] = out["baseline_value"].fillna(pd.to_numeric(out.get("baseline_value_from_baseline"), errors="coerce"))
    if "baseline_value_from_baseline" in out.columns:
        out = out.drop(columns=["baseline_value_from_baseline"])
    out["absolute_delta"] = out["value"] - out["baseline_value"]
    # For loss metrics lower is better; for scores higher is better.
    lower_better = out["metric"].astype(str).str.lower().isin({"mae", "rmse"})
    out["relative_drop_pct"] = np.where(
        out["baseline_value"].abs() > 1e-12,
        np.where(lower_better, (out["value"] - out["baseline_value"]) / out["baseline_value"].abs() * 100.0, (out["baseline_value"] - out["value"]) / out["baseline_value"].abs() * 100.0),
        np.nan,
    )
    out["baseline_condition"] = baseline_condition
    out["impact_direction"] = np.where(out["relative_drop_pct"] > 0, "性能下降/误差上升", "性能未下降或误差改善")
    totals = out.groupby("task", dropna=False)["relative_drop_pct"].transform(lambda s: s.abs().sum())
    out["impact_share_pct"] = np.where(totals > 1e-12, out["relative_drop_pct"].abs() / totals * 100.0, np.nan)
    return out
