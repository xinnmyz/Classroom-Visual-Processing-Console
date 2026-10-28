"""Detection and pose metrics computed from model prediction CSV files.

The main application intentionally does not assume a particular detector or
pose model.  This module provides a small, dependency-free evaluator for a
portable row format so mAP/AP/AR/PCK/OKS can be computed inside the report
page instead of being manually copied from a model's log.  It supports
one-row-per-prediction detection exports and one-row-per-person pose exports.
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def parse_box(value: Any) -> tuple[float, float, float, float] | None:
    """Parse ``x1,y1,x2,y2`` or JSON/list box representations."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                value = text.replace("|", ",").replace(";", ",").split(",")
    if isinstance(value, dict):
        values = [value.get(key) for key in ("x1", "y1", "x2", "y2")]
    else:
        try:
            values = list(value)
        except TypeError:
            return None
    if len(values) < 4:
        return None
    numbers = [_number(item) for item in values[:4]]
    return tuple(numbers) if all(math.isfinite(item) for item in numbers) else None  # type: ignore[return-value]


def box_from_row(row: pd.Series, prefix: str) -> tuple[float, float, float, float] | None:
    names = (
        (f"{prefix}_x1", f"{prefix}_y1", f"{prefix}_x2", f"{prefix}_y2"),
        (f"{prefix}x1", f"{prefix}y1", f"{prefix}x2", f"{prefix}y2"),
    )
    for fields in names:
        if all(field in row.index for field in fields):
            box = tuple(_number(row[field]) for field in fields)
            if all(math.isfinite(item) for item in box):
                return box  # type: ignore[return-value]
    for field in (f"{prefix}_box", f"{prefix}box", prefix):
        if field in row.index:
            box = parse_box(row[field])
            if box is not None:
                return box
    return None


def box_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _group_columns(frame: pd.DataFrame, group_cols: Iterable[str]) -> list[str]:
    return [column for column in group_cols if column in frame.columns]


def _average_precision(scores: list[tuple[float, int]], total_gt: int) -> float:
    if total_gt <= 0:
        return float("nan")
    scores = sorted(scores, key=lambda item: item[0], reverse=True)
    tp = fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _confidence, is_tp in scores:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / total_gt)
    # VOC-style all-points interpolation, sufficient for a single IoU level.
    area = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        area += precision * max(0.0, recall - previous_recall)
        previous_recall = recall
    return float(area)


def _compute_detection_at_iou(
    frame: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
    group_cols: Sequence[str] = ("task", "condition"),
) -> pd.DataFrame:
    """Compute precision/recall/mAP@IoU for detection rows.

    Recommended columns are ``sample_id``, ``condition``, ``class`` or
    ``label``, ``confidence``, ``gt_x1...gt_y2`` and ``pred_x1...pred_y2``.
    A row may contain an empty prediction (false negative) or empty ground
    truth (false positive).
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    groups = _group_columns(frame, group_cols)
    if not groups:
        frame = frame.copy(); frame["_group"] = "all"; groups = ["_group"]
    sample_col = _column(frame, "sample_id", "id")
    label_col = _column(frame, "class", "label", "category")
    confidence_col = _column(frame, "confidence", "score", "conf")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        detections: list[tuple[str, tuple[float, float, float, float] | None, tuple[float, float, float, float] | None, float]] = []
        gt_keys: set[tuple[str, str]] = set()
        for index, row in group.iterrows():
            sample = str(row[sample_col]) if sample_col else str(index)
            label = str(row[label_col]) if label_col else "person"
            gt_box = box_from_row(row, "gt")
            pred_box = box_from_row(row, "pred")
            if gt_box is not None:
                gt_keys.add((sample, label + "|" + "|".join(f"{v:.4f}" for v in gt_box)))
            confidence = _number(row[confidence_col], 1.0) if confidence_col else 1.0
            detections.append((sample + "|" + label, pred_box, gt_box, confidence))
        # Match predictions greedily within sample/class and prevent duplicate
        # predictions from claiming the same ground-truth object.
        matched: set[tuple[str, str]] = set()
        scored: list[tuple[float, int]] = []
        tp = fp = 0
        for identity, pred_box, gt_box, confidence in sorted(detections, key=lambda item: item[3], reverse=True):
            sample, label = identity.split("|", 1)
            candidates = []
            for gt_key in gt_keys:
                if gt_key in matched or gt_key[0] != sample:
                    continue
                # Recover the stored box from the current rows instead of
                # relying on fragile string parsing.
            best_key, best_iou = None, 0.0
            for idx, row in group.iterrows():
                row_sample = str(row[sample_col]) if sample_col else str(idx)
                row_label = str(row[label_col]) if label_col else "person"
                candidate = box_from_row(row, "gt")
                if row_sample != sample or row_label != label or candidate is None:
                    continue
                key = (sample, label + "|" + "|".join(f"{v:.4f}" for v in candidate))
                score = box_iou(pred_box, candidate)
                if key not in matched and score > best_iou:
                    best_key, best_iou = key, score
            if best_key is not None and best_iou >= float(iou_threshold):
                matched.add(best_key); tp += 1; scored.append((confidence, 1))
            elif pred_box is not None:
                fp += 1; scored.append((confidence, 0))
        fn = max(0, len(gt_keys) - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        ap = _average_precision(scored, len(gt_keys))
        rows.append({**dict(zip(groups, keys)), "metric": "Precision", "value": precision, "tp": tp, "fp": fp, "fn": fn, "sample_count": len(group), "source": "框匹配计算"})
        rows.append({**dict(zip(groups, keys)), "metric": "Recall", "value": recall, "tp": tp, "fp": fp, "fn": fn, "sample_count": len(group), "source": "框匹配计算"})
        rows.append({**dict(zip(groups, keys)), "metric": "mAP@0.5", "value": ap, "tp": tp, "fp": fp, "fn": fn, "sample_count": len(group), "source": "框匹配计算"})
    return pd.DataFrame(rows)


def bootstrap_confidence_interval(values: Sequence[float], samples: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Return a percentile bootstrap interval for metric values."""
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if int(samples) <= 0 or array.size == 1:
        value = float(np.mean(array))
        return value, value
    rng = np.random.default_rng(int(seed))
    means = np.asarray([rng.choice(array, size=array.size, replace=True).mean() for _ in range(int(samples))])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compute_detection_metrics(
    frame: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
    group_cols: Sequence[str] = ("task", "condition"),
    iou_thresholds: Sequence[float] | None = None,
    include_per_class: bool = True,
    bootstrap_samples: int = 0,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Compute Precision/Recall, mAP@0.5 and mAP@0.5:0.95.

    The original single-IoU output remains available. By default this wrapper
    also evaluates the 0.50..0.95 range and emits class-scoped rows when a
    class/label column exists.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    thresholds = [float(value) for value in (iou_thresholds or np.arange(0.50, 1.00, 0.05))]
    thresholds = list(dict.fromkeys(round(value, 2) for value in thresholds if 0.0 < value <= 1.0)) or [float(iou_threshold)]
    per_threshold = [_compute_detection_at_iou(frame, iou_threshold=threshold, group_cols=group_cols) for threshold in thresholds]
    per_threshold = [table for table in per_threshold if table is not None and not table.empty]
    if not per_threshold:
        return pd.DataFrame()
    base = per_threshold[thresholds.index(0.5)].copy() if 0.5 in thresholds else per_threshold[0].copy()
    if len(per_threshold) > 1:
        maps = pd.concat([table[table["metric"].astype(str).str.startswith("mAP@")] for table in per_threshold], ignore_index=True)
        if not maps.empty:
            keys = [column for column in group_cols if column in maps.columns]
            range_rows = maps.groupby(keys, dropna=False)["value"].mean().reset_index()
            range_rows["metric"] = "mAP@0.5:0.95"
            range_rows["sample_count"] = maps.groupby(keys, dropna=False)["sample_count"].max().to_numpy()
            range_rows["source"] = "多IoU阈值计算"
            base = pd.concat([base, range_rows], ignore_index=True, sort=False)
    class_col = _column(frame, "class", "label", "category")
    if include_per_class and class_col and class_col not in group_cols:
        class_tables = []
        for threshold in thresholds:
            table = _compute_detection_at_iou(frame, iou_threshold=threshold, group_cols=tuple(group_cols) + (class_col,))
            if not table.empty:
                if class_col != "class": table = table.rename(columns={class_col: "class"})
                class_tables.append(table)
        if class_tables:
            class_frame = pd.concat(class_tables, ignore_index=True)
            if len(class_tables) > 1:
                maps = class_frame[class_frame["metric"].astype(str).str.startswith("mAP@")] 
                if not maps.empty:
                    keys = [column for column in (*group_cols, "class") if column in maps.columns]
                    range_rows = maps.groupby(keys, dropna=False)["value"].mean().reset_index()
                    range_rows["metric"] = "mAP@0.5:0.95"
                    range_rows["sample_count"] = maps.groupby(keys, dropna=False)["sample_count"].max().to_numpy()
                    range_rows["source"] = "多IoU阈值/类别计算"
                    class_frame = pd.concat([class_frame, range_rows], ignore_index=True, sort=False)
            base = pd.concat([base, class_frame], ignore_index=True, sort=False)
    if bootstrap_samples > 0 and not base.empty:
        group_keys = [column for column in (*group_cols, "class", "metric") if column in base.columns]
        for keys, group in base.groupby(group_keys, dropna=False):
            if not isinstance(keys, tuple): keys = (keys,)
            low, high = bootstrap_confidence_interval(group["value"].tolist(), bootstrap_samples, random_seed)
            selector = np.ones(len(base), dtype=bool)
            for column, value in zip(group_keys, keys): selector &= base[column].astype(str).eq(str(value))
            base.loc[selector, "ci_low"] = low
            base.loc[selector, "ci_high"] = high
            base.loc[selector, "bootstrap_samples"] = int(bootstrap_samples)
    return base.reset_index(drop=True)


def parse_keypoints(value: Any) -> list[tuple[float, float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)): return []
    if isinstance(value, str):
        try: value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try: value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                value = [part.split(",") for part in value.replace(";", "|").split("|")]
    if isinstance(value, dict): value = value.get("keypoints", value.get("points", []))
    result = []
    try:
        for point in value:
            if isinstance(point, dict): x, y = point.get("x"), point.get("y")
            else: x, y = point[0], point[1]
            x, y = _number(x), _number(y)
            if math.isfinite(x) and math.isfinite(y): result.append((x, y))
    except (TypeError, IndexError):
        return []
    return result


def _pose_scale(row: pd.Series) -> float:
    area = _number(row.get("bbox_area"), float("nan"))
    if math.isfinite(area) and area > 0: return math.sqrt(area)
    width, height = _number(row.get("bbox_width")), _number(row.get("bbox_height"))
    if math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0: return math.sqrt(width * height)
    return 1.0


def _compute_pose_at_threshold(
    frame: pd.DataFrame,
    *,
    pck_threshold: float = 0.05,
    oks_threshold: float = 0.5,
    group_cols: Sequence[str] = ("task", "condition"),
) -> pd.DataFrame:
    """Compute PCK, OKS and one-person AP/AR from keypoint rows."""
    if frame is None or frame.empty: return pd.DataFrame()
    gt_col = _column(frame, "gt_keypoints", "ground_truth_keypoints", "keypoints_true")
    pred_col = _column(frame, "pred_keypoints", "keypoints_pred", "keypoints")
    if not gt_col or not pred_col: return pd.DataFrame()
    groups = _group_columns(frame, group_cols)
    if not groups: frame = frame.copy(); frame["_group"] = "all"; groups = ["_group"]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        pck_values, oks_values = [], []
        for _, row in group.iterrows():
            gt, pred = parse_keypoints(row[gt_col]), parse_keypoints(row[pred_col])
            count = min(len(gt), len(pred))
            if count == 0: continue
            scale = _pose_scale(row)
            distances = [math.hypot(gt[i][0] - pred[i][0], gt[i][1] - pred[i][1]) for i in range(count)]
            # If coordinates are normalized, use a unit scale.
            if max(max(abs(p[0]), abs(p[1])) for p in gt[:count]) <= 1.5: scale = max(scale, 1.0)
            pck_values.append(float(np.mean([distance <= pck_threshold * scale for distance in distances])))
            oks_values.append(float(np.mean([math.exp(-(distance * distance) / (2 * max(scale, 1e-6) ** 2 * 0.1 ** 2)) for distance in distances])))
        if not oks_values: continue
        ap = float(np.mean([value >= oks_threshold for value in oks_values]))
        rows.extend([
            {**dict(zip(groups, keys)), "metric": "PCK", "value": float(np.mean(pck_values)), "sample_count": len(pck_values), "source": "关键点距离计算"},
            {**dict(zip(groups, keys)), "metric": "OKS", "value": float(np.mean(oks_values)), "sample_count": len(oks_values), "source": "关键点距离计算"},
            {**dict(zip(groups, keys)), "metric": "AP", "value": ap, "sample_count": len(oks_values), "source": "OKS阈值计算"},
            {**dict(zip(groups, keys)), "metric": "AR", "value": ap, "sample_count": len(oks_values), "source": "OKS阈值计算"},
        ])
    return pd.DataFrame(rows)


def compute_pose_metrics(
    frame: pd.DataFrame,
    *,
    pck_threshold: float = 0.05,
    pck_thresholds: Sequence[float] | None = None,
    oks_threshold: float = 0.5,
    group_cols: Sequence[str] = ("task", "condition"),
    bootstrap_samples: int = 0,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Compute PCK@0.05/PCK@0.1, OKS and OKS-threshold AP/AR."""
    thresholds = list(pck_thresholds or (0.05, 0.1))
    tables = [_compute_pose_at_threshold(frame, pck_threshold=float(value), oks_threshold=oks_threshold, group_cols=group_cols) for value in thresholds]
    tables = [table for table in tables if table is not None and not table.empty]
    if not tables:
        return pd.DataFrame()
    base = tables[0].copy()
    if len(tables) > 1:
        pck_rows = []
        for threshold, table in zip(thresholds, tables):
            rows = table[table["metric"].eq("PCK")].copy()
            rows["metric"] = f"PCK@{float(threshold):g}"
            pck_rows.append(rows)
        base = pd.concat([base[~base["metric"].eq("PCK")], *pck_rows], ignore_index=True, sort=False)
    if bootstrap_samples > 0 and not base.empty:
        keys = [column for column in (*group_cols, "metric") if column in base.columns]
        for group_key, group in base.groupby(keys, dropna=False):
            if not isinstance(group_key, tuple): group_key = (group_key,)
            low, high = bootstrap_confidence_interval(group["value"].tolist(), bootstrap_samples, random_seed)
            selector = np.ones(len(base), dtype=bool)
            for column, value in zip(keys, group_key): selector &= base[column].astype(str).eq(str(value))
            base.loc[selector, "ci_low"] = low
            base.loc[selector, "ci_high"] = high
            base.loc[selector, "bootstrap_samples"] = int(bootstrap_samples)
    return base.reset_index(drop=True)


def compute_detection_pose_metrics(frame: pd.DataFrame, task_type: str = "auto", **kwargs: Any) -> pd.DataFrame:
    text = " ".join(str(value) for value in frame.get("task", pd.Series(dtype=str)).dropna()).lower()
    if task_type == "pose" or (task_type == "auto" and ("姿态" in text or "pose" in text or _column(frame, "gt_keypoints", "ground_truth_keypoints"))):
        return compute_pose_metrics(frame, **kwargs)
    return compute_detection_metrics(frame, **kwargs)


__all__ = ["box_iou", "parse_box", "compute_detection_metrics", "parse_keypoints", "compute_pose_metrics", "compute_detection_pose_metrics"]
