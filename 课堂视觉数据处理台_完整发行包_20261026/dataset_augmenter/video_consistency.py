"""Temporal augmentation helpers for video experiments.

The same noise mask and rectangle coordinates are reused for every frame in a
condition.  This avoids frame-to-frame flicker and makes duration statistics
meaningful.  The module is independent from Streamlit so it can be tested and
used by a batch script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TemporalAugmentationPlan:
    width: int
    height: int
    noise_percent: float
    occlusion_percent: float
    block_count: int
    position: str
    seed: int
    noise_mask: np.ndarray | None
    rectangles: tuple[tuple[int, int, int, int], ...]


def _rectangles(
    width: int,
    height: int,
    area_percent: float,
    block_count: int,
    position: str,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], ...]:
    if area_percent <= 0:
        return tuple()
    total_area = width * height * (area_percent / 100.0)
    side = max(1, int(max(1.0, total_area / max(1, block_count)) ** 0.5))
    result: list[tuple[int, int, int, int]] = []
    for _ in range(max(1, block_count)):
        block_w, block_h = min(width, side), min(height, side)
        x_max = max(1, width - block_w + 1)
        if position == "中心":
            x, y = (width - block_w) // 2, (height - block_h) // 2
        elif position == "上方":
            x = int(rng.integers(0, x_max))
            y = int(rng.integers(0, max(1, height // 2 - block_h + 1)))
        elif position == "下方":
            x = int(rng.integers(0, x_max))
            y_min = min(height - block_h, height // 2)
            y = int(rng.integers(y_min, max(y_min + 1, height - block_h + 1)))
        else:
            x = int(rng.integers(0, x_max))
            y = int(rng.integers(0, max(1, height - block_h + 1)))
        result.append((x, y, x + block_w, y + block_h))
    return tuple(result)


def make_temporal_plan(
    width: int,
    height: int,
    noise_percent: float,
    occlusion_percent: float,
    block_count: int = 1,
    position: str = "随机",
    seed: int = 0,
) -> TemporalAugmentationPlan:
    """Create one deterministic mask/rectangle layout for an entire video."""
    width, height = max(1, int(width)), max(1, int(height))
    rng = np.random.default_rng(int(seed))
    mask = None
    if noise_percent > 0:
        mask = rng.random((height, width)) < (float(noise_percent) / 100.0)
    rectangles = _rectangles(width, height, float(occlusion_percent), int(block_count), position, rng)
    return TemporalAugmentationPlan(
        width=width,
        height=height,
        noise_percent=float(noise_percent),
        occlusion_percent=float(occlusion_percent),
        block_count=int(block_count),
        position=str(position),
        seed=int(seed),
        noise_mask=mask,
        rectangles=rectangles,
    )


def apply_temporal_plan(image: Image.Image, plan: TemporalAugmentationPlan) -> Image.Image:
    """Apply a fixed plan to a frame; resize the mask only if dimensions differ."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb).copy()
    if plan.noise_mask is not None:
        mask = plan.noise_mask
        if mask.shape != array.shape[:2]:
            mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask = np.asarray(mask_image.resize((array.shape[1], array.shape[0]), Image.Resampling.NEAREST)) > 0
        array[mask] = 0
    output = Image.fromarray(array, mode="RGB")
    if plan.rectangles:
        draw = ImageDraw.Draw(output)
        sx = output.size[0] / max(1, plan.width)
        sy = output.size[1] / max(1, plan.height)
        for x1, y1, x2, y2 in plan.rectangles:
            draw.rectangle((round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)), fill=(0, 0, 0))
    return output


def action_duration_summary(
    rows: Iterable[dict[str, Any]],
    fps: float | None = None,
    person_key: str = "person_id",
    label_key: str = "label",
) -> list[dict[str, Any]]:
    """Summarize observed frame/time duration per person and action label."""
    import pandas as pd

    frame = pd.DataFrame(list(rows))
    if frame.empty or person_key not in frame.columns:
        return []
    if label_key not in frame.columns:
        frame[label_key] = "未标注"
    frame["timestamp_seconds"] = pd.to_numeric(frame.get("timestamp_seconds"), errors="coerce")
    frame["frame_index"] = pd.to_numeric(frame.get("frame_index"), errors="coerce")
    frame["person_id"] = frame[person_key].astype(str)
    frame["action_label"] = frame[label_key].fillna("未标注").astype(str)
    group_cols = ["person_id", "action_label"]
    rows_out: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        timestamps = group["timestamp_seconds"].dropna()
        if len(timestamps) >= 2:
            duration = float(timestamps.max() - timestamps.min())
        elif fps and fps > 0:
            duration = float(len(group) / fps)
        else:
            duration = float("nan")
        rows_out.append({
            "person_id": keys[0],
            "action_label": keys[1],
            "observed_frame_count": int(len(group)),
            "start_timestamp_seconds": float(timestamps.min()) if not timestamps.empty else float("nan"),
            "end_timestamp_seconds": float(timestamps.max()) if not timestamps.empty else float("nan"),
            "observed_duration_seconds": duration,
        })
    return rows_out


def write_video_from_frames(
    frame_paths: Iterable[str | Path],
    output_path: str | Path,
    fps: float,
) -> Path:
    """Encode ordered image frames into a video using OpenCV."""
    import cv2

    paths = [Path(p) for p in frame_paths]
    if not paths:
        raise ValueError("没有可合成视频的帧")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"无法读取帧：{paths[0]}")
    height, width = first.shape[:2]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), (width, height))
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return output

