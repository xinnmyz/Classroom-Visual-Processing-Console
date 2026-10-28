"""Optional multi-person detection and stable ID tracking.

The application keeps the core image processing dependencies small.  This module
uses Ultralytics YOLO when it is installed, and an IoU tracker for stable IDs
between frames.  If ``YOLO.track`` is available, ByteTrack IDs are preferred;
the local IoU tracker is used as a deterministic fallback (and for images).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - app already reports missing OpenCV
    cv2 = None


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = "person"
    person_id: int | None = None
    track_quality: float = 0.0
    association_iou: float = 0.0
    is_predicted: bool = False
    track_age: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_row(self, frame_index: int, timestamp_seconds: float, frame_width: int, frame_height: int) -> dict[str, Any]:
        return {
            "frame_index": int(frame_index),
            "timestamp_seconds": round(float(timestamp_seconds), 4),
            "person_id": int(self.person_id) if self.person_id is not None else "",
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "area_fraction": round(self.area / max(1.0, frame_width * frame_height), 6),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "track_quality": round(float(self.track_quality), 4),
            "association_iou": round(float(self.association_iou), 4),
            "is_predicted": bool(self.is_predicted),
            "track_age": int(self.track_age),
        }


def box_iou(a: Detection | tuple[float, float, float, float], b: Detection | tuple[float, float, float, float]) -> float:
    """Intersection over union for two xyxy boxes."""
    aa = (a.x1, a.y1, a.x2, a.y2) if isinstance(a, Detection) else a
    bb = (b.x1, b.y1, b.x2, b.y2) if isinstance(b, Detection) else b
    x1 = max(aa[0], bb[0])
    y1 = max(aa[1], bb[1])
    x2 = min(aa[2], bb[2])
    y2 = min(aa[3], bb[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, aa[2] - aa[0]) * max(0.0, aa[3] - aa[1])
    area_b = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Dependency-free tracker with velocity prediction and gap recovery."""

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 15) -> None:
        self.iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
        self.max_age = max(1, int(max_age))
        self._tracks: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._reacquisition_count = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._reacquisition_count = 0

    @staticmethod
    def _center(det: Detection) -> tuple[float, float]:
        return ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)

    @classmethod
    def _predict(cls, track: dict[str, Any]) -> Detection:
        det: Detection = track["detection"]
        vx, vy = track.get("velocity", (0.0, 0.0))
        age = int(track.get("age", 0))
        dx, dy = vx * max(1, age + 1), vy * max(1, age + 1)
        return Detection(det.x1 + dx, det.y1 + dy, det.x2 + dx, det.y2 + dy, confidence=det.confidence, label=det.label, person_id=det.person_id)

    @classmethod
    def _center_distance_ratio(cls, a: Detection, b: Detection) -> float:
        ax, ay = cls._center(a)
        bx, by = cls._center(b)
        distance = math.hypot(ax - bx, ay - by)
        scale = max(1.0, math.hypot(max(a.width, b.width), max(a.height, b.height)))
        return distance / scale

    def _association(self, detection: Detection, track: dict[str, Any]) -> tuple[float, float]:
        predicted = self._predict(track)
        iou = box_iou(detection, predicted)
        distance_ratio = self._center_distance_ratio(detection, predicted)
        gate = min(3.0, 1.25 + 0.35 * int(track.get("age", 0)))
        if iou < self.iou_threshold and distance_ratio > gate:
            return -1.0, iou
        distance_score = max(0.0, 1.0 - distance_ratio / max(gate, 1e-6))
        return max(iou, 0.35 * distance_score), iou

    def update(self, detections: Iterable[Detection]) -> list[Detection]:
        detections = list(detections)
        candidates: list[tuple[float, int, int]] = []
        for det_index, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                score, iou = self._association(det, track)
                if score >= self.iou_threshold or (score >= 0.20 and iou > 0.0):
                    candidates.append((score, det_index, track_id))
        candidates.sort(reverse=True)
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        for score, det_index, track_id in candidates:
            if det_index in matched_dets or track_id in matched_tracks:
                continue
            matched_dets.add(det_index)
            matched_tracks.add(track_id)
            det = detections[det_index]
            track = self._tracks[track_id]
            previous = track["detection"]
            old_cx, old_cy = self._center(previous)
            new_cx, new_cy = self._center(det)
            measured_velocity = (new_cx - old_cx, new_cy - old_cy)
            old_velocity = track.get("velocity", (0.0, 0.0))
            velocity = (0.35 * float(old_velocity[0]) + 0.65 * measured_velocity[0], 0.35 * float(old_velocity[1]) + 0.65 * measured_velocity[1])
            det.person_id = track_id
            det.association_iou = box_iou(det, self._predict(track))
            det.track_quality = max(0.0, min(1.0, 0.65 * score + 0.35 * det.confidence))
            det.track_age = int(track.get("hits", 0)) + 1
            if int(track.get("age", 0)) > 0:
                self._reacquisition_count += 1
            self._tracks[track_id] = {**track, "detection": det, "velocity": velocity, "age": 0, "hits": int(track.get("hits", 0)) + 1, "total_missed": int(track.get("total_missed", 0)) + int(track.get("age", 0)), "last_iou": det.association_iou}

        for det_index, det in enumerate(detections):
            if det_index in matched_dets:
                continue
            det.person_id = self._next_id
            det.track_quality = max(0.0, min(1.0, 0.45 * det.confidence))
            det.track_age = 1
            self._tracks[self._next_id] = {"detection": det, "velocity": (0.0, 0.0), "age": 0, "hits": 1, "total_missed": 0, "max_gap": 0, "last_iou": 0.0}
            self._next_id += 1

        for track_id in list(self._tracks):
            if track_id not in matched_tracks and not any(det.person_id == track_id for det in detections):
                self._tracks[track_id]["age"] += 1
                self._tracks[track_id]["max_gap"] = max(int(self._tracks[track_id].get("max_gap", 0)), int(self._tracks[track_id]["age"]))
                if self._tracks[track_id]["age"] > self.max_age:
                    del self._tracks[track_id]
        return detections

    def quality_summary(self) -> dict[str, Any]:
        qualities = [float(track["detection"].track_quality) for track in self._tracks.values()]
        return {"active_track_count": len(self._tracks), "reacquisition_count": int(self._reacquisition_count), "mean_track_quality": float(np.mean(qualities)) if qualities else 0.0, "max_track_gap_frames": max((int(track.get("max_gap", 0)) for track in self._tracks.values()), default=0)}


def _parse_ultralytics_boxes(result: Any, person_class: int = 0) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    ids = getattr(boxes, "id", None)
    if xyxy is None:
        return []
    xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    conf = conf.cpu().numpy() if hasattr(conf, "cpu") else (np.asarray(conf) if conf is not None else np.ones(len(xyxy)))
    cls = cls.cpu().numpy() if hasattr(cls, "cpu") else (np.asarray(cls) if cls is not None else np.zeros(len(xyxy)))
    ids = ids.cpu().numpy() if hasattr(ids, "cpu") else (np.asarray(ids) if ids is not None else None)
    output: list[Detection] = []
    for index, row in enumerate(xyxy):
        if len(row) < 4 or int(cls[index]) != person_class:
            continue
        person_id = None
        if ids is not None and index < len(ids) and not math.isnan(float(ids[index])):
            person_id = int(ids[index])
        output.append(Detection(*map(float, row[:4]), confidence=float(conf[index]), person_id=person_id))
    return output


def load_yolo_model(model_path: str = "yolo11n.pt") -> tuple[Any, str]:
    """Load Ultralytics YOLO and return (model, backend description).

    The first run may download the small pretrained weights.  No model weights
    are bundled with the app, so this remains an explicit opt-in operation.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path or "yolo11n.pt")
        return model, "Ultralytics YOLO + ByteTrack（若可用）"
    except Exception:
        # Keep a no-download path for quick tests and offline machines.  HOG is
        # less accurate than YOLO in a crowded classroom, so the UI labels it
        # clearly and users can install the optional tracking dependency later.
        if cv2 is not None and hasattr(cv2, "HOGDescriptor") and hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return hog, "OpenCV HOG + IoU稳定ID（建议安装YOLO提高课堂场景召回率）"
        # OpenCV 5 builds may omit the legacy HOG API.  Returning a null
        # detector keeps the UI usable for configuration and report review;
        # install_tracking_dependencies.bat enables actual detections.
        return None, "未加载检测模型（IoU跟踪器待接收检测框；建议安装YOLO）"


def detect_frame(model: Any, frame_bgr: np.ndarray, confidence: float = 0.25) -> list[Detection]:
    """Detect persons in one BGR frame, falling back to empty detections."""
    if model is None:
        return []
    if cv2 is not None and isinstance(model, cv2.HOGDescriptor):
        rects, weights = model.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        output: list[Detection] = []
        for index, rect in enumerate(rects):
            x, y, w, h = map(float, rect)
            score = float(weights[index]) if weights is not None and index < len(weights) else 1.0
            if score >= float(confidence):
                output.append(Detection(x, y, x + w, y + h, confidence=score))
        return output
    results = model(frame_bgr, classes=[0], conf=float(confidence), verbose=False)
    return _parse_ultralytics_boxes(results[0]) if results else []


def track_frame_with_model(
    model: Any,
    frame_bgr: np.ndarray,
    tracker: IoUTracker,
    confidence: float = 0.25,
    use_bytetrack: bool = True,
) -> tuple[list[Detection], str]:
    """Run ByteTrack when available; otherwise use deterministic IoU IDs."""
    if model is None:
        return [], "未检测（未加载模型）"
    if use_bytetrack:
        try:
            results = model.track(
                frame_bgr,
                classes=[0],
                conf=float(confidence),
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            detections = _parse_ultralytics_boxes(results[0]) if results else []
            if detections and all(det.person_id is not None for det in detections):
                for det in detections:
                    det.track_quality = max(0.0, min(1.0, float(det.confidence)))
                    det.track_age = 1
                return detections, "Ultralytics YOLO + ByteTrack"
        except Exception:
            pass
    detections = tracker.update(detect_frame(model, frame_bgr, confidence))
    return detections, "Ultralytics YOLO + IoU稳定ID"


def draw_detections(frame_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    if cv2 is None:
        return frame_bgr
    output = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        text = f"person_id={det.person_id} {det.confidence:.2f}"
        cv2.putText(output, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 0), 2)
    return output


def _write_rows(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("frame_index,timestamp_seconds,person_id\n", encoding="utf-8-sig")
        return
    # A mixed image/video summary has different optional path fields.  Build a
    # union of columns so DictWriter never fails on a later row that contains
    # ``output_image`` instead of ``output_video`` (or vice versa).
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def process_video_tracking(
    video_path: str | Path,
    output_dir: str | Path,
    model: Any,
    confidence: float = 0.25,
    iou_threshold: float = 0.25,
    max_frames: int = 300,
    frame_stride: int = 1,
    annotate: bool = True,
    use_bytetrack: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    """Track people through a video and save annotated MP4 + CSV manifest."""
    if cv2 is None:
        raise RuntimeError("未安装 OpenCV，无法执行视频跟踪。")
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_out = output_dir / f"{video_path.stem}_tracked.mp4"
    writer = None
    if annotate:
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = IoUTracker(iou_threshold=iou_threshold)
    rows: list[dict[str, Any]] = []
    backend = ""
    processed = 0
    frame_index = 0
    stride = max(1, int(frame_stride))
    try:
        while processed < max(1, int(max_frames)):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue
            detections, backend = track_frame_with_model(model, frame, tracker, confidence, use_bytetrack)
            timestamp = frame_index / fps if fps else 0.0
            rows.extend(det.to_row(frame_index, timestamp, width, height) for det in detections)
            if writer is not None:
                writer.write(draw_detections(frame, detections))
            processed += 1
            frame_index += 1
            if progress_callback:
                progress_callback(min(1.0, processed / max(1, min(max_frames, math.ceil(frame_count / stride) or max_frames))))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    manifest_path = output_dir / f"{video_path.stem}_tracking.csv"
    _write_rows(rows, manifest_path)
    quality = tracker.quality_summary()
    qualities = [float(row.get("track_quality", 0.0)) for row in rows]
    return {
        "source_path": str(video_path),
        "output_video": str(video_out) if writer is not None else "",
        "manifest_csv": str(manifest_path),
        "backend": backend or "无检测结果",
        "frame_count_processed": processed,
        "unique_person_count": len({row["person_id"] for row in rows}),
        "detection_count": len(rows),
        "fps": fps,
        "width": width,
        "height": height,
        "reacquisition_count": quality["reacquisition_count"],
        "mean_track_quality": float(np.mean(qualities)) if qualities else quality["mean_track_quality"],
        "max_track_gap_frames": quality["max_track_gap_frames"],
    }


def process_image_tracking(
    image_path: str | Path,
    output_dir: str | Path,
    model: Any,
    confidence: float = 0.25,
    annotate: bool = True,
) -> dict[str, Any]:
    """Detect all people in one image and assign stable IDs for the image."""
    if cv2 is None:
        raise RuntimeError("未安装 OpenCV，无法执行图片检测。")
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"无法打开图片：{image_path}")
    tracker = IoUTracker(iou_threshold=0.0)
    detections = tracker.update(detect_frame(model, frame, confidence))
    annotated_path = output_dir / f"{image_path.stem}_tracked.jpg"
    if annotate:
        cv2.imwrite(str(annotated_path), draw_detections(frame, detections))
    rows = [det.to_row(0, 0.0, frame.shape[1], frame.shape[0]) for det in detections]
    manifest_path = output_dir / f"{image_path.stem}_tracking.csv"
    _write_rows(rows, manifest_path)
    return {
        "source_path": str(image_path),
        "output_image": str(annotated_path) if annotate else "",
        "manifest_csv": str(manifest_path),
        "backend": "Ultralytics YOLO + 单帧ID",
        "frame_count_processed": 1,
        "unique_person_count": len(detections),
        "detection_count": len(detections),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "mean_track_quality": float(np.mean([det.track_quality for det in detections])) if detections else 0.0,
        "reacquisition_count": 0,
        "max_track_gap_frames": 0,
    }


def write_tracking_summary(results: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tracking_summary.json"
    csv_path = output_dir / "tracking_summary.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows(results, csv_path)
    return json_path, csv_path

