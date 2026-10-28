from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import secrets
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from augmentations import (
    anonymize_faces,
    anonymize_persons,
    apply_keypoint_occlusion,
    apply_region_occlusion,
    simulate_foreground_occlusion,
)
from annotation_formats import (
    parse_coco_json,
    parse_keypoints_csv,
    parse_keypoints_json,
    parse_voc_xml,
    parse_yolo_txt,
    write_unified_annotations,
)
from checkpoint import CheckpointStore, condition_key, load_existing_records
from evaluator_runner import (
    evaluate_runner_output,
    run_external_model,
    write_runner_report,
)
from experiment_charts import write_experiment_charts
from label_mapping import annotate_asset, registry_rows
from metrics_detection_pose import compute_detection_pose_metrics
from model_evaluation import (
    compare_condition_impact,
    compute_evaluation_metrics,
    normalize_evaluation_csv,
)
from model_registry import ModelRegistry
from reproducibility import APP_VERSION, write_reproducibility_artifacts
from quality_ui import render_quality_panel
from tracking_ui import render_tracking_panel
from tracking_metrics import evaluate_tracking, write_tracking_metrics_report
from video_consistency import (
    apply_temporal_plan,
    make_temporal_plan,
    write_video_from_frames,
)
from reporting import (
    build_dataset_master,
    build_raw_file_list,
    build_sample_index,
    create_portable_package_from_run,
    stratified_sample_assets,
    add_excel_hyperlinks,
    write_report_package,
)

try:
    import cv2
except ImportError:  # 图片功能仍可在未安装 OpenCV 时使用
    cv2 = None


st.set_page_config(
    page_title="课堂视觉数据处理台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
SCB_REPOSITORY_URL = "https://github.com/Whiffe/SCB-dataset"
SCB_CITATION = "Yang, Fan. SCB-dataset: A dataset for detecting student classroom behavior. arXiv:2304.02488, 2023."
IITH_DAISEE_URL = "https://people.iith.ac.in/vineethnb/resources/daisee/"
ARCHIVE_EXTENSIONS = {".zip"}
RESOLUTION_OPTIONS = {
    "原图": 1.0,
    "1/2（模拟远距离）": 0.5,
    "1/4（模拟远距离）": 0.25,
    "1/8（模拟远距离）": 0.125,
}
# Percentages refer to the whole image area. Small levels are intentional:
# a global 10% black block can cover an entire far-away student.
NOISE_OPTIONS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
OCCLUSION_OPTIONS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 25.0]
OCCLUSION_RECOMMENDATIONS = {
    "多人远距离/低分辨率": {
        "noise": [0.0, 0.1, 0.25, 0.5, 1.0],
        "occlusion": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        "note": "建议从0.1%–2%开始；5%及以上作为压力测试。",
    },
    "普通距离": {
        "noise": [0.0, 0.25, 0.5, 1.0, 2.0],
        "occlusion": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        "note": "建议从0.5%–5%开始；10%作为高强度条件。",
    },
    "近距离/目标较大": {
        "noise": [0.0, 0.5, 1.0, 2.0, 5.0],
        "occlusion": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
        "note": "建议从1%–10%分级；15%及以上只做极端压力测试。",
    },
}
OCCLUSION_EVIDENCE = (
    "依据说明：Random Erasing论文将擦除面积作为可调范围，经典设置覆盖约2%–33%；"
    "KeepAugment指出增强不应无差别破坏关键信息。这里针对远距离多人课堂场景采用更小档位，"
    "属于本课题的实验起点，不等同于真实遮挡比例。"
)
RESTORE_OPTIONS = [
    "保留低分辨率输出",
    "恢复到原尺寸（双线性）",
    "恢复到原尺寸（双三次）",
    "恢复到原尺寸（Lanczos）",
]
FOCUS_OPTIONS = ["全图", "按标注聚焦一个人物", "手动ROI聚焦一个人物"]
STRUCTURED_OCCLUSION_OPTIONS = [
    "全局黑色像素/方块",
    "标注框或ROI区域遮挡",
    "前景学生合成遮挡",
    "关键点区域遮挡",
]
PRIVACY_OPTIONS = [
    "不做隐私处理",
    "人脸模糊",
    "人脸马赛克",
    "人脸黑色遮挡",
    "人体区域模糊",
    "人体区域马赛克",
    "人体区域黑色遮挡",
]


def infer_task(asset: dict[str, Any], override: str = "自动识别") -> str:
    """Infer a research task without pretending that raw pixels are predictions."""
    if override and override != "自动识别":
        return override
    mapped_task = str(asset.get("task_namespace", "")).strip()
    if mapped_task and mapped_task != "待指定":
        return mapped_task
    text = " ".join(
        str(asset.get(key, ""))
        for key in ("dataset_name", "origin_name", "source_group", "source_path", "member", "label", "annotation_label")
    ).lower()
    if "daisee" in text or "投入" in text or "engagement" in text or "boredom" in text:
        return "投入度/情感状态"
    if "微表情" in text or "micro" in text:
        return "微表情"
    if "scb" in text or "课堂" in text or "行为" in text or asset.get("annotation_member"):
        return "课堂行为"
    return "待指定"


def crop_focus_region(
    image: Image.Image,
    asset: dict[str, Any],
    focus_mode: str,
    object_index: int = 1,
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> tuple[Image.Image, dict[str, Any]]:
    """Crop one person before degradation; supports XML/VOC boxes and manual ROI."""
    width, height = image.size
    box: tuple[int, int, int, int] | None = None
    source = "全图"
    if focus_mode == "按标注聚焦一个人物":
        objects = asset.get("objects") or []
        if objects:
            index = max(1, min(int(object_index), len(objects))) - 1
            selected = objects[index]
            try:
                x1 = max(0, min(width - 1, int(selected.get("xmin", 0))))
                y1 = max(0, min(height - 1, int(selected.get("ymin", 0))))
                x2 = max(x1 + 1, min(width, int(selected.get("xmax", width))))
                y2 = max(y1 + 1, min(height, int(selected.get("ymax", height))))
                box = (x1, y1, x2, y2)
                source = f"标注目标#{index + 1}:{selected.get('label_name', selected.get('label_code', 'unknown'))}"
            except (TypeError, ValueError):
                box = None
    elif focus_mode == "手动ROI聚焦一个人物":
        x1, y1, x2, y2 = roi
        left = max(0, min(width - 1, round(width * min(x1, x2))))
        top = max(0, min(height - 1, round(height * min(y1, y2))))
        right = max(left + 1, min(width, round(width * max(x1, x2))))
        bottom = max(top + 1, min(height, round(height * max(y1, y2))))
        box = (left, top, right, bottom)
        source = "手动ROI"
    if box is None:
        return image.copy(), {"focus_applied": "否", "focus_source": "全图", "focus_box": ""}
    return image.crop(box), {"focus_applied": "是", "focus_source": source, "focus_box": "|".join(map(str, box))}


def _asset_object_boxes(asset: dict[str, Any], image_size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    """Return valid annotation boxes in the current image coordinate system."""
    width, height = image_size
    boxes: list[tuple[float, float, float, float]] = []
    for item in asset.get("objects", []) or []:
        try:
            x1, y1 = float(item.get("xmin")), float(item.get("ymin"))
            x2, y2 = float(item.get("xmax")), float(item.get("ymax"))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _parse_keypoints(text: str) -> list[tuple[float, float]]:
    """Parse normalized or pixel keypoints entered as ``x,y;x,y``."""
    points: list[tuple[float, float]] = []
    for raw in str(text or "").replace("；", ";").split(";"):
        parts = raw.strip().replace("，", ",").split(",")
        if len(parts) < 2:
            continue
        try:
            points.append((float(parts[0].strip()), float(parts[1].strip())))
        except ValueError:
            continue
    return points


def _detect_face_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Detect faces only for privacy masking; no face identity is inferred."""
    if cv2 is None:
        return []
    try:
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            return []
        faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(16, 16))
        return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces]
    except Exception:  # noqa: BLE001
        return []


def _apply_privacy_transform(
    image: Image.Image,
    asset: dict[str, Any],
    privacy_mode: str,
    focus_applied: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply an explicit privacy transform and report its detection source."""
    if not privacy_mode or privacy_mode == "不做隐私处理":
        return image.copy(), {"privacy_applied": "否", "privacy_detection_status": "未启用"}
    is_face = privacy_mode.startswith("人脸")
    method = "blur" if "模糊" in privacy_mode else "pixelate" if "马赛克" in privacy_mode else "black"
    if is_face:
        boxes = _detect_face_boxes(image)
        source = "Haar人脸检测"
        if not boxes and focus_applied:
            boxes = [(0, 0, image.width, image.height)]
            source = "聚焦区域回退（未检测到人脸）"
        output, metadata = anonymize_faces(image, boxes, method=method)
        metadata.update({"privacy_applied": "是" if boxes else "否", "privacy_detection_status": source if boxes else "未检测到人脸"})
        return output, metadata
    boxes = _asset_object_boxes(asset, image.size)
    source = "XML/VOC标注框"
    if not boxes and focus_applied:
        boxes = [(0, 0, image.width, image.height)]
        source = "聚焦区域"
    output, metadata = anonymize_persons(image, boxes, method=method)
    metadata.update({"privacy_applied": "是" if boxes else "否", "privacy_detection_status": source if boxes else "未找到人体框"})
    return output, metadata


def _apply_structured_occlusion(
    image: Image.Image,
    asset: dict[str, Any],
    mode: str,
    keypoints: list[tuple[float, float]],
    rng: np.random.Generator,
    focus_applied: bool = False,
    source_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply region-aware synthetic occlusion without labeling it as real."""
    if mode == "全局黑色像素/方块":
        return image.copy(), {"structured_occlusion": "全局", "occluder_is_real": "否"}
    if focus_applied:
        boxes = [(0, 0, image.width, image.height)]
    else:
        source_width, source_height = source_size or image.size
        scale_x = image.width / max(1, source_width)
        scale_y = image.height / max(1, source_height)
        boxes = [
            (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            for x1, y1, x2, y2 in _asset_object_boxes(asset, (source_width, source_height))
        ]
    if mode == "标注框或ROI区域遮挡":
        if not boxes:
            return image.copy(), {"structured_occlusion": "未应用", "occlusion_source": "没有标注框"}
        # Use the first annotated target as the region of interest.
        output, metadata = apply_region_occlusion(image, boxes[0], method="black")
        metadata.update({"structured_occlusion": "标注框区域", "occluder_is_real": "否", "occlusion_source": "XML/VOC标注框"})
        return output, metadata
    if mode == "前景学生合成遮挡":
        if focus_applied:
            return image.copy(), {"structured_occlusion": "未应用", "occlusion_source": "聚焦后缺少两个场景框", "occluder_is_real": "否"}
        if len(boxes) < 2:
            return image.copy(), {"structured_occlusion": "未应用", "occlusion_source": "至少需要两个标注框", "occluder_is_real": "否"}
        output, metadata = simulate_foreground_occlusion(image, boxes[0], boxes[1], foreground_image=image)
        metadata.update({"structured_occlusion": "前景学生合成", "occlusion_source": "两个人体标注框"})
        return output, metadata
    if mode == "关键点区域遮挡":
        output, metadata = apply_keypoint_occlusion(image, keypoints, radius=max(4.0, min(image.size) * 0.03), fraction=1.0, method="black", rng=rng)
        metadata.update({"structured_occlusion": "关键点区域", "occlusion_source": "手动关键点" if keypoints else "未提供关键点", "occluder_is_real": "否"})
        return output, metadata
    return image.copy(), {"structured_occlusion": "未应用", "occluder_is_real": "否"}


def _safe_text(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^\w\-.\u4e00-\u9fff]+", "_", value, flags=re.UNICODE)
    return value[:max_len].strip("_.") or "sample"


def _source_group(path: str, member: str | None, root: str) -> str:
    if member:
        return Path(path).stem
    p = Path(path)
    try:
        relative = p.relative_to(Path(root))
        return relative.parts[0] if len(relative.parts) > 1 else p.parent.name
    except ValueError:
        return p.parent.name


def _infer_label(path: str, member: str | None, root: str) -> str:
    candidate = Path(member) if member else Path(path)
    parent = candidate.parent.name
    if parent and parent not in {".", ""}:
        return parent
    return Path(path).stem if member else _source_group(path, member, root)


def _annotation_candidate(member: str, members: set[str], xml_by_stem: dict[str, str]) -> str:
    """Find a matching Pascal VOC-style XML member for an image member."""
    image_path = Path(member)
    candidates = []
    text = member.replace("\\", "/")
    for src, dst in (("/images/", "/labels/"), ("/JPEGImages/", "/Annotations/"), ("/images/", "/annotations/")):
        if src in text:
            candidates.append(text.replace(src, dst).rsplit(".", 1)[0] + ".xml")
    candidates.append(str(image_path.with_suffix(".xml")).replace("\\", "/"))
    stem = image_path.stem.lower()
    for candidate in candidates:
        if candidate in members:
            return candidate
    return xml_by_stem.get(stem, "")


def _read_xml_annotation(xml_bytes: bytes) -> tuple[str, int, int | None, int | None, list[dict[str, Any]]]:
    try:
        root = ET.fromstring(xml_bytes)
        objects: list[dict[str, Any]] = []
        labels: list[str] = []
        for index, node in enumerate(root.findall(".//object"), start=1):
            name_node = node.find("name")
            label_code = name_node.text.strip() if name_node is not None and name_node.text else "unknown"
            labels.append(label_code)
            box = node.find("bndbox")

            def integer_value(field: str) -> int | None:
                if box is None:
                    return None
                value = box.find(field)
                try:
                    return int(float(value.text)) if value is not None and value.text else None
                except (TypeError, ValueError):
                    return None

            truncated_node = node.find("truncated")
            difficult_node = node.find("difficult")
            objects.append(
                {
                    "object_id": index,
                    "label_code": label_code,
                    "xmin": integer_value("xmin"),
                    "ymin": integer_value("ymin"),
                    "xmax": integer_value("xmax"),
                    "ymax": integer_value("ymax"),
                    "truncated": truncated_node.text.strip() if truncated_node is not None and truncated_node.text else "",
                    "difficult": difficult_node.text.strip() if difficult_node is not None and difficult_node.text else "",
                }
            )
        width_node = root.find(".//size/width")
        height_node = root.find(".//size/height")
        width = int(width_node.text) if width_node is not None and width_node.text else None
        height = int(height_node.text) if height_node is not None and height_node.text else None
        unique_labels = list(dict.fromkeys(labels))
        return "|".join(unique_labels), len(objects), width, height, objects
    except (ET.ParseError, ValueError, TypeError):
        return "", 0, None, None, []


def _read_class_map(archive: zipfile.ZipFile, members: set[str]) -> dict[str, str]:
    class_members = [name for name in members if Path(name).name.lower() == "classes.txt"]
    for member in class_members:
        try:
            text = archive.read(member).decode("utf-8", errors="replace")
            mapping: dict[str, str] = {}
            for line in text.splitlines():
                match = re.match(r"\s*\d+\s+#\s*(.*?)\s+([A-Za-z0-9_]+)\s+\d+\s*$", line)
                if match:
                    mapping[match.group(2)] = match.group(1).strip()
            if mapping:
                return mapping
        except (OSError, UnicodeError):
            continue
    return {}


def _display_labels(raw_labels: str, class_map: dict[str, str]) -> str:
    if not raw_labels:
        return ""
    values = []
    for code in raw_labels.split("|"):
        name = class_map.get(code)
        values.append(f"{name}（{code}）" if name else code)
    return "|".join(values)


def _label_names(raw_labels: str, class_map: dict[str, str]) -> list[str]:
    if not raw_labels:
        return []
    return list(dict.fromkeys(class_map.get(code, code) for code in raw_labels.split("|")))


def parse_source_specs(config_text: str) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|||")]
        root_path = parts[0]
        if not root_path:
            continue
        specs.append(
            {
                "root_path": root_path,
                "dataset_name": parts[1] if len(parts) > 1 and parts[1] else Path(root_path).name,
                "origin_name": parts[2] if len(parts) > 2 and parts[2] else "未登记",
                "origin_url": parts[3] if len(parts) > 3 else "",
            }
        )
    return specs


def apply_known_source_metadata(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Register the user-provided provenance for the current classroom archives."""
    for asset in assets:
        dataset_name = str(asset.get("dataset_name", ""))
        origin_name = str(asset.get("origin_name", ""))
        path_text = f"{asset.get('source_path', '')} {asset.get('member', '')}".lower()
        default_classroom = dataset_name in {"课堂行为数据集", "未登记", ""} or origin_name in {"待确认", "未登记", ""}
        if default_classroom and "vocdevkit" in path_text:
            asset.update(
                {
                    "dataset_name": "VOCdevKit",
                    "origin_name": "Indian Institute of Technology Hyderabad（IITH）官方数据集页面",
                    "origin_url": IITH_DAISEE_URL,
                    "citation": "来源按项目组提供的信息登记；具体数据集论文待核对。",
                    "license_or_access": "以 IITH 官方页面的访问和研究使用条款为准",
                }
            )
        elif default_classroom:
            asset.update(
                {
                    "dataset_name": "SCB-dataset",
                    "origin_name": "GitHub/Whiffe/SCB-dataset",
                    "origin_url": SCB_REPOSITORY_URL,
                    "citation": SCB_CITATION,
                    "license_or_access": "以 GitHub 仓库 README 和原始数据条款为准",
                }
            )
        else:
            asset.setdefault("citation", "待补充")
            asset.setdefault("license_or_access", "待核对")
    return assets


@st.cache_data(show_spinner=False)
def scan_assets(config_text: str, include_archives: bool, include_videos: bool = False) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for spec in parse_source_specs(config_text):
        root = Path(spec["root_path"]).expanduser()
        if not root.exists() or not root.is_dir():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                folder_label = _infer_label(str(path), None, str(root))
                assets.append(
                    {
                        "asset_id": hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12],
                        "media_type": "image",
                        "source_kind": "文件夹",
                        "source_path": str(path),
                        "root_path": spec["root_path"],
                        "dataset_name": spec["dataset_name"],
                        "origin_name": spec["origin_name"],
                        "origin_url": spec["origin_url"],
                        "member": "",
                        "file_name": path.name,
                        "source_group": _source_group(str(path), None, str(root)),
                        "sampling_group": _source_group(str(path), None, str(root)),
                        "label": folder_label,
                        "label_names": [folder_label],
                        "label_codes": [folder_label],
                        "annotation_label": "",
                        "annotation_label_raw": "",
                        "bbox_count": 0,
                        "annotation_member": "",
                        "objects": [],
                        "size_bytes": path.stat().st_size,
                    }
                )
            elif include_videos and suffix in VIDEO_EXTENSIONS:
                folder_label = _infer_label(str(path), None, str(root))
                assets.append(
                    {
                        "asset_id": hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12],
                        "media_type": "video",
                        "source_kind": "视频文件",
                        "source_path": str(path),
                        "root_path": spec["root_path"],
                        "dataset_name": spec["dataset_name"],
                        "origin_name": spec["origin_name"],
                        "origin_url": spec["origin_url"],
                        "member": "",
                        "file_name": path.name,
                        "source_group": _source_group(str(path), None, str(root)),
                        "sampling_group": _source_group(str(path), None, str(root)),
                        "label": folder_label,
                        "label_names": [folder_label],
                        "label_codes": [folder_label],
                        "annotation_label": "",
                        "annotation_label_raw": "",
                        "bbox_count": 0,
                        "annotation_member": "",
                        "objects": [],
                        "size_bytes": path.stat().st_size,
                    }
                )
            elif include_archives and suffix in ARCHIVE_EXTENSIONS:
                try:
                    with zipfile.ZipFile(path) as archive:
                        member_names = set(archive.namelist())
                        class_map = _read_class_map(archive, member_names)
                        xml_by_stem = {
                            Path(name).stem.lower(): name
                            for name in member_names
                            if name.lower().endswith(".xml")
                        }
                        for info in archive.infolist():
                            if info.is_dir():
                                continue
                            member_suffix = Path(info.filename).suffix.lower()
                            if member_suffix in IMAGE_EXTENSIONS:
                                annotation_member = _annotation_candidate(info.filename, member_names, xml_by_stem)
                                annotation_label = ""
                                raw_annotation_label = ""
                                bbox_count = 0
                                xml_width = None
                                xml_height = None
                                objects: list[dict[str, Any]] = []
                                if annotation_member:
                                    raw_annotation_label, bbox_count, xml_width, xml_height, objects = _read_xml_annotation(
                                        archive.read(annotation_member)
                                    )
                                    annotation_label = _display_labels(raw_annotation_label, class_map)
                                label_names = _label_names(raw_annotation_label, class_map)
                                label_codes = list(dict.fromkeys(raw_annotation_label.split("|"))) if raw_annotation_label else []
                                for item in objects:
                                    item["label_name"] = class_map.get(item["label_code"], item["label_code"])
                                folder_label = _infer_label(str(path), info.filename, str(root))
                                member_id = f"{path}|{info.filename}"
                                assets.append(
                                    {
                                        "asset_id": hashlib.sha1(member_id.encode("utf-8")).hexdigest()[:12],
                                        "media_type": "image",
                                        "source_kind": "ZIP 压缩包",
                                        "source_path": str(path),
                                        "root_path": spec["root_path"],
                                        "dataset_name": spec["dataset_name"],
                                        "origin_name": spec["origin_name"],
                                        "origin_url": spec["origin_url"],
                                        "member": info.filename,
                                        "file_name": Path(info.filename).name,
                                        "source_group": _source_group(str(path), info.filename, str(root)),
                                        "sampling_group": _source_group(str(path), info.filename, str(root)),
                                        "label": annotation_label or folder_label,
                                        "label_names": label_names or [folder_label],
                                        "label_codes": label_codes or [folder_label],
                                        "annotation_label": annotation_label,
                                        "annotation_label_raw": raw_annotation_label,
                                        "bbox_count": bbox_count,
                                        "annotation_member": annotation_member,
                                        "annotation_width": xml_width,
                                        "annotation_height": xml_height,
                                        "objects": objects,
                                        "size_bytes": info.file_size,
                                    }
                                )
                            elif include_videos and member_suffix in VIDEO_EXTENSIONS:
                                member_id = f"{path}|{info.filename}"
                                video_label = Path(info.filename).parent.name or Path(path).stem
                                assets.append(
                                    {
                                        "asset_id": hashlib.sha1(member_id.encode("utf-8")).hexdigest()[:12],
                                        "media_type": "video",
                                        "source_kind": "ZIP 视频",
                                        "source_path": str(path),
                                        "root_path": spec["root_path"],
                                        "dataset_name": spec["dataset_name"],
                                        "origin_name": spec["origin_name"],
                                        "origin_url": spec["origin_url"],
                                        "member": info.filename,
                                        "file_name": Path(info.filename).name,
                                        "source_group": _source_group(str(path), info.filename, str(root)),
                                        "sampling_group": _source_group(str(path), info.filename, str(root)),
                                        "label": video_label,
                                        "label_names": [video_label],
                                        "label_codes": [video_label],
                                        "annotation_label": "",
                                        "annotation_label_raw": "",
                                        "bbox_count": 0,
                                        "annotation_member": "",
                                        "objects": [],
                                        "size_bytes": info.file_size,
                                    }
                                )
                except (OSError, zipfile.BadZipFile):
                    continue
    assets = apply_known_source_metadata(assets)
    # Keep dataset-specific task/label namespaces attached to every asset.
    # Similar labels from DAiSEE, SCB and Tianchi must not be silently merged.
    return [annotate_asset(asset) for asset in assets]


def load_asset(asset: dict[str, Any]) -> Image.Image:
    if asset["member"]:
        with zipfile.ZipFile(asset["source_path"]) as archive:
            data = archive.read(asset["member"])
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(asset["source_path"]).convert("RGB")


@contextmanager
def materialize_video(asset: dict[str, Any]):
    """Yield a local video path; ZIP members are extracted only for the current operation."""
    if not asset.get("member"):
        yield asset["source_path"]
        return
    temporary_path = None
    try:
        suffix = Path(asset["member"]).suffix or ".mp4"
        with zipfile.ZipFile(asset["source_path"]) as archive:
            data = archive.read(asset["member"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(data)
            temporary_path = temporary.name
        yield temporary_path
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def extract_video_frames(
    asset: dict[str, Any],
    interval_seconds: float,
    max_frames: int,
    seed: int,
) -> tuple[list[tuple[int, float, Image.Image]], dict[str, Any]]:
    if cv2 is None:
        raise RuntimeError("未安装 OpenCV，无法抽取视频帧。请运行 install_dependencies.bat。")
    with materialize_video(asset) as video_path:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError("无法打开视频文件")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = fps if fps > 0 else 25.0
        step = max(1, int(round(fps * max(0.1, interval_seconds))))
        candidate_indices = list(range(0, max(1, frame_count), step))
        if len(candidate_indices) > max_frames:
            rng = random.Random(seed)
            candidate_indices = sorted(rng.sample(candidate_indices, max_frames))

        frames: list[tuple[int, float, Image.Image]] = []
        for frame_index in candidate_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_index, frame_index / fps, Image.fromarray(rgb)))
        capture.release()
        metadata = {
            "video_fps": fps,
            "video_frame_count": frame_count,
            "video_width": width,
            "video_height": height,
            "video_duration_seconds": frame_count / fps if fps else 0.0,
            "frame_interval_seconds": interval_seconds,
            "max_frames_per_video": max_frames,
        }
        return frames, metadata


def apply_low_resolution(
    image: Image.Image,
    scale: float,
    restore_mode: str = "恢复到原尺寸（双线性）",
) -> tuple[Image.Image, dict[str, Any]]:
    """Create a low-resolution view and optionally restore it for comparison.

    The returned metadata separates the simulated sensor resolution from the
    final output resolution, so a restored image is not mistaken for an
    original high-resolution observation.
    """
    width, height = image.size
    if scale >= 0.999:
        return image.copy(), {
            "low_resolution_width": width,
            "low_resolution_height": height,
            "resolution_ratio": 1.0,
            "restored_to_original": "否",
        }
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = image.resize(small_size, Image.Resampling.BOX)
    if restore_mode == "保留低分辨率输出":
        return small, {
            "low_resolution_width": small_size[0],
            "low_resolution_height": small_size[1],
            "resolution_ratio": scale,
            "restored_to_original": "否",
        }
    interpolation = {
        "恢复到原尺寸（双线性）": Image.Resampling.BILINEAR,
        "恢复到原尺寸（双三次）": Image.Resampling.BICUBIC,
        "恢复到原尺寸（Lanczos）": Image.Resampling.LANCZOS,
    }.get(restore_mode, Image.Resampling.BILINEAR)
    restored = small.resize((width, height), interpolation)
    return restored, {
        "low_resolution_width": small_size[0],
        "low_resolution_height": small_size[1],
        "resolution_ratio": scale,
        "restored_to_original": "是",
    }


def add_black_pixels(
    image: Image.Image,
    percent: float,
    rng: np.random.Generator,
    base_array: np.ndarray | None = None,
) -> Image.Image:
    """Apply sparse black-pixel noise, reusing an RGB array when available."""
    if percent <= 0:
        return image.copy()
    # A run may apply several occlusion levels to the same resolution/noise
    # version.  Avoid converting the same PIL image to NumPy for each level.
    array = (base_array if base_array is not None else np.asarray(image.convert("RGB"))).copy()
    height, width = array.shape[:2]
    mask = rng.random((height, width)) < (percent / 100.0)
    array[mask] = 0
    return Image.fromarray(array, mode="RGB")


def add_block_occlusion(
    image: Image.Image,
    area_percent: float,
    block_count: int,
    position: str,
    rng: np.random.Generator,
) -> Image.Image:
    if area_percent <= 0:
        return image.copy()
    output = image.copy()
    width, height = output.size
    draw = ImageDraw.Draw(output)
    total_area = width * height * (area_percent / 100.0)
    each_area = max(1.0, total_area / max(1, block_count))
    side = max(1, int(each_area**0.5))

    for index in range(max(1, block_count)):
        block_w = min(width, side)
        block_h = min(height, side)
        if position == "中心":
            x = (width - block_w) // 2
            y = (height - block_h) // 2
        elif position == "上方":
            x = int(rng.integers(0, max(1, width - block_w + 1)))
            y = int(rng.integers(0, max(1, height // 2 - block_h + 1)))
        elif position == "下方":
            x = int(rng.integers(0, max(1, width - block_w + 1)))
            y_min = min(height - block_h, height // 2)
            y = int(rng.integers(y_min, max(y_min + 1, height - block_h + 1)))
        else:
            x = int(rng.integers(0, max(1, width - block_w + 1)))
            y = int(rng.integers(0, max(1, height - block_h + 1)))
        draw.rectangle((x, y, x + block_w, y + block_h), fill=(0, 0, 0))
    return output


def read_image_size(asset: dict[str, Any]) -> tuple[int, int]:
    image = load_asset(asset)
    return image.size


def build_condition_name(scale_name: str, noise: float, occlusion: float) -> str:
    resolution = {"原图": "clean", "1/2（模拟远距离）": "lowres_2", "1/4（模拟远距离）": "lowres_4", "1/8（模拟远距离）": "lowres_8"}[scale_name]
    noise_text = f"noise_{noise:g}pct" if noise > 0 else "noise_0pct"
    occ_text = f"occ_{occlusion:g}pct" if occlusion > 0 else "occ_0pct"
    return f"{resolution}_{noise_text}_{occ_text}"


def build_impact_summary(
    records: list[dict[str, Any]],
    prediction_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize processing-condition shares and, when supplied, model impact.

    Image degradation alone cannot produce behavior, micro-expression, or
    engagement accuracy. If a prediction CSV is supplied it must contain
    ``sample_id``, ``baseline_score`` and ``condition_score``. The optional
    ``task`` and ``metric`` columns are carried into the report.
    """
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([{k: v for k, v in item.items() if k != "objects"} for item in records])
    if frame.empty:
        return frame
    def col(name: str, default: Any) -> pd.Series:
        if name in frame.columns:
            return frame[name]
        return pd.Series([default] * len(frame), index=frame.index)

    frame["task"] = frame.apply(lambda row: infer_task(row.to_dict()), axis=1)
    frame["media_type"] = col("media_type", "image").fillna("image")
    frame["camera_view"] = col("sampling_group", "").fillna("")
    frame["condition"] = col("condition", "未知条件")
    frame["resolution_setting"] = col("resolution_setting", "未处理")
    frame["resolution_ratio"] = pd.to_numeric(col("resolution_ratio", 1.0), errors="coerce").fillna(1.0)
    frame["black_pixel_percent"] = pd.to_numeric(col("black_pixel_percent", 0.0), errors="coerce").fillna(0.0)
    frame["occlusion_area_percent"] = pd.to_numeric(col("occlusion_area_percent", 0.0), errors="coerce").fillna(0.0)
    frame["label"] = col("label_from_annotation_or_folder", "未标注").fillna("未标注")
    frame["distance_proxy"] = np.where(
        frame["resolution_ratio"] >= 0.999,
        "原始距离代理",
        "远距离模拟",
    )
    group_cols = ["task", "media_type", "label", "camera_view", "distance_proxy", "condition", "resolution_setting"]
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            sample_count=("sample_id", "count"),
            mean_resolution_ratio=("resolution_ratio", "mean"),
            mean_black_pixel_percent=("black_pixel_percent", "mean"),
            mean_occlusion_area_percent=("occlusion_area_percent", "mean"),
        )
        .reset_index()
    )
    total = max(1, int(summary["sample_count"].sum()))
    summary["output_share_pct"] = summary["sample_count"] / total * 100.0
    summary["analysis_status"] = "仅统计处理条件占比；未提供模型预测结果"

    if prediction_df is not None and not prediction_df.empty:
        required = {"sample_id", "baseline_score", "condition_score"}
        if required.issubset(prediction_df.columns):
            pred = prediction_df.copy()
            pred["baseline_score"] = pd.to_numeric(pred["baseline_score"], errors="coerce")
            pred["condition_score"] = pd.to_numeric(pred["condition_score"], errors="coerce")
            pred = pred.dropna(subset=["sample_id", "baseline_score", "condition_score"])
            joined = frame.merge(pred, on="sample_id", how="inner", suffixes=("", "_prediction"))
            if not joined.empty:
                joined["score_delta"] = joined["condition_score"] - joined["baseline_score"]
                joined["relative_drop_pct"] = np.where(
                    joined["baseline_score"].abs() > 1e-12,
                    (joined["baseline_score"] - joined["condition_score"]) / joined["baseline_score"].abs() * 100.0,
                    np.nan,
                )
                metric_group = (
                    joined.groupby(group_cols, dropna=False)
                    .agg(
                        prediction_count=("sample_id", "count"),
                        baseline_score=("baseline_score", "mean"),
                        condition_score=("condition_score", "mean"),
                        score_delta=("score_delta", "mean"),
                        relative_drop_pct=("relative_drop_pct", "mean"),
                    )
                    .reset_index()
                )
                abs_total = max(1e-12, float(metric_group["relative_drop_pct"].abs().sum()))
                metric_group["impact_share_pct"] = metric_group["relative_drop_pct"].abs() / abs_total * 100.0
                summary = summary.merge(metric_group, on=group_cols, how="left")
                summary["analysis_status"] = np.where(
                    summary["prediction_count"].fillna(0) > 0,
                    "已根据模型预测结果计算",
                    summary["analysis_status"],
                )
            else:
                summary["prediction_note"] = "预测CSV与处理样本的sample_id没有交集"
        else:
            summary["prediction_note"] = "预测CSV缺少sample_id/baseline_score/condition_score"
    return summary


def update_run_model_metadata(
    run_dir: Path,
    prediction_df: pd.DataFrame,
    records: list[dict[str, Any]],
    selected_assets: list[dict[str, Any]],
) -> None:
    """Refresh reproducibility artifacts with external model metadata."""
    metadata: dict[str, Any] = {"status": "已导入外部模型评估结果"}
    for column in ("model_name", "model_version", "split", "metric_direction"):
        if column in prediction_df.columns:
            values = [str(value) for value in prediction_df[column].dropna().unique().tolist() if str(value).strip()]
            if values:
                metadata[column] = values if len(values) > 1 else values[0]
    settings: dict[str, Any] = {}
    config_path = run_dir / "experiment_config.json"
    if config_path.exists():
        try:
            settings = json.loads(config_path.read_text(encoding="utf-8")).get("settings", {})
        except (OSError, json.JSONDecodeError):
            settings = {}
    write_reproducibility_artifacts(run_dir, settings, records, selected_assets, metadata)


def create_excel(
    records: list[dict[str, Any]],
    settings: dict[str, Any],
    output_path: Path,
    prediction_df: pd.DataFrame | None = None,
) -> None:
    object_rows: list[dict[str, Any]] = []
    for record in records:
        for obj in record.get("objects", []) or []:
            object_rows.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "file_name": record.get("file_name", ""),
                    "dataset_name": record.get("dataset_name", ""),
                    "origin_name": record.get("origin_name", ""),
                    "origin_url": record.get("origin_url", ""),
                    "source_group": record.get("source_group", ""),
                    "sampling_group": record.get("sampling_group", ""),
                    "condition": record.get("condition", ""),
                    "output_path": record.get("output_path", ""),
                    **obj,
                }
            )
    manifest = pd.DataFrame(records).drop(columns=["objects"], errors="ignore")
    source_summary = (
        manifest.groupby(
            ["dataset_name", "origin_name", "origin_url", "sampling_group", "source_group", "label_from_annotation_or_folder"],
            dropna=False,
        )
        .size()
        .reset_index(name="output_count")
        if not manifest.empty
        else pd.DataFrame()
    )
    settings_df = pd.DataFrame([settings])
    impact_summary = build_impact_summary(records, prediction_df)
    image_manifest = manifest[manifest.get("media_type", pd.Series(dtype=str)) == "image"] if not manifest.empty else manifest
    video_manifest = manifest[manifest.get("media_type", pd.Series(dtype=str)) == "video"] if not manifest.empty else manifest
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        manifest.to_excel(writer, sheet_name="sample_manifest", index=False)
        image_manifest.to_excel(writer, sheet_name="image_manifest", index=False)
        video_manifest.to_excel(writer, sheet_name="video_manifest", index=False)
        pd.DataFrame(object_rows).to_excel(writer, sheet_name="object_manifest", index=False)
        source_summary.to_excel(writer, sheet_name="source_summary", index=False)
        impact_summary.to_excel(writer, sheet_name="impact_summary", index=False)
        pd.DataFrame(registry_rows()).to_excel(writer, sheet_name="label_registry", index=False)
        settings_df.to_excel(writer, sheet_name="settings", index=False)
    add_excel_hyperlinks(output_path)
    manifest.to_csv(output_path.with_name("sample_manifest.csv"), index=False, encoding="utf-8-sig")
    image_manifest.to_csv(output_path.with_name("image_manifest.csv"), index=False, encoding="utf-8-sig")
    video_manifest.to_csv(output_path.with_name("video_manifest.csv"), index=False, encoding="utf-8-sig")
    impact_summary.to_csv(output_path.with_name("impact_summary.csv"), index=False, encoding="utf-8-sig")


def run_processing(
    selected_assets: list[dict[str, Any]],
    output_root: str,
    resolution_names: list[str],
    noise_levels: list[float],
    occlusion_levels: list[float],
    block_count: int,
    occlusion_position: str,
    seed: int,
    output_format: str,
    progress_callback=None,
    video_interval_seconds: float = 2.0,
    video_max_frames: int = 20,
    seed_mode: str = "固定种子（可复现）",
    restore_mode: str = "恢复到原尺寸（双线性）",
    focus_mode: str = "全图",
    focus_object_index: int = 1,
    focus_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    task_override: str = "自动识别",
    distance_profile: str = "多人远距离/低分辨率",
    structured_occlusion_mode: str = "全局黑色像素/方块",
    privacy_mode: str = "不做隐私处理",
    keypoints_text: str = "",
    temporal_consistency: bool = False,
    synthesize_processed_videos: bool = False,
    resume_run_dir: str | None = None,
) -> tuple[Path, list[dict[str, Any]], list[tuple[dict[str, Any], Image.Image, Image.Image]]]:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(resume_run_dir).expanduser() if resume_run_dir else Path(output_root).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    previews: list[tuple[dict[str, Any], Image.Image, Image.Image]] = []
    total = max(
        1,
        len(selected_assets)
        * max(1, video_max_frames)
        * len(resolution_names)
        * len(noise_levels)
        * len(occlusion_levels),
    )
    completed = 0
    # Frequent Streamlit progress updates can become visible overhead for
    # large batches.  Keep at most about 100 updates per run.
    progress_step = max(1, total // 100)
    keypoints = _parse_keypoints(keypoints_text)
    video_output_rows: list[dict[str, Any]] = []
    checkpoint_settings = {
        "app_version": APP_VERSION,
        "resolution_names": resolution_names,
        "noise_levels": noise_levels,
        "occlusion_levels": occlusion_levels,
        "block_count": block_count,
        "occlusion_position": occlusion_position,
        "seed": seed,
        "output_format": output_format,
        "restore_mode": restore_mode,
        "focus_mode": focus_mode,
        "structured_occlusion_mode": structured_occlusion_mode,
        "privacy_mode": privacy_mode,
        "temporal_consistency": temporal_consistency,
    }
    checkpoint = CheckpointStore(run_dir / "checkpoint.json", settings=checkpoint_settings, resume=bool(resume_run_dir))
    checkpoint.save()
    existing_records = load_existing_records(run_dir)
    if not existing_records:
        existing_records = checkpoint.completed_records()
    existing_by_key: dict[str, dict[str, Any]] = {}
    for existing in existing_records:
        existing_by_key[condition_key(existing.get("asset_id", existing.get("sample_id", "")), existing.get("video_frame_index", ""), existing.get("condition", ""))] = existing

    for asset in selected_assets:
        try:
            if asset.get("media_type", "image") == "video":
                frame_items, video_metadata = extract_video_frames(
                    asset, video_interval_seconds, video_max_frames, seed
                )
                frame_items = [
                    (frame_index, timestamp, image, video_metadata)
                    for frame_index, timestamp, image in frame_items
                ]
            else:
                original = load_asset(asset)
                frame_items = [(None, None, original, {})]
        except Exception as exc:  # noqa: BLE001
            records.append({**asset, "status": "读取失败", "error": str(exc)})
            continue

        is_video_asset = asset.get("media_type", "image") == "video"
        video_frame_paths: dict[str, list[str]] = {}
        video_record_refs: dict[str, list[dict[str, Any]]] = {}
        temporal_plans: dict[tuple[str, float, float], Any] = {}
        video_fps = float((frame_items[0][3] if frame_items else {}).get("video_fps", 25.0) or 25.0)

        for frame_index, timestamp, original, video_metadata in frame_items:
            full_width, full_height = original.size
            focused, focus_metadata = crop_focus_region(
                original, asset, focus_mode, focus_object_index, focus_roi
            )
            original_width, original_height = focused.size
            privacy_image, privacy_metadata = _apply_privacy_transform(
                focused,
                asset,
                privacy_mode,
                focus_applied=focus_metadata.get("focus_applied") == "是",
            )
            # Values that do not depend on the degradation condition are
            # computed once per frame.  This matters when several noise and
            # occlusion levels are selected for the same image.
            base_name = _safe_text(Path(asset["file_name"]).stem)
            frame_suffix = f"_frame_{frame_index:08d}" if frame_index is not None else ""
            output_group = "images" if asset.get("media_type", "image") == "image" else "videos"
            suffix = ".png" if output_format == "PNG" else ".jpg"
            sample_id = asset["asset_id"] + (f"_frame_{frame_index:08d}" if frame_index is not None else "")
            task = infer_task(asset, task_override)
            condition_dirs = {
                build_condition_name(resolution_name, noise, occlusion):
                run_dir / output_group / build_condition_name(resolution_name, noise, occlusion)
                for resolution_name in resolution_names
                for noise in noise_levels
                for occlusion in occlusion_levels
            }
            for condition_dir in condition_dirs.values():
                condition_dir.mkdir(parents=True, exist_ok=True)
            for resolution_name in resolution_names:
                # Resize/restoration is the most expensive image operation in
                # this loop.  Cache it once per resolution instead of repeating
                # it for every noise × occlusion combination.
                resolution_image, resolution_metadata = apply_low_resolution(
                    privacy_image,
                    RESOLUTION_OPTIONS[resolution_name],
                    restore_mode,
                )
                resolution_array = np.asarray(resolution_image.convert("RGB"))
                for noise in noise_levels:
                    for occlusion in occlusion_levels:
                        condition = build_condition_name(resolution_name, noise, occlusion)
                        output_dir = condition_dirs[condition]
                        output_path = output_dir / f"{asset['asset_id']}{frame_suffix}_{base_name}{suffix}"
                        resume_key = condition_key(asset.get("asset_id", ""), frame_index, condition)
                        if checkpoint.is_done(resume_key, output_path):
                            previous = existing_by_key.get(resume_key)
                            if previous:
                                records.append(previous)
                            completed += 1
                            if progress_callback and (completed == total or completed % progress_step == 0):
                                progress_callback(min(1.0, completed / total))
                            continue
                        temporal_plan = None
                        if temporal_consistency and is_video_asset:
                            plan_key = (resolution_name, float(noise), float(occlusion))
                            temporal_plan = temporal_plans.get(plan_key)
                            if temporal_plan is None:
                                plan_seed = int.from_bytes(
                                    hashlib.sha256(
                                        f"{asset.get('asset_id', '')}|{condition}|{seed}".encode("utf-8")
                                    ).digest()[:4],
                                    "big",
                                )
                                temporal_plan = make_temporal_plan(
                                    resolution_image.width,
                                    resolution_image.height,
                                    noise,
                                    occlusion,
                                    block_count,
                                    occlusion_position,
                                    plan_seed,
                                )
                                temporal_plans[plan_key] = temporal_plan
                            image = apply_temporal_plan(resolution_image, temporal_plan)
                        else:
                            noise_image = add_black_pixels(
                                resolution_image, noise, rng, base_array=resolution_array
                            )
                            image = add_block_occlusion(noise_image, occlusion, block_count, occlusion_position, rng)
                        image, structured_metadata = _apply_structured_occlusion(
                        image,
                        asset,
                        structured_occlusion_mode,
                        keypoints,
                        rng,
                        focus_applied=focus_metadata.get("focus_applied") == "是",
                        source_size=original.size,
                        )

                        if output_format == "PNG":
                            image.save(output_path, format="PNG")
                        else:
                            image.save(output_path, format="JPEG", quality=95)

                        record = {
                "asset_id": asset["asset_id"],
                "sample_id": sample_id,
                "file_name": asset["file_name"],
                "media_type": asset.get("media_type", "image"),
                "source_kind": asset["source_kind"],
                "source_path": asset["source_path"],
                "root_path": asset.get("root_path", ""),
                "dataset_name": asset.get("dataset_name", ""),
                "dataset_id": asset.get("dataset_id", "unknown"),
                "dataset_display_name": asset.get("dataset_display_name", asset.get("dataset_name", "")),
                "task_namespace": asset.get("task_namespace", task),
                "label_namespace": asset.get("label_namespace", "未登记"),
                "source_label": asset.get("source_label", asset.get("label", "未标注")),
                "canonical_label": asset.get("canonical_label", asset.get("label", "未标注")),
                "qualified_label": asset.get("qualified_label", ""),
                "label_mapping_note": asset.get("label_mapping_note", ""),
                "task": task,
                "behavior_label": asset.get("label", "") if task == "课堂行为" else "",
                "micro_expression_label": asset.get("label", "") if task == "微表情" else "",
                "engagement_label": asset.get("label", "") if task == "投入度/情感状态" else "",
                "origin_name": asset.get("origin_name", ""),
                "origin_url": asset.get("origin_url", ""),
                "citation": asset.get("citation", ""),
                "license_or_access": asset.get("license_or_access", ""),
                "archive_member": asset["member"],
                "source_group": asset["source_group"],
                "sampling_group": asset.get("sampling_group", asset["source_group"]),
                "camera_view": asset.get("sampling_group", asset.get("source_group", "")),
                "distance_proxy": "原始距离代理" if RESOLUTION_OPTIONS[resolution_name] >= 0.999 else "远距离模拟",
                "distance_profile": distance_profile,
                "occlusion_recommendation": OCCLUSION_RECOMMENDATIONS.get(distance_profile, {}).get("note", ""),
                "label_from_annotation_or_folder": asset["label"],
                "label_codes": "|".join(asset.get("label_codes", []) or []),
                "annotation_label": asset.get("annotation_label", ""),
                "bbox_count": asset.get("bbox_count", 0),
                "annotation_member": asset.get("annotation_member", ""),
                "objects": asset.get("objects", []),
                "video_frame_index": frame_index if frame_index is not None else "",
                "timestamp_seconds": round(timestamp, 3) if timestamp is not None else "",
                **video_metadata,
                "full_original_width": full_width,
                "full_original_height": full_height,
                "original_width": original_width,
                "original_height": original_height,
                "focused_width": focused.size[0],
                "focused_height": focused.size[1],
                **focus_metadata,
                "privacy_mode": privacy_mode,
                **privacy_metadata,
                **resolution_metadata,
                "condition": condition,
                "resolution_setting": resolution_name,
                "resolution_output_mode": restore_mode,
                "black_pixel_percent": noise,
                "occlusion_area_percent": occlusion,
                "occlusion_blocks": block_count if occlusion > 0 else 0,
                "occlusion_position": occlusion_position if occlusion > 0 else "无",
                "original_occlusion": "未自动判定",
                "structured_occlusion_mode": structured_occlusion_mode,
                **structured_metadata,
                "temporal_consistency": "是" if temporal_consistency and is_video_asset else "否",
                "temporal_plan_seed": getattr(temporal_plan, "seed", "") if temporal_plan is not None else "",
                "temporal_rectangles": ";".join("|".join(str(value) for value in rect) for rect in getattr(temporal_plan, "rectangles", ())) if temporal_plan is not None else "",
                "keypoints_input_count": len(keypoints),
                "processed_width": image.size[0],
                "processed_height": image.size[1],
                "output_media_group": output_group,
                "output_path": str(output_path),
                "status": "完成",
                "error": "",
                }
                        if is_video_asset:
                            video_frame_paths.setdefault(condition, []).append(str(output_path))
                            video_record_refs.setdefault(condition, []).append(record)
                        records.append(record)
                        checkpoint.mark_done(resume_key, record)
                        if completed % 25 == 0:
                            checkpoint.save()
                        if len(previews) < 8:
                            previews.append((record, focused.copy(), image.copy()))
                        completed += 1
                        # Updating a Streamlit progress widget for every saved
                        # file is surprisingly expensive for large runs.
                        if progress_callback and (completed == total or completed % progress_step == 0):
                            progress_callback(min(1.0, completed / total))

        if is_video_asset and synthesize_processed_videos:
            for condition, frame_paths in video_frame_paths.items():
                if not frame_paths:
                    continue
                video_dir = run_dir / "processed_videos" / condition
                video_path = video_dir / f"{asset['asset_id']}_{_safe_text(Path(asset['file_name']).stem)}.mp4"
                try:
                    write_video_from_frames(frame_paths, video_path, video_fps)
                except Exception as exc:  # noqa: BLE001
                    video_output_rows.append({
                        "asset_id": asset.get("asset_id", ""),
                        "file_name": asset.get("file_name", ""),
                        "condition": condition,
                        "status": "合成失败",
                        "error": str(exc),
                    })
                    continue
                video_output_rows.append({
                    "asset_id": asset.get("asset_id", ""),
                    "file_name": asset.get("file_name", ""),
                    "condition": condition,
                    "output_video_path": str(video_path),
                    "fps": video_fps,
                    "frame_count": len(frame_paths),
                    "status": "完成",
                })
                for record in video_record_refs.get(condition, []):
                    record["processed_video_path"] = str(video_path)

    if progress_callback:
        progress_callback(1.0)

    settings = {
        "app_version": APP_VERSION,
        "run_id": run_id,
        "sample_count": len(selected_assets),
        "resolution_settings": ", ".join(resolution_names),
        "black_pixel_levels": ", ".join(f"{v:g}%" for v in noise_levels),
        "occlusion_levels": ", ".join(f"{v:g}%" for v in occlusion_levels),
        "occlusion_blocks": block_count,
        "occlusion_position": occlusion_position,
        "random_seed": seed,
        "seed_mode": seed_mode,
        "output_format": output_format,
        "video_interval_seconds": video_interval_seconds,
        "video_max_frames": video_max_frames,
        "restore_mode": restore_mode,
        "focus_mode": focus_mode,
        "focus_object_index": focus_object_index,
        "focus_roi": "|".join(f"{v:.4f}" for v in focus_roi),
        "task_override": task_override,
        "distance_profile": distance_profile,
        "structured_occlusion_mode": structured_occlusion_mode,
        "privacy_mode": privacy_mode,
        "keypoints_text": keypoints_text,
        "temporal_consistency": temporal_consistency,
        "synthesize_processed_videos": synthesize_processed_videos,
        "occlusion_evidence": OCCLUSION_EVIDENCE,
        "note": "原始数据只读；图片和视频分开输出；结构化遮挡和隐私处理均记录来源；视频可按固定方案跨帧保持一致。",
    }
    excel_path = run_dir / "sample_manifest.xlsx"
    create_excel(records, settings, excel_path)
    if video_output_rows:
        pd.DataFrame(video_output_rows).to_csv(run_dir / "processed_video_manifest.csv", index=False, encoding="utf-8-sig")
    checkpoint.finish("completed")
    write_reproducibility_artifacts(run_dir, settings, records, selected_assets)
    return run_dir, records, previews


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## 🧪 课堂视觉数据处理台")
        st.caption("抽样、降分辨率、黑色像素干扰、矩形遮挡和结果记录")
        st.divider()

        source_config = st.text_area(
            "数据源列表",
            value=r"D:\数据处理1\学生行为数据集1|||学生行为数据集1|||GitHub/DAiSEE/Tianchi 数据|||",
            height=120,
            help=(
                "每行一个数据源，格式：本地路径|||数据集名称|||原始来源|||来源链接。"
                "例如：D:\\DAiSEE|||DAiSEE|||DAiSEE官方网站|||https://..."
            ),
        )
        st.caption("可填写多个根目录；使用 ||| 分隔本地路径、数据集名称、来源名称和来源链接。")
        include_archives = st.checkbox("扫描 ZIP 压缩包中的图片", value=True)
        include_videos = st.checkbox("扫描视频文件（可抽帧）", value=True)
        st.caption("扫描视频只读取目录信息；真正处理时才抽取帧，不会一次性解压全部视频。")
        sample_count = st.number_input("随机抽样数量", min_value=1, max_value=10000, value=20, step=1)
        sampling_mode = st.radio(
            "抽样方式",
            ["按媒体类型均衡", "按子文件夹均衡", "混合随机", "按来源尽量均衡", "按标签尽量均衡"],
            index=0,
            help="按媒体类型均衡会尽量保证图片和视频都进入抽样结果。",
        )
        seed_mode = st.radio(
            "随机种子模式",
            ["固定种子（可复现）", "每次自动生成"],
            index=0,
            help="固定模式便于复现实验；自动模式每次点击抽样都会生成不同结果。",
        )
        seed = st.number_input(
            "固定随机种子",
            min_value=0,
            max_value=99999999,
            value=20261024,
            step=1,
            disabled=seed_mode == "每次自动生成",
        )
        if seed_mode == "每次自动生成":
            st.caption("点击抽样时自动生成新的随机种子，并写入 Excel。")
        scan_button = st.button("🔍 扫描数据", type="primary", use_container_width=True)
        scan_signature = f"{source_config}||{include_archives}||{include_videos}"
        if st.session_state.get("scan_signature") not in {None, scan_signature}:
            st.session_state["needs_rescan"] = True
        if scan_button:
            with st.spinner("正在扫描文件夹和 ZIP 目录……"):
                assets = scan_assets(source_config, include_archives, include_videos)
            st.session_state["assets"] = assets
            st.session_state["source_config"] = source_config
            st.session_state["scan_signature"] = scan_signature
            st.session_state["needs_rescan"] = False
            st.session_state.pop("sampled_assets", None)
            st.session_state.pop("effective_seed", None)
            for filter_key in ["dataset_filter", "origin_filter", "source_filter", "media_filter", "label_filter"]:
                st.session_state.pop(filter_key, None)
            media_counts = pd.Series([item.get("media_type", "image") for item in assets]).value_counts().to_dict()
            st.success(
                f"已找到 {len(assets):,} 个媒体样本：图片 {media_counts.get('image', 0):,}，视频 {media_counts.get('video', 0):,}"
            )

        st.divider()
        st.markdown("### 生成内容保存路径")
        output_root = st.text_input(
            "输出根目录",
            value=st.session_state.get("output_root_input", r"D:\classroom_project\outputs"),
            key="output_root_input",
            help="图片、视频、Excel、质量报告、跟踪结果和模型评估结果都会保存到此目录或其运行子目录。",
        )
        path_cols = st.columns(2)
        with path_cols[0]:
            if st.button("📁 选择文件夹", use_container_width=True, key="choose_output_folder"):
                try:
                    import tkinter as tk
                    from tkinter import filedialog

                    root = tk.Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    initial_path = Path(output_root).expanduser()
                    if not initial_path.exists():
                        initial_path = initial_path.parent if initial_path.parent.exists() else Path.cwd()
                    selected = filedialog.askdirectory(initialdir=str(initial_path))
                    root.destroy()
                    if selected:
                        st.session_state["output_root_input"] = selected
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"无法打开系统文件夹选择器，请直接输入路径：{exc}")
        with path_cols[1]:
            if st.button("✅ 验证/创建", use_container_width=True, key="validate_output_folder"):
                try:
                    target = Path(output_root).expanduser()
                    target.mkdir(parents=True, exist_ok=True)
                    st.success(f"可保存：{target}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"保存路径不可用：{exc}")
        return {
            "source_config": source_config,
            "include_videos": include_videos,
            "sample_count": int(sample_count),
            "sampling_mode": sampling_mode,
            "seed": int(seed),
            "seed_mode": seed_mode,
            "output_root": output_root,
        }


def main() -> None:
    st.markdown(
        """
        <style>
        .main-title {font-size: 2.2rem; font-weight: 750; margin-bottom: 0.2rem;}
        .subtitle {color: #64748b; margin-bottom: 1.4rem;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        </style>
        <div class="main-title">课堂视觉数据处理台</div>
        <div class="subtitle">把大数据集变成可复现的小样本退化实验，并自动生成 Excel 记录。</div>
        """,
        unsafe_allow_html=True,
    )

    settings = render_sidebar()
    assets: list[dict[str, Any]] = st.session_state.get("assets", [])

    if st.session_state.get("needs_rescan"):
        st.warning("数据源或视频扫描选项已经变化，请回到左侧点击“扫描数据”，再进行抽样。")

    if not assets:
        st.info("请先在左侧填写一个或多个数据源，然后点击“扫描数据”。支持普通图片文件夹，也支持直接扫描 ZIP 压缩包中的图片。")
        st.markdown(
            """
            **推荐流程**

            1. 扫描数据并检查来源；  
            2. 设置抽样数量和干扰条件；  
            3. 生成处理结果；  
            4. 查看样例并下载 Excel/CSV 清单。
            """
        )
        return

    df_assets = pd.DataFrame(assets)
    if {"source_kind"}.issubset(df_assets.columns) and df_assets["source_kind"].astype(str).str.startswith("ZIP").any() and (df_assets["source_kind"] == "文件夹").any():
        st.warning("当前同时扫描了解压文件和ZIP成员，可能产生重复样本。已解压数据时建议关闭左侧‘扫描 ZIP 压缩包中的图片’后重新扫描。")
    if settings.get("include_videos") and not (df_assets["media_type"] == "video").any():
        st.warning("当前扫描结果没有视频样本。请确认视频文件位于数据源根目录或 ZIP 内，并重新点击“扫描数据”。")
    dataset_options = sorted(df_assets["dataset_name"].dropna().unique().tolist())
    origin_options = sorted(df_assets["origin_name"].dropna().unique().tolist())
    source_options = sorted(df_assets["source_group"].dropna().unique().tolist())
    label_options = sorted({label for labels in df_assets["label_names"] for label in (labels or [])})
    if not label_options:
        label_options = ["未标注"]

    tab_overview, tab_settings, tab_result, tab_report, tab_tracking, tab_quality = st.tabs(
        ["① 数据概览", "② 处理设置", "③ 结果与导出", "④ 汇报三表", "⑤ 多人跟踪", "⑥ 数据质量"]
    )

    with tab_overview:
        metrics = st.columns(4)
        metrics[0].metric("样本数量", f"{len(df_assets):,}")
        metrics[1].metric("数据集数量", f"{df_assets['dataset_name'].nunique():,}")
        metrics[2].metric("原始来源数量", f"{df_assets['origin_name'].nunique():,}")
        metrics[3].metric("视频数量", f"{(df_assets['media_type'] == 'video').sum():,}")
        st.caption(f"当前可用于均衡抽样的一级子文件夹/ZIP 分组：{df_assets['sampling_group'].nunique():,} 个")
        with st.expander("查看任务与标签命名空间", expanded=False):
            st.caption("DAiSEE、SCB、Tianchi 和微表情数据集使用独立命名空间，原始标签会保留在清单中。")
            st.dataframe(pd.DataFrame(registry_rows()), use_container_width=True, hide_index=True)

        st.markdown("### 数据筛选")
        media_options = sorted(df_assets["media_type"].dropna().unique().tolist())
        filter_cols = st.columns(4)
        with filter_cols[0]:
            selected_datasets = st.multiselect("数据集筛选", dataset_options, default=dataset_options, key="dataset_filter")
        with filter_cols[1]:
            selected_origins = st.multiselect("原始来源筛选", origin_options, default=origin_options, key="origin_filter")
        with filter_cols[2]:
            selected_sources = st.multiselect("本地来源筛选", source_options, default=source_options, key="source_filter")
        with filter_cols[3]:
            selected_media = st.multiselect("媒体类型", media_options, default=media_options, key="media_filter")
        label_filter_col = st.container()
        with label_filter_col:
            selected_labels = st.multiselect("标签筛选", label_options, default=label_options, key="label_filter")
        reset_filter_cols = st.columns([1, 4])
        with reset_filter_cols[0]:
            if st.button("重置全部筛选"):
                for filter_key in ["dataset_filter", "origin_filter", "source_filter", "media_filter", "label_filter"]:
                    st.session_state.pop(filter_key, None)
                st.rerun()
        st.caption(
            f"当前筛选范围：数据集 {len(selected_datasets)}/{len(dataset_options)}，"
            f"本地分组 {len(selected_sources)}/{len(source_options)}，"
            f"媒体类型 {len(selected_media)}/{len(media_options)}"
        )
        if len(selected_sources) == 1 and len(source_options) > 1:
            st.warning("当前只选择了一个本地分组，抽样结果会只来自这一组；请点击“重置全部筛选”。")
        filtered = df_assets[
            df_assets["dataset_name"].isin(selected_datasets)
            & df_assets["origin_name"].isin(selected_origins)
            & df_assets["source_group"].isin(selected_sources)
            & df_assets["media_type"].isin(selected_media)
            & df_assets["label_names"].apply(lambda labels: any(label in (labels or []) for label in selected_labels))
        ]
        st.caption(f"当前筛选后可抽样：{len(filtered):,} 个样本")
        st.dataframe(
            filtered[["dataset_name", "origin_name", "sampling_group", "source_group", "label", "annotation_label", "bbox_count", "file_name", "source_path", "member"]].head(200),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("🎲 按当前筛选随机抽样", type="primary"):
            if settings["seed_mode"] == "每次自动生成":
                effective_seed = secrets.randbelow(100000000)
            else:
                effective_seed = settings["seed"]
            st.session_state["effective_seed"] = effective_seed
            rng = random.Random(effective_seed)
            records = filtered.to_dict("records")
            if settings["sampling_mode"] == "混合随机" or not records:
                picked = rng.sample(records, min(settings["sample_count"], len(records)))
            elif settings["sampling_mode"] == "按媒体类型均衡":
                groups = {}
                for record in records:
                    groups.setdefault(record.get("media_type", "image"), []).append(record)
                group_names = list(groups)
                picked = []
                target_each = max(1, settings["sample_count"] // max(1, len(group_names)))
                for group_name in group_names:
                    rng.shuffle(groups[group_name])
                    picked.extend(groups[group_name][:target_each])
                rng.shuffle(records)
                picked.extend(records[: max(0, settings["sample_count"] - len(picked))])
                picked = picked[: settings["sample_count"]]
            elif settings["sampling_mode"] == "按子文件夹均衡":
                groups = {}
                for record in records:
                    groups.setdefault(record.get("sampling_group", record.get("source_group", "未分组")), []).append(record)
                group_names = list(groups)
                picked = []
                target_each = max(1, settings["sample_count"] // max(1, len(group_names)))
                remainder = settings["sample_count"] % max(1, len(group_names))
                rng.shuffle(group_names)
                for index, group_name in enumerate(group_names):
                    rng.shuffle(groups[group_name])
                    quota = target_each + (1 if index < remainder else 0)
                    picked.extend(groups[group_name][:quota])
                if len(picked) < settings["sample_count"]:
                    selected_ids = {item["asset_id"] for item in picked}
                    remaining = [item for item in records if item["asset_id"] not in selected_ids]
                    rng.shuffle(remaining)
                    picked.extend(remaining[: settings["sample_count"] - len(picked)])
                picked = picked[: settings["sample_count"]]
            elif settings["sampling_mode"] == "按来源尽量均衡":
                groups = {}
                for record in records:
                    groups.setdefault(record["source_group"], []).append(record)
                group_names = list(groups)
                picked = []
                target_each = max(1, settings["sample_count"] // max(1, len(group_names)))
                for group_name in group_names:
                    rng.shuffle(groups[group_name])
                    picked.extend(groups[group_name][:target_each])
                rng.shuffle(records)
                picked.extend(records[: max(0, settings["sample_count"] - len(picked))])
                picked = picked[: settings["sample_count"]]
            else:
                # 一张图片可能包含多个目标，因此按标签均衡时按“包含该标签的图片”抽样，
                # 但最终每张图片只保留一份，避免重复图片。
                target = min(settings["sample_count"], len(records))
                candidates_by_label = {
                    label: [r for r in records if label in (r.get("label_names") or [])]
                    for label in selected_labels
                }
                usable_labels = [label for label, items in candidates_by_label.items() if items]
                for items in candidates_by_label.values():
                    rng.shuffle(items)
                selected_ids: set[str] = set()
                picked = []
                while len(picked) < target and usable_labels:
                    usable_labels.sort(
                        key=lambda label: sum(
                            1 for item in picked if label in (item.get("label_names") or [])
                        )
                    )
                    selected = False
                    for label in usable_labels:
                        candidate = next(
                            (item for item in candidates_by_label[label] if item["asset_id"] not in selected_ids),
                            None,
                        )
                        if candidate is not None:
                            picked.append(candidate)
                            selected_ids.add(candidate["asset_id"])
                            selected = True
                            if len(picked) >= target:
                                break
                    if not selected:
                        break
                if len(picked) < target:
                    remaining = [item for item in records if item["asset_id"] not in selected_ids]
                    rng.shuffle(remaining)
                    picked.extend(remaining[: target - len(picked)])
            st.session_state["sampled_assets"] = picked
            st.success(f"已抽取 {len(picked)} 个样本；本次随机种子：{effective_seed}")

        st.markdown("### 手动选择图片或视频")
        st.caption("先用筛选条件缩小范围，再按文件名/路径搜索并勾选。手动选择会覆盖当前随机抽样结果。")
        manual_query = st.text_input("搜索文件名或路径", value="", key="manual_media_query")
        manual_candidates = filtered.copy()
        if manual_query.strip():
            pattern = re.escape(manual_query.strip())
            manual_candidates = manual_candidates[
                manual_candidates.apply(
                    lambda row: bool(
                        re.search(
                            pattern,
                            f"{row.get('file_name', '')} {row.get('source_path', '')} {row.get('member', '')}",
                            flags=re.IGNORECASE,
                        )
                    ),
                    axis=1,
                )
            ]
        manual_candidates = manual_candidates.head(300)
        manual_options = {
            str(row["asset_id"]): f"{row.get('media_type', 'image')} | {row.get('file_name', '')} | {row.get('source_group', '')}"
            for _, row in manual_candidates.iterrows()
        }
        selected_manual_ids = st.multiselect(
            f"勾选媒体（当前显示 {len(manual_options):,} 条，最多显示300条）",
            list(manual_options),
            format_func=lambda asset_id: manual_options.get(asset_id, asset_id),
            key="manual_media_ids",
        )
        if st.button("✅ 使用手动选择", disabled=not selected_manual_ids):
            asset_by_id = {str(item.get("asset_id")): item for item in assets}
            st.session_state["sampled_assets"] = [asset_by_id[item_id] for item_id in selected_manual_ids if item_id in asset_by_id]
            st.session_state["effective_seed"] = settings["seed"]
            st.success(f"已使用手动选择：{len(st.session_state['sampled_assets'])} 个媒体")

        sampled = st.session_state.get("sampled_assets", [])
        if sampled:
            st.markdown("### 当前抽样结果")
            media_summary = pd.Series([item.get("media_type", "image") for item in sampled]).value_counts().to_dict()
            st.caption(f"抽样媒体分布：图片 {media_summary.get('image', 0)}，视频 {media_summary.get('video', 0)}")
            sampled_df = pd.DataFrame(sampled)
            image_tab, video_tab = st.tabs([f"图片（{media_summary.get('image', 0)}）", f"视频（{media_summary.get('video', 0)}）"])
            for media_tab, media_name in ((image_tab, "image"), (video_tab, "video")):
                with media_tab:
                    view = sampled_df[sampled_df["media_type"] == media_name]
                    if view.empty:
                        st.info("当前抽样没有此类媒体。")
                    else:
                        st.dataframe(
                            view[["dataset_name", "origin_name", "sampling_group", "source_group", "label", "file_name"]],
                            use_container_width=True,
                            hide_index=True,
                        )
            distribution = []
            for label in label_options:
                count = sum(label in (item.get("label_names") or []) for item in sampled)
                if count:
                    distribution.append({"label": label, "image_count": count})
            if distribution:
                st.markdown("### 抽样标签覆盖情况")
                st.dataframe(pd.DataFrame(distribution), use_container_width=True, hide_index=True)
            group_distribution = (
                pd.DataFrame(sampled)["sampling_group"].value_counts().rename_axis("sampling_group").reset_index(name="sample_count")
            )
            st.markdown("### 抽样分组覆盖情况")
            st.dataframe(group_distribution, use_container_width=True, hide_index=True)

    with tab_settings:
        sampled = st.session_state.get("sampled_assets", [])
        if not sampled:
            st.warning("请先在“数据概览”中完成随机抽样。")
        else:
            st.markdown("### 设置要生成的版本")
            st.caption("为了避免十几 GB 数据被复制成几十 GB，建议先抽样 20～100 张进行测试。")
            processing_profile = st.radio(
                "处理速度档位",
                ["快速试跑（推荐）", "标准实验"],
                horizontal=True,
                help="快速试跑只预选少量条件，适合检查流程；你仍可以在下方手动增加条件。",
            )
            if processing_profile == "快速试跑（推荐）":
                default_resolution_names = ["原图", "1/4（模拟远距离）"]
                default_noise_levels = [0.0, 1.0]
                default_occlusion_levels = [0.0, 1.0]
                st.info("快速试跑默认生成 2×2×2=8 个版本/媒体；确认结果后再切换到标准实验。")
            else:
                default_resolution_names = ["原图", "1/2（模拟远距离）", "1/4（模拟远距离）"]
                default_noise_levels = None
                default_occlusion_levels = None
            resolution_names = st.multiselect(
                "降分辨率版本",
                list(RESOLUTION_OPTIONS),
                default=default_resolution_names,
                key=f"resolution_names_{processing_profile}",
            )
            restore_mode = st.selectbox(
                "低分辨率输出方式",
                RESTORE_OPTIONS,
                index=1,
                help="保留低分辨率用于真实远距离输入；恢复到原尺寸用于比较超分辨率/插值恢复后的分析结果。",
            )
            task_override = st.selectbox(
                "人物分析任务",
                ["自动识别", "课堂行为", "微表情", "投入度/情感状态", "姿态", "待指定"],
                index=0,
            )
            focus_mode = st.selectbox(
                "多人场景聚焦方式",
                FOCUS_OPTIONS,
                index=0,
                help="标注目标使用XML/VOC框；手动ROI对当前抽样的图片和视频帧使用同一归一化区域。",
            )
            focus_object_index = 1
            focus_roi = (0.0, 0.0, 1.0, 1.0)
            if focus_mode == "按标注聚焦一个人物":
                max_objects = max([len(item.get("objects") or []) for item in sampled] or [1])
                focus_object_index = st.number_input(
                    "标注目标序号（从1开始）",
                    min_value=1,
                    max_value=max(1, min(50, max_objects)),
                    value=1,
                    step=1,
                    help="对有XML/VOC目标框的图片生效；没有标注框的样本会保留全图。",
                )
            elif focus_mode == "手动ROI聚焦一个人物":
                roi_cols = st.columns(4)
                with roi_cols[0]:
                    roi_x1 = st.slider("ROI左", 0.0, 1.0, 0.0, 0.01)
                with roi_cols[1]:
                    roi_y1 = st.slider("ROI上", 0.0, 1.0, 0.0, 0.01)
                with roi_cols[2]:
                    roi_x2 = st.slider("ROI右", 0.0, 1.0, 1.0, 0.01)
                with roi_cols[3]:
                    roi_y2 = st.slider("ROI下", 0.0, 1.0, 1.0, 0.01)
                focus_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
            st.markdown("### 结构化遮挡与隐私保护")
            structured_occlusion_mode = st.selectbox(
                "结构化遮挡方式",
                STRUCTURED_OCCLUSION_OPTIONS,
                index=0,
                help="标注框/ROI、前景学生和关键点遮挡都会记录来源；没有真实前景图时不会标记为真实遮挡。",
            )
            keypoints_text = ""
            if structured_occlusion_mode == "关键点区域遮挡":
                keypoints_text = st.text_input(
                    "关键点坐标（x,y;x,y；支持0～1归一化坐标）",
                    value="",
                    help="例如 0.45,0.30;0.55,0.30。没有关键点输入时不会伪造关键点遮挡。",
                )
            privacy_mode = st.selectbox(
                "隐私保护方式",
                PRIVACY_OPTIONS,
                index=0,
                help="人脸优先使用OpenCV Haar检测；没有检测到人脸且已聚焦ROI时会使用聚焦区域回退并在清单中标明。",
            )
            distance_profile = st.selectbox(
                "遮挡面积推荐场景",
                list(OCCLUSION_RECOMMENDATIONS),
                index=0,
                help=OCCLUSION_EVIDENCE,
            )
            recommendation = OCCLUSION_RECOMMENDATIONS[distance_profile]
            st.info(
                f"当前推荐：黑色像素 {recommendation['noise'][1]:g}%–{recommendation['noise'][-1]:g}%；"
                f"黑色方块 {recommendation['occlusion'][1]:g}%–{recommendation['occlusion'][-1]:g}%。"
                f"{recommendation['note']}"
            )
            st.caption(OCCLUSION_EVIDENCE)
            noise_levels = st.multiselect(
                "随机黑色像素比例",
                NOISE_OPTIONS,
                default=default_noise_levels if default_noise_levels is not None else recommendation["noise"],
                format_func=lambda x: f"{x:g}%",
                key=f"noise_levels_{processing_profile}",
                help="随机把指定比例的像素设置为黑色。这里是像素比例，不是一个连续遮挡方块。",
            )
            occ_levels = st.multiselect(
                "黑色方框遮挡面积比例",
                OCCLUSION_OPTIONS,
                default=default_occlusion_levels if default_occlusion_levels is not None else recommendation["occlusion"],
                format_func=lambda x: f"{x:g}%",
                key=f"occlusion_levels_{processing_profile}",
                help="黑色方框总面积占整张图片面积的比例。远距离多人场景优先使用0.1%–2%。",
            )
            cols = st.columns(3)
            with cols[0]:
                block_count = st.number_input("方框数量", min_value=1, max_value=5, value=1, step=1)
            with cols[1]:
                occ_position = st.selectbox("遮挡位置", ["随机", "中心", "上方", "下方"])
            with cols[2]:
                output_format = st.selectbox(
                    "输出格式",
                    ["JPG", "PNG"],
                    index=0,
                    help="JPG 写盘速度和占用空间通常明显低于 PNG，适合批量预处理；需要无损像素时再选择 PNG。",
                )

            video_sample_count = sum(item.get("media_type", "image") == "video" for item in sampled)
            if video_sample_count:
                st.markdown("### 视频抽帧设置")
                video_cols = st.columns(2)
                with video_cols[0]:
                    video_interval = st.number_input(
                        "抽帧间隔（秒）",
                        min_value=0.1,
                        max_value=60.0,
                        value=2.0,
                        step=0.5,
                        help="例如 2 秒表示每隔约 2 秒抽取一帧。",
                    )
                with video_cols[1]:
                    video_max_frames = st.number_input(
                        "每个视频最多抽帧数",
                        min_value=1,
                        max_value=500,
                        value=20,
                        step=1,
                    )
                st.caption("视频会先按时间间隔抽帧，再对帧执行相同的降分辨率、黑色像素和遮挡处理。")
                temporal_consistency = st.checkbox(
                    "保持视频中噪声和遮挡位置跨帧一致",
                    value=True,
                    help="开启后，同一视频和处理条件会复用固定噪声掩码与遮挡矩形，减少闪烁。",
                )
                synthesize_processed_videos = st.checkbox(
                    "将处理后的帧重新合成为 MP4",
                    value=False,
                    help="会额外写入 processed_videos 目录，适合动作持续时间和视频模型评估。",
                )
            else:
                video_interval = 2.0
                video_max_frames = 20
                temporal_consistency = False
                synthesize_processed_videos = False

            estimated_frames = len(sampled) + video_sample_count * max(0, int(video_max_frames) - 1)
            combinations = estimated_frames * max(1, len(resolution_names)) * max(1, len(noise_levels)) * max(1, len(occ_levels))
            if combinations > 5000:
                st.warning(f"当前设置将生成约 {combinations:,} 张图片，可能需要大量磁盘空间。建议先减少抽样数量或处理条件。")
            else:
                st.info(f"预计生成 {combinations:,} 张图片。")
            st.caption(
                "性能提示：同一帧的降分辨率结果会在不同噪声和遮挡条件之间复用；进度条也会按批次刷新。"
                "如果仍然卡顿，请先减少抽样数量和处理条件组合。"
            )
            resume_run_dir = None
            output_root_path = Path(settings["output_root"]).expanduser()
            candidate_runs = sorted(
                [path for path in output_root_path.glob("run_*") if path.is_dir() and (path / "checkpoint.json").exists()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ) if output_root_path.exists() else []
            if candidate_runs:
                resume_options = ["新建运行"] + [f"恢复：{path.name}" for path in candidate_runs[:10]]
                resume_choice = st.selectbox(
                    "断点续跑",
                    resume_options,
                    help="恢复同一组参数且输出文件仍存在的条件；参数改变时不会跳过旧结果。",
                )
                if resume_choice != "新建运行":
                    resume_run_dir = str(candidate_runs[resume_options.index(resume_choice) - 1])
                    st.info(f"将恢复已有运行：{resume_run_dir}")

            run_button = st.button("▶ 开始处理并生成 Excel", type="primary", use_container_width=True)
            if run_button:
                if not resolution_names or not noise_levels or not occ_levels:
                    st.error("请至少选择一个分辨率、黑色像素比例和遮挡比例。")
                else:
                    try:
                        Path(settings["output_root"]).expanduser().mkdir(parents=True, exist_ok=True)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"生成内容保存路径不可用：{exc}")
                    else:
                        progress_bar = st.progress(0.0)
                        with st.spinner("正在读取样本并生成处理版本……"):
                            run_dir, records, previews = run_processing(
                            sampled,
                            settings["output_root"],
                            resolution_names,
                            noise_levels,
                            occ_levels,
                            int(block_count),
                            occ_position,
                            (
                                st.session_state.get("effective_seed", settings["seed"])
                                if settings["seed_mode"] == "每次自动生成"
                                else settings["seed"]
                            ),
                            output_format,
                            progress_callback=progress_bar.progress,
                            video_interval_seconds=float(video_interval),
                            video_max_frames=int(video_max_frames),
                            seed_mode=settings["seed_mode"],
                            restore_mode=restore_mode,
                            focus_mode=focus_mode,
                            focus_object_index=int(focus_object_index),
                            focus_roi=focus_roi,
                            task_override=task_override,
                            distance_profile=distance_profile,
                            structured_occlusion_mode=structured_occlusion_mode,
                            privacy_mode=privacy_mode,
                            keypoints_text=keypoints_text,
                            temporal_consistency=bool(temporal_consistency),
                            synthesize_processed_videos=bool(synthesize_processed_videos),
                            resume_run_dir=resume_run_dir,
                            )
                        progress_bar.progress(1.0)
                        st.session_state["last_run_dir"] = str(run_dir)
                        st.session_state["last_records"] = records
                        st.session_state["last_previews"] = previews
                        st.success(f"处理完成：{run_dir}")

    with tab_result:
        records = st.session_state.get("last_records", [])
        previews = st.session_state.get("last_previews", [])
        last_run_dir = st.session_state.get("last_run_dir", "")
        if not records:
            st.info("完成一次处理后，这里会显示处理样例、统计和下载按钮。")
        else:
            result_df = pd.DataFrame(records)
            result_cols = st.columns(4)
            result_cols[0].metric("成功输出", f"{(result_df['status'] == '完成').sum():,}")
            result_cols[1].metric("读取失败", f"{(result_df['status'] == '读取失败').sum():,}")
            result_cols[2].metric("处理条件", f"{result_df['condition'].nunique():,}")
            result_cols[3].metric("图片/视频输出", f"{(result_df['media_type'] == 'image').sum():,} / {(result_df['media_type'] == 'video').sum():,}")

            st.markdown("### 处理样例")
            for record, original, processed in previews:
                cols = st.columns(2)
                caption = f"{record['file_name']} · {record['condition']} · {record['label_from_annotation_or_folder']}"
                cols[0].image(ImageOps.contain(original, (520, 360)), caption="原图", use_container_width=True)
                cols[1].image(ImageOps.contain(processed, (520, 360)), caption=caption, use_container_width=True)

            st.markdown("### 清单预览")
            st.dataframe(result_df.head(200), use_container_width=True, hide_index=True)
            excel_path = Path(last_run_dir) / "sample_manifest.xlsx"
            csv_path = Path(last_run_dir) / "sample_manifest.csv"
            download_cols = st.columns(4)
            if excel_path.exists():
                download_cols[0].download_button(
                    "⬇ 下载 Excel 清单",
                    data=excel_path.read_bytes(),
                    file_name="sample_manifest.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            if csv_path.exists():
                download_cols[1].download_button(
                    "⬇ 下载 CSV 清单",
                    data=csv_path.read_bytes(),
                    file_name="sample_manifest.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            for col_index, file_name, label in (
                (2, "image_manifest.csv", "⬇ 图片清单"),
                (3, "video_manifest.csv", "⬇ 视频清单"),
            ):
                path = Path(last_run_dir) / file_name
                if path.exists():
                    download_cols[col_index].download_button(
                        label,
                        data=path.read_bytes(),
                        file_name=file_name,
                        mime="text/csv",
                        use_container_width=True,
                    )
            video_manifest_path = Path(last_run_dir) / "processed_video_manifest.csv"
            if video_manifest_path.exists():
                st.download_button(
                    "⬇ 处理后视频清单",
                    data=video_manifest_path.read_bytes(),
                    file_name="processed_video_manifest.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            reproducibility_path = Path(last_run_dir) / "reproducibility.json"
            if reproducibility_path.exists():
                st.download_button(
                    "⬇ 复现配置 JSON",
                    data=reproducibility_path.read_bytes(),
                    file_name="reproducibility.json",
                    mime="application/json",
                    use_container_width=True,
                )
            impact_path = Path(last_run_dir) / "impact_summary.csv"
            if impact_path.exists():
                st.markdown("### 处理条件影响占比（未导入模型结果时为样本/条件占比）")
                st.dataframe(pd.read_csv(impact_path).head(200), use_container_width=True, hide_index=True)
            st.caption("原图的真实遮挡状态不会被自动臆测；本程序会记录全局/结构化遮挡、隐私处理和视频时间一致性参数。")
            if st.button("📦 生成便携实验包", use_container_width=True):
                with st.spinner("正在复制实验结果并改写为相对路径……"):
                    portable_dir, portable_zip = create_portable_package_from_run(last_run_dir)
                st.session_state["portable_dir"] = str(portable_dir)
                st.session_state["portable_zip"] = str(portable_zip)
                st.success(f"便携实验包已生成：{portable_dir}")
            if st.session_state.get("portable_zip"):
                st.caption(f"可复制整个文件夹，或发送 ZIP：{st.session_state['portable_zip']}")

    with tab_report:
        st.markdown("### 汇报用三类清单")
        st.caption("总表按数据集汇总；样本索引只抽取实验子集；全量文件清单只记录路径、数量和完整性。")
        st.markdown("### 真实模型评估与条件影响分析")
        st.caption(
            "处理台不替模型运行推理；请上传外部模型的评估CSV。支持两种格式："
            "逐样本格式（sample_id、condition、task、y_true、y_pred），或汇总格式（task、condition、metric、value）。"
            "分类任务计算 Accuracy、Precision、Recall、UAR、Macro-F1；投入度回归计算 MAE、RMSE；"
            "检测/姿态任务可直接导入 mAP、AP、AR、PCK、OKS 等汇总指标。"
        )
        prediction_upload = st.file_uploader(
            "上传模型评估CSV（建议使用发行包中的model_evaluation_template.csv）",
            type=["csv"],
            key="prediction_upload",
        )
        if prediction_upload is not None:
            try:
                uploaded_eval_df = pd.read_csv(prediction_upload)
                normalized_eval, eval_warnings = normalize_evaluation_csv(
                    uploaded_eval_df, st.session_state.get("last_records", [])
                )
                for warning in eval_warnings:
                    st.warning(warning)
                if not normalized_eval.empty:
                    condition_options = list(dict.fromkeys(normalized_eval["condition"].astype(str).tolist()))
                    # Legacy paired CSVs carry the original-image score in
                    # ``baseline_score`` while only naming the degraded
                    # condition.  Expose a synthetic 原图 baseline so the
                    # relative drop is calculated instead of comparing the
                    # degraded row with itself.
                    has_paired_baseline = "baseline_value" in normalized_eval.columns and normalized_eval["baseline_value"].notna().any()
                    if has_paired_baseline and "原图" not in condition_options:
                        condition_options.insert(0, "原图")
                    default_baseline = "原图" if "原图" in condition_options else condition_options[0]
                    baseline_condition = st.selectbox(
                        "基准条件（用于计算性能下降）",
                        condition_options,
                        index=condition_options.index(default_baseline),
                        key="evaluation_baseline_condition",
                        help="例如原图、clean；低分辨率/遮挡条件将与它比较。",
                    )
                    if st.button("📈 计算真实模型指标与影响", key="compute_real_evaluation"):
                        metrics_df = compute_evaluation_metrics(normalized_eval)
                        comparison_df = compare_condition_impact(metrics_df, baseline_condition)
                        impact_dir = Path(st.session_state.get("last_run_dir", settings["output_root"]))
                        impact_dir.mkdir(parents=True, exist_ok=True)
                        normalized_eval.to_csv(impact_dir / "model_evaluation_input_normalized.csv", index=False, encoding="utf-8-sig")
                        metrics_df.to_csv(impact_dir / "model_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
                        comparison_df.to_csv(impact_dir / "model_evaluation_condition_impact.csv", index=False, encoding="utf-8-sig")
                        with pd.ExcelWriter(impact_dir / "model_evaluation_report.xlsx", engine="openpyxl") as writer:
                            normalized_eval.to_excel(writer, sheet_name="normalized_input", index=False)
                            metrics_df.to_excel(writer, sheet_name="metrics", index=False)
                            comparison_df.to_excel(writer, sheet_name="condition_impact", index=False)
                        update_run_model_metadata(
                            impact_dir,
                            uploaded_eval_df,
                            st.session_state.get("last_records", []),
                            st.session_state.get("sampled_assets", []),
                        )
                        chart_index = write_experiment_charts(metrics_df, comparison_df, impact_dir / "charts")
                        st.session_state["chart_index"] = chart_index
                        st.session_state["chart_dir"] = str(impact_dir / "charts")
                        st.session_state["model_evaluation_metrics"] = metrics_df
                        st.session_state["model_evaluation_comparison"] = comparison_df
                        st.session_state["model_evaluation_dir"] = str(impact_dir)
                        st.success(f"真实模型评估报告已生成：{impact_dir / 'model_evaluation_report.xlsx'}")
                    st.markdown("#### 已识别的评估输入")
                    st.dataframe(normalized_eval.head(200), use_container_width=True, hide_index=True)
                else:
                    st.error("未能从CSV识别有效评估记录。请检查模板字段。")
            except Exception as exc:  # noqa: BLE001
                st.error(f"模型评估CSV无法解析：{exc}")
        with st.expander("自动运行外部模型并生成评估结果", expanded=False):
            st.caption(
                "程序会把当前处理清单写成 model_input_manifest.csv，然后用 shell=False 调用你提供的模型命令。"
                "模型必须把预测结果写到 {output_csv}，支持占位符 {manifest_path}、{output_csv}、{output_dir}、{task}、{dataset_name}、{model_name}、{model_version}。"
            )
            runner_command = st.text_input(
                "模型命令模板",
                value='python model_infer.py --manifest "{manifest_path}" --output "{output_csv}"',
                key="runner_command_template",
                help="建议使用绝对路径指定模型脚本；模型自身的权重和依赖由模型环境负责。",
            )
            runner_cols = st.columns(4)
            with runner_cols[0]:
                runner_task = st.text_input("任务", value="", key="runner_task")
            with runner_cols[1]:
                runner_model_name = st.text_input("模型名称", value="", key="runner_model_name")
            with runner_cols[2]:
                runner_model_version = st.text_input("模型版本", value="", key="runner_model_version")
            with runner_cols[3]:
                runner_timeout = st.number_input("超时（秒）", min_value=10, max_value=86400, value=3600, step=60, key="runner_timeout")
            if st.button("▶ 运行外部模型并评估", key="run_external_model"):
                runner_records = st.session_state.get("last_records", [])
                if not runner_records:
                    st.warning("请先完成一次图片/视频处理，再运行外部模型。")
                else:
                    runner_dir = Path(st.session_state.get("last_run_dir", settings["output_root"])) / "model_runner"
                    with st.spinner("正在运行外部模型并等待预测 CSV……"):
                        runner_result, runner_prediction = run_external_model(
                            runner_records,
                            runner_command,
                            runner_dir,
                            task=runner_task,
                            model_name=runner_model_name,
                            model_version=runner_model_version,
                            timeout_seconds=float(runner_timeout),
                        )
                        runner_report_path = write_runner_report(runner_result, runner_dir)
                    st.session_state["runner_result"] = runner_result.to_dict()
                    st.session_state["runner_report_dir"] = str(runner_dir)
                    if runner_result.status != "完成":
                        st.error(f"外部模型未完成：{runner_result.status}；{runner_result.error}")
                        if runner_result.stderr:
                            st.code(runner_result.stderr[-4000:])
                    else:
                        runner_metrics, runner_comparison, runner_warnings = evaluate_runner_output(
                            runner_prediction,
                            runner_records,
                            baseline_condition="原图",
                        )
                        for warning in runner_warnings:
                            st.warning(warning)
                        runner_prediction.to_csv(runner_dir / "model_predictions.csv", index=False, encoding="utf-8-sig")
                        runner_metrics.to_csv(runner_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
                        runner_comparison.to_csv(runner_dir / "model_condition_impact.csv", index=False, encoding="utf-8-sig")
                        chart_index = write_experiment_charts(runner_metrics, runner_comparison, runner_dir / "charts")
                        with pd.ExcelWriter(runner_dir / "model_runner_evaluation.xlsx", engine="openpyxl") as writer:
                            runner_prediction.to_excel(writer, sheet_name="predictions", index=False)
                            runner_metrics.to_excel(writer, sheet_name="metrics", index=False)
                            runner_comparison.to_excel(writer, sheet_name="condition_impact", index=False)
                        update_run_model_metadata(
                            Path(st.session_state.get("last_run_dir", runner_dir)),
                            runner_prediction,
                            runner_records,
                            st.session_state.get("sampled_assets", []),
                        )
                        st.session_state["runner_metrics"] = runner_metrics
                        st.session_state["runner_comparison"] = runner_comparison
                        st.session_state["chart_index"] = chart_index
                        st.session_state["chart_dir"] = str(runner_dir / "charts")
                        st.success(f"外部模型评估完成：{runner_dir}")
            if st.session_state.get("runner_result"):
                st.json(st.session_state["runner_result"])
            runner_dir_state = Path(st.session_state.get("runner_report_dir", ""))
            runner_xlsx = runner_dir_state / "model_runner_evaluation.xlsx"
            if runner_xlsx.exists():
                st.download_button(
                    "⬇ 下载自动评估报告",
                    runner_xlsx.read_bytes(),
                    file_name="model_runner_evaluation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_runner_evaluation",
                )
        with st.expander("可选基础模型插件与权重登记", expanded=False):
            registry = ModelRegistry()
            st.caption("主程序不强制捆绑大型权重；这里登记模型任务、版本、权重来源和输出协议。")
            st.dataframe(pd.DataFrame(registry.rows()), use_container_width=True, hide_index=True)
            registry_path = Path(st.session_state.get("last_run_dir", settings["output_root"])) / "model_registry.json"
            if st.button("保存模型注册表", key="save_model_registry"):
                registry.save(registry_path)
                st.success(f"模型注册表已保存：{registry_path}")
        with st.expander("自动计算检测/姿态指标", expanded=False):
            st.caption("检测需要 gt_x1...gt_y2/pred_x1...pred_y2；姿态需要 gt_keypoints/pred_keypoints 和可选 bbox_area。")
            detection_pose_upload = st.file_uploader("上传检测/姿态原始预测 CSV", type=["csv"], key="detection_pose_upload")
            detection_pose_task = st.selectbox("指标类型", ["自动识别", "检测", "姿态"], key="detection_pose_task")
            detection_pose_bootstrap = st.number_input("Bootstrap 次数（0=不计算置信区间）", min_value=0, max_value=10000, value=0, step=100, key="detection_pose_bootstrap")
            if detection_pose_upload is not None and st.button("计算检测/姿态指标", key="compute_detection_pose"):
                try:
                    detection_pose_df = pd.read_csv(detection_pose_upload)
                    computed_pose_detection = compute_detection_pose_metrics(
                        detection_pose_df,
                        task_type=detection_pose_task.lower() if detection_pose_task != "自动识别" else "auto",
                        bootstrap_samples=int(detection_pose_bootstrap),
                        random_seed=int(settings.get("seed", 20261024)),
                    )
                    metric_dir = Path(st.session_state.get("last_run_dir", settings["output_root"])) / "detection_pose_metrics"
                    metric_dir.mkdir(parents=True, exist_ok=True)
                    computed_pose_detection.to_csv(metric_dir / "detection_pose_metrics.csv", index=False, encoding="utf-8-sig")
                    with pd.ExcelWriter(metric_dir / "detection_pose_metrics.xlsx", engine="openpyxl") as writer:
                        detection_pose_df.to_excel(writer, sheet_name="raw_input", index=False)
                        computed_pose_detection.to_excel(writer, sheet_name="metrics", index=False)
                    st.session_state["detection_pose_metrics"] = computed_pose_detection
                    st.session_state["detection_pose_metric_dir"] = str(metric_dir)
                    st.success(f"检测/姿态指标已生成：{metric_dir}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"检测/姿态指标计算失败：{exc}")
            if st.session_state.get("detection_pose_metrics") is not None:
                st.dataframe(st.session_state["detection_pose_metrics"], use_container_width=True, hide_index=True)
        metrics_state = st.session_state.get("model_evaluation_metrics")
        comparison_state = st.session_state.get("model_evaluation_comparison")
        if metrics_state is not None and not metrics_state.empty:
            st.markdown("#### 模型指标（按任务和处理条件）")
            st.dataframe(metrics_state.head(500), use_container_width=True, hide_index=True)
            if comparison_state is not None and not comparison_state.empty:
                st.markdown("#### 相对基准的性能变化")
                st.dataframe(comparison_state.head(500), use_container_width=True, hide_index=True)
                eval_dir = Path(st.session_state.get("model_evaluation_dir", settings["output_root"]))
                report_path = eval_dir / "model_evaluation_report.xlsx"
                if report_path.exists():
                    st.download_button(
                        "⬇ 下载模型评估报告 Excel",
                        data=report_path.read_bytes(),
                        file_name="model_evaluation_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_model_evaluation_report",
                    )
        elif not st.session_state.get("last_records"):
            st.info("可以直接上传汇总格式评估CSV；完成一次处理后，逐样本CSV中的sample_id还能与处理条件自动关联。")
        chart_metrics = metrics_state if metrics_state is not None and not metrics_state.empty else st.session_state.get("detection_pose_metrics")
        chart_comparison = comparison_state if comparison_state is not None else pd.DataFrame()
        if chart_metrics is not None and not chart_metrics.empty:
            st.markdown("#### 实验曲线与影响图")
            chart_dir = Path(st.session_state.get("last_run_dir", settings["output_root"])) / "experiment_charts"
            if st.button("📈 生成实验曲线和影响柱状图", key="generate_experiment_charts"):
                chart_index = write_experiment_charts(chart_metrics, chart_comparison, chart_dir)
                st.session_state["chart_index"] = chart_index
                st.session_state["chart_dir"] = str(chart_dir)
            chart_view = chart_metrics.copy()
            if {"condition", "metric", "value"}.issubset(chart_view.columns):
                chart_view["chart_series"] = chart_view.get("task", "任务").astype(str) + "|" + chart_view["metric"].astype(str)
                chart_view = chart_view.pivot_table(index="condition", columns="chart_series", values="value", aggfunc="mean")
                st.line_chart(chart_view)
            chart_index = st.session_state.get("chart_index")
            if chart_index:
                st.caption(f"图表状态：{chart_index.get('status', '完成')}，数量：{chart_index.get('chart_count', 0)}")
                chart_dir_state = Path(st.session_state.get("chart_dir", chart_dir))
                for chart_path in sorted(chart_dir_state.glob("*.png")):
                    st.download_button(
                        f"下载 {chart_path.name}",
                        chart_path.read_bytes(),
                        file_name=chart_path.name,
                        mime="image/png",
                        key=f"download_chart_{chart_path.name}",
                    )
        report_cols = st.columns(2)
        with report_cols[0]:
            target_per_dataset = st.number_input(
                "每个数据集实验样本数",
                min_value=50,
                max_value=1000,
                value=500,
                step=50,
                help="数据集不足该数量时使用全部样本；大数据集最多抽取该数量。",
            )
        with report_cols[1]:
            report_seed = st.number_input(
                "实验子集随机种子",
                min_value=0,
                max_value=99999999,
                value=int(st.session_state.get("effective_seed", settings["seed"])),
                step=1,
            )
        st.info("样本级索引按数据集、类别、分辨率、视角/来源分组和小目标状态分层；原始距离和遮挡没有标注时会写为‘未标注’。")
        if st.button("📊 生成汇报三表", type="primary", use_container_width=True):
            with st.spinner("正在生成数据集总表、实验子集索引和全量文件清单……"):
                master_df = build_dataset_master(assets)
                selected_for_report = stratified_sample_assets(assets, int(target_per_dataset), int(report_seed))
                sample_df = build_sample_index(selected_for_report)
                raw_df = build_raw_file_list(parse_source_specs(settings["source_config"]))
                report_settings = {
                    "target_per_dataset": int(target_per_dataset),
                    "selected_sample_count": len(sample_df),
                    "raw_file_record_count": len(raw_df),
                    "random_seed": int(report_seed),
                    "source_config": settings["source_config"],
                    "note": "全量数据只生成文件清单；样本索引为分层实验子集。距离和原始遮挡未标注时保留为未标注。",
                }
                report_dir = write_report_package(
                    master_df,
                    sample_df,
                    raw_df,
                    report_settings,
                    settings["output_root"],
                )
            st.session_state["report_dir"] = str(report_dir)
            st.session_state["report_master"] = master_df
            st.session_state["report_sample"] = sample_df
            st.session_state["report_raw"] = raw_df
            st.success(f"汇报三表已生成：{report_dir}")

        report_dir = st.session_state.get("report_dir", "")
        if report_dir:
            master_df = st.session_state.get("report_master", pd.DataFrame())
            sample_df = st.session_state.get("report_sample", pd.DataFrame())
            raw_df = st.session_state.get("report_raw", pd.DataFrame())
            counts = st.columns(3)
            counts[0].metric("数据集总表行数", f"{len(master_df):,}")
            counts[1].metric("实验子集行数", f"{len(sample_df):,}")
            counts[2].metric("全量文件清单行数", f"{len(raw_df):,}")
            st.dataframe(master_df, use_container_width=True, hide_index=True)
            st.caption(f"文件目录：{report_dir}")

    with tab_tracking:
        # Tracking is optional; the YOLO model is loaded only after the user runs it.
        render_tracking_panel(
            st.session_state.get("sampled_assets", []),
            settings["output_root"],
        )

    with tab_quality:
        render_quality_panel(assets, settings["output_root"])
        with st.expander("统一标注格式转换", expanded=False):
            st.caption("将 VOC XML、YOLO TXT、COCO JSON 或关键点 JSON 转为统一 CSV，保留原始标签和数据集命名空间。")
            annotation_format = st.selectbox(
                "标注格式",
                ["VOC XML", "YOLO TXT", "COCO JSON", "关键点 JSON"],
                key="annotation_format",
            )
            annotation_upload = st.file_uploader("上传标注文件", type=["xml", "txt", "json"], key="annotation_upload")
            annotation_dataset_id = st.text_input("数据集 ID", value="unknown", key="annotation_dataset_id")
            annotation_namespace = st.text_input("标签命名空间", value="", key="annotation_namespace")
            annotation_cols = st.columns(2)
            with annotation_cols[0]:
                annotation_width = st.number_input("YOLO/关键点图像宽度", min_value=1, value=1920, step=1, key="annotation_width")
            with annotation_cols[1]:
                annotation_height = st.number_input("YOLO/关键点图像高度", min_value=1, value=1080, step=1, key="annotation_height")
            annotation_image_id = st.text_input("COCO image_id（留空=全部）", value="", key="annotation_image_id")
            if annotation_upload is not None and st.button("转换为统一标注 CSV", key="convert_annotations"):
                try:
                    raw = annotation_upload.getvalue()
                    if annotation_format == "VOC XML":
                        unified_rows = parse_voc_xml(raw, dataset_id=annotation_dataset_id, label_namespace=annotation_namespace)
                    elif annotation_format == "YOLO TXT":
                        unified_rows = parse_yolo_txt(
                            raw,
                            image_width=int(annotation_width),
                            image_height=int(annotation_height),
                            dataset_id=annotation_dataset_id,
                            label_namespace=annotation_namespace,
                        )
                    elif annotation_format == "COCO JSON":
                        image_id = annotation_image_id if annotation_image_id.strip() else None
                        unified_rows = parse_coco_json(raw.decode("utf-8", errors="replace"), image_id=image_id, dataset_id=annotation_dataset_id, label_namespace=annotation_namespace)
                    else:
                        unified_rows = parse_keypoints_json(raw.decode("utf-8", errors="replace"), dataset_id=annotation_dataset_id, label_namespace=annotation_namespace)
                    annotation_dir = Path(settings["output_root"]) / "annotation_conversion"
                    unified_path = annotation_dir / "unified_annotations.csv"
                    write_unified_annotations(unified_rows, unified_path)
                    st.session_state["unified_annotations"] = pd.DataFrame(unified_rows)
                    st.session_state["unified_annotations_path"] = str(unified_path)
                    st.success(f"统一标注已生成：{unified_path}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"标注转换失败：{exc}")
            if st.session_state.get("unified_annotations") is not None:
                st.dataframe(st.session_state["unified_annotations"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()


