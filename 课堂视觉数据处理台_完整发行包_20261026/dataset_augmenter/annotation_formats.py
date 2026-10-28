"""Unified annotation readers for VOC XML, YOLO TXT and COCO JSON.

The readers preserve the source label and dataset namespace so converted
annotations can be evaluated without merging unrelated class vocabularies.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterable


def _read_text(source: str | Path | bytes) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8", errors="replace")
    path = Path(source)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else str(source)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _namespace(source_label: Any, dataset_id: str, label_namespace: str) -> dict[str, str]:
    raw = str(source_label or "未标注").strip() or "未标注"
    namespace = str(label_namespace or dataset_id or "未登记")
    return {"source_label": raw, "canonical_label": raw, "label_namespace": namespace, "qualified_label": f"{namespace}::{raw}"}


def _row(*, source_format: str, source_label: Any, dataset_id: str = "unknown", label_namespace: str = "", image_path: str = "", class_id: Any = "", image_width: Any = "", image_height: Any = "", xmin: Any = "", ymin: Any = "", xmax: Any = "", ymax: Any = "", keypoints: Any = "", **extra: Any) -> dict[str, Any]:
    row = {"image_path": str(image_path), "source_format": source_format, "class_id": class_id, "image_width": image_width, "image_height": image_height, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "keypoints": keypoints}
    row.update(_namespace(source_label, dataset_id, label_namespace)); row.update(extra); return row


def parse_voc_xml(source: str | Path | bytes, *, image_path: str = "", class_map: Mapping[Any, str] | None = None, dataset_id: str = "unknown", label_namespace: str = "") -> list[dict[str, Any]]:
    root = ET.fromstring(_read_text(source)); size = root.find("size")
    width = size.findtext("width", default="") if size is not None else ""; height = size.findtext("height", default="") if size is not None else ""
    rows = []
    for index, obj in enumerate(root.findall("object"), 1):
        raw = obj.findtext("name", default="unknown").strip(); label = class_map.get(raw, raw) if class_map else raw; box = obj.find("bndbox")
        values = {field: box.findtext(field, default="") if box is not None else "" for field in ("xmin", "ymin", "xmax", "ymax")}
        rows.append(_row(source_format="voc_xml", source_label=label, dataset_id=dataset_id, label_namespace=label_namespace, image_path=image_path or root.findtext("filename", default=""), class_id=raw, image_width=width, image_height=height, difficult=obj.findtext("difficult", default=""), truncated=obj.findtext("truncated", default=""), object_index=index, raw_source_label=raw, **values))
    return rows


def parse_yolo_txt(source: str | Path | bytes, *, image_width: int, image_height: int, image_path: str = "", class_names: Sequence[str] | Mapping[Any, str] | None = None, dataset_id: str = "unknown", label_namespace: str = "") -> list[dict[str, Any]]:
    names = class_names or {}; rows = []
    for index, line in enumerate(_read_text(source).splitlines(), 1):
        parts = line.strip().replace(",", " ").split()
        if not parts or parts[0].startswith("#") or len(parts) < 5: continue
        class_id = parts[0]; values = [_float(value) for value in parts[1:5]]
        if not all(math.isfinite(value) for value in values): continue
        cx, cy, box_w, box_h = values; xmin, ymin = (cx - box_w / 2) * image_width, (cy - box_h / 2) * image_height; xmax, ymax = (cx + box_w / 2) * image_width, (cy + box_h / 2) * image_height
        try: label = names.get(int(float(class_id)), class_id) if isinstance(names, Mapping) else names[int(float(class_id))]
        except (KeyError, IndexError, TypeError, ValueError): label = class_id
        rows.append(_row(source_format="yolo_txt", source_label=label, dataset_id=dataset_id, label_namespace=label_namespace, image_path=image_path, class_id=class_id, image_width=image_width, image_height=image_height, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, yolo_cx=cx, yolo_cy=cy, yolo_width=box_w, yolo_height=box_h, object_index=index, raw_source_label=class_id))
    return rows


def parse_coco_json(source: str | Path | Mapping[str, Any], *, image_path: str = "", image_id: Any = None, dataset_id: str = "unknown", label_namespace: str = "") -> list[dict[str, Any]]:
    data = dict(source) if isinstance(source, Mapping) else json.loads(_read_text(source)); images = {item.get("id"): item for item in data.get("images", [])}; categories = {item.get("id"): item for item in data.get("categories", [])}; annotations = data.get("annotations", [])
    if image_id is not None: annotations = [item for item in annotations if item.get("image_id") == image_id]
    rows = []
    for index, annotation in enumerate(annotations, 1):
        image = images.get(annotation.get("image_id"), {}); category = categories.get(annotation.get("category_id"), {}); bbox = list(annotation.get("bbox") or [])
        if len(bbox) < 4: continue
        x, y, width, height = map(_float, bbox[:4]); label = category.get("name", annotation.get("category_id", "unknown"))
        rows.append(_row(source_format="coco_json", source_label=label, dataset_id=dataset_id, label_namespace=label_namespace, image_path=image_path or image.get("file_name", ""), class_id=annotation.get("category_id", ""), image_width=image.get("width", ""), image_height=image.get("height", ""), xmin=x, ymin=y, xmax=x + width, ymax=y + height, keypoints=annotation.get("keypoints", ""), annotation_id=annotation.get("id", ""), image_id=annotation.get("image_id", ""), segmentation=annotation.get("segmentation", ""), raw_source_label=label, object_index=index))
    return rows


def _parse_points(data: Any) -> list[Any]:
    if isinstance(data, str):
        try: data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            try: data = ast.literal_eval(data)
            except (SyntaxError, ValueError): data = []
    if isinstance(data, Mapping): data = data.get("keypoints", data.get("points", data.get("people", [data])))
    return list(data or []) if isinstance(data, (list, tuple)) else []


def parse_keypoints_json(source: str | Path | Mapping[str, Any] | Sequence[Any], *, image_path: str = "", dataset_id: str = "unknown", label_namespace: str = "") -> list[dict[str, Any]]:
    data = source if isinstance(source, (Mapping, list, tuple)) else json.loads(_read_text(source)); items = data.get("annotations", data.get("people", data.get("instances", [data]))) if isinstance(data, Mapping) else data
    rows = []
    for index, item in enumerate(items or [], 1):
        item = item if isinstance(item, Mapping) else {"keypoints": item}; label = item.get("label", item.get("category", "person")); points = item.get("keypoints", item.get("points", []))
        rows.append(_row(source_format="keypoints_json", source_label=label, dataset_id=dataset_id, label_namespace=label_namespace, image_path=image_path or item.get("image_path", ""), class_id=item.get("class_id", ""), keypoints=json.dumps(points, ensure_ascii=False), person_id=item.get("person_id", ""), person_index=index, raw_source_label=label))
    return rows


def parse_keypoints_csv(source: str | Path, *, dataset_id: str = "unknown", label_namespace: str = "") -> list[dict[str, Any]]:
    with Path(source).open("r", encoding="utf-8-sig", newline="") as handle: source_rows = list(csv.DictReader(handle))
    return [_row(source_format="keypoints_csv", source_label=row.get("label", row.get("category", "person")), dataset_id=dataset_id, label_namespace=label_namespace, image_path=row.get("image_path", row.get("file_name", "")), class_id=row.get("class_id", ""), keypoints=row.get("keypoints", row.get("points", "")), person_id=row.get("person_id", ""), person_index=index, raw_source_label=row.get("label", row.get("category", "person"))) for index, row in enumerate(source_rows, 1)]


def write_unified_annotations(records: Iterable[Mapping[str, Any]], output_path: str | Path) -> Path:
    rows = [dict(record) for record in records]; target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    if not rows: target.write_text("", encoding="utf-8-sig"); return target
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    return target


__all__ = ["parse_voc_xml", "parse_yolo_txt", "parse_coco_json", "parse_keypoints_json", "parse_keypoints_csv", "write_unified_annotations"]
