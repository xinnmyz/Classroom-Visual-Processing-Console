"""Read-only dataset quality checks for classroom experiments."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_asset_bytes(asset: dict[str, Any]) -> bytes:
    source = Path(str(asset.get("source_path", "")))
    member = str(asset.get("member", "") or "")
    if member:
        with zipfile.ZipFile(source) as archive:
            return archive.read(member)
    return source.read_bytes()


def _split_name(asset: dict[str, Any]) -> str:
    explicit = str(asset.get("split", "") or "").strip().lower()
    if explicit:
        return explicit
    text = f"{asset.get('source_path', '')} {asset.get('member', '')}".lower().replace("\\", "/")
    for name in ("train", "val", "valid", "validation", "test"):
        if f"/{name}/" in f"/{text}/" or text.endswith(f"/{name}"):
            return "val" if name in {"val", "valid", "validation"} else name
    return "unknown"


def inspect_asset(asset: dict[str, Any], *, min_width: int = 64, min_height: int = 64, check_hash: bool = True) -> dict[str, Any]:
    """Inspect one image/video/ZIP member without changing the source."""
    row: dict[str, Any] = {
        "asset_id": asset.get("asset_id", ""),
        "dataset_name": asset.get("dataset_name", ""),
        "dataset_id": asset.get("dataset_id", "unknown"),
        "media_type": asset.get("media_type", "image"),
        "source_path": asset.get("source_path", ""),
        "member": asset.get("member", ""),
        "file_name": asset.get("file_name", Path(str(asset.get("source_path", ""))).name),
        "label": asset.get("canonical_label", asset.get("label", "")),
        "source_label": asset.get("source_label", asset.get("label", "")),
        "split": _split_name(asset),
        "width": "", "height": "", "size_bytes": asset.get("size_bytes", ""),
        "sha256": "", "status": "ok", "issue_codes": "", "error": "",
    }
    issues: list[str] = []
    try:
        data = _read_asset_bytes(asset)
        row["size_bytes"] = len(data)
        if check_hash:
            row["sha256"] = _sha256_bytes(data)
        if str(asset.get("media_type", "image")) == "image":
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                row["width"], row["height"] = image.size
        elif str(asset.get("media_type", "image")) == "video":
            if cv2 is None:
                issues.append("opencv_unavailable")
            else:
                suffix = Path(str(asset.get("file_name", "video.mp4"))).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
                    handle.write(data); handle.flush()
                    capture = cv2.VideoCapture(handle.name)
                    if not capture.isOpened():
                        issues.append("video_unreadable")
                    else:
                        row["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                        row["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                        capture.release()
        else:
            issues.append("unsupported_media_type")
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"; row["error"] = str(exc); issues.append("unreadable")
    try:
        if row["width"] != "" and int(row["width"]) < int(min_width): issues.append("width_too_small")
        if row["height"] != "" and int(row["height"]) < int(min_height): issues.append("height_too_small")
    except (TypeError, ValueError):
        issues.append("invalid_resolution")
    if not str(row.get("label", "") or "").strip() or str(row.get("label")) in {"未标注", "待指定"}:
        issues.append("missing_label")
    if issues and row["status"] == "ok": row["status"] = "warning"
    row["issue_codes"] = "|".join(dict.fromkeys(issues))
    return row


def check_dataset_quality(assets: Iterable[dict[str, Any]], *, min_width: int = 64, min_height: int = 64, check_hash: bool = True, max_assets: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect assets and flag unreadable files, duplicates and split leakage."""
    items = list(assets)
    if max_assets is not None: items = items[: max(0, int(max_assets))]
    frame = pd.DataFrame([inspect_asset(item, min_width=min_width, min_height=min_height, check_hash=check_hash) for item in items])
    if frame.empty:
        return frame, {"asset_count": 0, "issue_count": 0, "duplicate_group_count": 0, "split_leakage_count": 0}
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, value in frame["sha256"].items():
        if str(value).strip(): by_hash[str(value)].append(int(index))
    duplicate_groups = split_leakages = 0
    for indexes in by_hash.values():
        if len(indexes) < 2: continue
        duplicate_groups += 1
        for index in indexes:
            frame.at[index, "issue_codes"] = "|".join(filter(None, [str(frame.at[index, "issue_codes"]), "duplicate_content"]))
            if frame.at[index, "status"] == "ok": frame.at[index, "status"] = "warning"
        known_splits = {str(frame.at[index, "split"]) for index in indexes} - {"unknown", ""}
        if len(known_splits) > 1:
            split_leakages += 1
            for index in indexes:
                frame.at[index, "issue_codes"] = "|".join(filter(None, [str(frame.at[index, "issue_codes"]), "split_leakage"]))
    summary = {
        "asset_count": int(len(frame)), "ok_count": int((frame["status"] == "ok").sum()),
        "warning_count": int((frame["status"] == "warning").sum()), "error_count": int((frame["status"] == "error").sum()),
        "issue_count": int((frame["issue_codes"].astype(str).str.len() > 0).sum()),
        "duplicate_group_count": duplicate_groups, "split_leakage_count": split_leakages,
    }
    return frame, summary


def write_quality_report(frame: pd.DataFrame, summary: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path, json_path = root / "dataset_quality_findings.csv", root / "dataset_quality_report.xlsx", root / "dataset_quality_summary.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="findings", index=False); pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"findings_csv": csv_path, "report_xlsx": xlsx_path, "summary_json": json_path}
