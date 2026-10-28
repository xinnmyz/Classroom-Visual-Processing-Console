from __future__ import annotations

import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
ANNOTATION_EXTENSIONS = {".xml", ".txt", ".json", ".csv"}
SCB_URL = "https://github.com/Whiffe/SCB-dataset"
SCB_CITATION = "Yang, Fan. SCB-dataset: A dataset for detecting student classroom behavior. arXiv:2304.02488, 2023."
IITH_URL = "https://people.iith.ac.in/vineethnb/resources/daisee/"


def asset_path(asset: dict[str, Any]) -> str:
    if asset.get("member"):
        return f"{asset.get('source_path', '')}::{asset.get('member', '')}"
    return str(asset.get("source_path", ""))


def _resolution(asset: dict[str, Any]) -> str:
    width = asset.get("annotation_width")
    height = asset.get("annotation_height")
    if width and height:
        return f"{int(width)}x{int(height)}"
    return "视频/图片分辨率待读取"


def _bbox_list(asset: dict[str, Any]) -> list[dict[str, Any]]:
    return asset.get("objects", []) or []


def _small_target(asset: dict[str, Any]) -> str:
    objects = _bbox_list(asset)
    width = asset.get("annotation_width")
    height = asset.get("annotation_height")
    if not objects or not width or not height:
        return "未计算"
    image_area = max(1, int(width) * int(height))
    for obj in objects:
        try:
            box_area = max(0, int(obj.get("xmax", 0)) - int(obj.get("xmin", 0))) * max(
                0, int(obj.get("ymax", 0)) - int(obj.get("ymin", 0))
            )
            if box_area / image_area < 0.02:
                return "是"
        except (TypeError, ValueError):
            continue
    return "否"


def _split(asset: dict[str, Any]) -> str:
    text = f"{asset.get('member', '')}/{asset.get('source_path', '')}".lower()
    for name in ("train", "training", "val", "valid", "validation", "test", "testing"):
        if f"/{name}/" in text or text.endswith(f"/{name}"):
            return {"training": "train", "validation": "val", "testing": "test"}.get(name, name)
    return "未划分"


def _display_labels(asset: dict[str, Any]) -> str:
    if asset.get("media_type") == "video" and not asset.get("annotation_member"):
        return "未标注"
    labels = asset.get("label_names") or asset.get("label_codes") or []
    return "|".join(dict.fromkeys(str(label) for label in labels))


def _task_name(asset: dict[str, Any]) -> str:
    mapped = str(asset.get("task_namespace", "")).strip()
    if mapped and mapped != "待指定":
        return mapped
    text = " ".join(
        str(asset.get(key, ""))
        for key in ("dataset_name", "origin_name", "source_group", "source_path", "member", "label")
    ).lower()
    if "daisee" in text or "engagement" in text or "boredom" in text or "投入" in text:
        return "投入度/情感状态"
    if "微表情" in text or "micro" in text:
        return "微表情"
    if "scb" in text or "课堂" in text or "行为" in text or asset.get("annotation_member"):
        return "课堂行为"
    return "待指定"


def build_dataset_master(assets: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        grouped[str(asset.get("dataset_name", "未登记"))].append(asset)

    rows: list[dict[str, Any]] = []
    for dataset_name, items in sorted(grouped.items()):
        sources = sorted({str(item.get("source_group", "")) for item in items if item.get("source_group")})
        origins = sorted({str(item.get("origin_name", "未登记")) for item in items})
        urls = sorted({str(item.get("origin_url", "")) for item in items if item.get("origin_url")})
        citations = sorted({str(item.get("citation", "待补充")) for item in items})
        licenses = sorted({str(item.get("license_or_access", "待核对")) for item in items})
        has_annotation_labels = any(item.get("annotation_member") for item in items)
        label_items = [item for item in items if item.get("annotation_member")] if has_annotation_labels else items
        labels = sorted({label for item in label_items for label in (item.get("label_names") or [])})
        resolutions = sorted({_resolution(item) for item in items if _resolution(item) != "视频/图片分辨率待读取"})
        modalities = sorted({"视频" if item.get("media_type") == "video" else "图片" for item in items})
        rows.append(
            {
                "dataset_name": dataset_name,
                "origin_name": "；".join(origins),
                "origin_url": "；".join(urls),
                "license_or_access": "；".join(licenses),
                "citation": "；".join(citations),
                "modality": "+".join(modalities),
                "resolution": "；".join(resolutions[:12]) or "待读取/多种分辨率",
                "annotation_type": "XML目标框" if any(item.get("annotation_member") for item in items) else "未提供或待确认",
                "class_count": len(labels),
                "class_names": "|".join(labels),
                "sample_count": len(items),
                "image_count": sum(item.get("media_type") == "image" for item in items),
                "video_count": sum(item.get("media_type") == "video" for item in items),
                "has_occlusion": "原始数据未明确，后续可合成",
                "has_distance": "原始数据未明确，后续可用降采样模拟",
                "has_multi_view": "是" if len(sources) > 1 else "未明确",
                "view_or_source_groups": "|".join(sources),
                "can_degradation_evaluate": "是：降分辨率、黑色像素、矩形遮挡",
                "notes": "原始遮挡/远近条件需在样本级进一步标记或通过脚本构造。",
            }
        )
    return pd.DataFrame(rows)


def _stratum_key(asset: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(asset.get("media_type", "image")),
        _display_labels(asset) or "未标注",
        _resolution(asset),
        str(asset.get("sampling_group", asset.get("source_group", "未分组"))),
        _small_target(asset),
        "未标注遮挡",
        "未标注远近",
    )


def stratified_sample_assets(assets: list[dict[str, Any]], target_per_dataset: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        by_dataset[str(asset.get("dataset_name", "未登记"))].append(asset)

    selected: list[dict[str, Any]] = []
    for dataset_name, items in sorted(by_dataset.items()):
        target = min(max(1, int(target_per_dataset)), len(items))

        def sample_group(group_items: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
            strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for item in group_items:
                strata[_stratum_key(item)].append(item)
            for values in strata.values():
                rng.shuffle(values)
            keys = list(strata)
            rng.shuffle(keys)
            chosen: list[dict[str, Any]] = []
            for key in keys:
                if len(chosen) >= quota:
                    break
                if strata[key]:
                    chosen.append(strata[key].pop())
            remaining = [item for values in strata.values() for item in values]
            rng.shuffle(remaining)
            chosen.extend(remaining[: max(0, quota - len(chosen))])
            return chosen[:quota]

        by_media: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_media[str(item.get("media_type", "image"))].append(item)
        media_names = list(by_media)
        rng.shuffle(media_names)
        base_quota = target // max(1, len(media_names))
        remainder = target % max(1, len(media_names))
        chosen: list[dict[str, Any]] = []
        for index, media_name in enumerate(media_names):
            quota = base_quota + (1 if index < remainder else 0)
            chosen.extend(sample_group(by_media[media_name], min(quota, len(by_media[media_name]))))
        if len(chosen) < target:
            selected_ids = {item.get("asset_id") for item in chosen}
            remaining = [item for item in items if item.get("asset_id") not in selected_ids]
            rng.shuffle(remaining)
            chosen.extend(remaining[: target - len(chosen)])
        selected.extend(chosen[:target])
    return selected


def build_sample_index(assets: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        rows.append(
            {
                "sample_id": asset.get("asset_id", ""),
                "file_path": asset_path(asset),
                "open_path": asset.get("source_path", ""),
                "dataset": asset.get("dataset_name", ""),
                "dataset_id": asset.get("dataset_id", "unknown"),
                "label_namespace": asset.get("label_namespace", "未登记"),
                "task": _task_name(asset),
                "origin_name": asset.get("origin_name", ""),
                "origin_url": asset.get("origin_url", ""),
                "category": _display_labels(asset),
                "source_label": asset.get("source_label", _display_labels(asset)),
                "canonical_label": asset.get("canonical_label", _display_labels(asset)),
                "qualified_label": asset.get("qualified_label", ""),
                "resolution": _resolution(asset),
                "bbox": json.dumps(_bbox_list(asset), ensure_ascii=False),
                "bbox_count": asset.get("bbox_count", 0),
                "is_small_target": _small_target(asset),
                "occlusion": "未标注",
                "distance": "未标注",
                "view": asset.get("sampling_group", asset.get("source_group", "")),
                "split": _split(asset),
                "media_type": asset.get("media_type", "image"),
                "source_kind": asset.get("source_kind", ""),
                "original_width": asset.get("annotation_width", ""),
                "original_height": asset.get("annotation_height", ""),
            }
        )
    return pd.DataFrame(rows)


def _media_type(extension: str) -> str:
    ext = extension.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in ANNOTATION_EXTENSIONS:
        return "annotation"
    if ext == ".zip":
        return "archive"
    return "other"


def _raw_spec_metadata(spec: dict[str, str], path_text: str) -> dict[str, str]:
    dataset_name = spec.get("dataset_name", "未登记")
    origin_name = spec.get("origin_name", "未登记")
    origin_url = spec.get("origin_url", "")
    lower = path_text.lower()
    if dataset_name in {"课堂行为数据集", "未登记", ""} or origin_name in {"待确认", "未登记", ""}:
        if "vocdevkit" in lower:
            return {
                "dataset_name": "VOCdevKit",
                "origin_name": "Indian Institute of Technology Hyderabad（IITH）官方数据集页面",
                "origin_url": IITH_URL,
            }
        return {"dataset_name": "SCB-dataset", "origin_name": "GitHub/Whiffe/SCB-dataset", "origin_url": SCB_URL}
    return {"dataset_name": dataset_name, "origin_name": origin_name, "origin_url": origin_url}


def build_raw_file_list(source_specs: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in source_specs:
        root = Path(spec["root_path"]).expanduser()
        if not root.exists() or not root.is_dir():
            rows.append(
                {
                    "root_path": spec["root_path"],
                    "dataset_name": spec.get("dataset_name", ""),
                    "origin_name": spec.get("origin_name", ""),
                    "origin_url": spec.get("origin_url", ""),
                    "record_type": "root",
                    "relative_path": "",
                    "archive_path": "",
                    "archive_member": "",
                    "extension": "",
                    "size_bytes": "",
                    "media_type": "missing_root",
                    "is_annotation": False,
                    "integrity_status": "根目录不存在",
                }
            )
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            meta = _raw_spec_metadata(spec, str(path))
            relative = str(path.relative_to(root))
            ext = path.suffix.lower()
            record = {
                "root_path": spec["root_path"],
                **meta,
                "record_type": "file",
                "relative_path": relative,
                "archive_path": "",
                "archive_member": "",
                "open_path": str(path),
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "media_type": _media_type(ext),
                "is_annotation": ext in ANNOTATION_EXTENSIONS,
                "integrity_status": "存在",
            }
            rows.append(record)
            if ext != ".zip":
                continue
            try:
                with zipfile.ZipFile(path) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        member_ext = Path(info.filename).suffix.lower()
                        rows.append(
                            {
                                "root_path": spec["root_path"],
                                **_raw_spec_metadata(spec, f"{path} {info.filename}"),
                                "record_type": "archive_member",
                                "relative_path": relative,
                                "archive_path": str(path),
                                "archive_member": info.filename,
                                "open_path": str(path),
                                "extension": member_ext,
                                "size_bytes": info.file_size,
                                "media_type": _media_type(member_ext),
                                "is_annotation": member_ext in ANNOTATION_EXTENSIONS,
                                "integrity_status": "ZIP成员可读取",
                            }
                        )
            except (OSError, zipfile.BadZipFile):
                rows[-1]["integrity_status"] = "ZIP无法读取"
    return pd.DataFrame(rows)


def write_report_package(
    dataset_master: pd.DataFrame,
    sample_index: pd.DataFrame,
    raw_file_list: pd.DataFrame,
    settings: dict[str, Any],
    output_root: str,
) -> Path:
    run_dir = Path(output_root).expanduser() / f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings_df = pd.DataFrame([settings])
    workbook = run_dir / "dataset_report.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        dataset_master.to_excel(writer, sheet_name="dataset_master", index=False)
        sample_index.to_excel(writer, sheet_name="sample_index", index=False)
        raw_file_list.to_excel(writer, sheet_name="raw_file_list", index=False)
        settings_df.to_excel(writer, sheet_name="settings", index=False)
    add_excel_hyperlinks(workbook)
    dataset_master.to_csv(run_dir / "dataset_master.csv", index=False, encoding="utf-8-sig")
    sample_index.to_csv(run_dir / "sample_index.csv", index=False, encoding="utf-8-sig")
    raw_file_list.to_csv(run_dir / "raw_file_list.csv", index=False, encoding="utf-8-sig")
    return run_dir


def add_excel_hyperlinks(workbook_path: Path) -> None:
    """Turn local path columns into clickable Excel links without embedding media."""
    workbook = load_workbook(workbook_path)
    path_headers = {"file_path", "open_path", "source_path", "archive_path", "output_path"}
    for worksheet in workbook.worksheets:
        headers = {cell.value: cell.column for cell in worksheet[1] if cell.value}
        for row in range(2, worksheet.max_row + 1):
            fallback = worksheet.cell(row, headers["open_path"]).value if "open_path" in headers else ""
            for header in path_headers:
                if header not in headers:
                    continue
                cell = worksheet.cell(row, headers[header])
                value = cell.value
                if not value:
                    continue
                target = str(value)
                if "::" in target:
                    target = target.split("::", 1)[0]
                if header == "file_path" and fallback:
                    target = str(fallback)
                path = Path(target)
                if path.exists():
                    # Excel on Windows handles a plain absolute path more reliably
                    # than a percent-encoded file:/// URI, especially for Chinese paths.
                    cell.hyperlink = str(path.resolve())
                    cell.style = "Hyperlink"
    workbook.save(workbook_path)


def create_portable_package_from_run(run_dir: str | Path) -> tuple[Path, Path]:
    """Package processed outputs with relative Excel links for use on another computer."""
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"运行目录不存在：{run_path}")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    package_dir = run_path.parent / f"{run_path.name}_portable_{timestamp}"
    media_dir = package_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    copied_outputs: dict[str, str] = {}
    for source in run_path.rglob("*"):
        if not source.is_file() or source.suffix.lower() in {".xlsx", ".csv"}:
            continue
        try:
            relative = source.relative_to(run_path)
        except ValueError:
            continue
        destination = media_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_outputs[str(source.resolve())] = (Path("media") / relative).as_posix()

    source_workbook = run_path / "sample_manifest.xlsx"
    portable_workbook = package_dir / "dataset_report.xlsx"
    if source_workbook.exists():
        shutil.copy2(source_workbook, portable_workbook)
        workbook = load_workbook(portable_workbook)
        for worksheet in workbook.worksheets:
            headers = {cell.value: cell.column for cell in worksheet[1] if cell.value}
            if "open_path" not in headers:
                open_column = worksheet.max_column + 1
                worksheet.cell(1, open_column).value = "open_path"
                headers["open_path"] = open_column
            output_column = headers.get("output_path")
            if not output_column:
                continue
            for row in range(2, worksheet.max_row + 1):
                output_value = worksheet.cell(row, output_column).value
                if not output_value:
                    continue
                source_key = str(Path(str(output_value)).resolve())
                relative_output = copied_outputs.get(source_key, "")
                if relative_output:
                    cell = worksheet.cell(row, headers["open_path"])
                    cell.value = relative_output
                    cell.hyperlink = relative_output
                    cell.style = "Hyperlink"
        workbook.save(portable_workbook)

    source_csv = run_path / "sample_manifest.csv"
    if source_csv.exists():
        manifest = pd.read_csv(source_csv)
        manifest["open_path"] = manifest.get("output_path", pd.Series(dtype=str)).map(
            lambda value: copied_outputs.get(str(Path(str(value)).resolve()), "") if pd.notna(value) else ""
        )
        manifest.to_csv(package_dir / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    readme = package_dir / "README.txt"
    readme.write_text(
        "便携实验包使用说明\n\n"
        "1. 请保持 dataset_report.xlsx 与 media 文件夹在同一目录。\n"
        "2. 打开 dataset_report.xlsx，在 sample_manifest 或 object_manifest 表中点击 open_path。\n"
        "3. 这些链接使用相对路径，因此整个文件夹可以复制到其他电脑。\n"
        "4. 原始数据来源和原始路径仍保留在表格中；media 文件夹保存本次实验实际生成的处理结果。\n",
        encoding="utf-8",
    )
    zip_path = Path(shutil.make_archive(str(package_dir), "zip", root_dir=package_dir.parent, base_dir=package_dir.name))
    return package_dir, zip_path
