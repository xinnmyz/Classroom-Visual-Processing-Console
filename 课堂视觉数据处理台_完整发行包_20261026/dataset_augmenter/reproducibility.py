"""Reproducibility metadata and output checksums for each experiment run."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

APP_VERSION = "2026.10.26"
SCHEMA_VERSION = "1.0"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a whole media file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_signature(assets: Iterable[dict[str, Any]]) -> str:
    rows = []
    for asset in assets:
        rows.append(
            {
                "asset_id": asset.get("asset_id", ""),
                "source_path": asset.get("source_path", ""),
                "member": asset.get("member", ""),
                "dataset_name": asset.get("dataset_name", ""),
                "origin_url": asset.get("origin_url", ""),
                "size_bytes": asset.get("size_bytes", ""),
            }
        )
    payload = json.dumps(sorted(rows, key=lambda row: str(row["asset_id"])), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_reproducibility_artifacts(
    run_dir: str | Path,
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    selected_assets: Iterable[dict[str, Any]] | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a machine-readable config, run metadata and output checksums.

    Metadata files are excluded from their own checksum list to avoid a
    self-referential hash. The input signature is derived from stable source
    metadata and does not read every raw dataset byte.
    """
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    assets = list(selected_assets or [])
    completed = [record for record in records if record.get("status") == "完成"]
    files: list[dict[str, Any]] = []
    excluded = {"reproducibility.json", "experiment_config.json", "output_hashes.csv"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    input_rows = [
        {
            key: asset.get(key, "")
            for key in ("asset_id", "media_type", "source_path", "member", "dataset_name", "origin_name", "origin_url", "size_bytes")
        }
        for asset in assets
    ]
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "created_at_utc": now,
        "python_version": sys.version,
        "platform": platform.platform(),
        "run_id": settings.get("run_id", root.name),
        "random_seed": settings.get("random_seed", ""),
        "seed_mode": settings.get("seed_mode", ""),
        "dataset_version": settings.get("dataset_version", "未提供；以来源链接和输入签名为准"),
        "input_asset_count": len(input_rows),
        "input_completed_record_count": len(completed),
        "input_signature_sha256": _input_signature(assets),
        "model_metadata": model_metadata or {"status": "未提供；处理阶段未运行人物分析模型"},
        "settings": settings,
        "input_assets": input_rows,
        "output_files": files,
    }
    (root / "reproducibility.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "run_id": metadata["run_id"],
        "created_at_utc": now,
        "random_seed": metadata["random_seed"],
        "dataset_version": metadata["dataset_version"],
        "model_metadata": metadata["model_metadata"],
        "settings": settings,
    }
    (root / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    import csv

    with (root / "output_hashes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(files)
    return metadata

