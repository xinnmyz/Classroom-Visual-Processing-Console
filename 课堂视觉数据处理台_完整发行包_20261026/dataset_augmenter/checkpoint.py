"""Resumable experiment checkpoints.

The store is append-safe JSON and uses a stable condition key.  It can be
used by the Streamlit runner or a headless batch job to skip outputs that are
already present and record failures without losing the rest of a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def condition_key(asset_id: Any, frame_index: Any, condition: Any) -> str:
    return f"{asset_id}|frame={frame_index if frame_index not in (None, '') else 'image'}|condition={condition}"


def settings_fingerprint(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def output_is_valid(path: str | Path, min_bytes: int = 1) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size >= int(min_bytes)
    except OSError:
        return False


class CheckpointStore:
    """Atomic JSON checkpoint with completed and failed condition records."""

    def __init__(self, path: str | Path, *, settings: dict[str, Any] | None = None, resume: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_hash = settings_fingerprint(settings or {})
        self.state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": "",
            "settings_sha256": self.settings_hash,
            "status": "running",
            "completed": {},
            "failed": {},
        }
        if resume and self.path.exists():
            self._load()
            if self.state.get("settings_sha256") not in {"", self.settings_hash}:
                # A changed experiment must not accidentally skip outputs from
                # a different configuration.
                self.state["status"] = "settings_changed"
                self.state["completed"] = {}
                self.state["failed"] = {}

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict): self.state.update(data)
        except (OSError, json.JSONDecodeError):
            self.state["status"] = "checkpoint_corrupt"

    def is_done(self, key: str, output_path: str | Path | None = None) -> bool:
        item = self.state.get("completed", {}).get(str(key))
        if not item: return False
        if output_path is not None and not output_is_valid(output_path): return False
        return True

    def mark_done(self, key: str, record: dict[str, Any] | None = None) -> None:
        snapshot = {}
        if record:
            # Keep enough information to rebuild a manifest after an
            # interrupted run, while dropping the nested annotation objects.
            snapshot = {str(k): v for k, v in record.items() if k != "objects"}
        self.state.setdefault("completed", {})[str(key)] = {
            "output_path": str((record or {}).get("output_path", "")),
            "sample_id": (record or {}).get("sample_id", ""),
            "record": snapshot,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.state.setdefault("failed", {}).pop(str(key), None)

    def mark_failed(self, key: str, error: str, record: dict[str, Any] | None = None) -> None:
        self.state.setdefault("failed", {})[str(key)] = {
            "error": str(error),
            "sample_id": (record or {}).get("sample_id", ""),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def save(self) -> Path:
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(self.state, ensure_ascii=False, indent=2, default=str)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle: handle.write(payload)
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name): os.unlink(temp_name)
        return self.path

    def finish(self, status: str = "completed") -> Path:
        self.state["status"] = status
        return self.save()

    def completed_records(self) -> list[dict[str, Any]]:
        """Return manifest snapshots stored in the checkpoint."""
        output: list[dict[str, Any]] = []
        for item in self.state.get("completed", {}).values():
            record = item.get("record") if isinstance(item, dict) else None
            if isinstance(record, dict) and record:
                output.append(dict(record))
        return output


def load_existing_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load prior sample records for a resumed run, if available."""
    path = Path(run_dir) / "sample_manifest.csv"
    if not path.exists(): return []
    try:
        import pandas as pd
        return pd.read_csv(path).fillna("").to_dict("records")
    except Exception:  # noqa: BLE001
        return []
