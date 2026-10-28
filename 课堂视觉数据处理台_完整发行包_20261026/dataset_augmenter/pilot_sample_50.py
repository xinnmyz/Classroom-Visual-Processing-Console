from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

import app


SOURCE_CONFIG = r"D:\学生行为数据集1|||课堂行为数据集|||GitHub/SCB-Dataset（待确认）|||https://github.com/Whiffe/SCB-dataset"
SOURCE_ARCHIVE = "dataset 学生课堂行为 8884"
OUTPUT_ROOT = r"D:\classroom_project\pilot_50"
SEED = 20261024


def balanced_pick(records: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    labels = sorted({label for item in records for label in (item.get("label_names") or [])})
    by_label = {label: [item for item in records if label in (item.get("label_names") or [])] for label in labels}
    for items in by_label.values():
        rng.shuffle(items)
    selected_ids: set[str] = set()
    picked: list[dict] = []
    usable_labels = [label for label in labels if by_label[label]]
    while len(picked) < min(count, len(records)) and usable_labels:
        usable_labels.sort(key=lambda label: sum(label in (item.get("label_names") or []) for item in picked))
        selected = False
        for label in usable_labels:
            candidate = next((item for item in by_label[label] if item["asset_id"] not in selected_ids), None)
            if candidate is not None:
                picked.append(candidate)
                selected_ids.add(candidate["asset_id"])
                selected = True
                if len(picked) >= count:
                    break
        if not selected:
            break
    if len(picked) < count:
        remaining = [item for item in records if item["asset_id"] not in selected_ids]
        rng.shuffle(remaining)
        picked.extend(remaining[: count - len(picked)])
    return picked[:count]


def make_contact_sheet(records: list[dict], output_path: Path) -> None:
    thumbs = []
    for record in records:
        try:
            image = Image.open(record["output_path"]).convert("RGB")
            image.thumbnail((240, 135))
            canvas = Image.new("RGB", (260, 175), "white")
            x = (260 - image.width) // 2
            canvas.paste(image, (x, 5))
            draw = ImageDraw.Draw(canvas)
            label = record.get("label_codes", "")
            draw.text((8, 145), f"{record['sample_id']} {label[:22]}", fill="black")
            thumbs.append(canvas)
        except Exception:
            continue
    columns = 5
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 260, rows * 175), "#e5e7eb")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 260, (index // columns) * 175))
    sheet.save(output_path, quality=92)


def main() -> None:
    assets = app.scan_assets(SOURCE_CONFIG, True)
    source_assets = [item for item in assets if SOURCE_ARCHIVE in item["source_group"]]
    if len(source_assets) < 50:
        source_assets = assets
    picked = balanced_pick(source_assets, 50, SEED)
    run_dir, records, _ = app.run_processing(
        picked,
        OUTPUT_ROOT,
        ["原图"],
        [0.0],
        [0.0],
        1,
        "随机",
        SEED,
        "PNG",
    )
    label_counter = Counter(label for item in picked for label in (item.get("label_names") or []))
    pd.DataFrame([{"label": label, "image_count": count} for label, count in sorted(label_counter.items())]).to_csv(
        run_dir / "label_summary.csv", index=False, encoding="utf-8-sig"
    )
    make_contact_sheet(records, run_dir / "sample_50_contact_sheet.jpg")
    print(run_dir)
    print(f"picked={len(picked)} outputs={len(records)}")
    print(dict(label_counter))


if __name__ == "__main__":
    main()
