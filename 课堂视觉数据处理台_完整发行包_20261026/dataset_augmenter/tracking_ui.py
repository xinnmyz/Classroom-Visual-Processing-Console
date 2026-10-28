"""Streamlit panel for optional multi-person tracking."""

from __future__ import annotations

import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from video_consistency import action_duration_summary

from tracking import (
    load_yolo_model,
    process_image_tracking,
    process_video_tracking,
    write_tracking_summary,
)
from tracking_metrics import evaluate_tracking, write_tracking_metrics_report


@contextmanager
def materialize_tracking_asset(asset: dict[str, Any]):
    """Yield a path for local files or ZIP members and clean temporary files."""
    source_path = Path(str(asset.get("source_path", "")))
    member = asset.get("member")
    if not member:
        yield source_path
        return
    suffix = Path(str(member)).suffix or ".bin"
    temporary_path = None
    try:
        with zipfile.ZipFile(source_path) as archive:
            data = archive.read(str(member))
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(data)
            temporary_path = Path(handle.name)
        yield temporary_path
    finally:
        if temporary_path:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def render_tracking_panel(selected_assets: list[dict[str, Any]], output_root: str | Path) -> None:
    """Render tracking controls and execute selected files on demand."""
    st.markdown("### 多人检测、跟踪与固定 person_id")
    st.caption(
        "可选安装 Ultralytics YOLO 后运行。视频优先使用 YOLO + ByteTrack，"
        "如果跟踪器不可用则回退到 IoU 稳定 ID；同一次视频运行中 person_id 会持续保持。"
    )
    if not selected_assets:
        st.info("请先在“数据概览”中完成抽样或手动选择媒体。")
        return
    tracking_assets = [
        item for item in selected_assets if item.get("media_type", "image") in {"image", "video"}
    ]
    if not tracking_assets:
        st.info("当前没有可跟踪的图片或视频。")
        return
    id_to_asset = {
        f"{item.get('asset_id', '')} | {item.get('file_name', Path(item.get('source_path', '')).name)}": item
        for item in tracking_assets
    }
    selected_labels = st.multiselect(
        "选择要跟踪的媒体（建议先选少量视频验证）",
        list(id_to_asset),
        default=list(id_to_asset)[: min(3, len(id_to_asset))],
        key="tracking_asset_selection",
    )
    controls = st.columns(4)
    with controls[0]:
        model_path = st.text_input("YOLO 模型权重", value="yolo11n.pt", key="tracking_model_path")
    with controls[1]:
        confidence = st.slider("检测置信度", 0.05, 0.95, 0.25, 0.05, key="tracking_confidence")
    with controls[2]:
        iou_threshold = st.slider("IoU 关联阈值", 0.05, 0.90, 0.25, 0.05, key="tracking_iou_threshold")
    with controls[3]:
        frame_stride = st.number_input("视频帧步长", min_value=1, max_value=60, value=1, step=1, key="tracking_frame_stride")
    max_frames = st.number_input("每个视频最多处理帧数", min_value=1, max_value=10000, value=300, step=10, key="tracking_max_frames")
    annotate = st.checkbox("保存带 person_id 框的视频/图片", value=True, key="tracking_annotate")
    st.caption("person_id 是运行内的轨迹编号；跨视频、跨运行不能直接当作同一个真实学生身份。")

    if st.button("▶ 开始多人检测与跟踪", type="primary", key="run_tracking"):
        if not selected_labels:
            st.warning("请至少选择一个媒体文件。")
            return
        try:
            with st.spinner("正在加载 YOLO 模型（首次运行可能下载权重）……"):
                model, backend = load_yolo_model(str(model_path).strip() or "yolo11n.pt")
        except Exception as exc:  # noqa: BLE001
            st.error(f"多人检测模型加载失败：{exc}")
            st.info("请先运行发行包中的 install_tracking_dependencies.bat，或在命令行执行 pip install ultralytics。")
            return
        run_dir = Path(output_root) / f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        progress = st.progress(0.0)
        for index, label in enumerate(selected_labels):
            asset = id_to_asset[label]
            try:
                with materialize_tracking_asset(asset) as local_path:
                    if asset.get("media_type", "image") == "video":
                        result = process_video_tracking(
                            local_path,
                            run_dir / "videos",
                            model,
                            confidence=float(confidence),
                            iou_threshold=float(iou_threshold),
                            max_frames=int(max_frames),
                            frame_stride=int(frame_stride),
                            annotate=bool(annotate),
                            progress_callback=lambda value: progress.progress((index + value) / len(selected_labels)),
                        )
                    else:
                        result = process_image_tracking(
                            local_path,
                            run_dir / "images",
                            model,
                            confidence=float(confidence),
                            annotate=bool(annotate),
                        )
                result.update(
                    {
                        "asset_id": asset.get("asset_id", ""),
                        "file_name": asset.get("file_name", ""),
                        "dataset_name": asset.get("dataset_name", ""),
                        "task": asset.get("task_namespace", asset.get("task", "")),
                        "media_type": asset.get("media_type", "image"),
                        "tracking_backend": backend,
                    }
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "asset_id": asset.get("asset_id", ""),
                    "file_name": asset.get("file_name", ""),
                    "media_type": asset.get("media_type", "image"),
                    "status": "失败",
                    "error": str(exc),
                })
            progress.progress((index + 1) / len(selected_labels))
        json_path, csv_path = write_tracking_summary(results, run_dir)
        result_df = pd.DataFrame(results)
        # Merge per-file frame rows into one manifest so the same person_id,
        # timestamp and box columns can be consumed by the downstream reports.
        manifest_frames: list[pd.DataFrame] = []
        for item in results:
            manifest_path = item.get("manifest_csv")
            if not manifest_path or not Path(manifest_path).exists():
                continue
            try:
                frame_df = pd.read_csv(manifest_path)
                for key in ("asset_id", "file_name", "dataset_name", "task", "media_type"):
                    frame_df[key] = item.get(key, "")
                manifest_frames.append(frame_df)
            except Exception:
                continue
        if manifest_frames:
            tracking_manifest = pd.concat(manifest_frames, ignore_index=True)
        else:
            tracking_manifest = pd.DataFrame()
        manifest_csv = run_dir / "person_tracking_manifest.csv"
        manifest_xlsx = run_dir / "person_tracking_manifest.xlsx"
        tracking_manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
        duration_rows = action_duration_summary(tracking_manifest.to_dict("records")) if not tracking_manifest.empty else []
        duration_df = pd.DataFrame(duration_rows)
        duration_csv = run_dir / "person_duration_summary.csv"
        duration_df.to_csv(duration_csv, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(manifest_xlsx, engine="openpyxl") as writer:
            tracking_manifest.to_excel(writer, sheet_name="person_tracking", index=False)
            duration_df.to_excel(writer, sheet_name="person_duration", index=False)
        st.session_state["tracking_results"] = result_df
        st.session_state["tracking_manifest"] = tracking_manifest
        st.session_state["tracking_run_dir"] = str(run_dir)
        st.success(f"跟踪完成：{run_dir}")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button("下载跟踪汇总 CSV", csv_path.read_bytes(), file_name="tracking_summary.csv", mime="text/csv")
        st.download_button("下载跟踪汇总 JSON", json_path.read_bytes(), file_name="tracking_summary.json", mime="application/json")
        st.download_button("下载 person_id 逐帧清单 CSV", manifest_csv.read_bytes(), file_name="person_tracking_manifest.csv", mime="text/csv")
        st.download_button("下载 person_id 持续时间 CSV", duration_csv.read_bytes(), file_name="person_duration_summary.csv", mime="text/csv")
        st.download_button(
            "下载 person_id 逐帧清单 Excel",
            manifest_xlsx.read_bytes(),
            file_name="person_tracking_manifest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        if not duration_df.empty:
            st.markdown("#### person_id 观测持续时间")
            st.dataframe(duration_df, use_container_width=True, hide_index=True)
        with st.expander("跟踪指标评估（需要真实 track_id 标注）", expanded=False):
            st.caption("上传包含 asset_id、frame_index、track_id、gt_x1、gt_y1、gt_x2、gt_y2 的真实标注 CSV。")
            gt_upload = st.file_uploader("上传真实跟踪标注 CSV", type=["csv"], key="tracking_ground_truth_upload")
            if gt_upload is not None and st.button("计算 IDF1/MOTA/HOTA", key="compute_tracking_metrics"):
                try:
                    gt_df = pd.read_csv(gt_upload)
                    metric_dir = Path(st.session_state.get("tracking_run_dir", run_dir)) / "tracking_metrics"
                    metric_paths = write_tracking_metrics_report(tracking_manifest, gt_df, metric_dir)
                    tracking_metrics_df = evaluate_tracking(tracking_manifest, gt_df)
                    st.session_state["tracking_metrics_df"] = tracking_metrics_df
                    st.success(f"跟踪指标报告已生成：{metric_dir}")
                    st.dataframe(tracking_metrics_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "下载跟踪指标 Excel",
                        metric_paths["xlsx"].read_bytes(),
                        file_name="tracking_metrics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_tracking_metrics",
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"跟踪指标计算失败：{exc}")
            elif st.session_state.get("tracking_metrics_df") is not None:
                st.dataframe(st.session_state["tracking_metrics_df"], use_container_width=True, hide_index=True)
    elif st.session_state.get("tracking_results") is not None:
        st.dataframe(st.session_state["tracking_results"], use_container_width=True, hide_index=True)

