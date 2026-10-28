"""Optional Streamlit panel for read-only dataset quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from quality_checks import check_dataset_quality, write_quality_report


def render_quality_panel(assets: list[dict[str, Any]], output_root: str | Path) -> None:
    st.markdown("### 数据质量检查与断点恢复准备")
    st.caption("检查只读源数据，不修改图片、视频或 ZIP；报告会记录损坏文件、分辨率、标签、重复内容和数据划分泄漏。")
    if not assets:
        st.info("请先扫描数据。")
        return
    cols = st.columns(3)
    with cols[0]:
        min_width = st.number_input("最小宽度", min_value=1, max_value=10000, value=64, step=8, key="quality_min_width")
    with cols[1]:
        min_height = st.number_input("最小高度", min_value=1, max_value=10000, value=64, step=8, key="quality_min_height")
    with cols[2]:
        max_assets = st.number_input("最多检查数量（0=全部）", min_value=0, max_value=1000000, value=0, step=100, key="quality_max_assets")
    check_hash = st.checkbox("计算 SHA-256（可发现重复内容和划分泄漏）", value=True, key="quality_check_hash")
    if st.button("🔎 开始数据质量检查", type="primary", key="run_quality_check"):
        with st.spinner("正在检查文件完整性和标签/划分问题……"):
            findings, summary = check_dataset_quality(
                assets,
                min_width=int(min_width),
                min_height=int(min_height),
                check_hash=bool(check_hash),
                max_assets=None if int(max_assets) == 0 else int(max_assets),
            )
            report_dir = Path(output_root) / "quality_check"
            paths = write_quality_report(findings, summary, report_dir)
        st.session_state["quality_findings"] = findings
        st.session_state["quality_summary"] = summary
        st.session_state["quality_report_dir"] = str(report_dir)
        st.success(f"质量检查完成：{report_dir}")
    summary = st.session_state.get("quality_summary")
    findings = st.session_state.get("quality_findings")
    if summary:
        metrics = st.columns(4)
        metrics[0].metric("检查数量", f"{summary.get('asset_count', 0):,}")
        metrics[1].metric("问题数量", f"{summary.get('issue_count', 0):,}")
        metrics[2].metric("重复内容组", f"{summary.get('duplicate_group_count', 0):,}")
        metrics[3].metric("划分泄漏组", f"{summary.get('split_leakage_count', 0):,}")
    if findings is not None:
        st.dataframe(findings.head(500), use_container_width=True, hide_index=True)
        report_dir = Path(st.session_state.get("quality_report_dir", output_root))
        for filename, label, mime in (
            ("dataset_quality_findings.csv", "下载质量问题 CSV", "text/csv"),
            ("dataset_quality_report.xlsx", "下载质量报告 Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ):
            path = report_dir / filename
            if path.exists():
                st.download_button(label, path.read_bytes(), file_name=filename, mime=mime, key=f"download_{filename}")
