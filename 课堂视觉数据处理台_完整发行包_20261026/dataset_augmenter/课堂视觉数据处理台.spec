# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

SRC = Path(r"D:\dataset_augmenter")

# Streamlit and its data files are loaded dynamically at runtime.  Explicitly
# collect the main scientific packages used by app.py so the EXE works on a
# computer without a Python installation.
datas = []
binaries = []
hiddenimports = []
for package in (
    "streamlit", "pandas", "numpy", "PIL", "cv2", "openpyxl",
    "pyarrow", "altair", "watchdog", "requests", "sklearn",
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

# Streamlit executes the application script by path, so keep the application
# and its local helper modules as files beside the frozen launcher.
for name in (
    "app.py", "reporting.py", "model_evaluation.py", "tracking.py",
    "tracking_ui.py", "augmentations.py", "label_mapping.py",
    "video_consistency.py", "reproducibility.py", "evaluator_runner.py",
    "quality_checks.py", "quality_ui.py", "checkpoint.py",
    "metrics_detection_pose.py", "experiment_charts.py", "model_registry.py",
    "model_plugins.py", "model_plugin_template.py", "annotation_formats.py",
    "tracking_metrics.py",
):
    datas.append((str(SRC / name), "."))

a = Analysis(
    [str(SRC / "launcher_exe.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["ultralytics", "torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="课堂视觉数据处理台",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
