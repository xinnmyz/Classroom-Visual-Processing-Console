from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path


def _bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def main() -> None:
    root = _bundle_root()
    app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"找不到内置程序文件：{app_path}")
    port = int(os.environ.get("CLASSROOM_APP_PORT", "8502"))
    url = f"http://localhost:{port}"
    # Streamlit starts its server inside cli.main; opening the browser shortly
    # after startup keeps the EXE double-click experience self-contained.
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
