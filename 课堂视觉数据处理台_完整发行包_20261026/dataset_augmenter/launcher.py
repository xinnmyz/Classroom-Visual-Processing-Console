"""Launch the Streamlit app from a normal Python process or a PyInstaller EXE."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _app_root() -> Path:
    # In a one-folder PyInstaller build the bundled files sit next to the EXE.
    # In a one-file build PyInstaller extracts data under _MEIPASS.
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def main() -> None:
    root = _app_root()
    app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"找不到程序文件：{app_path}")
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    port = _find_port(8502)
    threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()
    # Streamlit's CLI entry point works in both ordinary Python and a bundled
    # executable.  The browser is opened by the caller's normal browser after
    # the local server is ready; headless keeps the EXE window uncluttered.
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    stcli.main()


def _find_port(start: int, limit: int = 20) -> int:
    for port in range(int(start), int(start) + int(limit)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.15)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("没有找到可用的本地端口")


def _open_browser_when_ready(port: int) -> None:
    for _ in range(80):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                webbrowser.open(f"http://localhost:{port}")
                return
        except OSError:
            time.sleep(0.25)


if __name__ == "__main__":
    main()
