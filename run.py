"""YOLO数据处理平台本地启动器。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"


def build_frontend() -> None:
    if (DIST / "index.html").exists():
        return

    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("首次运行需要 Node.js 18+，用于构建可视化界面。")

    if not (FRONTEND / "node_modules").exists():
        print("首次运行：正在安装界面依赖……")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True)

    print("正在构建界面……")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)


def open_browser(port: int) -> None:
    time.sleep(0.8)
    webbrowser.open(f"http://127.0.0.1:{port}")


def run_web() -> None:
    os.chdir(ROOT)
    from backend.server import serve

    port = int(os.environ.get("PROCESSING_VIEW_PORT", "8765"))
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    print(f"YOLO数据处理平台已启动：http://127.0.0.1:{port}")
    print("按 Ctrl+C 可停止服务。")
    serve(port=port, static_dir=DIST)


def main() -> None:
    os.chdir(ROOT)
    build_frontend()
    if "--web" in sys.argv[1:]:
        run_web()
        return

    from desktop import run_desktop

    port = int(os.environ.get("PROCESSING_VIEW_DESKTOP_PORT", "0"))
    print("正在打开YOLO数据处理平台……")
    run_desktop(static_dir=DIST, port=port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n工作台已停止。")
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
