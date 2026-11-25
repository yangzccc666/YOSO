"""YOLO项目平台本地启动器。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


FROZEN = bool(getattr(sys, "frozen", False))
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"


def build_frontend() -> None:
    if (DIST / "index.html").exists():
        return
    if FROZEN:
        raise RuntimeError("安装包缺少已构建的前端资源，请重新下载完整发行包。")

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
    print(f"YOLO项目平台已启动：http://127.0.0.1:{port}")
    print("按 Ctrl+C 可停止服务。")
    serve(port=port, static_dir=DIST)


def main() -> None:
    if sys.argv[1:2] == ["--internal-worker"]:
        from backend.worker_runtime import run_worker
        run_worker(sys.argv[2:])
        return
    os.chdir(ROOT)
    build_frontend()
    if "--check" in sys.argv[1:]:
        from backend.server import workspace_payload

        functions = workspace_payload().get("functions", [])
        if not functions:
            raise RuntimeError("安装包未能加载功能目录。")
        if "--require-fcitx" in sys.argv[1:]:
            from PySide6.QtCore import qVersion
            from desktop_input_method import configure_input_method, private_plugin_root

            os.environ["PROCESSING_VIEW_QT_IM_MODULE"] = "fcitx"
            report = configure_input_method()
            expected = private_plugin_root(ROOT, qVersion()).resolve()
            if not report.plugin or not Path(report.plugin).resolve().is_relative_to(expected):
                raise RuntimeError("安装包未能加载内置 Fcitx 插件。" + report.describe())
            print(report.describe())
        if "--require-ai" in sys.argv[1:]:
            import importlib
            from backend.distribution import AI_MODULES, module_present
            missing = [name for name in AI_MODULES if not module_present(name)]
            if missing:
                raise RuntimeError("AI 集成检查未通过，缺少：" + ", ".join(missing))
            for name in AI_MODULES:
                try:
                    importlib.import_module(name)
                except Exception as exc:
                    raise RuntimeError(f"AI 集成检查未通过，{name} 无法加载：{exc}") from exc
        print(f"YOLO项目平台安装包检查通过：{len(functions)} 个功能。")
        return
    if "--dependencies" in sys.argv[1:]:
        import json
        from backend.distribution import dependency_report
        print(json.dumps(dependency_report(), ensure_ascii=False, indent=2))
        return
    if "--web" in sys.argv[1:]:
        run_web()
        return

    from desktop import run_desktop

    port = int(os.environ.get("PROCESSING_VIEW_DESKTOP_PORT", "0"))
    print("正在打开YOLO项目平台……")
    run_desktop(static_dir=DIST, port=port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n工作台已停止。")
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
