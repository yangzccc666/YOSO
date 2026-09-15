"""PySide6 desktop shell for YOLO Data Processing Platform."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


APP_DISPLAY_NAME = "YOLO数据处理平台"
APP_PROCESS_NAME = "yolo-data-platform"


def configure_linux_input_method() -> str | None:
    """Use Fcitx's XIM bridge when bundled Qt has no matching Fcitx plugin."""
    import os
    import sys

    override = os.environ.get("PROCESSING_VIEW_QT_IM_MODULE", "").strip()
    if override:
        os.environ["QT_IM_MODULE"] = override
        return override
    if not sys.platform.startswith("linux"):
        return None
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11":
        return None
    current = os.environ.get("QT_IM_MODULE", "").lower()
    if current not in {"fcitx", "fcitx5"}:
        return None

    try:
        from PySide6.QtCore import QLibraryInfo

        plugin_root = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        input_plugins = plugin_root / "platforminputcontexts"
        has_native_fcitx = input_plugins.is_dir() and any(input_plugins.glob("*fcitx*.so"))
    except (ImportError, OSError):
        has_native_fcitx = False

    if has_native_fcitx:
        return current

    # Fcitx/Sogou publishes an XIM endpoint through XMODIFIERS. PySide6 ships
    # a version-matched XIM plugin even when its bundle omits the Fcitx plugin.
    os.environ.setdefault("XMODIFIERS", "@im=fcitx")
    os.environ["QT_IM_MODULE"] = "xim"
    return "xim"


@dataclass
class DialogRequest:
    finished: threading.Event = field(default_factory=threading.Event)
    paths: list[str] = field(default_factory=list)
    error: Exception | None = None


def run_desktop(static_dir: Path, port: int = 0) -> int:
    input_method = configure_linux_input_method()
    if input_method:
        print(f"桌面输入法模式：{input_method}", flush=True)
    try:
        from PySide6.QtCore import QObject, QUrl, Signal, Slot, Qt
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        raise RuntimeError(
            "桌面组件 PySide6 尚未安装，请先运行：pip install -r requirements.txt"
        ) from exc

    from backend.server import create_server, set_path_chooser

    # Supplying an ASCII process name prevents Linux taskbars from falling back
    # to an incorrectly decoded Python/terminal process title.
    app = QApplication.instance() or QApplication([APP_PROCESS_NAME])
    app.setApplicationName(APP_PROCESS_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setDesktopFileName(APP_PROCESS_NAME)
    app.setOrganizationName("YOLO")

    class DesktopWindow(QMainWindow):
        def __init__(self, on_close: Callable[[], None]) -> None:
            super().__init__()
            self._on_close = on_close

        def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
            self._on_close()
            super().closeEvent(event)

    server = create_server(port=port, static_dir=static_dir)
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="yolo-data-platform-server",
        daemon=True,
    )
    server_thread.start()
    host, actual_port = server.server_address[:2]
    server_url = f"http://{host}:{actual_port}/"
    print(f"桌面界面地址：{server_url}", flush=True)

    stopped = False

    def stop_server() -> None:
        nonlocal stopped
        if stopped:
            return
        stopped = True
        set_path_chooser(None)
        server.shutdown()
        server.server_close()

    window = DesktopWindow(stop_server)
    window.setWindowTitle(APP_DISPLAY_NAME)
    window.resize(1360, 860)
    window.setMinimumSize(960, 640)

    class QtPathChooser(QObject):
        requested = Signal(str, object)

        def __init__(self) -> None:
            super().__init__()
            self.requested.connect(self._show_dialog, Qt.ConnectionType.QueuedConnection)

        def choose(self, mode: str) -> list[str]:
            request = DialogRequest()
            self.requested.emit(mode, request)
            if not request.finished.wait(timeout=300):
                raise RuntimeError("文件选择窗口等待超时，请重新选择。")
            if request.error:
                raise request.error
            return request.paths

        @Slot(str, object)
        def _show_dialog(self, mode: str, request: DialogRequest) -> None:
            try:
                if mode in {"directory", "save_directory"}:
                    value = QFileDialog.getExistingDirectory(window, "选择文件夹")
                    request.paths = [value] if value else []
                elif mode == "file":
                    value, _selected_filter = QFileDialog.getOpenFileName(window, "选择文件")
                    request.paths = [value] if value else []
                else:
                    values, _selected_filter = QFileDialog.getOpenFileNames(window, "选择一个或多个文件")
                    request.paths = values
            except Exception as exc:  # pragma: no cover - native dialog failures vary by OS
                request.error = exc
            finally:
                request.finished.set()

    chooser = QtPathChooser()
    set_path_chooser(chooser.choose)

    web_view = QWebEngineView(window)
    window.setCentralWidget(web_view)
    web_view.load(QUrl(server_url))

    def show_load_error(ok: bool) -> None:
        if not ok:
            QMessageBox.critical(window, "启动失败", "桌面界面加载失败，请关闭后重新启动。")

    web_view.loadFinished.connect(show_load_error)
    app.aboutToQuit.connect(stop_server)
    window.show()

    try:
        return app.exec()
    finally:
        stop_server()
