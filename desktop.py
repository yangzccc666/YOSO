"""PySide6 desktop shell for YOLO Data Processing Platform."""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


APP_DISPLAY_NAME = "YOLO数据处理平台"
APP_PROCESS_NAME = "yolo-data-platform"


def configure_linux_input_method() -> str | None:
    """Use Fcitx's XIM bridge when bundled Qt has no matching Fcitx plugin."""
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
        from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Slot, Qt
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import (
            QApplication, QFileDialog, QMainWindow, QMessageBox, QStyle, QSystemTrayIcon,
        )
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        raise RuntimeError(
            "桌面组件 PySide6 尚未安装，请先运行：pip install -r requirements.txt"
        ) from exc

    server_module = importlib.import_module("backend.server")

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

    server = server_module.create_server(port=port, static_dir=static_dir)
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
    hot_reloader: QObject | None = None
    notifier: QObject | None = None

    def stop_server() -> None:
        nonlocal stopped, hot_reloader
        if stopped:
            return
        stopped = True
        if hot_reloader is not None:
            hot_reloader.stop()  # type: ignore[attr-defined]
            hot_reloader = None
        server_module.set_path_chooser(None)
        server_module.set_system_notifier(None)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)

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
    server_module.set_path_chooser(chooser.choose)

    class QtSystemNotifier(QObject):
        requested = Signal(str, str)

        def __init__(self) -> None:
            super().__init__()
            self._fallbacks: set[QMessageBox] = set()
            self._queue: list[tuple[str, str]] = []
            self._showing = False
            self._tray: QSystemTrayIcon | None = None
            if QSystemTrayIcon.isSystemTrayAvailable():
                icon = window.windowIcon()
                if icon.isNull():
                    icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                self._tray = QSystemTrayIcon(icon, window)
                self._tray.setToolTip(APP_DISPLAY_NAME)
                self._tray.show()
            self.requested.connect(self._enqueue, Qt.ConnectionType.QueuedConnection)

        def notify(self, title: str, message: str) -> None:
            self.requested.emit(str(title), str(message))

        @Slot(str, str)
        def _enqueue(self, title: str, message: str) -> None:
            self._queue.append((title, message))
            if not self._showing:
                self._show_next()

        def _show_next(self) -> None:
            if not self._queue:
                self._showing = False
                return
            self._showing = True
            title, message = self._queue.pop(0)
            if self._tray is not None and self._tray.supportsMessages():
                self._tray.showMessage(
                    title,
                    message,
                    QSystemTrayIcon.MessageIcon.Information,
                    5000,
                )
                QTimer.singleShot(5000, self._advance)
                return

            # Some Linux desktops expose no system tray. Keep a non-modal,
            # auto-closing fallback so task completion is still visible.
            popup = QMessageBox(QMessageBox.Icon.Information, title, message, parent=window)
            popup.setModal(False)
            popup.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            popup.setStandardButtons(QMessageBox.StandardButton.NoButton)
            self._fallbacks.add(popup)
            popup.finished.connect(lambda _result, item=popup: self._fallbacks.discard(item))
            popup.show()
            QTimer.singleShot(5000, popup.close)
            QTimer.singleShot(5000, self._advance)

        @Slot()
        def _advance(self) -> None:
            self._showing = False
            self._show_next()

    notifier = QtSystemNotifier()
    server_module.set_system_notifier(notifier.notify)

    web_view = QWebEngineView(window)
    window.setCentralWidget(web_view)
    web_view.load(QUrl(server_url))

    def backend_files() -> dict[str, tuple[int, int]]:
        """Return a cheap snapshot of reloadable backend Python files."""
        backend_dir = static_dir.parent.parent / "backend"
        snapshot: dict[str, tuple[int, int]] = {}
        for path in backend_dir.rglob("*.py"):
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def frontend_files() -> dict[str, tuple[int, int]]:
        """Return a snapshot of the compiled frontend served to the window."""
        snapshot: dict[str, tuple[int, int]] = {}
        if not static_dir.exists():
            return snapshot
        for path in static_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def import_fresh_server_module():
        """Import a new backend module graph while the old server stays usable."""
        for name in tuple(sys.modules):
            if name == "backend" or name.startswith("backend."):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()
        return importlib.import_module("backend.server")

    def restart_backend() -> tuple[bool, str]:
        """Replace the backend server without closing the desktop window."""
        nonlocal server, server_thread, server_module

        old_module = server_module
        try:
            fresh_module = import_fresh_server_module()
            fresh_module.set_path_chooser(chooser.choose)
            fresh_module.set_system_notifier(notifier.notify)  # type: ignore[union-attr]
        except Exception as exc:
            return False, f"后台代码加载失败，仍保留当前版本：{exc}"

        old_module.set_path_chooser(None)
        old_module.set_system_notifier(None)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)

        try:
            fresh_server = fresh_module.create_server(port=actual_port, static_dir=static_dir)
        except Exception as exc:
            # Binding or importing the new version failed after the old listener
            # stopped. Bring the known-good version back so the UI remains usable.
            try:
                old_module.set_path_chooser(chooser.choose)
                old_module.set_system_notifier(notifier.notify)  # type: ignore[union-attr]
                fallback_server = old_module.create_server(port=actual_port, static_dir=static_dir)
                fallback_thread = threading.Thread(
                    target=fallback_server.serve_forever,
                    name="yolo-data-platform-server",
                    daemon=True,
                )
                fallback_thread.start()
                server = fallback_server
                server_thread = fallback_thread
                server_module = old_module
            except Exception as fallback_exc:
                return False, f"热更新失败，服务恢复也失败：{exc}；{fallback_exc}"
            return False, f"热更新失败，已恢复上一版本：{exc}"

        fresh_thread = threading.Thread(
            target=fresh_server.serve_forever,
            name="yolo-data-platform-server",
            daemon=True,
        )
        fresh_thread.start()
        server = fresh_server
        server_thread = fresh_thread
        server_module = fresh_module
        return True, ""

    class HotReloadController(QObject):
        """Debounce file changes and apply them once the active task is idle."""

        def __init__(self) -> None:
            super().__init__()
            self._backend_snapshot = backend_files()
            self._frontend_snapshot = frontend_files()
            self._backend_pending = False
            self._frontend_pending = False
            self._last_change = 0.0
            self._waiting_logged = False
            self._timer = QTimer(self)
            self._timer.setInterval(500)
            self._timer.timeout.connect(self._poll)
            self._timer.start()

        def stop(self) -> None:
            self._timer.stop()

        @Slot()
        def _poll(self) -> None:
            if stopped:
                return

            backend_snapshot = backend_files()
            frontend_snapshot = frontend_files()
            changed = False
            if backend_snapshot != self._backend_snapshot:
                self._backend_snapshot = backend_snapshot
                self._backend_pending = True
                changed = True
            if frontend_snapshot != self._frontend_snapshot:
                self._frontend_snapshot = frontend_snapshot
                self._frontend_pending = True
                changed = True
            if changed:
                self._last_change = time.monotonic()
                self._waiting_logged = False
                return

            if not (self._backend_pending or self._frontend_pending):
                return
            if time.monotonic() - self._last_change < 0.8:
                return

            active_task = server_module.platform_task_manager.active()
            if active_task:
                if not self._waiting_logged:
                    print("检测到功能更新，当前任务结束后自动生效。", flush=True)
                    self._waiting_logged = True
                return

            backend_pending = self._backend_pending
            self._backend_pending = False
            self._frontend_pending = False
            self._waiting_logged = False

            if backend_pending:
                ok, message = restart_backend()
                if not ok:
                    QMessageBox.warning(window, "热更新未生效", message)
                    return

            # Keep the same host and port so local/session storage and all
            # remembered form values survive the refresh.
            hot_url = f"{server_url}?hotReload={int(time.time() * 1000)}"
            web_view.load(QUrl(hot_url))
            print("平台功能已热更新，无需重新启动。", flush=True)

    hot_reload_enabled = os.environ.get("PROCESSING_VIEW_HOT_RELOAD", "1").lower() not in {
        "0", "false", "off", "no",
    }
    if hot_reload_enabled:
        hot_reloader = HotReloadController()
        print("热更新已开启：功能完成后会自动生效。", flush=True)

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
