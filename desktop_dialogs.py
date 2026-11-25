"""Hand a single HTTP picker request to the Qt event loop."""
from concurrent.futures import Future
from threading import Lock

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QWidget


class DesktopPathChooser(QObject):
    requested = Signal(str, object)

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.window = window
        self._lock = Lock()
        self._pending: Future | None = None
        self._stopped = False
        self._dialog: QFileDialog | None = None
        self.requested.connect(self._show, Qt.ConnectionType.QueuedConnection)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending is not None

    def choose(self, mode: str) -> list[str]:
        # Called by HTTP worker threads. Reserve the slot before posting a Qt
        # event so rapid clicks cannot queue a second picker behind the first.
        with self._lock:
            if self._stopped or self._pending is not None:
                return []
            result = Future()
            self._pending = result
        self.requested.emit(mode, result)
        return result.result()

    def _finish(self, result: Future, paths: list[str], error: Exception | None = None):
        with self._lock:
            if result is not self._pending:
                return
            self._pending = None
            if error is None:
                result.set_result(paths)
            else:
                result.set_exception(error)

    @Slot(str, object)
    def _show(self, mode: str, result: Future):
        if result.done():
            return
        if QApplication.activeModalWidget() or QApplication.modalWindow():
            self._finish(result, [])
            return
        try:
            dialog = QFileDialog(self.window)
            self._dialog = dialog
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            dialog.setOption(QFileDialog.Option.DontUseCustomDirectoryIcons)
            if mode in {"directory", "save_directory"}:
                dialog.setWindowTitle("选择输出文件夹" if mode == "save_directory" else "选择文件夹")
                dialog.setFileMode(QFileDialog.FileMode.Directory)
                dialog.setOption(QFileDialog.Option.ShowDirsOnly)
            elif mode == "file":
                dialog.setWindowTitle("选择文件")
                dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            else:
                dialog.setWindowTitle("选择一个或多个文件")
                dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)

            def finished(code: int):
                paths = dialog.selectedFiles() if code == QDialog.DialogCode.Accepted else []
                self._dialog = None
                dialog.deleteLater()
                self._finish(result, paths)

            dialog.finished.connect(finished)
            # Do not nest the Qt event loop inside the queued HTTP callback.
            dialog.open()
        except Exception as exc:
            if self._dialog:
                self._dialog.deleteLater()
                self._dialog = None
            self._finish(result, [], exc)

    def stop(self):
        # GUI thread. Release waiting HTTP workers even if Qt has not yet
        # delivered the requested signal when the window starts shutting down.
        with self._lock:
            self._stopped = True
            result = self._pending
        if self._dialog is not None:
            self._dialog.reject()
        if result is not None:
            self._finish(result, [])
