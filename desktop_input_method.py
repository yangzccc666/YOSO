"""Process-local Qt input-method setup. Never modifies the user's IME service.

Qt's compose plugin also advertises 'xim', but is NOT a Fcitx/XIM bridge.
Fcitx needs its native plugin, compiled for the bundled Qt private ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


@dataclass(frozen=True)
class InputMethodReport:
    module: str = ""
    plugin: str = ""
    warning: str = ""

    def describe(self) -> str:
        if self.warning:
            return f"桌面输入法提示：{self.warning}"
        if self.plugin:
            return f"桌面输入法模式：{self.module}（匹配当前 Qt 的原生插件已加载）"
        return f"桌面输入法模式：{self.module}（保留指定配置）" if self.module else ""


def private_plugin_root(base: Path, version: str, arch: str | None = None) -> Path:
    return base / ".desktop-ime" / f"qt-{version}-{arch or platform.machine()}"


def validate_private_plugin(root: Path, version: str, arch: str | None = None) -> Path:
    """Do not load stale artifacts after a Qt/architecture upgrade."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("qt") != version or manifest.get("arch") != (arch or platform.machine()):
        raise ValueError("插件的 Qt 版本或 CPU 架构不匹配")
    filename = manifest.get("plugin", "")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".so"):
        raise ValueError("插件清单中的文件名无效")
    plugin = root / "platforminputcontexts" / filename
    if hashlib.sha256(plugin.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("输入法插件校验失败，请重新构建")
    return plugin


def load_fcitx_plugin(path: Path) -> tuple[bool, str, tuple[str, ...]]:
    # Import Gui first so plugin dependencies resolve to the same Qt as the app.
    from PySide6 import QtGui, QtDBus  # noqa: F401
    from PySide6.QtCore import QPluginLoader

    loader = QPluginLoader(str(path))
    metadata = loader.metaData()
    keys = tuple(str(key).lower() for key in metadata.get("MetaData", {}).get("Keys", []))
    if not {"fcitx", "fcitx5"}.intersection(keys):
        return False, "不是 Fcitx 原生输入法插件", keys
    if not loader.load():
        return False, loader.errorString(), keys
    # The Qt factory opens it again on QApplication construction. QPluginLoader
    # retains its PreventUnloadHint; no native IM context is instantiated here.
    return True, "", keys


def configure_input_method() -> InputMethodReport:
    if not sys.platform.startswith("linux"):
        return InputMethodReport()
    override = os.environ.get("PROCESSING_VIEW_QT_IM_MODULE", "").strip().lower()
    current = override or os.environ.get("QT_IM_MODULE", "").strip().lower()
    wants_fcitx = current in {"fcitx", "fcitx5"} or (
        os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland"
        and (not current or current in {"xim", "compose"})
        and "fcitx" in os.environ.get("XMODIFIERS", "").lower()
    )
    if override:
        os.environ["QT_IM_MODULE"] = override
        if override not in {"fcitx", "fcitx5"}:
            warning = "手动指定的 xim/compose 不是搜狗/Fcitx 中文输入通道；请取消该覆盖值。" if override in {"xim", "compose"} else ""
            return InputMethodReport(override, warning=warning)
    if not wants_fcitx:
        return InputMethodReport(current)

    try:
        from PySide6.QtCore import QCoreApplication, QLibraryInfo, qVersion

        version = qVersion()
        base = Path(__file__).resolve().parent
        private = private_plugin_root(base, version)
        candidates: list[tuple[Path, Path | None]] = []
        errors: list[str] = []
        if private.exists():
            try:
                candidates.append((validate_private_plugin(private, version), private))
            except (OSError, ValueError, TypeError) as exc:
                errors.append(str(exc))
        # Only examine existing Qt search roots, never add the system's whole Qt
        # plugin tree (which could replace graphics/platform plugins as well).
        roots = [Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)),
                 *(Path(path) for path in QCoreApplication.libraryPaths())]
        seen: set[Path] = set()
        for root in roots:
            for plugin in sorted((root / "platforminputcontexts").glob("*fcitx*.so")):
                if plugin not in seen:
                    candidates.append((plugin, None))
                    seen.add(plugin)
        for plugin, extra_root in candidates:
            try:
                loaded, error, keys = load_fcitx_plugin(plugin)
            except (OSError, RuntimeError, ImportError) as exc:
                loaded, error, keys = False, str(exc), ()
            if not loaded:
                errors.append(error)
                continue
            if extra_root is not None:
                QCoreApplication.addLibraryPath(str(extra_root))
            selected = current if current in keys else "fcitx" if "fcitx" in keys else "fcitx5"
            os.environ["QT_IM_MODULE"] = selected
            return InputMethodReport(selected, str(plugin))
        detail = f"（{'；'.join(errors)[:500]}）" if errors else ""
        return InputMethodReport(current, warning=(
            f"Qt {version} 缺少可加载的 Fcitx 原生插件{detail}，未切换到无效的 xim 模式。"
            "请按 setup/linux-ime/README.md 构建匹配插件；临时可用系统浏览器打开本次桌面界面地址，保持平台开启。"
        ))
    except (ImportError, OSError, RuntimeError) as exc:
        return InputMethodReport(current, warning=f"输入法插件检查失败：{exc}")


def fcitx_service_warning() -> str:
    """Read-only status check, called after QApplication creates the event loop."""
    try:
        from PySide6.QtDBus import QDBusConnection

        bus = QDBusConnection.sessionBus()
        if not bus.isConnected():
            return "无法连接桌面会话总线，输入法可能不可用：" + bus.lastError().message()
        # Fcitx4 uses a display-specific service; Fcitx5 uses org.fcitx.Fcitx5.
        display = os.environ.get("DISPLAY", ":0").rsplit(":", 1)[-1].split(".", 1)[0]
        services = (f"org.fcitx.Fcitx-{display}", "org.fcitx.Fcitx", "org.fcitx.Fcitx5", "org.freedesktop.portal.Fcitx")
        interface = bus.interface()
        interface.setTimeout(1000)
        for name in services:
            reply = interface.isServiceRegistered(name)
            if reply.isValid() and reply.value():
                return ""
        return "未检测到当前桌面会话中的 Fcitx 服务；平台不会自动启动或重启输入法。"
    except (ImportError, RuntimeError, AttributeError) as exc:
        return f"无法检查 Fcitx 会话服务：{exc}"
