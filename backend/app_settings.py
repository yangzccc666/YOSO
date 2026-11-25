"""Per-computer platform settings and configurable external app launcher."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from .platform_paths import platform_config_dir
from .work_clock import DEFAULT_WORK_CLOCK, calendar_payload, normalize_work_clock


APP_DEFINITIONS: dict[str, dict[str, str]] = {
    "xanylabeling": {
        "name": "X-AnyLabeling", "description": "图片标注与人工校正",
        "executable": "", "workingDirectory": "", "arguments": "",
        "condaEnvironment": "", "condaExecutable": "",
    },
}
_ALLOWED_FIELDS = {"executable", "workingDirectory", "arguments", "condaEnvironment", "condaExecutable"}
_APP_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')


def _app_defaults(app_id):
    return {"name": "Antigravity" if app_id == "antigravity" else "", "description": "",
            **{key: "" for key in _ALLOWED_FIELDS}, **APP_DEFINITIONS.get(app_id, {})}


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _resolve_command(value: str, label: str) -> str:
    expanded = os.path.expandvars(value.strip())
    if not expanded:
        raise ValueError(f"请先在设置中填写{label}。")
    path = _expand_path(expanded)
    if path.is_absolute() or len(path.parts) > 1:
        if not path.is_file():
            raise ValueError(f"{label}不存在：{path}")
        if not os.access(path, os.X_OK):
            raise ValueError(f"{label}不可执行：{path}")
        return str(path.resolve())
    resolved = shutil.which(expanded)
    if not resolved:
        raise ValueError(f"未在 PATH 中找到{label}：{expanded}")
    return resolved


def _find_conda(configured: str) -> str:
    if configured.strip():
        return _resolve_command(configured, "Conda 可执行文件")
    detected = shutil.which("conda")
    if detected:
        return detected
    for candidate in (
        Path.home() / "miniconda3/bin/conda", Path.home() / "anaconda3/bin/conda",
        Path.home() / "mambaforge/bin/conda", Path.home() / "miniforge3/bin/conda",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ValueError("未找到 Conda，请在外部工具设置中填写 Conda 可执行文件。")


class AppSettingsStore:
    def __init__(self, storage_file: Path | None = None) -> None:
        self.storage_file = storage_file or platform_config_dir() / "settings.json"
        self.log_directory = self.storage_file.parent / "logs"
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def payload(self) -> dict[str, Any]:
        with self._lock:
            saved = self._load()
            apps = []
            for app_id, values in saved.items():
                if _APP_ID.fullmatch(app_id) and isinstance(values, dict):
                    apps.append({"id": app_id, **_app_defaults(app_id),
                                 **{key: str(value) for key, value in values.items()
                                    if key in _ALLOWED_FIELDS | {"name", "description"}}})
            return {"externalApps": apps, "configFile": str(self.storage_file), "aiPython": self._ai_python(),
                    "theme": "dark" if self._document().get("theme") == "dark" else "light"}

    def _ai_python(self) -> str:
        try:
            raw = json.loads(self.storage_file.read_text(encoding="utf-8"))
            return str(raw.get("aiPython", "")).strip() if isinstance(raw, dict) else ""
        except (OSError, json.JSONDecodeError):
            return ""

    def _document(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.storage_file.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def work_clock_payload(self) -> dict[str, Any]:
        with self._lock:
            try:
                settings = normalize_work_clock(self._document().get("workClock", {}))
            except ValueError:
                settings = deepcopy(DEFAULT_WORK_CLOCK)
            return {"settings": settings, "calendar": calendar_payload()}

    def update_work_clock(self, raw: dict[str, Any]) -> dict[str, Any]:
        settings = normalize_work_clock(raw)
        with self._lock:
            document = self._document()
            # Migrate the historical app-id-at-root form without losing paths.
            if "externalApps" not in document:
                document = {"externalApps": {key: document[key] for key in (*APP_DEFINITIONS, 'antigravity') if key in document},
                            "aiPython": document.get("aiPython", "")}
            document["workClock"] = settings
            self._write_document(document)
            return self.work_clock_payload()

    def update(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("本机设置格式不正确。")
        apps = raw.get("externalApps")
        if not isinstance(apps, list) or len(apps) > 40:
            raise ValueError("外部工具设置格式不正确。")
        normalized: dict[str, dict[str, str]] = {}
        for item in apps:
            if not isinstance(item, dict):
                raise ValueError("外部工具设置格式不正确。")
            app_id = str(item.get("id", "")).strip()
            if not _APP_ID.fullmatch(app_id):
                raise ValueError("外部工具 ID 只能包含字母、数字、横线和下划线，长度不超过64。")
            if app_id in normalized:
                raise ValueError("外部工具 ID 重复，请重新添加工具。")
            values: dict[str, str] = {}
            for key in _ALLOWED_FIELDS | {'name', 'description'}:
                value = str(item.get(key, _app_defaults(app_id).get(key, ""))).strip()
                maximum = 80 if key == 'name' else 300 if key == 'description' else 8192
                if "\x00" in value or len(value) > maximum or (key != 'arguments' and ('\n' in value or '\r' in value)):
                    raise ValueError("外部工具设置包含无效字符或内容过长。")
                values[key] = value
            if not values['name']:
                raise ValueError("请填写外部工具的显示名称。")
            normalized[app_id] = values
        ai_python = str(raw.get("aiPython", self._ai_python())).strip()
        if "\x00" in ai_python or "\n" in ai_python or "\r" in ai_python:
            raise ValueError("AI Python 路径包含无效字符。")
        theme = raw.get("theme")
        if theme is not None and theme not in ("light", "dark"):
            raise ValueError("请选择浅色或深色主题。")
        self._save(normalized, ai_python, theme)
        return self.payload()

    def launch(self, app_id: str) -> dict[str, str]:
        if not isinstance(app_id, str) or not _APP_ID.fullmatch(app_id):
            raise ValueError("未知的外部工具。")
        with self._lock:
            settings = next((item for item in self.payload()["externalApps"] if item["id"] == app_id), None)
            if settings is None:
                raise ValueError("外部工具未配置或已删除，请在本机设置中添加并保存。")
            process = self._processes.get(app_id)
            name = settings['name']
            if process is not None and process.poll() is None:
                return {"message": f"{name} 已启动，请切换到其窗口。"}
            executable_value = str(settings["executable"]).strip()
            environment_name = str(settings["condaEnvironment"]).strip()
            executable_path = _expand_path(os.path.expandvars(executable_value))
            if environment_name and executable_value and not executable_path.is_absolute() and len(executable_path.parts) == 1:
                # Keep commands such as "python" unresolved so `conda run` uses
                # the interpreter from the selected environment, not the host.
                executable = executable_value
            else:
                executable = _resolve_command(executable_value, f"{name} 启动程序")
            working_raw = str(settings["workingDirectory"]).strip()
            working_directory = _expand_path(working_raw).resolve() if working_raw else Path.cwd()
            if not working_directory.is_dir():
                raise ValueError(f"{name} 工作目录不存在：{working_directory}")
            arguments = [line.strip() for line in str(settings["arguments"]).splitlines() if line.strip()]
            command = [executable, *arguments]
            if environment_name:
                conda = _find_conda(str(settings["condaExecutable"]))
                command = [conda, "run", "--no-capture-output", "-n", environment_name, *command]
            self.log_directory.mkdir(parents=True, exist_ok=True)
            log = self.log_directory / f"{app_id}.log"
            environment = os.environ.copy()
            for key in (
                "ELECTRON_RUN_AS_NODE", "NODE_OPTIONS", "NODE_PATH", "QT_PLUGIN_PATH",
                "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "PYTHONHOME", "PYTHONPATH",
            ):
                environment.pop(key, None)
            with log.open("w", encoding="utf-8") as output:
                process = subprocess.Popen(
                    command, cwd=working_directory, env=environment,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self._processes[app_id] = process
        try:
            code = process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            return {"message": f"已启动 {name}，请切换到其窗口。", "log": str(log)}
        if code == 0:
            return {"message": f"已发送 {name} 打开请求。", "log": str(log)}
        detail = log.read_text(encoding="utf-8", errors="replace")[-1800:]
        raise RuntimeError(f"{name} 启动失败（{code}）。日志：{log}\n{detail}")

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.storage_file.is_file():
            return {key: {} for key in APP_DEFINITIONS}
        try:
            content = json.loads(self.storage_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {key: {} for key in APP_DEFINITIONS}
        if not isinstance(content, dict):
            return {key: {} for key in APP_DEFINITIONS}
        apps = content.get("externalApps", content)
        if content.get('externalAppsVersion') == 2:
            return deepcopy(apps) if isinstance(apps, dict) else {}
        # Older versions always injected Antigravity. Hide that legacy entry
        # without modifying the source file or the user's X-AnyLabeling paths.
        return {key: deepcopy(apps.get(key, {})) if isinstance(apps, dict) else {} for key in APP_DEFINITIONS}

    def _save(self, apps: dict[str, dict[str, str]], ai_python: str = "", theme: str | None = None) -> None:
        with self._lock:
            document = self._document()
            if document.get('externalAppsVersion') != 2:
                previous = document.get('externalApps', document)
                if isinstance(previous, dict) and isinstance(previous.get('antigravity'), dict):
                    document.setdefault('hiddenLegacyExternalApps', {})['antigravity'] = previous['antigravity']
            if theme is not None:
                document["theme"] = theme
            document.update({"externalApps": apps, "externalAppsVersion": 2, "aiPython": ai_python})
            self._write_document(document)

    def _write_document(self, document: dict[str, Any]) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_file)


settings_store = AppSettingsStore()
