"""Per-computer platform settings and configurable external app launcher."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any


APP_DEFINITIONS: dict[str, dict[str, str]] = {
    "xanylabeling": {
        "name": "X-AnyLabeling", "description": "图片标注与人工校正",
        "executable": "python", "workingDirectory": "", "arguments": "app.py",
        "condaEnvironment": "xlabel", "condaExecutable": "",
    },
    "antigravity": {
        "name": "Antigravity", "description": "打开本地开发工具",
        "executable": "", "workingDirectory": "", "arguments": "",
        "condaEnvironment": "", "condaExecutable": "",
    },
}
_ALLOWED_FIELDS = {"executable", "workingDirectory", "arguments", "condaEnvironment", "condaExecutable"}


def platform_config_dir() -> Path:
    override = os.environ.get("PROCESSING_VIEW_CONFIG_DIR", "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return root / "yolo-data-platform"


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
            for app_id, defaults in APP_DEFINITIONS.items():
                apps.append({"id": app_id, **defaults, **saved.get(app_id, {})})
            return {"externalApps": apps, "configFile": str(self.storage_file)}

    def update(self, raw: dict[str, Any]) -> dict[str, Any]:
        apps = raw.get("externalApps")
        if not isinstance(apps, list):
            raise ValueError("外部工具设置格式不正确。")
        normalized: dict[str, dict[str, str]] = {}
        for item in apps:
            if not isinstance(item, dict):
                continue
            app_id = str(item.get("id", "")).strip()
            if app_id not in APP_DEFINITIONS:
                continue
            values: dict[str, str] = {}
            for key in _ALLOWED_FIELDS:
                value = str(item.get(key, "")).strip()
                if "\x00" in value:
                    raise ValueError("外部工具设置不能包含空字符。")
                values[key] = value
            normalized[app_id] = values
        self._save(normalized)
        return self.payload()

    def launch(self, app_id: str) -> dict[str, str]:
        if app_id not in APP_DEFINITIONS:
            raise ValueError("未知的外部工具。")
        with self._lock:
            process = self._processes.get(app_id)
            name = APP_DEFINITIONS[app_id]["name"]
            if process is not None and process.poll() is None:
                return {"message": f"{name} 已启动，请切换到其窗口。"}
            settings = next(item for item in self.payload()["externalApps"] if item["id"] == app_id)
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
            return {}
        try:
            content = json.loads(self.storage_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(content, dict):
            return {}
        apps = content.get("externalApps", content)
        return deepcopy(apps) if isinstance(apps, dict) else {}

    def _save(self, apps: dict[str, dict[str, str]]) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({"externalApps": apps}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_file)


settings_store = AppSettingsStore()
