"""Per-function user defaults that survive desktop ports and browser profiles."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any
import uuid

from .platform_paths import platform_config_dir


def _private_parameter(key: str) -> bool:
    return key != "remember_password" and bool(re.search(r"(?:^|_)(?:password|passphrase|token|secret)$", key, re.I))


def _parameters(raw: dict[str, Any], definition: dict[str, Any]) -> dict[str, Any]:
    private_ids = {field["id"] for field in definition.get("parameters", []) if field.get("type") == "password"}
    parameters = {}
    for key, value in raw.items():
        if not isinstance(key, str) or key in private_ids or _private_parameter(key):
            continue
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"参数 {key} 的默认值必须是文本、数字或开关。")
        if definition.get("handlerId") == "yolo.train" and key == "device":
            value = ""
        if definition.get("handlerId") == "yolo.train" and key == "training_parameters":
            try:
                training = json.loads(str(value))
            except (ValueError, TypeError) as exc:
                raise ValueError("训练默认参数格式不正确。") from exc
            if not isinstance(training, dict):
                raise ValueError("训练默认参数格式不正确。")
            value = json.dumps({**_parameters({key: value for key, value in training.items() if key != "training_parameters"}, definition), "device": ""}, ensure_ascii=False)
        parameters[key] = value
    return parameters


class FunctionDefaultsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or platform_config_dir() / "function_defaults.json"
        self._lock = threading.RLock()

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, function_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._read().get(function_id)
            return deepcopy(value) if isinstance(value, dict) else None

    def save(self, definition: dict[str, Any], raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict) or not isinstance(raw.get("paths"), dict) or not isinstance(raw.get("parameters"), dict):
            raise ValueError("默认配置应包含有效的数据路径和参数。")
        paths = {field["id"]: "" for field in definition.get("pathFields", [])}
        for key, value in raw["paths"].items():
            if not isinstance(key, str) or not isinstance(value, str) or "\x00" in value:
                raise ValueError("默认路径必须是有效的文本。")
            paths[key] = value
        parameters = _parameters({**{field["id"]: field.get("default", "") for field in definition.get("parameters", [])}, **raw["parameters"]}, definition)
        record = {"values": {"paths": paths, "parameters": parameters},
                  "savedAt": datetime.now().astimezone().isoformat(timespec="seconds"), "revision": uuid.uuid4().hex}
        with self._lock:
            records = self._read()
            records[str(definition["id"])] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(records, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return deepcopy(record)


defaults_store = FunctionDefaultsStore()
