"""Configurable local/SSH Ultralytics training with reusable scenario profiles."""

from __future__ import annotations

import json
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .handlers import RunContext
from .remote_inference import RemoteInferenceManager
from .remote_inference import manager as remote_connection_manager
from .remote_yolo_dataset import (
    is_remote_yolo_parameters,
    resolve_tested_yolo_connection,
)
from .task_manager import TaskCancelled


DEFAULT_TRAINING_VALUES: dict[str, Any] = {
    "yolo_executable": "yolo",
    "task": "detect",
    "data": "/home/dell/yzc_ws/data/Longcheng/shentou2-gw2-state/data.yaml",
    "model": "/home/dell/yzc_ws/ultralytics/yolo26n.pt",
    "epochs": 200,
    "patience": 30,
    "imgsz": 640,
    "batch": 8,
    "nbs": 128,
    "workers": 12,
    "device": "2",
    "project": "/home/dell/yzc_ws/data/Longcheng/shentou2-gw2-state",
    "run_name": "yolo26n_0915",
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 3.0,
    "translate": 0.08,
    "scale": 0.25,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "cls": 0.6,
    "box": 7.5,
    "dfl": 1.5,
    "cos_lr": True,
    "lr0": 0.005,
    "lrf": 0.05,
    "warmup_epochs": 3.0,
    "multi_scale": 0.1,
    "weight_decay": 0.0005,
    "optimizer": "SGD",
}

TRAIN_ARGUMENT_ORDER = [
    "data", "model", "epochs", "patience", "imgsz", "batch", "nbs", "workers",
    "device", "project", "run_name", "mosaic", "mixup", "copy_paste", "degrees",
    "translate", "scale", "shear", "perspective", "fliplr", "flipud", "cls", "box",
    "dfl", "cos_lr", "lr0", "lrf", "warmup_epochs", "multi_scale", "weight_decay",
    "optimizer",
]

INTEGER_RANGES = {
    "epochs": (1, 1_000_000),
    "patience": (0, 1_000_000),
    "imgsz": (32, 16384),
    "nbs": (1, 1_000_000),
    "workers": (0, 4096),
}

FLOAT_RANGES = {
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
    "degrees": (0.0, 180.0),
    "translate": (0.0, 1.0),
    "scale": (0.0, 10.0),
    "shear": (0.0, 180.0),
    "perspective": (0.0, 0.001),
    "fliplr": (0.0, 1.0),
    "flipud": (0.0, 1.0),
    "cls": (0.0, 1000.0),
    "box": (0.0, 1000.0),
    "dfl": (0.0, 1000.0),
    "lr0": (0.0, 1.0),
    "lrf": (0.0, 1.0),
    "warmup_epochs": (0.0, 100_000.0),
    "multi_scale": (0.0, 1.0),
    "weight_decay": (0.0, 1.0),
}


def _number(value: Any, key: str, minimum: float, maximum: float, integer: bool = False) -> int | float:
    label = key
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"训练参数 {label} 必须是{'整数' if integer else '数字'}。") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"训练参数 {label} 必须在 {minimum:g} 到 {maximum:g} 之间。")
    return parsed


def normalize_training_values(raw: dict[str, Any]) -> dict[str, Any]:
    values = {**DEFAULT_TRAINING_VALUES, **{key: raw[key] for key in DEFAULT_TRAINING_VALUES if key in raw}}
    for key, bounds in INTEGER_RANGES.items():
        values[key] = _number(values[key], key, *bounds, integer=True)
    for key, bounds in FLOAT_RANGES.items():
        values[key] = _number(values[key], key, *bounds)
    batch = _number(values["batch"], "batch", -1, 1_000_000)
    if batch != -1 and not (0 < batch <= 1) and not float(batch).is_integer():
        raise ValueError("训练参数 batch 应为正整数、-1（自动批量）或 0 到 1 之间的显存比例。")
    values["batch"] = int(batch) if float(batch).is_integer() else batch
    cos_lr = values["cos_lr"]
    if isinstance(cos_lr, str):
        lowered = cos_lr.strip().lower()
        if lowered not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError("训练参数 cos_lr 必须是开启或关闭。")
        values["cos_lr"] = lowered in {"true", "1", "yes", "on"}
    else:
        values["cos_lr"] = bool(cos_lr)
    for key in ("yolo_executable", "task", "data", "model", "device", "project", "run_name", "optimizer"):
        values[key] = str(values[key]).strip()
        if not values[key]:
            raise ValueError(f"训练参数 {key} 不能为空。")
    if values["task"] not in {"detect", "segment", "classify", "pose", "obb"}:
        raise ValueError("训练任务类型只能是 detect、segment、classify、pose 或 obb。")
    return values


def training_task_key(parameters: dict[str, Any]) -> str:
    """Serialize only trainings targeting the same host and output directory."""
    values = normalize_training_values(parameters)
    remote = is_remote_yolo_parameters(parameters)
    location = (
        f"{parameters.get('remote_username', '')}@{parameters.get('remote_host', '')}:{parameters.get('remote_port', 22)}"
        if remote else "local"
    )
    destination = str(Path(str(values["project"])) / str(values["run_name"]))
    return f"yolo-training:{location}:{destination}"


def build_training_command(raw: dict[str, Any]) -> list[str]:
    values = normalize_training_values(raw)
    command = [values["yolo_executable"], values["task"], "train"]
    for key in TRAIN_ARGUMENT_ORDER:
        argument_name = "name" if key == "run_name" else key
        value = values[key]
        rendered = "True" if value is True else "False" if value is False else str(value)
        command.append(f"{argument_name}={rendered}")
    return command


class TrainingProfileStore:
    def __init__(self, storage_file: Path) -> None:
        self.storage_file = storage_file
        self._lock = threading.RLock()
        self._profiles = self._load()

    @staticmethod
    def _default_profile() -> dict[str, Any]:
        return {
            "id": "general-training",
            "name": "通用训练",
            "description": "通用目标检测训练参数：最多 200 轮，验证指标连续 30 轮未提升则早停；可复制后按场景调整。",
            "values": deepcopy(DEFAULT_TRAINING_VALUES),
            "updatedAt": datetime.now().isoformat(timespec="seconds"),
        }

    def _load(self) -> list[dict[str, Any]]:
        if not self.storage_file.exists():
            return [self._default_profile()]
        try:
            content = json.loads(self.storage_file.read_text(encoding="utf-8"))
            if isinstance(content, list):
                profiles = []
                for raw in content:
                    if not isinstance(raw, dict):
                        continue
                    profiles.append(self._normalize(raw, str(raw.get("id", "")) or None))
                return profiles
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return [self._default_profile()]

    def _normalize(self, raw: dict[str, Any], profile_id: str | None = None) -> dict[str, Any]:
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("请填写训练场景名称。")
        return {
            "id": profile_id or uuid.uuid4().hex[:12],
            "name": name,
            "description": str(raw.get("description", "")).strip(),
            "values": normalize_training_values(dict(raw.get("values", {}))),
            "updatedAt": datetime.now().isoformat(timespec="seconds"),
        }

    def _save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_file)

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return {"profiles": deepcopy(self._profiles), "defaults": deepcopy(DEFAULT_TRAINING_VALUES)}

    def create(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            profile = self._normalize(raw)
            if any(item["name"] == profile["name"] for item in self._profiles):
                raise ValueError("训练场景名称已存在，请换一个名称。")
            self._profiles.append(profile)
            self._save()
            return deepcopy(profile)

    def update(self, profile_id: str, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for index, existing in enumerate(self._profiles):
                if existing["id"] != profile_id:
                    continue
                profile = self._normalize(raw, profile_id)
                if any(item["id"] != profile_id and item["name"] == profile["name"] for item in self._profiles):
                    raise ValueError("训练场景名称已存在，请换一个名称。")
                self._profiles[index] = profile
                self._save()
                return deepcopy(profile)
        raise ValueError("训练场景不存在。")

    def delete(self, profile_id: str) -> None:
        with self._lock:
            before = len(self._profiles)
            self._profiles = [item for item in self._profiles if item["id"] != profile_id]
            if len(self._profiles) == before:
                raise ValueError("训练场景不存在。")
            self._save()


@dataclass
class TrainingSession:
    id: str
    remote: bool
    model: str = ""
    device: str = ""
    output: str = ""
    host: str = ""
    port: int = 22
    username: str = ""
    remote_dir: str = ""
    status: str = "starting"
    message: str = "正在准备训练环境……"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    error: str | None = None
    result: dict[str, Any] | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    def add_log(self, message: str) -> None:
        clean = str(message).strip()
        if clean:
            self.logs.append(clean)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "remote": self.remote,
            "model": self.model,
            "device": self.device,
            "output": self.output,
            "host": self.host,
            "remoteDir": self.remote_dir,
            "status": self.status,
            "message": self.message,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "logs": list(self.logs),
            "error": self.error,
            "result": self.result,
        }


def _run_local(context: RunContext, values: dict[str, Any]) -> dict[str, Any]:
    command = build_training_command(values)
    context.report(f"本地训练命令：{shlex.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"找不到 YOLO 命令：{command[0]}。请填写正确的可执行文件路径。") from exc

    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            output_queue.put(line.rstrip())
        output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    finished_output = False
    while not finished_output or process.poll() is None:
        if context.stop_event is not None and context.stop_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
            raise TaskCancelled("YOLO 模型训练已被用户终止，已生成的训练结果会保留。")
        try:
            line = output_queue.get(timeout=0.15)
        except queue.Empty:
            continue
        if line is None:
            finished_output = True
        elif line:
            context.report(line)
    exit_code = process.wait()
    if process.stdout is not None:
        process.stdout.close()
    if exit_code != 0:
        raise RuntimeError(f"YOLO 训练异常结束（退出码 {exit_code}）。")
    return {"message": "YOLO 模型训练完成。", "project": values["project"], "name": values["run_name"]}


def _remote_capture(client: Any, command: str) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command, get_pty=False)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    return stdout.channel.recv_exit_status(), output, error


def select_remote_yolo_candidate(candidates: list[str], model: str) -> str | None:
    unique = list(dict.fromkeys(path.strip() for path in candidates if path.strip()))
    if not unique:
        return None
    family_match = re.search(r"(yolo\d+)", model.lower())
    if family_match:
        family = family_match.group(1)
        matching = [path for path in unique if f"/{family}/" in path.lower()]
        if len(matching) == 1:
            return matching[0]
    if len(unique) == 1:
        return unique[0]
    return None


def _resolve_remote_yolo_executable(
    client: Any,
    requested: str,
    model: str,
    report: Callable[[str], None],
) -> str:
    if "/" in requested:
        status, _output, _error = _remote_capture(
            client, f"test -x {shlex.quote(requested)}"
        )
        if status != 0:
            raise ValueError(f"远程 YOLO 可执行文件不存在或不可执行：{requested}")
        report(f"远程环境检查：使用指定 YOLO 命令 {requested}")
        return requested

    lookup_script = f"command -v {shlex.quote(requested)}"
    status, output, _error = _remote_capture(
        client, "bash -lc " + shlex.quote(lookup_script)
    )
    if status == 0 and output:
        resolved = output.splitlines()[0].strip()
        report(f"远程环境检查：找到 YOLO 命令 {resolved}")
        return resolved

    search_script = (
        'find "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/.conda" '
        "-maxdepth 7 -type f -name yolo -path '*/bin/yolo' -perm -u+x 2>/dev/null"
    )
    _status, output, _error = _remote_capture(
        client, "bash -lc " + shlex.quote(search_script)
    )
    candidates = output.splitlines()
    resolved = select_remote_yolo_candidate(candidates, model)
    if resolved:
        report(f"远程环境检查：{requested} 不在 PATH，已根据模型自动选择 {resolved}")
        return resolved
    if candidates:
        choices = "、".join(candidates[:8])
        raise ValueError(
            f"远程登录环境找不到“{requested}”，并发现多个 YOLO 环境：{choices}。"
            "请在“YOLO 命令或可执行文件路径”中选择一个完整路径。"
        )
    raise ValueError(
        f"远程登录环境找不到 YOLO 命令“{requested}”。"
        "请先在服务器安装 Ultralytics，或填写虚拟环境中 bin/yolo 的绝对路径。"
    )


def _check_remote_training_paths(client: Any, values: dict[str, Any]) -> None:
    data_path = str(values["data"])
    status, _output, _error = _remote_capture(client, f"test -f {shlex.quote(data_path)}")
    if status != 0:
        raise ValueError(f"远程 data.yaml 不存在或不是文件：{data_path}")

    project_path = str(values["project"])
    project_parent = str(Path(project_path).parent)
    check_project = (
        f"if test -e {shlex.quote(project_path)}; then "
        f"test -d {shlex.quote(project_path)} && test -w {shlex.quote(project_path)}; "
        f"else test -d {shlex.quote(project_parent)} && test -w {shlex.quote(project_parent)}; fi"
    )
    status, _output, _error = _remote_capture(client, check_project)
    if status != 0:
        raise ValueError(f"远程训练输出目录不可用或没有写入权限：{project_path}")

    model_path = str(values["model"])
    if "/" in model_path:
        status, _output, _error = _remote_capture(client, f"test -f {shlex.quote(model_path)}")
        if status != 0:
            raise ValueError(f"远程预训练模型或模型配置不存在：{model_path}")


def _conda_environment_for_yolo(executable: str) -> tuple[str, str] | None:
    """Return the conda activation script and environment name for an env YOLO binary."""
    match = re.fullmatch(r"(.+)/envs/([^/]+)/bin/yolo", executable)
    if not match:
        return None
    return f"{match.group(1)}/etc/profile.d/conda.sh", match.group(2)


def _remote_training_shell(command: list[str], token: str) -> tuple[str, str | None]:
    executable = command[0]
    conda_environment = _conda_environment_for_yolo(executable)
    launch = f"exec -a {shlex.quote(token)} " + " ".join(shlex.quote(part) for part in command)
    pieces = ["export PYTHONUNBUFFERED=1"]
    environment_name: str | None = None
    if conda_environment is not None:
        conda_script, environment_name = conda_environment
        pieces.extend((
            f"source {shlex.quote(conda_script)}",
            f"conda activate {shlex.quote(environment_name)}",
        ))
    model_argument = next((part.removeprefix("model=") for part in command if part.startswith("model=")), "")
    if model_argument.startswith("/"):
        pieces.append(f"cd {shlex.quote(str(Path(model_argument).parent))}")
    pieces.append(launch)
    return "bash -lc " + shlex.quote(" && ".join(pieces)), environment_name


def _remote_output_lines(buffer: bytearray, final: bool = False) -> list[str]:
    """Consume both newline and carriage-return progress output from a remote command."""
    lines: list[str] = []
    while True:
        separators = [index for index in (buffer.find(b"\n"), buffer.find(b"\r")) if index >= 0]
        if not separators:
            break
        index = min(separators)
        raw = bytes(buffer[:index])
        del buffer[:index + 1]
        while buffer and buffer[0] in {10, 13}:
            del buffer[:1]
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            lines.append(line)
    if final and buffer:
        line = bytes(buffer).decode("utf-8", errors="replace").strip()
        buffer.clear()
        if line:
            lines.append(line)
    return lines


def _validate_remote_model_archive(client: Any, values: dict[str, Any]) -> None:
    """Fail early when a relative or absolute PyTorch checkpoint already exists but is truncated."""
    model = str(values["model"])
    if not model.lower().endswith(".pt"):
        return
    if "/" in model:
        candidate = model
    else:
        script = f"candidate=\"$HOME\"/{shlex.quote(model)}; test -f \"$candidate\" && realpath \"$candidate\""
        status, output, _error = _remote_capture(client, "bash -lc " + shlex.quote(script))
        if status != 0 or not output:
            return
        candidate = output.splitlines()[0].strip()
    python_path = str(Path(str(values["yolo_executable"])).with_name("python"))
    validation = "import sys, zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 2)"
    status, _output, _error = _remote_capture(
        client,
        f"{shlex.quote(python_path)} -c {shlex.quote(validation)} {shlex.quote(candidate)}",
    )
    if status != 0:
        raise ValueError(
            f"远程模型文件无法读取或下载不完整：{candidate}。"
            "请删除损坏文件后重新下载，或在“预训练模型或模型配置”中填写有效模型的绝对路径。"
        )
    values["model"] = candidate


def _run_remote(context: RunContext, values: dict[str, Any]) -> dict[str, Any]:
    connection = resolve_tested_yolo_connection(context.parameters, "开始远程模型训练")
    paramiko = RemoteInferenceManager._load_paramiko()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    session = context.parameters.get("_training_session")
    token = f"yolo-training-{session.id}"
    channel = None
    try:
        context.report(f"正在连接训练服务器：{connection['username']}@{connection['host']}:{connection['port']}")
        try:
            client.connect(
                hostname=connection["host"], port=connection["port"],
                username=connection["username"], password=connection["password"],
                look_for_keys=False, allow_agent=False, timeout=12,
                banner_timeout=12, auth_timeout=12,
            )
        except Exception as exc:
            raise ValueError(RemoteInferenceManager._connection_error(
                exc, paramiko, connection["host"], connection["port"]
            )) from exc
        connection["password"] = ""
        context.parameters["remote_password"] = ""
        values = dict(values)
        values["yolo_executable"] = _resolve_remote_yolo_executable(
            client, values["yolo_executable"], values["model"], context.report
        )
        _check_remote_training_paths(client, values)
        _validate_remote_model_archive(client, values)
        context.report("远程环境检查完成：data.yaml 与训练输出目录可用，模型参数已通过检查。")
        command = build_training_command(values)
        remote_command, conda_environment = _remote_training_shell(command, token)
        if conda_environment:
            context.report(f"已激活远程 Conda 环境：{conda_environment}")
        context.report(f"远程训练命令：{shlex.join(command)}")
        status, _, _ = _remote_capture(client, "command -v tmux")
        if status:
            raise ValueError("远程服务器缺少 tmux，请先安装 tmux；训练未启动。")
        remote_dir = f"$HOME/.local/state/yolo-processing/training/{session.id}"
        session.remote_dir = remote_dir
        wrapper = f"{remote_command} > run.log 2>&1; result=$?; printf '%s\\n' \"$result\" > exit.code"
        launch = (
            f"mkdir -p {remote_dir} && cd {remote_dir} && "
            f"tmux new-session -d -s {shlex.quote(token)} "
            f"{shlex.quote('bash -lc ' + shlex.quote(wrapper))}"
        )
        status, _, error = _remote_capture(client, launch)
        if status:
            raise RuntimeError(f"启动远程 tmux 训练失败：{error}")
        context.parameters["_training_manager"]._save()
        context.report(f"训练已在远端 tmux 会话 {token} 中启动；关闭本机平台不会停止训练。")
        try:
            _follow_remote_session(client, session, context.report, context.stop_event)
        except (OSError, EOFError, paramiko.SSHException) as exc:
            raise RemoteDetached(f"SSH 日志连接断开：{exc}。远端 tmux 训练未被停止，请重新连接。") from exc
        return {"message": "远程 YOLO 模型训练完成。", "project": values["project"], "name": values["run_name"]}
    finally:
        connection["password"] = ""
        context.parameters["remote_password"] = ""
        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass
        client.close()


class RemoteDetached(RuntimeError):
    """Local log connection was lost; the remote tmux job remains independent."""


def _follow_remote_session(client: Any, session: TrainingSession, report: Callable[[str], None], stop_event: threading.Event | None) -> None:
    token = f"yolo-training-{session.id}"
    directory = session.remote_dir
    last_lines: list[str] = []
    while True:
        if stop_event is not None and stop_event.is_set():
            _remote_capture(client, f"tmux send-keys -t {shlex.quote(token)} C-c")
            time.sleep(0.3)
            _remote_capture(client, f"tmux kill-session -t {shlex.quote(token)}")
            raise TaskCancelled("远程训练已手动终止；已有结果保留在服务器。")
        status, output, error = _remote_capture(client, f"cd {directory} && tail -n 500 run.log 2>/dev/null")
        if status == 0:
            lines = output.splitlines()
            overlap = next((n for n in range(min(len(lines), len(last_lines)), 0, -1) if last_lines[-n:] == lines[:n]), 0)
            for line in lines[overlap:]:
                report(line)
            last_lines = lines
        status, output, _ = _remote_capture(client, f"test -f {directory}/exit.code && cat {directory}/exit.code")
        if status == 0:
            if output.strip() != "0":
                raise RuntimeError(f"远程 YOLO 训练异常结束（退出码 {output.strip()}），请查看训练日志。")
            return
        status, _, _ = _remote_capture(client, f"tmux has-session -t {shlex.quote(token)} 2>/dev/null")
        if status != 0:
            raise RuntimeError("远程 tmux 会话已结束但未写入退出码，请检查远程训练日志。")
        time.sleep(2)


def run_training(context: RunContext) -> dict[str, Any]:
    values = normalize_training_values(context.parameters)
    context.report("训练参数校验完成，正在启动 Ultralytics YOLO。")
    if is_remote_yolo_parameters(context.parameters):
        return _run_remote(context, values)
    return _run_local(context, values)


class TrainingManager:
    def __init__(self, storage_file: Path | None = None) -> None:
        self._sessions: dict[str, TrainingSession] = {}
        self._lock = threading.RLock()
        self._storage = storage_file or Path(__file__).resolve().parents[1] / "runtime" / "remote_training_sessions.json"
        self._load()

    def _save(self) -> None:
        with self._lock:
            records = [{"id": s.id, "remote": s.remote, "model": s.model, "device": s.device,
                        "output": s.output, "host": s.host, "port": s.port,
                        "username": s.username, "remote_dir": s.remote_dir,
                        "started_at": s.started_at, "status": s.status,
                        "message": s.message, "finished_at": s.finished_at,
                        "error": s.error, "logs": list(s.logs)}
                       for s in self._sessions.values() if s.status in {"completed", "failed", "stopped"}
                       or (s.remote and s.remote_dir)]
            self._storage.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._storage.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self._storage)

    def _load(self) -> None:
        try:
            records = json.loads(self._storage.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for record in records if isinstance(records, list) else []:
            try:
                remote = bool(record.get("remote", True))
                previous_status = record.get("status", "running")
                terminal = previous_status in {"completed", "failed", "stopped"}
                session = TrainingSession(
                    id=record["id"], remote=remote, model=record["model"],
                    device=record["device"], output=record["output"],
                    host=record["host"], port=int(record["port"]),
                    username=record["username"], remote_dir=record["remote_dir"],
                    started_at=record["started_at"],
                    status=previous_status if terminal else "disconnected" if remote else "stopped",
                    message=record.get("message", "") if terminal else
                    "远程训练可能仍在运行；请用当前 SSH 凭据重新连接以查看日志。" if remote else
                    "本地训练已随平台关闭而停止。",
                    finished_at=record.get("finished_at") if terminal else None,
                    error=record.get("error") if terminal else None,
                    logs=deque(record.get("logs", []), maxlen=500),
                )
                self._sessions[session.id] = session
            except (KeyError, TypeError, ValueError):
                continue

    def reconnect(self, session_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        session = self.get(session_id)
        if not session.remote or not session.remote_dir:
            raise ValueError("该任务不是已启动的远程 tmux 训练。")
        if (str(parameters.get("remote_host", "")).strip(), int(parameters.get("remote_port", 22)),
                str(parameters.get("remote_username", "")).strip()) != (session.host, session.port, session.username):
            raise ValueError("SSH 连接信息与该训练任务的服务器不一致。")
        connection = remote_connection_manager.resolve_connection({
            "host": session.host, "port": session.port, "username": session.username,
            "password": parameters.get("remote_password", ""),
            "remember_password": parameters.get("remember_password", True),
        })
        paramiko = RemoteInferenceManager._load_paramiko()
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=session.host, port=session.port, username=session.username,
                           password=connection["password"], look_for_keys=False, allow_agent=False, timeout=12)
        finally:
            connection["password"] = ""
            client.close()
        if session.status == "disconnected":
            session.status = "running"
            session.message = "已重新连接远程 tmux 训练。"
            def follow() -> None:
                connected = remote_connection_manager.resolve_connection({
                    "host": session.host, "port": session.port, "username": session.username,
                    "password": parameters.get("remote_password", ""), "remember_password": parameters.get("remember_password", True),
                })
                ssh = paramiko.SSHClient()
                ssh.load_system_host_keys()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                try:
                    ssh.connect(hostname=session.host, port=session.port, username=session.username,
                                password=connected["password"], look_for_keys=False, allow_agent=False, timeout=12)
                    connected["password"] = ""
                    _follow_remote_session(ssh, session, session.add_log, session.stop_event)
                    session.status, session.message = "completed", "远程 YOLO 模型训练完成。"
                except TaskCancelled as exc:
                    session.status, session.message = "stopped", str(exc)
                except Exception as exc:
                    session.status, session.message = "disconnected", f"远程状态暂不可用：{exc}；训练未被停止。"
                finally:
                    connected["password"] = ""
                    ssh.close()
                    if session.status in {"completed", "stopped"}:
                        session.finished_at = datetime.now().isoformat(timespec="seconds")
                        self._save()
            threading.Thread(target=follow, name=f"reconnect-training-{session.id}", daemon=True).start()
        return session.snapshot()

    def start(self, parameters: dict[str, Any], on_finished: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        values = normalize_training_values(parameters)
        remote = is_remote_yolo_parameters(parameters)
        if remote:
            resolve_tested_yolo_connection(parameters, "开始远程模型训练")["password"] = ""
        session = TrainingSession(
            id=uuid.uuid4().hex[:16], remote=remote,
            model=str(values["model"]), device=str(values["device"]),
            output=str(Path(str(values["project"])) / str(values["run_name"])),
            host=str(parameters.get("remote_host", "")) if remote else "本地",
            port=int(parameters.get("remote_port", 22)) if remote else 22,
            username=str(parameters.get("remote_username", "")) if remote else "",
        )
        with self._lock:
            if any(existing.remote == session.remote and existing.host == session.host
                   and existing.output == session.output and existing.status not in {"completed", "stopped", "failed"}
                   for existing in self._sessions.values()):
                raise ValueError("同一服务器上的该训练输出目录已有活动任务；请先确认远程任务状态，或更换训练名称。")
            self._sessions[session.id] = session

        def worker() -> None:
            context = RunContext(
                function_id="yolo_dataset_split",
                paths={},
                parameters={**parameters, "_training_session": session, "_training_manager": self},
                report=session.add_log,
                stop_event=session.stop_event,
            )
            session.status = "running"
            session.message = "YOLO 模型训练正在运行。"
            try:
                session.result = run_training(context)
                session.status = "completed"
                session.message = str(session.result.get("message", "训练完成。"))
            except TaskCancelled as exc:
                session.status = "stopped"
                session.message = str(exc)
            except RemoteDetached as exc:
                session.status = "disconnected"
                session.message = str(exc)
            except Exception as exc:
                session.status = "failed"
                session.error = str(exc)
                session.message = "训练失败。"
                session.add_log(f"错误：{exc}")
            finally:
                parameters["remote_password"] = ""
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                self._save()
                if on_finished is not None:
                    on_finished(session.snapshot())

        threading.Thread(target=worker, name=f"yolo-training-{session.id}", daemon=True).start()
        return session.snapshot()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        sessions.sort(key=lambda item: item.started_at, reverse=True)
        active = [item for item in sessions if item.status not in {"completed", "stopped", "failed"}]
        finished = [item for item in sessions if item.status in {"completed", "stopped", "failed"}][:10]
        return [{key: value for key, value in item.snapshot().items() if key != "logs"} for item in [*active, *finished]]

    def get(self, session_id: str) -> TrainingSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("训练任务不存在或已过期。")
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("训练任务不存在或已被删除。")
            if session.status not in {"completed", "stopped", "failed"}:
                raise ValueError("训练尚未结束，请先终止训练并等待停止完成，不能删除运行中的任务记录。")
            del self._sessions[session_id]
            try:
                self._save()
            except OSError:
                self._sessions[session_id] = session
                raise

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if session.status in {"completed", "stopped", "failed"}:
            return session.snapshot()
        if session.remote and session.status == "disconnected":
            raise ValueError("远端连接已断开，请先重新连接，再手动终止训练。")
        session.status = "stopping"
        session.message = "正在终止训练……"
        session.stop_event.set()
        return session.snapshot()

    def stop_all(self) -> None:
        with self._lock:
            ids = [s.id for s in self._sessions.values() if not s.remote]
        for session_id in ids:
            self.stop(session_id)


manager = TrainingManager()
