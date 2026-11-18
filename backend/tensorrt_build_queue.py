"""Persistent FIFO scheduling and safe reattachment to independent remote builds."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import posixpath
import json
import threading
import time
import uuid
from typing import Any, Callable

from .handlers import RunContext
from .remote_tensorrt_build import (
    _configuration, run_remote_tensorrt_build, reconnect_remote_build,
    discover_remote_builds, validate_remote_job, RemoteBuildDisconnected, RemoteBuildFailed,
)
from .remote_inference import manager as remote_connection_manager
from .task_manager import ActiveTask, PlatformTaskManager, TaskCancelled

TERMINAL = {"completed", "failed", "stopped"}
SAFE_PARAMETERS = {"host", "port", "username", "remember_password", "operation", "onnx_files",
                   "trtexec_path", "remote_workspace", "build_mode", "run_benchmark",
                   "keep_remote_files", "overwrite_output"}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class QueueEntry:
    id: str
    function_id: str
    name: str
    source: str
    host: str
    port: int
    username: str
    mode: str
    parameters: dict[str, Any] = field(repr=False)
    paths: dict[str, Path] = field(repr=False)
    status: str = "queued"
    submitted_at: str = field(default_factory=now)
    started_at: str | None = None
    finished_at: str | None = None
    task_id: str | None = None
    message: str = "等待前面的构建任务完成。"
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=500), repr=False)
    outputs: list[str] = field(default_factory=list)
    remote_job: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, str, str]:
        return self.host.casefold(), self.port, self.username, posixpath.normpath(self.source)

    def snapshot(self) -> dict[str, Any]:
        return {"id": self.id, "functionId": self.function_id, "source": self.source,
                "host": self.host, "port": self.port, "username": self.username, "buildMode": self.mode,
                "status": self.status, "submittedAt": self.submitted_at, "startedAt": self.started_at,
                "finishedAt": self.finished_at, "taskId": self.task_id, "message": self.message,
                "logs": list(self.logs)[-160:], "outputs": list(self.outputs),
                "remoteDirectory": self.remote_job.get("remoteDirectory", ""),
                "reconnectable": bool(self.remote_job)}


class TensorRTBuildQueue:
    MAX_PENDING = 200

    def __init__(self, tasks: PlatformTaskManager, *, runner: Callable | None = None,
                 resolve_connection: Callable | None = None, on_finished: Callable | None = None,
                 storage: Path | None = None, reconnect_runner: Callable | None = None,
                 history_contains: Callable[[str], bool] | None = None) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._tasks = tasks
        self._runner = runner or run_remote_tensorrt_build
        self._resolve_connection = resolve_connection or remote_connection_manager.resolve_connection
        self._on_finished = on_finished
        self._history_contains = history_contains
        self._reconnect_runner = reconnect_runner or reconnect_remote_build
        self._storage = storage
        self._detach = threading.Event()
        self._followers: list[threading.Thread] = []
        self._last_saved_log = 0.0
        self._storage_error = ""
        self._items: list[QueueEntry] = []
        self._paused = False
        self._manual_paused = False
        self._pause_kind = "none"
        self._reason = ""
        self._closed = False
        self._version = 0
        self._instance_id = uuid.uuid4().hex
        self._worker: threading.Thread | None = None
        self._load()

    def _save(self, *, closing: bool = False) -> None:
        """Caller holds the condition lock; credentials never reach the JSON file."""
        if self._storage is None:
            return
        if self._storage_error:
            raise ValueError(self._storage_error)
        if self._closed and not closing:
            return  # A former process must not overwrite a newly reopened queue.
        records = []
        for item in self._items:
            records.append({**item.snapshot(), "name": item.name,
                "parameters": {key: value for key, value in item.parameters.items() if key in SAFE_PARAMETERS},
                "remoteJob": dict(item.remote_job)})
        self._storage.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._storage.with_suffix('.tmp')
        temporary.write_text(json.dumps({"items": records, "paused": self._paused,
            "manualPaused": self._manual_paused, "pauseKind": self._pause_kind}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self._storage)

    def _load(self) -> None:
        if self._storage is None or not self._storage.exists():
            return
        try:
            payload = json.loads(self._storage.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise ValueError("记录格式无效")
            for record in payload["items"]:
                if not isinstance(record, dict):
                    raise ValueError("任务记录格式无效")
                if (not isinstance(record.get("parameters", {}), dict)
                        or not isinstance(record.get("logs", []), list)
                        or not isinstance(record.get("outputs", []), list)):
                    raise ValueError("任务参数或日志格式无效")
                state = record.get("status")
                if state not in TERMINAL | {"queued", "running", "stopping", "disconnected"}:
                    raise ValueError("任务状态无效")
                parameters = {key: value for key, value in record.get("parameters", {}).items() if key in SAFE_PARAMETERS}
                parameters.update({"host": str(record["host"]), "port": int(record["port"]),
                                   "username": str(record["username"]), "password": "",
                                   "onnx_files": str(record["source"]), "build_mode": str(record["buildMode"])})
                _configuration(RunContext(str(record["functionId"]), {}, parameters, lambda _: None), require_paths=True)
                remote_job = validate_remote_job(record["remoteJob"]) if record.get("remoteJob") else {}
                if remote_job and remote_job["source"] != str(record["source"]):
                    raise ValueError("远端任务与 ONNX 路径不匹配")
                restored_state = state if state in TERMINAL | {"queued"} else "disconnected"
                item = QueueEntry(str(record["id"]), str(record["functionId"]), str(record.get("name", "TensorRT 构建")),
                    str(record["source"]), str(record["host"]), int(record["port"]), str(record["username"]),
                    str(record["buildMode"]), parameters, {}, status=restored_state,
                    submitted_at=str(record["submittedAt"]), started_at=record.get("startedAt"),
                    finished_at=record.get("finishedAt") if state in TERMINAL else None,
                    task_id=record.get("taskId") if state in TERMINAL else None,
                    message=str(record.get("message", "")), remote_job=remote_job,
                    logs=deque((str(line) for line in record.get("logs", [])[-500:]), maxlen=500),
                    outputs=[str(path) for path in record.get("outputs", [])])
                if restored_state == "disconnected":
                    item.message = "待重新连接：远端可能仍在构建；重连只检查原任务，不会重新构建。"
                elif restored_state == "queued":
                    item.message = "已恢复等待任务；自动执行开启时会按顺序开始，缺少 SSH 凭据时会明确提示。"
                self._items.append(item)
            # Legacy files only stored 'paused', including the automatic pause
            # on shutdown. Do not mistake that old shutdown flag for user intent.
            self._manual_paused = payload.get("manualPaused") is True
            previous_kind = payload.get("pauseKind", "recovery")
            if any(item.status == "disconnected" for item in self._items):
                self._paused, self._pause_kind = True, "recovery"
                self._reason = "后续任务尚未执行：请先重连确认原任务状态；确认成功后将自动继续（手动暂停除外）。"
            elif self._manual_paused:
                self._paused, self._pause_kind = True, "manual"
                self._reason = "已恢复你手动暂停的队列，等待任务不会执行；点击继续队列或加入新任务可恢复自动执行。"
            elif previous_kind in {"error", "credentials", "storage"} and payload.get("paused"):
                self._paused, self._pause_kind = True, previous_kind
                self._reason = "后续任务尚未执行：上次因异常或凭据问题暂停；修正后点击继续队列或加入新任务。"
            else:
                self._paused, self._pause_kind, self._reason = False, "none", ""
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Preserve a damaged file for recovery instead of replacing it with
            # an empty queue and silently forgetting a live remote process.
            self._storage_error = f"无法安全恢复 TensorRT 队列，请保留并检查 {self._storage}：{exc}"
            self._paused, self._pause_kind, self._reason = True, "storage", self._storage_error

    def start_pending(self) -> None:
        """Called only after API license checks, never on module import."""
        with self._condition:
            if self._closed or self._storage_error:
                return
            # Also clean older completed cards, including a crash between
            # writing history and removing the queue record. Never discard an
            # unarchived result just because its status says 'completed'.
            if self._history_contains:
                archived = []
                for item in self._items:
                    if item.status == "completed" and item.task_id:
                        try:
                            if self._history_contains(item.task_id):
                                archived.append(item)
                        except Exception:
                            pass  # History is unavailable: preserve the result.
                if archived:
                    previous = self._items
                    self._items = [item for item in self._items if item not in archived]
                    self._version += 1
                    try:
                        self._save()
                    except OSError as exc:
                        self._items = previous
                        self._paused, self._pause_kind, self._reason = True, "storage", f"队列清理保存失败，已保留原记录：{exc}"
            if not self._closed and not self._paused and not self._storage_error and any(item.status == "queued" for item in self._items):
                self._start_worker()
                self._condition.notify_all()

    def _start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="tensorrt-build-queue", daemon=True)
            self._worker.start()

    def reconnect(self, entry_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            if self._closed:
                raise ValueError("平台正在关闭，不能重新连接。")
            if self._storage_error:
                raise ValueError(self._storage_error)
            item = next((entry for entry in self._items if entry.id == entry_id), None)
            if item is None or item.status != "disconnected":
                raise ValueError("该任务不是待重新连接状态。")
            if not item.remote_job:
                raise ValueError("该记录缺少远端任务信息，请填写原 ONNX 路径后点击“查找远端任务”。")
            self._check_identity(item, parameters)
            connection = self._resolve_connection(parameters)
            item.parameters.update({key: connection[key] for key in ("host", "port", "username", "password")})
            task = self._tasks.start(item.function_id, f"{item.name} · {posixpath.basename(item.source)}", "remote-build", task_key=f"trt:{item.id}")
            task.started_at, task.started_monotonic = item.started_at or task.started_at, None
            item.status, item.task_id = "running", task.id
            item.message = "正在重新连接原远端任务；不会重新执行构建。"
            self._version += 1
            try:
                self._save()
            except Exception:
                item.status, item.task_id = "disconnected", None
                item.parameters["password"] = ""
                self._tasks.finish(task.id)
                raise
            # A user-supplied password during reattachment is also valid for
            # waiting tasks on this exact device. Keep it in memory only, so
            # auto-continuation does not immediately ask for the same password.
            for pending in self._items:
                if pending.status == "queued" and pending.key[:3] == item.key[:3]:
                    pending.parameters["password"] = connection["password"]
            thread = threading.Thread(target=self._run_entry, args=(item, task, True), name=f"reconnect-trt-{item.id}", daemon=True)
            self._followers = [follower for follower in self._followers if follower.is_alive()]
            self._followers.append(thread)
            thread.start()
            return self.snapshot()

    @staticmethod
    def _check_identity(item: QueueEntry, parameters: dict[str, Any]) -> None:
        if (str(parameters.get("host", "")).strip().casefold(), int(parameters.get("port") or 22),
                str(parameters.get("username", "")).strip()) != (item.host.casefold(), item.port, item.username):
            raise ValueError(f"请先在上方选择此任务的 SSH 设备：{item.username}@{item.host}:{item.port}，再重连。")

    def discover(self, function_id: str, name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        parameters = deepcopy(parameters)
        connection = self._resolve_connection(parameters)
        parameters.update({key: connection[key] for key in ("host", "port", "username", "password")})
        result = discover_remote_builds(parameters)
        added = 0
        with self._condition:
            if self._closed:
                raise ValueError("平台正在关闭，不能找回任务。")
            for state in result["jobs"]:
                existing = next((item for item in self._items if (item.host, item.port, item.username, item.remote_job.get("remoteDirectory")) ==
                                (connection["host"], connection["port"], connection["username"], state["remoteDirectory"])), None)
                if existing:
                    continue
                # At most one unresolved build per source: an old directory
                # must not replace the active record for a newer build.
                if any(item.key == (connection["host"].casefold(), connection["port"], connection["username"], state["source"])
                       and item.status not in TERMINAL for item in self._items):
                    continue
                if sum(item.status not in TERMINAL for item in self._items) >= self.MAX_PENDING:
                    break
                copied = {key: value for key, value in parameters.items() if key in SAFE_PARAMETERS}
                copied["onnx_files"] = state["source"]
                item = QueueEntry(uuid.uuid4().hex[:16], function_id, name, state["source"], connection["host"],
                    connection["port"], connection["username"], str(parameters.get("build_mode") or "自动最佳（--best）"),
                    copied, {}, status="disconnected", remote_job=state,
                    message="已找回远端任务信息，请点击重新连接检查日志和最终结果。")
                self._items.append(item)
                added += 1
            if added:
                self._paused = True
                self._pause_kind = "recovery"
                self._reason = "已找回远端任务；请先重新连接，确认原任务成功结束后自动继续（手动暂停除外）。"
                self._version += 1
                self._save()
            message = f"已找回 {added} 个任务。"
            if result["skipped"]:
                message += f"另有 {result['skipped']} 个旧目录缺少元数据；如需找回，请在 ONNX 路径框填入原路径后再次查找。"
            return {**self.snapshot(), "message": message}

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {"instanceId": self._instance_id, "version": self._version, "paused": self._paused, "pauseReason": self._reason,
                    "pauseKind": self._pause_kind,
                    "closed": self._closed, "queuedCount": sum(item.status == "queued" for item in self._items),
                    "runningCount": sum(item.status in {"running", "stopping"} for item in self._items),
                    "disconnectedCount": sum(item.status == "disconnected" for item in self._items),
                    "items": [item.snapshot() for item in self._items]}

    def enqueue(self, function_id: str, name: str, paths: dict[str, Path], parameters: dict[str, Any]) -> dict[str, Any]:
        parameters = deepcopy(parameters)
        parameters["operation"] = "构建 TensorRT 引擎"
        context = RunContext(function_id, dict(paths), parameters, lambda _: None)
        config = _configuration(context, require_paths=True)
        # Resolve the remembered-password marker now, not after the user changes devices.
        connection = self._resolve_connection(parameters)
        parameters.update({key: connection[key] for key in ("host", "port", "username", "password")})
        with self._condition:
            if self._closed:
                raise ValueError("平台正在关闭，不能再加入构建队列。")
            if self._storage_error:
                raise ValueError(self._storage_error)
            occupied = {item.key for item in self._items if item.status not in TERMINAL}
            entries, skipped = [], []
            for job in config["jobs"]:
                source = str(job["source"])
                key = connection["host"].casefold(), connection["port"], connection["username"], posixpath.normpath(source)
                if key in occupied:
                    skipped.append(source)
                    continue
                occupied.add(key)
                copied = deepcopy(parameters)
                copied["onnx_files"] = source
                entries.append(QueueEntry(uuid.uuid4().hex[:16], function_id, name, source,
                                          connection["host"], connection["port"], connection["username"],
                                          config["mode"], copied, dict(paths)))
            if sum(item.status == "queued" for item in self._items) + len(entries) > self.MAX_PENDING:
                raise ValueError(f"等待队列最多保留 {self.MAX_PENDING} 个 ONNX，请等待部分任务完成后再添加。")
            self._items.extend(entries)
            pending = [item for item in self._items if item.status == "queued"]
            if entries or pending:
                old_pause = self._paused, self._manual_paused, self._pause_kind, self._reason
                # Clicking "join and auto-run" is explicit permission to
                # resume an idle/paused queue. Unknown remote work still blocks.
                self._manual_paused = False
                if any(item.status in {"disconnected", "stopping"} for item in self._items):
                    self._paused, self._pause_kind = True, "recovery"
                    self._reason = "任务已加入但后续尚未执行：请先重连确认或等待当前任务终止，避免重复构建。"
                else:
                    self._paused, self._pause_kind, self._reason = False, "none", ""
                for item in pending:
                    if (item.host.casefold(), item.port, item.username) == (connection["host"].casefold(), connection["port"], connection["username"]):
                        item.parameters["password"] = connection["password"]
                self._version += 1
                try:
                    self._save()
                except Exception:
                    self._items = [item for item in self._items if item not in entries]
                    self._paused, self._manual_paused, self._pause_kind, self._reason = old_pause
                    raise
                self._start_worker()
                self._condition.notify_all()
            message = f"已加入 {len(entries)} 个构建任务。"
            if skipped:
                message += f"{len(skipped)} 个相同设备上的 ONNX 已在运行或等待，未重复添加。"
            if self._paused:
                message += f"尚未执行：{self._reason}"
            elif pending:
                message += "已开启自动执行，无需再点继续队列；当前任务结束后会按顺序执行等待项。"
            return {**self.snapshot(), "addedIds": [item.id for item in entries], "skipped": skipped, "message": message}

    def set_paused(self, paused: bool, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(paused, bool):
            raise ValueError("队列暂停状态必须是布尔值。")
        with self._condition:
            if self._closed:
                raise ValueError("平台已经关闭，不能继续调度。")
            if self._storage_error:
                raise ValueError(self._storage_error)
            if not paused and any(item.status == "disconnected" for item in self._items):
                raise ValueError("请先重新连接待确认的远端任务；确认状态前不能启动后续构建，避免重复占用设备。")
            if not paused and parameters:
                matching = [item for item in self._items if item.status == "queued" and
                    (item.host.casefold(), item.port, item.username) ==
                    (str(parameters.get("host", "")).strip().casefold(), int(parameters.get("port") or 22), str(parameters.get("username", "")).strip())]
                if matching:
                    connection = self._resolve_connection(parameters)
                    for item in matching:
                        item.parameters["password"] = connection["password"]
            self._paused = paused
            self._manual_paused = paused
            self._pause_kind = "manual" if paused else "none"
            self._reason = "已暂停后续任务；当前构建不会被中断。" if paused else ""
            self._version += 1
            self._save()
            if not paused:
                self._start_worker()
            self._condition.notify_all()
            return self.snapshot()

    def remove(self, entry_id: str, *, forget: bool = False) -> dict[str, Any]:
        with self._condition:
            item = next((item for item in self._items if item.id == entry_id), None)
            if item is None:
                raise ValueError("构建队列记录不存在。")
            if item.status in {"running", "stopping"} or (item.status == "disconnected" and not forget):
                raise ValueError("正在构建的任务不能移除，请先终止当前构建。")
            item.parameters.clear()
            self._items.remove(item)
            self._version += 1
            self._save()
            return self.snapshot()

    def clear_waiting(self) -> dict[str, Any]:
        with self._condition:
            for item in self._items:
                if item.status == "queued":
                    item.parameters.clear()
            self._items = [item for item in self._items if item.status != "queued"]
            self._version += 1
            self._save()
            return self.snapshot()

    def stop(self, entry_id: str) -> dict[str, Any]:
        with self._condition:
            item = next((item for item in self._items if item.id == entry_id), None)
            if item is None or item.status not in {"running", "stopping"} or item.task_id is None:
                raise ValueError("该任务当前不在运行。")
            self._paused = True
            self._manual_paused = True
            self._pause_kind = "manual"
            self._reason = "已请求终止当前构建；后续队列已暂停。"
            item.status = "stopping"
            self._version += 1
            self._save()
            task_id = item.task_id
        try:
            self._tasks.stop(task_id)
        except ValueError:
            pass  # The job finished between the button press and cancellation.
        return self.snapshot()

    def close(self) -> None:
        """Do not kill remote tmux/setsid jobs, but never start another after shutdown."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._paused = True
            self._detach.set()
            if not self._storage_error:
                try:
                    self._save(closing=True)
                except OSError as exc:
                    # Keep the previous atomic checkpoint; a disk failure must
                    # not interrupt the remaining application's shutdown.
                    self._reason = f"关闭时队列保存失败，请保留之前的记录或通过远端目录找回：{exc}"
                    print(f"[YOLO数据处理平台] {self._reason}", flush=True)
            for item in self._items:
                if item.status == "queued":
                    item.parameters = {key: value for key, value in item.parameters.items() if key in SAFE_PARAMETERS}
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (not self._paused
                    and not any(item.status in {"running", "stopping", "disconnected"} for item in self._items)
                    and any(item.status == "queued" for item in self._items)))
                if self._closed:
                    return
                item = next(item for item in self._items if item.status == "queued")
                try:
                    task = self._tasks.start(item.function_id, f"{item.name} · {posixpath.basename(item.source)}", "remote-build")
                except ValueError:
                    # An environment check or a legacy request still holds the function slot.
                    self._condition.wait(0.25)
                    continue
                item.status, item.task_id, item.started_at = "running", task.id, now()
                item.message = "正在连接设备并构建 TensorRT 引擎。"
                self._version += 1
                try:
                    self._save()
                except Exception as exc:
                    item.status, item.task_id = "queued", None
                    self._paused, self._pause_kind, self._reason = True, "storage", f"队列保存失败，未启动远端任务：{exc}"
                    self._tasks.finish(task.id)
                    continue
            self._run_entry(item, task)

    def _run_entry(self, item: QueueEntry, task: ActiveTask, reconnect: bool = False) -> None:
        status, message, outputs = "failed", "", []
        def report(text: str) -> None:
            with self._condition:
                item.logs.append(str(text))
                task.add_log(str(text))
                self._version += 1
                if time.monotonic() - self._last_saved_log > 1:
                    self._save()
                    self._last_saved_log = time.monotonic()
        def checkpoint(state: dict[str, Any]) -> None:
            with self._condition:
                item.remote_job = validate_remote_job(state)
                self._version += 1
                self._save()
        try:
            task_name = item.source
            report(f"队列开始：{task_name}；设备：{item.username}@{item.host}:{item.port}；精度：{item.mode}")
            # The server's finished callback emits one notification per queue item.
            # Suppress the handler's per-file notification to avoid duplicates.
            context = RunContext(item.function_id, dict(item.paths), item.parameters, report, task.stop_event,
                                 checkpoint=checkpoint, detach_event=self._detach)
            context.check_cancelled()
            if not reconnect:
                try:
                    connection = self._resolve_connection(item.parameters)
                except ValueError as exc:
                    status, message = "queued", f"等待 SSH 凭据：{exc}；请填写此设备的密码后继续队列。"
                    return
                item.parameters.update({key: connection[key] for key in ("host", "port", "username", "password")})
            result = self._reconnect_runner(context, item.remote_job) if reconnect else self._runner(context)
            context.check_cancelled()
            status, message = "completed", str(result.get("message", "TensorRT 构建完成。"))
            outputs = list(dict.fromkeys(str(path) for path in [*result.get("outputFiles", []), *result.get("outputFolders", [])]))
        except TaskCancelled as exc:
            status, message = "stopped", str(exc)
        except RemoteBuildFailed as exc:
            status, message = "failed", str(exc)
        except Exception as exc:
            message = str(exc)
            if task.stop_event.is_set():
                status = "stopped"
            elif item.remote_job or isinstance(exc, RemoteBuildDisconnected):
                status = "disconnected" if item.remote_job else "queued"
                message = f"待重新连接：{message}。远端任务未被主动终止，不会自动重新构建。"
        finally:
            # Neither completed queue items nor JSON snapshots retain a password.
            if status in TERMINAL:
                item.parameters.clear()
            else:
                item.parameters["password"] = ""
            try:
                report(message)
            except OSError as exc:
                message += f"；本地记录保存失败：{exc}，可通过远端任务目录找回。"
            with self._condition:
                item.status, item.message, item.outputs, item.finished_at = status, message, outputs, now() if status in TERMINAL else None
                if status != "completed":
                    self._paused = True
                    self._pause_kind = "recovery" if status == "disconnected" else "credentials" if status == "queued" else "error"
                    self._reason = ("后续尚未执行：远端状态待确认，请先重新连接原任务；确认成功结束后自动继续（手动暂停除外）。" if status == "disconnected" else
                                    message if status == "queued" else "当前任务失败或被终止，后续队列已暂停；请确认远端没有遗留构建进程后再继续。")
                elif self._pause_kind == "recovery" and not self._manual_paused and not any(entry.status == "disconnected" for entry in self._items):
                    self._paused, self._pause_kind, self._reason = False, "none", ""
                    self._start_worker()
                elif self._manual_paused:
                    self._paused, self._pause_kind = True, "manual"
                    self._reason = "后续任务已手动暂停，点击继续队列或加入新任务后恢复自动执行。"
                self._version += 1
                try:
                    self._save()
                except OSError as exc:
                    self._paused, self._pause_kind, self._reason = True, "storage", f"任务记录保存失败：{exc}；远端结果可通过原路径检查。"
            history_saved = False
            history_error = ""
            try:
                if self._on_finished and status in TERMINAL:
                    # False explicitly means the durable history write failed;
                    # legacy callbacks returning None still mean success.
                    history_saved = self._on_finished(task, {**item.snapshot(), "logs": list(item.logs)}) is not False
            except Exception as exc:
                # A failed history/notification callback must not wedge the scheduler.
                history_error = str(exc)
            finally:
                self._tasks.finish(task.id)
            with self._condition:
                if status == "completed":
                    if history_saved:
                        if item in self._items:
                            self._items.remove(item)
                            self._version += 1
                    else:
                        warning = "运行历史未成功保存，暂保留此队列记录，避免结果丢失。"
                        if history_error:
                            warning += f"原因：{history_error}"
                        item.message = f"{message}；{warning}"
                        item.logs.append(warning)
                        self._version += 1
                # Failed/stopped tasks remain actionable. Unarchived successes
                # must not be evicted by this display limit.
                finished = [entry for entry in self._items if entry.status in {"failed", "stopped"}]
                for old in finished[:-5]:
                    self._items.remove(old)
                    self._version += 1
                self._condition.notify_all()
                try:
                    self._save()
                except OSError:
                    pass  # Reported above; never strand the active task slot.
