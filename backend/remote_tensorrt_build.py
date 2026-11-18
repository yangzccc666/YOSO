"""Build a TensorRT engine on a remote Jetson device through SSH and tmux."""

from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from .handlers import RunContext, register_handler
from .remote_inference import RemoteInferenceManager, manager as remote_connection_manager
from .task_manager import TaskCancelled


HANDLER_ID = "remote.build_tensorrt"
DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
DEFAULT_REMOTE_ROOT = "~/.local/state/yolo-processing/tensorrt"


class RemoteBuildDisconnected(RuntimeError):
    """Monitoring ended, not necessarily the independently running build."""


class RemoteBuildFailed(RuntimeError):
    """A definite remote exit failure or invalid result, not an SSH interruption."""


def validate_remote_job(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("远端构建任务信息必须是对象。")
    source = PurePosixPath(str(raw.get("source", "")))
    directory = PurePosixPath(str(raw.get("remoteDirectory", "")))
    token = str(raw.get("token", ""))
    if (not source.is_absolute() or source.suffix.lower() != ".onnx"
            or not re.fullmatch(r"yolo-trt-[0-9a-f]{12}", token)
            or not directory.is_absolute() or directory.name != f"{source.stem}-{token[9:]}"
            or raw.get("launcher") not in {"tmux", "setsid"}
            or any("\x00" in str(value) for value in raw.values())):
        raise ValueError("远端构建任务信息无效，不能安全重连。")
    return {"source": str(source), "plan": str(source.with_suffix(".plan")),
            "log": str(source.with_suffix(".trtexec.log")), "remoteDirectory": str(directory),
            "token": token, "launcher": raw["launcher"], "keepRemote": raw.get("keepRemote") is not False}


def _remote_capture(client: Any, command: str) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command, get_pty=False, timeout=20)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    return stdout.channel.recv_exit_status(), output, error


def _configuration(context: RunContext, *, require_paths: bool) -> dict[str, Any]:
    trtexec = str(context.parameters.get("trtexec_path") or DEFAULT_TRTEXEC).strip()
    if not trtexec.startswith("/") or "\x00" in trtexec:
        raise ValueError("远端 trtexec 路径必须是绝对路径。")
    remote_root = str(context.parameters.get("remote_workspace") or DEFAULT_REMOTE_ROOT).strip()
    if not (remote_root.startswith("~/") or remote_root.startswith("/")) or "\x00" in remote_root:
        raise ValueError("远端任务目录必须是绝对路径或以 ~/ 开头。")
    mode = str(context.parameters.get("build_mode") or "自动最佳（--best）")
    mode_flags = {
        "自动最佳（--best）": ["--best"],
        "FP16": ["--fp16"],
        "FP32": ["--noTF32"],
    }
    if mode not in mode_flags:
        raise ValueError("请选择有效的 TensorRT 构建精度。")
    config: dict[str, Any] = {
        "trtexec": trtexec,
        "remote_root": remote_root,
        "mode": mode,
        "mode_flags": mode_flags[mode],
        "benchmark": bool(context.parameters.get("run_benchmark", True)),
        "keep_remote": bool(context.parameters.get("keep_remote_files", True)),
        "overwrite": bool(context.parameters.get("overwrite_output", False)),
    }
    if not require_paths:
        return config
    raw_sources = context.parameters.get("onnx_files", "[]")
    if isinstance(raw_sources, str):
        stripped_sources = raw_sources.strip()
        if stripped_sources.startswith("["):
            try:
                source_values = json.loads(stripped_sources)
            except json.JSONDecodeError as exc:
                raise ValueError("ONNX 路径格式不正确，请每行填写一个完整路径。") from exc
        else:
            source_values = [line.strip() for line in raw_sources.splitlines() if line.strip()]
    else:
        source_values = raw_sources
    if source_values in (None, []):
        legacy_source = context.paths.get("input_onnx")
        source_values = [str(legacy_source)] if legacy_source else []
    if not isinstance(source_values, list) or not 1 <= len(source_values) <= 50:
        raise ValueError("请至少填写 1 个、最多填写 50 个远端 ONNX 文件路径，每行一个。")
    sources: list[PurePosixPath] = []
    seen_sources: set[PurePosixPath] = set()
    for index, value in enumerate(source_values, 1):
        raw_source = str(value).strip()
        source = PurePosixPath(raw_source)
        if not source.is_absolute() or source.suffix.lower() != ".onnx" or "\x00" in raw_source:
            raise ValueError(f"第 {index} 个 ONNX 必须是远程服务器上的 .onnx 绝对路径：{raw_source}")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)
    if not sources:
        raise ValueError("请至少填写一个远端 ONNX 文件路径。")
    jobs: list[dict[str, PurePosixPath]] = []
    planned_outputs: set[PurePosixPath] = set()
    for source in sources:
        output_plan = source.with_suffix(".plan")
        output_log = output_plan.with_suffix(".trtexec.log")
        if output_plan in planned_outputs:
            raise ValueError(f"多个 ONNX 会生成同名远程结果：{output_plan}。")
        planned_outputs.add(output_plan)
        jobs.append({
            "source": source,
            "output_plan": output_plan,
            "output_log": output_log,
        })
    config.update({
        "jobs": jobs,
    })
    return config


def _connect(parameters: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    connection = remote_connection_manager.resolve_connection(parameters)
    paramiko = RemoteInferenceManager._load_paramiko()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=connection["host"], port=connection["port"],
            username=connection["username"], password=connection["password"],
            look_for_keys=False, allow_agent=False, timeout=12,
            banner_timeout=12, auth_timeout=12,
        )
    except Exception as exc:
        connection["password"] = ""
        client.close()
        raise RemoteBuildDisconnected(RemoteInferenceManager._connection_error(
            exc, paramiko, connection["host"], connection["port"]
        )) from exc
    connection["password"] = ""
    parameters["password"] = ""
    return client, connection


def _check_environment(client: Any, config: dict[str, Any], report: Any) -> str:
    status, _, _ = _remote_capture(client, f"test -x {shlex.quote(config['trtexec'])}")
    if status:
        raise ValueError(f"远端 trtexec 不存在或不可执行：{config['trtexec']}")
    status, tmux_path, _ = _remote_capture(client, "command -v tmux")
    if status == 0 and tmux_path:
        config["launcher"] = "tmux"
        launcher_text = f"tmux={tmux_path}"
    else:
        status, setsid_path, _ = _remote_capture(client, "command -v setsid")
        if status or not setsid_path:
            raise ValueError("AI 推理盒子既没有 tmux，也没有 setsid，无法安全启动独立后台构建。")
        config["launcher"] = "setsid"
        launcher_text = f"tmux 未安装，自动使用 setsid={setsid_path} + nohup"
    status, output, error = _remote_capture(
        client, f"{shlex.quote(config['trtexec'])} --help 2>&1 | head -n 1"
    )
    version_text = "\n".join(value for value in (output, error) if value).strip()
    if status or "TensorRT" not in version_text:
        raise RuntimeError(f"无法读取远端 TensorRT 版本：{version_text or f'退出码 {status}'}")
    version_line = next((line.strip() for line in version_text.splitlines() if "TensorRT" in line), "TensorRT 可用")
    report(f"远端环境检查完成：trtexec={config['trtexec']}；{launcher_text}；{version_line}")
    return version_line


def _validate_remote_jobs(client: Any, config: dict[str, Any], report: Any) -> None:
    sftp = client.open_sftp()
    try:
        for index, job in enumerate(config["jobs"], 1):
            source = str(job["source"])
            try:
                attributes = sftp.stat(source)
            except OSError as exc:
                raise ValueError(f"第 {index} 个远端 ONNX 文件不存在或无权访问：{source}") from exc
            if getattr(attributes, "st_size", 0) <= 0:
                raise ValueError(f"第 {index} 个远端 ONNX 文件为空：{source}")
            if config["mode"] in {"FP16", "FP32"}:
                probe = Path(__file__).with_name('onnx_precision_probe.py').read_text(encoding='utf-8')
                status, output, error = _remote_capture(client, shlex.join(['python3', '-c', probe, source]))
                if status:
                    raise ValueError(f"无法检查远端 ONNX 精度：{source}；{error or output}。FP16/FP32 构建需盒子可运行 python3（无需 onnx 包）。")
                try:
                    precision = json.loads(output)
                    if not isinstance(precision, dict) or type(precision.get('quantized')) is not bool or not isinstance(precision.get('reducedFloatTypes'), list):
                        raise ValueError('invalid probe response')
                except (ValueError, TypeError) as exc:
                    raise ValueError('远端 ONNX 精度检查未返回有效结果，已停止构建。') from exc
                if precision['quantized']:
                    raise ValueError(f"{source} 含量化算子 {precision.get('quantizedOps')}，不能通过选择 {config['mode']} 恢复精度。请使用 PT 直接导出的未量化 FP32 ONNX，跳过 Docker ONNX 量化。")
                if config['mode'] == 'FP32' and precision['reducedFloatTypes']:
                    raise ValueError(f"{source} 已含低精度浮点张量，不能保证 FP32 构建。请从原 PT 重新导出 FP32 ONNX。")
                report(f"精度检查通过：未发现量化算子；{config['mode']} 构建，FP32 时禁用 TF32。最终层精度请以引擎检查为准。")
            if not config["overwrite"]:
                for output in (job["output_plan"], job["output_log"]):
                    try:
                        sftp.stat(str(output))
                    except OSError:
                        continue
                    raise ValueError(f"远端输出已存在：{output}。请开启“允许覆盖远端同名结果”后重试。")
            report(f"[{index}/{len(config['jobs'])}] 远端 ONNX 已确认：{source}")
    finally:
        sftp.close()


def _resolved_remote_root(sftp: Any, raw: str) -> PurePosixPath:
    home = PurePosixPath(sftp.normalize("."))
    value = PurePosixPath(raw)
    result = home / str(value)[2:] if str(value).startswith("~/") else value
    if not result.is_absolute() or str(result) == "/":
        raise ValueError("远端任务目录无效。")
    return result


def _mkdir_p(sftp: Any, directory: PurePosixPath) -> None:
    current = PurePosixPath("/")
    for part in directory.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except OSError:
            sftp.mkdir(str(current))


def _follow_job(
    client: Any,
    token: str,
    remote_dir: str,
    remote_log: str,
    remote_plan: str,
    launcher: str,
    context: RunContext,
) -> None:
    previous: list[str] = []
    while True:
        if context.detach_event is not None and context.detach_event.is_set():
            raise RemoteBuildDisconnected("平台已关闭，已断开日志监控；远端构建未被终止，可重新连接。")
        if context.stop_event is not None and context.stop_event.is_set():
            if launcher == "tmux":
                _remote_capture(client, f"tmux send-keys -t {shlex.quote(token)} C-c 2>/dev/null || true")
                time.sleep(0.3)
                _remote_capture(client, f"tmux kill-session -t {shlex.quote(token)} 2>/dev/null || true")
            else:
                pid_file = shlex.quote(remote_dir + "/launcher.pid")
                _remote_capture(client, f"test -f {pid_file} && kill -TERM -- -$(cat {pid_file}) 2>/dev/null || true")
            _remote_capture(client, f"rm -f {shlex.quote(remote_plan)}")
            raise TaskCancelled("远程 TensorRT 构建已终止；远端日志会保留。")
        status, output, _ = _remote_capture(
            client, f"tail -n 500 {shlex.quote(remote_log)} 2>/dev/null"
        )
        if status == 0 and output:
            lines = output.splitlines()
            overlap = next(
                (amount for amount in range(min(len(lines), len(previous)), 0, -1)
                 if previous[-amount:] == lines[:amount]),
                0,
            )
            for line in lines[overlap:]:
                context.report(line)
            previous = lines
        status, exit_code, _ = _remote_capture(
            client, f"test -f {shlex.quote(remote_dir + '/exit.code')} && cat {shlex.quote(remote_dir + '/exit.code')}"
        )
        if status == 0 and exit_code.strip():
            if not exit_code.strip().isdigit():
                raise RemoteBuildDisconnected("远端退出状态内容异常，请检查任务目录后重连。")
            if exit_code.strip() != "0":
                raise RemoteBuildFailed(f"远端 TensorRT 构建失败（退出码 {exit_code.strip()}），请查看 trtexec 日志。")
            return
        if launcher == "tmux":
            status, _, _ = _remote_capture(client, f"tmux has-session -t {shlex.quote(token)} 2>/dev/null")
        else:
            pid_file = shlex.quote(remote_dir + "/launcher.pid")
            status, _, _ = _remote_capture(client, f"test -f {pid_file} && kill -0 $(cat {pid_file}) 2>/dev/null")
        if status:
            # The wrapper may have finished between the exit-file check and
            # the liveness check. Re-read before declaring the state unknown.
            status, exit_code, _ = _remote_capture(client, f"cat {shlex.quote(remote_dir + '/exit.code')} 2>/dev/null")
            if status == 0 and exit_code.strip() == "0":
                return
            if status == 0 and exit_code.strip():
                if not exit_code.strip().isdigit():
                    raise RemoteBuildDisconnected("远端退出状态内容异常，请检查任务目录后重连。")
                raise RemoteBuildFailed(f"远端 TensorRT 构建失败（退出码 {exit_code.strip()}）。")
            raise RemoteBuildDisconnected("远端会话未找到且没有退出状态，无法确认是否完成；请检查日志后重连，不会自动重新构建。")
        if context.detach_event is not None:
            context.detach_event.wait(2)
        else:
            time.sleep(2)


def _follow_existing_job(client: Any, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
    state = validate_remote_job(state)
    _follow_job(client, state["token"], state["remoteDirectory"], state["log"],
                state["plan"], state["launcher"], context)
    status, size_text, error = _remote_capture(
        client, f"test -s {shlex.quote(state['plan'])} && "
        f"test \"$(head -c 4 {shlex.quote(state['plan'])})\" = ftrt && stat -c %s {shlex.quote(state['plan'])}")
    if status:
        raise RemoteBuildFailed(f"trtexec 已结束但没有生成有效 .plan：{error or state['plan']}")
    size = int(size_text.strip())
    context.report(f"TensorRT 引擎已在远端生成：{state['plan']}（{size / 1024 / 1024:.2f} MB）")
    # Retain the recovery metadata even when users opt to clean task files:
    # an exit.code + job.json is tiny and allows post-restart status lookup.
    return {"message": "TensorRT 构建完成。", "outputFiles": [state["plan"], state["log"]],
            "outputFolders": [str(PurePosixPath(state["source"]).parent)], "engineSize": size}


def reconnect_remote_build(context: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    state = validate_remote_job(state)
    client = None
    try:
        client, _ = _connect(context.parameters)
        context.report(f"重新接管远端任务：{state['token']}；只读取状态和日志，不重新执行构建。")
        return _follow_existing_job(client, state, context)
    finally:
        context.parameters["password"] = ""
        if client is not None:
            client.close()


def discover_remote_builds(parameters: dict[str, Any]) -> dict[str, Any]:
    """Read only: recover metadata, or legacy directories matching supplied ONNX paths."""
    context = RunContext("remote_tensorrt_build", {}, parameters, lambda _: None)
    config = _configuration(context, require_paths=False)
    candidates: dict[str, PurePosixPath] = {}
    if str(parameters.get("onnx_files", "")).strip() not in {"", "[]"}:
        for job in _configuration(context, require_paths=True)["jobs"]:
            source = job["source"]
            if source.stem in candidates and candidates[source.stem] != source:
                raise ValueError("旧任务查找时请勿填写文件名相同、目录不同的 ONNX，请分开查找。")
            candidates[source.stem] = source
    client, _ = _connect(parameters)
    sftp = None
    found, skipped = [], 0
    try:
        sftp = client.open_sftp()
        root = _resolved_remote_root(sftp, config["remote_root"])
        try:
            entries = sorted(sftp.listdir_attr(str(root)), key=lambda entry: getattr(entry, "st_mtime", 0), reverse=True)[:500]
        except OSError as exc:
            raise ValueError(f"无法读取远端任务目录 {root}，请确认使用原任务根目录且有读取权限。") from exc
        candidate_sources = {str(source) for source in candidates.values()}
        for entry in entries:
            name = entry.filename
            match = re.fullmatch(r"(.+)-([0-9a-f]{12})", name)
            if not match:
                continue
            directory = str(root / name)
            try:
                with sftp.open(directory + "/job.json", "r") as handle:
                    raw = json.loads(handle.read(65537))
                state = validate_remote_job(raw)
                if state["remoteDirectory"] != directory:
                    raise ValueError("任务目录不匹配")
            except (OSError, ValueError, TypeError, KeyError):
                source = candidates.get(match[1])
                if source is None:
                    skipped += 1
                    continue
                try:
                    sftp.stat(directory + "/launcher.pid")
                    launcher = "setsid"
                except OSError:
                    launcher = "tmux"
                state = validate_remote_job({"source": str(source), "remoteDirectory": directory,
                    "token": "yolo-trt-" + match[2], "launcher": launcher, "keepRemote": True})
            if candidates and state["source"] not in candidate_sources:
                continue  # A targeted lookup must not import unrelated models.
            found.append(state)
        return {"jobs": found, "skipped": skipped}
    finally:
        if sftp is not None:
            sftp.close()
        client.close()


def _trtexec_arguments(config: dict[str, Any], remote_input: str, remote_plan: str) -> list[str]:
    arguments = [
        config["trtexec"], f"--onnx={remote_input}", f"--saveEngine={remote_plan}",
        *config["mode_flags"],
    ]
    if not config["benchmark"]:
        arguments.append("--skipInference")
    return arguments


def _run_one_job(
    client: Any,
    root: PurePosixPath,
    config: dict[str, Any],
    job: dict[str, PurePosixPath],
    context: RunContext,
    index: int,
    total: int,
) -> dict[str, Any]:
    source = job["source"]
    job_id = uuid.uuid4().hex[:12]
    job_dir = root / f"{source.stem}-{job_id}"
    remote_dir = str(job_dir)
    remote_input = str(source)
    remote_plan = str(job["output_plan"])
    remote_log = str(job["output_log"])
    remote_exit = str(job_dir / "exit.code")

    context.report(f"── [{index}/{total}] 开始构建：{source.name} ──")
    context.check_cancelled()
    sftp = client.open_sftp()
    try:
        _mkdir_p(sftp, job_dir)
    finally:
        sftp.close()
    context.report(f"[{index}/{total}] 远端 ONNX：{remote_input}")
    context.report(f"[{index}/{total}] 远端 Plan：{remote_plan}")
    context.report(f"[{index}/{total}] 远端 Log：{remote_log}")
    if config["overwrite"]:
        _remote_capture(client, f"rm -f -- {shlex.quote(remote_plan)} {shlex.quote(remote_log)}")

    arguments = _trtexec_arguments(config, remote_input, remote_plan)
    command = shlex.join(arguments)
    wrapper = (
        f"{command} > {shlex.quote(remote_log)} 2>&1; result=$?; "
        f"printf '%s\\n' \"$result\" > {shlex.quote(remote_exit)}; exit \"$result\""
    )
    token = f"yolo-trt-{job_id}"
    state = validate_remote_job({"source": remote_input, "remoteDirectory": remote_dir,
        "token": token, "launcher": config["launcher"], "keepRemote": config["keep_remote"]})
    # Save before executing SSH launch: a crash after command acceptance must
    # never lead to automatically re-executing the same model on restart.
    sftp = client.open_sftp()
    try:
        with sftp.open(remote_dir + "/job.json", "w") as handle:
            handle.write(json.dumps(state, ensure_ascii=False))
    finally:
        sftp.close()
    if context.checkpoint is not None:
        context.checkpoint(state)
    if context.detach_event is not None and context.detach_event.is_set():
        raise RemoteBuildDisconnected("平台已关闭，未继续提交远端启动命令；请重连确认状态。")
    if config["launcher"] == "tmux":
        launch = (
            f"tmux new-session -d -s {shlex.quote(token)} "
            f"{shlex.quote('bash -lc ' + shlex.quote(wrapper))}"
        )
    else:
        background = "bash -lc " + shlex.quote(wrapper)
        launch = (
            f"nohup setsid {background} </dev/null >/dev/null 2>&1 & "
            f"printf '%s\\n' \"$!\" > {shlex.quote(remote_dir + '/launcher.pid')}"
        )
    status, _, error = _remote_capture(client, launch)
    if status:
        raise RemoteBuildFailed(
            f"第 {index}/{total} 个模型启动远程后台构建失败：{error or f'退出码 {status}'}"
        )
    launcher_message = (
        f"tmux 会话 {token}" if config["launcher"] == "tmux" else "setsid + nohup 独立后台任务"
    )
    context.report(f"[{index}/{total}] 已在远程{launcher_message}中启动。")
    context.report(f"[{index}/{total}] TensorRT 命令：{command}")
    result = _follow_existing_job(client, state, context)
    remote_size = result["engineSize"]
    return {
        "source": str(source),
        "plan": remote_plan,
        "log": remote_log,
        "remoteDirectory": remote_dir if config["keep_remote"] else "",
        "remotePlan": remote_plan,
        "engineSize": remote_size,
    }


def run_remote_tensorrt_build(context: RunContext) -> dict[str, Any]:
    operation = str(context.parameters.get("operation") or "构建 TensorRT 引擎")
    if operation not in {"检查远程环境", "构建 TensorRT 引擎"}:
        raise ValueError("请选择有效的操作方式。")
    config = _configuration(context, require_paths=operation == "构建 TensorRT 引擎")
    client = None
    sftp = None
    try:
        client, connection = _connect(context.parameters)
        context.report(f"SSH 已连接：{connection['username']}@{connection['host']}:{connection['port']}")
        version = _check_environment(client, config, context.report)
        if operation == "检查远程环境":
            return {
                "message": f"AI 推理盒子的 TensorRT 构建环境可用：{version}。",
                "host": connection["host"],
                "trtexec": config["trtexec"],
            }

        context.check_cancelled()
        sftp = client.open_sftp()
        root = _resolved_remote_root(sftp, config["remote_root"])
        sftp.close()
        sftp = None
        jobs = config["jobs"]
        _validate_remote_jobs(client, config, context.report)
        context.report(f"共 {len(jobs)} 个远端 ONNX 待构建，将严格按填写顺序逐个执行。")
        results: list[dict[str, Any]] = []
        for index, job in enumerate(jobs, 1):
            context.check_cancelled()
            result = _run_one_job(client, root, config, job, context, index, len(jobs))
            results.append(result)
            context.send_notification(
                "远程 TensorRT 引擎构建",
                f"[{index}/{len(jobs)}] {job['source'].name} 转换完成\n"
                f"远端结果：{result['plan']}",
            )
        output_files = [path for result in results for path in (result["plan"], result["log"])]
        output_folders = list(dict.fromkeys(str(job["source"].parent) for job in jobs))
        return {
            "message": f"远程 TensorRT 批量构建完成：{len(results)} 个模型。",
            "outputFiles": output_files,
            "outputFolders": output_folders,
            "jobs": results,
        }
    finally:
        context.parameters["password"] = ""
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        if client is not None:
            client.close()


register_handler(HANDLER_ID, run_remote_tensorrt_build)
