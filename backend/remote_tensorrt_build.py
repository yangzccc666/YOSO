"""Build a TensorRT engine on a remote Jetson device through SSH and tmux."""

from __future__ import annotations

import json
import shlex
import time
import uuid
from pathlib import PurePosixPath
from typing import Any

from .handlers import RunContext, register_handler
from .remote_inference import RemoteInferenceManager, manager as remote_connection_manager
from .task_manager import TaskCancelled


HANDLER_ID = "remote.build_tensorrt"
DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
DEFAULT_REMOTE_ROOT = "~/.local/state/yolo-processing/tensorrt"


def _remote_capture(client: Any, command: str) -> tuple[int, str, str]:
    _stdin, stdout, stderr = client.exec_command(command, get_pty=False)
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
        "FP32": [],
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
        raise ValueError("请至少填写 1 个、最多填写 50 个量化 ONNX 文件路径，每行一个。")
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
        raise ValueError("请至少填写一个量化 ONNX 文件路径。")
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
        raise ValueError(RemoteInferenceManager._connection_error(
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
        if status == 0:
            if exit_code.strip() != "0":
                raise RuntimeError(f"远端 TensorRT 构建失败（退出码 {exit_code.strip()}），请查看 trtexec 日志。")
            return
        if launcher == "tmux":
            status, _, _ = _remote_capture(client, f"tmux has-session -t {shlex.quote(token)} 2>/dev/null")
        else:
            pid_file = shlex.quote(remote_dir + "/launcher.pid")
            status, _, _ = _remote_capture(client, f"test -f {pid_file} && kill -0 $(cat {pid_file}) 2>/dev/null")
        if status:
            raise RuntimeError("远端后台任务已结束，但没有生成退出状态；请检查远端日志。")
        time.sleep(2)


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
        raise RuntimeError(
            f"第 {index}/{total} 个模型启动远程后台构建失败：{error or f'退出码 {status}'}"
        )
    launcher_message = (
        f"tmux 会话 {token}" if config["launcher"] == "tmux" else "setsid + nohup 独立后台任务"
    )
    context.report(f"[{index}/{total}] 已在远程{launcher_message}中启动。")
    context.report(f"[{index}/{total}] TensorRT 命令：{command}")
    try:
        _follow_job(
            client, token, remote_dir, remote_log, remote_plan,
            config["launcher"], context,
        )
    except (RuntimeError, TaskCancelled) as exc:
        context.report(f"[{index}/{total}] 请检查远端日志：{remote_log}")
        if isinstance(exc, TaskCancelled):
            raise
        raise RuntimeError(
            f"第 {index}/{total} 个模型 {source.name} 构建失败：{exc}"
        ) from exc

    status, size_text, error = _remote_capture(
        client,
        f"test -s {shlex.quote(remote_plan)} && "
        f"test \"$(head -c 4 {shlex.quote(remote_plan)})\" = ftrt && "
        f"stat -c %s {shlex.quote(remote_plan)}",
    )
    if status:
        raise RuntimeError(
            f"第 {index}/{total} 个模型的 trtexec 已结束但未在远端生成有效 .plan："
            f"{error or remote_plan}"
        )
    remote_size = int(size_text.strip())
    context.report(
        f"[{index}/{total}] TensorRT 引擎已在远端生成：{remote_plan}"
        f"（{remote_size / 1024 / 1024:.2f} MB）"
    )
    if not config["keep_remote"]:
        _remote_capture(client, f"rm -rf -- {shlex.quote(remote_dir)}")
        context.report(f"[{index}/{total}] 远程任务目录已清理。")
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
