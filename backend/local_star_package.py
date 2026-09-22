"""Package a local TensorRT plan into a STAR model with package_tool."""

from __future__ import annotations

import ast
import json
import math
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import tomllib
import uuid
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "model.package_star"
DEFAULT_TOOL = Path("/mnt/disk2/code/gen_package/package_tool")
DEFAULT_TEMPLATE = Path("/mnt/disk2/code/gen_package/trt.toml")
SAFE_TITLE = re.compile(r"^[^/\\\x00\r\n]+$")
LABEL_SEPARATOR = re.compile(r"[,\uff0c\r\n]+")
PRECISIONS = {"INT4 (0)": 0, "INT8 (1)": 1, "FP8 (2)": 2, "FP16 (3)": 3, "FP32 (4)": 4}
COLOR_FORMATS = {"NV12 (0)": 0, "NV21 (1)": 1, "RGB (2)": 2, "BGR (3)": 3, "GRAY (4)": 4}
LABEL_FILE_NAMES = (
    "classes.txt",
    "labels.txt",
    "data.yaml",
    "data.yml",
    "dataset.yaml",
    "dataset.yml",
    "metadata.json",
    "model_metadata.json",
    "config.json",
    "args.yaml",
    "args.yml",
    "trt.toml",
)


def _integer(raw: Any, label: str, default: int, minimum: int, maximum: int) -> int:
    try:
        number = float(default if raw in (None, "") else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}\u5fc5\u987b\u662f\u6574\u6570\u3002") from exc
    if not math.isfinite(number) or not number.is_integer() or not minimum <= number <= maximum:
        raise ValueError(f"{label}\u5e94\u4e3a {minimum} \u5230 {maximum} \u4e4b\u95f4\u7684\u6574\u6570\u3002")
    return int(number)


def _text(raw: Any, label: str, default: str = "") -> str:
    value = str(default if raw in (None, "") else raw).strip()
    if not value or "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{label}\u4e0d\u80fd\u4e3a\u7a7a\u6216\u5305\u542b\u6362\u884c\u3002")
    return value


def _labels(raw: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in LABEL_SEPARATOR.split(str(raw or "")):
        value = item.strip().strip("'\"").strip()
        if not value:
            continue
        if "\x00" in value:
            raise ValueError("\u7c7b\u522b\u540d\u4e0d\u80fd\u5305\u542b\u7a7a\u5b57\u7b26\u3002")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("\u8bf7\u81f3\u5c11\u586b\u5199\u4e00\u4e2a\u7c7b\u522b\u540d\uff0c\u53ef\u4ee5\u6bcf\u884c\u4e00\u4e2a\u6216\u7528\u9017\u53f7\u5206\u9694\u3002")
    return values


def _unique_label_values(raw_values: Any) -> list[str]:
    if isinstance(raw_values, dict):
        entries = list(raw_values.items())
        if entries and all(str(key).strip().isdigit() for key, _value in entries):
            entries.sort(key=lambda item: int(str(item[0]).strip()))
        raw_values = [value for _key, value in entries]
    elif isinstance(raw_values, str):
        raw_values = LABEL_SEPARATOR.split(raw_values)
    if not isinstance(raw_values, (list, tuple)):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        value = str(raw).strip().strip("'\"").strip()
        if value and "\x00" not in value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _without_yaml_comment(value: str) -> str:
    position = _comment_position(value)
    return value[:position].rstrip() if position is not None else value.rstrip()


def _parse_inline_labels(raw: str) -> list[str]:
    value = _without_yaml_comment(raw).strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None
    labels = _unique_label_values(parsed)
    if labels:
        return labels
    if value[:1] in "[{" and value[-1:] in "]}":
        value = value[1:-1]
    parts = []
    for item in LABEL_SEPARATOR.split(value):
        item = item.strip()
        if not item:
            continue
        key, separator, remainder = item.partition(":")
        parts.append(remainder if separator and key.strip().strip("'\"").isdigit() else item)
    return _unique_label_values(parts)


def _parse_yaml_names(text: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)names\s*:\s*(.*)$", line)
        if not match:
            continue
        inline = _parse_inline_labels(match.group(2))
        if inline:
            return inline
        base_indent = len(match.group(1))
        block: list[tuple[str, str]] = []
        sequence: list[str] = []
        for child in lines[index + 1:]:
            if not child.strip() or child.lstrip().startswith("#"):
                continue
            indent = len(child) - len(child.lstrip())
            if indent <= base_indent:
                break
            value = _without_yaml_comment(child.strip())
            if value.startswith("-"):
                sequence.append(value[1:].strip())
                continue
            key, separator, remainder = value.partition(":")
            if separator and remainder.strip():
                block.append((key.strip().strip("'\""), remainder.strip()))
        if sequence:
            return _unique_label_values(sequence)
        if block:
            ordered: dict[str, str] = {key: value for key, value in block}
            return _unique_label_values(ordered)
    return []


def _labels_from_candidate(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return []
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml_names(text)
    if suffix == ".json":
        try:
            content = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(content, dict):
            for key in ("names", "labels", "class_names", "classes"):
                labels = _unique_label_values(content.get(key))
                if labels:
                    return labels
            model = content.get("model")
            if isinstance(model, dict):
                for key in ("names", "labels", "class_names", "classes"):
                    labels = _unique_label_values(model.get(key))
                    if labels:
                        return labels
        return []
    if suffix == ".toml":
        try:
            content = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return []
        model = content.get("model", {})
        return _unique_label_values(model.get("labels") if isinstance(model, dict) else None)

    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prefix, separator, remainder = line.partition(":")
        values.append(remainder if separator and prefix.strip().isdigit() else line)
    return _unique_label_values(values)


def detect_plan_labels(plan_value: str | Path) -> dict[str, Any]:
    """Find class names in sidecar dataset metadata; TensorRT plans do not retain them."""
    plan = Path(plan_value).expanduser()
    if plan.suffix.lower() != ".plan" or not plan.is_file():
        raise ValueError("请先选择存在的本地 TensorRT .plan 模型文件。")
    plan = plan.resolve()
    stem_candidates = (
        plan.with_suffix(".labels.txt"),
        plan.with_suffix(".names"),
        plan.with_name(f"{plan.stem}_labels.txt"),
    )
    candidates: list[Path] = list(stem_candidates)
    folder = plan.parent
    for _depth in range(5):
        candidates.extend(folder / name for name in LABEL_FILE_NAMES)
        parent = folder.parent
        if parent == folder:
            break
        folder = parent

    checked: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if not candidate.is_file():
            continue
        existing.append(candidate)
        labels = _labels_from_candidate(candidate)
        if labels:
            return {
                "ok": True,
                "labels": labels,
                "source": str(candidate),
                "count": len(labels),
                "message": f"已从 {candidate.name} 识别 {len(labels)} 个类别。",
            }

    detail = f"找到 {len(existing)} 个候选配置文件，但其中没有可识别的类别名称。" if existing else "附近未找到类别配置文件。"
    raise ValueError(
        "PLAN 文件本身通常不保存类别名称，无法直接从模型二进制中还原。"
        f"{detail}请将 classes.txt 或 data.yaml 放在 PLAN 同目录或上级目录后重试。"
    )


def _choice(raw: Any, label: str, choices: dict[str, int], default: str) -> int:
    value = str(raw or default).strip()
    if value not in choices:
        raise ValueError(f"\u8bf7\u9009\u62e9\u6709\u6548\u7684{label}\u3002")
    return choices[value]


def _configuration(context: RunContext) -> dict[str, Any]:
    plan = context.paths.get("model_file")
    if not plan or plan.suffix.lower() != ".plan" or not plan.is_file():
        raise ValueError("\u8bf7\u9009\u62e9\u5b58\u5728\u7684\u672c\u5730 .plan \u6a21\u578b\u6587\u4ef6\u3002")
    tool = Path(str(context.parameters.get("package_tool_path") or DEFAULT_TOOL)).expanduser()
    if not tool.is_file():
        raise ValueError(f"\u6253\u5305\u5de5\u5177\u4e0d\u5b58\u5728\uff1a{tool}")
    if not os.access(tool, os.X_OK):
        raise ValueError(f"\u6253\u5305\u5de5\u5177\u4e0d\u53ef\u6267\u884c\uff1a{tool}\u3002\u8bf7\u5148\u6dfb\u52a0\u6267\u884c\u6743\u9650\u3002")
    template = Path(str(context.parameters.get("template_file") or DEFAULT_TEMPLATE)).expanduser()
    if not template.is_file() or template.suffix.lower() != ".toml":
        raise ValueError(f"TOML \u6a21\u677f\u4e0d\u5b58\u5728\u6216\u683c\u5f0f\u4e0d\u6b63\u786e\uff1a{template}")
    try:
        tomllib.loads(template.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"TOML \u6a21\u677f\u65e0\u6cd5\u89e3\u6790\uff1a{exc}") from exc
    title = _text(context.parameters.get("title"), "\u6253\u5305\u540d\u79f0 title")
    if not SAFE_TITLE.fullmatch(title):
        raise ValueError("\u6253\u5305\u540d\u79f0 title \u4e0d\u80fd\u5305\u542b\u659c\u6760\u6216\u6362\u884c\u3002")
    category = str(context.parameters.get("category") or "generic").strip()
    if category not in {"generic", "seg", "classify"}:
        raise ValueError("\u6a21\u578b\u7c7b\u522b\u53ea\u80fd\u662f generic\u3001seg \u6216 classify\u3002")
    return {
        "plan": plan.expanduser().resolve(),
        "tool": tool.resolve(),
        "template": template.resolve(),
        "title": title,
        "labels": _labels(context.parameters.get("labels")),
        "hardware_name": _text(context.parameters.get("hardware_name"), "\u786c\u4ef6\u5e73\u53f0", "TensorRT"),
        "architecture": _text(context.parameters.get("architecture"), "\u786c\u4ef6\u67b6\u6784", "Turing"),
        "driver": _text(context.parameters.get("driver"), "\u9a71\u52a8\u4fe1\u606f", "CUDA12.2"),
        "model_name": _text(context.parameters.get("model_name"), "\u6a21\u578b\u67b6\u6784\u540d\u79f0", "yolo26"),
        "category": category,
        "version": _text(context.parameters.get("version"), "\u7248\u672c\u4fe1\u606f", "1"),
        "precision": _choice(context.parameters.get("precision"), "\u6a21\u578b\u7cbe\u5ea6", PRECISIONS, "INT8 (1)"),
        "n": _integer(context.parameters.get("input_n"), "\u8f93\u5165 N", 1, 1, 1024),
        "c": _integer(context.parameters.get("input_c"), "\u8f93\u5165 C", 3, 1, 4096),
        "h": _integer(context.parameters.get("input_h"), "\u8f93\u5165 H", 640, 1, 16384),
        "w": _integer(context.parameters.get("input_w"), "\u8f93\u5165 W", 640, 1, 16384),
        "color_fmt": _choice(context.parameters.get("color_format"), "\u989c\u8272\u683c\u5f0f", COLOR_FORMATS, "RGB (2)"),
    }


def _toml_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _comment_position(value: str) -> int | None:
    quote = ""
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote == '"' and character == "\\":
            escaped = True
            continue
        if character in {'"', "'"}:
            if not quote:
                quote = character
            elif quote == character:
                quote = ""
            continue
        if character == "#" and not quote:
            return index
    return None


def _render_template(template_text: str, config: dict[str, Any]) -> str:
    replacements = {
        ("", "title"): _toml_string(config["title"]),
        ("hardware", "name"): _toml_string(config["hardware_name"]),
        ("hardware", "architecture"): _toml_string(config["architecture"]),
        ("hardware", "driver"): _toml_string(config["driver"]),
        ("model", "name"): _toml_string(config["model_name"]),
        ("model", "category"): _toml_string(config["category"]),
        ("model", "version"): _toml_string(config["version"]),
        ("model", "model_file"): _toml_string(config["plan"]),
        ("model", "labels"): _toml_array(config["labels"]),
        ("model", "precision"): str(config["precision"]),
        ("model.input_tensor", "n"): str(config["n"]),
        ("model.input_tensor", "c"): str(config["c"]),
        ("model.input_tensor", "h"): str(config["h"]),
        ("model.input_tensor", "w"): str(config["w"]),
        ("model.input_tensor", "color_fmt"): str(config["color_fmt"]),
    }
    assignment = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*)$")
    section = ""
    found: set[tuple[str, str]] = set()
    lines: list[str] = []
    for line in template_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            section = stripped[2:-2].strip()
        elif stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
        match = assignment.match(line)
        key = (section, match.group(2)) if match else None
        if match and key in replacements:
            raw_value = match.group(4)
            comment_at = _comment_position(raw_value)
            comment = raw_value[comment_at:].rstrip() if comment_at is not None else ""
            suffix = f" {comment}" if comment else ""
            line = f"{match.group(1)}{match.group(2)}{match.group(3)}{replacements[key]}{suffix}"
            found.add(key)
        lines.append(line)
    missing = [f"{section_name or '<root>'}.{key}" for section_name, key in replacements if (section_name, key) not in found]
    if missing:
        raise ValueError(f"TOML \u6a21\u677f\u7f3a\u5c11\u5fc5\u8981\u5b57\u6bb5\uff1a{', '.join(missing)}")
    rendered = "\n".join(lines) + "\n"
    try:
        tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"\u751f\u6210\u7684 TOML \u914d\u7f6e\u65e0\u6548\uff1a{exc}") from exc
    return rendered


def _snapshot_stars(folders: set[Path]) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.star"):
            try:
                info = path.stat()
            except OSError:
                continue
            snapshot[path.resolve()] = (info.st_mtime_ns, info.st_size)
    return snapshot


def _run_process(command: list[str], cwd: Path, context: RunContext) -> list[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()
    recent: list[str] = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    threading.Thread(target=read_output, name="star-package-log", daemon=True).start()
    closed = False
    try:
        while not closed or process.poll() is None:
            context.check_cancelled()
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                closed = True
            elif line:
                recent.append(line)
                del recent[:-30]
                context.report(line)
        return_code = process.wait()
        if return_code != 0:
            detail = recent[-1] if recent else f"\u9000\u51fa\u7801 {return_code}"
            raise RuntimeError(f"STAR \u6253\u5305\u5931\u8d25\uff1a{detail}")
        return recent
    except TaskCancelled:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def run_local_star_package(context: RunContext) -> dict[str, Any]:
    config = _configuration(context)
    template_text = config["template"].read_text(encoding="utf-8")
    rendered = _render_template(template_text, config)
    folders = {config["tool"].parent, config["plan"].parent}
    before = _snapshot_stars(folders)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".toml",
            prefix=f".yolo-package-{uuid.uuid4().hex[:8]}-",
            dir=config["tool"].parent, delete=False,
        ) as stream:
            stream.write(rendered)
            temporary = Path(stream.name)
        context.report(f"\u6253\u5305\u5de5\u5177\uff1a{config['tool']}")
        context.report(f"TOML \u6a21\u677f\uff1a{config['template']}")
        context.report(f"PLAN \u6a21\u578b\uff1a{config['plan']}")
        context.report(f"\u6253\u5305\u540d\u79f0\uff1a{config['title']}\uff1b\u7c7b\u522b\u6570\uff1a{len(config['labels'])}")
        context.report(f"\u6267\u884c\u547d\u4ee4\uff1a./{config['tool'].name} {temporary.name}")
        _run_process([str(config["tool"]), temporary.name], config["tool"].parent, context)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    after = _snapshot_stars(folders)
    generated = [path for path, state in after.items() if before.get(path) != state and state[1] > 0]
    if not generated:
        raise RuntimeError("\u6253\u5305\u547d\u4ee4\u5df2\u7ed3\u675f\uff0c\u4f46\u6ca1\u6709\u68c0\u6d4b\u5230\u65b0\u751f\u6210\u7684 .star \u6587\u4ef6\u3002")
    generated.sort(key=lambda path: after[path][0], reverse=True)
    output = generated[0]
    if output.parent != config["plan"].parent:
        destination = config["plan"].parent / output.name
        if destination.exists():
            raise RuntimeError(f"PLAN \u540c\u7ea7\u76ee\u5f55\u5df2\u5b58\u5728\u540c\u540d STAR\uff1a{destination}")
        output.replace(destination)
        output = destination
        context.report(f"STAR \u5df2\u79fb\u52a8\u5230 PLAN \u540c\u7ea7\u76ee\u5f55\u3002")
    size_mb = output.stat().st_size / 1024 / 1024
    context.report(f"STAR \u6253\u5305\u5b8c\u6210\uff1a{output}\uff08{size_mb:.1f} MB\uff09")
    return {
        "message": f"STAR \u6253\u5305\u5b8c\u6210\uff1a{output.name}\u3002",
        "outputFiles": [str(output)],
        "outputFolders": [str(output.parent)],
        "title": config["title"],
        "labelCount": len(config["labels"]),
    }


register_handler(HANDLER_ID, run_local_star_package)
