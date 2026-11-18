"""Sequence-aware, reversible image deduplication for YOLO datasets."""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "image.smart_dedup"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIDECAR_SUFFIXES = (".txt", ".xml", ".json")
BACKUP_FOLDER = ".yolo_dedup_backup"
STRENGTHS = {
    "保守": 0.992,
    "较保守": 0.985,
    "平衡（推荐）": 0.970,
    "较强": 0.950,
    "强力": 0.920,
}


@dataclass(frozen=True)
class ImageFeature:
    gray: np.ndarray
    edges: np.ndarray
    histogram: np.ndarray
    sharpness: float


@dataclass(frozen=True)
class Candidate:
    path: Path
    feature: ImageFeature
    annotation_signature: tuple[str, ...] | None


def _number(raw: Any, label: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(default if raw in (None, "") else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字。") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label}应在 {minimum:g} 到 {maximum:g} 之间。")
    return value


def _integer(raw: Any, label: str, default: int, minimum: int, maximum: int) -> int:
    value = _number(raw, label, default, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f"{label}必须是整数。")
    return int(value)


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in re.split(r"(\d+)", path.name))


def _sequence_key(path: Path) -> str:
    modern = re.match(r"^(.*?)_(?:fps[\dp]+|every[\dp]+s_.*|priority[\dp]+_bg[\dp]+|step\d+)_t(?:\d+ms|unknown)_f\d+$", path.stem)
    if modern:
        return modern.group(1).casefold()
    match = re.match(r"^(.*?)[_-](\d+)$", path.stem)
    return match.group(1).casefold() if match else "__folder_sequence__"


def _read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, cv2.error):
        return None


def _feature(image: np.ndarray) -> ImageFeature:
    reduced = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    reduced = cv2.GaussianBlur(reduced, (5, 5), 0)
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).astype(np.float32)
    cv2.normalize(histogram, histogram, 1.0, 0.0, cv2.NORM_L1)
    sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return ImageFeature(gray=gray, edges=edges, histogram=histogram, sharpness=sharpness)


def _similarity(left: ImageFeature, right: ImageFeature) -> float:
    gray_similarity = 1.0 - min(1.0, float(np.mean(np.abs(left.gray - right.gray))) / 255.0)
    edge_similarity = 1.0 - min(1.0, float(np.mean(np.abs(left.edges - right.edges))) / 128.0)
    histogram_similarity = (float(cv2.compareHist(left.histogram, right.histogram, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
    return max(0.0, min(1.0, 0.65 * gray_similarity + 0.20 * edge_similarity + 0.15 * histogram_similarity))


def _annotation_signature(image: Path) -> tuple[str, ...] | None:
    from .image_quality_dedup import annotation_key
    try:
        return (annotation_key(image),)
    except Exception:
        return ('__protected_annotation__', str(image))


def _select_sequence(
    candidates: list[Candidate], threshold: float
) -> tuple[set[Path], dict[Path, dict[str, Any]]]:
    count = len(candidates)
    if count <= 2:
        return {item.path for item in candidates}, {}
    kept_indices = [0]
    removed: dict[int, dict[str, Any]] = {}
    for index in range(1, count - 1):
        candidate = candidates[index]
        reference_index = kept_indices[-1]
        reference = candidates[reference_index]
        similarity = _similarity(reference.feature, candidate.feature)
        annotations_compatible = candidate.annotation_signature == reference.annotation_signature
        should_keep = not annotations_compatible or similarity < threshold
        if not should_keep:
            from .image_quality_dedup import read_original, verify_pair
            duplicate, _reason, _metrics = verify_pair(read_original(reference.path), read_original(candidate.path))
            should_keep = not duplicate
        if should_keep:
            kept_indices.append(index)
        else:
            removed[index] = {
                "similarity": round(similarity, 6),
                "reference": reference.path.name,
                "reason": "与最近保留帧相似，且通过原图整体/局部复核",
                "sharpness": round(candidate.feature.sharpness, 3),
            }
    kept_indices.append(count - 1)

    kept = {candidates[index].path for index in kept_indices}
    return kept, {candidates[index].path: detail for index, detail in removed.items()}


def _safe_relative(folder: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("去重备份清单包含不安全路径，已停止回滚。")
    resolved = (folder / relative).resolve()
    if folder.resolve() not in resolved.parents:
        raise RuntimeError("去重备份清单路径超出图片文件夹，已停止回滚。")
    return resolved


def _backup_root(folder: Path) -> Path:
    # Keep backups outside the image tree so recursive YOLO scans cannot see
    # images that were intentionally removed from the training set.
    return folder.parent / f".{folder.name}{BACKUP_FOLDER}"


def _rollback_moves(folder: Path, manifest_path: Path, manifest: dict) -> list[tuple[Path, Path]]:
    """Preflight a restore before moving any image (also used by batch rollback)."""
    if manifest.get('sourceFolder') != str(folder) or manifest.get('status') != 'active':
        raise RuntimeError('去重备份的来源或状态不匹配，已停止回滚。')
    moves = []
    for entry in manifest.get('movedFiles', []):
        source = _safe_relative(manifest_path.parent, str(entry.get('backup', '')))
        destination = _safe_relative(folder, str(entry.get('original', '')))
        if not source.is_file():
            raise RuntimeError(f'备份文件缺失，无法安全回滚：{source.name}')
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f'原位置已有同名文件，未执行回滚：{destination.name}')
        moves.append((source, destination))
    return moves


def _rollback(folder: Path, context: RunContext) -> dict[str, Any]:
    backup_root = _backup_root(folder)
    batches = []
    if backup_root.is_dir():
        for manifest_path in backup_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "active":
                    batches.append((manifest_path.parent.name, manifest_path, manifest))
            except (OSError, json.JSONDecodeError):
                continue
    if not batches:
        raise ValueError("没有找到可以回滚的去重记录。")
    _batch_name, manifest_path, manifest = max(batches, key=lambda item: item[0])
    moves = _rollback_moves(folder, manifest_path, manifest)
    restored_images = 0
    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in moves:
            context.check_cancelled()
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            completed.append((source, destination))
            if destination.suffix.lower() in IMAGE_SUFFIXES:
                restored_images += 1
    except Exception:
        for source, destination in reversed(completed):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
        raise
    manifest["status"] = "rolled_back"
    manifest["rolledBackAt"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    context.report(f"已回滚批次 {manifest_path.parent.name}，恢复 {restored_images} 张图片。")
    return {
        "message": f"图片去重已回滚：恢复 {restored_images} 张图片。",
        "restoredCount": restored_images,
        "outputFolders": [str(folder)],
        "backupManifest": str(manifest_path),
    }


def run_image_dedup(context: RunContext) -> dict[str, Any]:
    recursive = context.parameters.get('recursive', False)
    if recursive not in (True, False):
        raise ValueError('包含子文件夹必须为开启或关闭。')
    if recursive:
        from .image_dedup_batch import run_batch
        return run_batch(context)
    return _run_single_folder(context)


def _run_single_folder(context: RunContext) -> dict[str, Any]:
    folder = context.paths.get("images_folder")
    if not folder:
        raise ValueError("请选择需要去重的图片文件夹。")
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"图片文件夹不存在：{folder}")
    operation = str(context.parameters.get("operation", "分析并去重"))
    if operation == "回滚最近一次":
        return _rollback(folder, context)
    if operation not in {"分析并去重", "仅分析（不移动文件）"}:
        raise ValueError("请选择有效的操作方式。")

    from .image_quality_dedup import AUTO_MODE, EXACT_MODE, RULE_VERSION, plan_quality
    from .image_process_dedup import PROCESS_MODE, plan_process
    mode = str(context.parameters.get("dedup_mode", AUTO_MODE))
    automatic = mode in {AUTO_MODE, EXACT_MODE}
    if automatic:
        strength = None
    elif mode == PROCESS_MODE:
        strength = str(context.parameters.get('process_strength', '平衡（推荐）'))
    else:
        strength = str(context.parameters.get('dedup_strength', '保守'))
    if not automatic and strength not in STRENGTHS:
        raise ValueError("请选择有效的去重力度。")
    threshold = None if automatic or mode == PROCESS_MODE else STRENGTHS[strength]
    if mode not in {AUTO_MODE, EXACT_MODE, PROCESS_MODE, "连续相似帧去重", "完整工序循环去重"}:
        raise ValueError("请选择有效的去重模式。")
    images = sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=_natural_key,
    )
    if not images:
        raise ValueError('图片文件夹中没有支持的图片。')
    from collections import Counter
    stems = Counter(path.stem.casefold() for path in images)
    if any(stems[path.stem.casefold()] > 1 and (mode == '完整工序循环去重' or any(path.with_suffix(s).exists() for s in SIDECAR_SUFFIXES)) for path in images):
        raise ValueError('文件夹存在同名但扩展名不同的图片共用标注，请先分开，避免误移动标签。')
    snapshot = _snapshot_images(images)

    process_settings = {}
    if mode == PROCESS_MODE:
        removed, unreadable, process_settings = plan_process(images, folder, context, strength)
        cycles = []
    elif mode == "完整工序循环去重":
        from .image_cycles import plan_cycles
        removed, cycles, unreadable = plan_cycles(images, folder, context, threshold)
    elif automatic:
        removed, unreadable = plan_quality(images, context, exact_only=mode == EXACT_MODE)
        cycles = []
    else:
        removed, unreadable = _plan_adjacent(images, context, strength, threshold)
        cycles = []
    current_images = sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES], key=_natural_key)
    if current_images != images or _snapshot_images(images) != snapshot:
        raise ValueError('分析期间图片或标注发生变化，未移动文件。请停止其他写入操作后重新分析。')
    # Persist an inspectable report for both preview and execution. Reports are
    # outside the image tree and contain no image copies until execution begins.
    settings = {"mode": mode, "strength": strength, "similarityThreshold": threshold,
                "quantityQuotaApplied": False, 'verificationRule': RULE_VERSION}
    if process_settings:
        settings.update({'workstation': process_settings, 'verificationRule': process_settings['rule']})
    if unreadable:
        context.report(f'警告：{len(unreadable)} 张图片或标注无法安全核验，全部原样保留。')
    if mode == "完整工序循环去重":
        settings.update({"referenceStart": str(context.paths.get("cycle_start_image", "")),
                         "referenceEnd": str(context.paths.get("cycle_end_image", "")),
                         "minimumCycleRatio": context.parameters.get("cycle_min_ratio", .4),
                         "maximumCycleRatio": context.parameters.get("cycle_max_ratio", 2.5)})
    preview = operation == "仅分析（不移动文件）"
    context.check_cancelled()
    if preview or not removed:
        if not preview and mode == "连续相似帧去重":
            context.report("没有发现达到当前力度阈值的重复图片，文件夹未发生变化。")
            return {"message": "图片去重完成：未发现需要移除的重复图片。", "imageCount": len(images),
                    "keptCount": len(images), "removedCount": 0, "unreadableCount": len(unreadable),
                    "outputFolders": [str(folder)]}
        report = {"status": "analysis", "settings": settings, "cycles": cycles,
                  "originalImageCount": len(images), "proposedRemovedCount": len(removed),
                  "unreadableImages": unreadable,
                  "removedImages": [{"name": path.name, **detail} for path, detail in removed.items()]}
        batch = _backup_root(folder) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        batch.mkdir(parents=True, exist_ok=False)
        report_path = batch / "analysis.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        message = f"分析完成：建议保留 {len(images) - len(removed)} 张，移除 {len(removed)} 张；尚未移动任何图片。" if preview else "图片去重完成：未发现通过核验的重复样本，原数据未变。"
        context.report(message)
        context.report(f"详细分析报告：{report_path}")
        return {"message": message, "imageCount": len(images), "keptCount": len(images), "removedCount": 0,
                "proposedRemovedCount": len(removed), "preview": preview, "cycleDetails": cycles,
                "unreadableCount": len(unreadable), "analysisReport": str(report_path),
                "outputFiles": [str(report_path)], "outputFolders": [str(folder)]}
    return _apply_plan(folder, images, removed, unreadable, cycles, settings, context)


def _snapshot_images(images):
    snapshot = {}
    for image in images:
        for path in [image, *(image.with_suffix(suffix) for suffix in SIDECAR_SUFFIXES)]:
            try:
                stat = path.lstat()
                snapshot[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_mode)
            except FileNotFoundError:
                snapshot[path] = None
    return snapshot


def _plan_adjacent(images, context, strength, threshold):
    context.report(
        f"找到 {len(images)} 张图片；力度：{strength}，相似度阈值 {threshold:.3f}，"
        '不设最低保留比例或连续移除配额，候选再通过原图复核。'
    )
    sequences: dict[str, list[Candidate]] = {}
    unreadable: list[str] = []
    for index, path in enumerate(images, 1):
        context.check_cancelled()
        image = None if path.is_symlink() else _read_image(path)
        if image is None or image.size == 0:
            unreadable.append(path.name)
            continue
        sequences.setdefault(_sequence_key(path), []).append(Candidate(
            path=path,
            feature=_feature(image),
            annotation_signature=_annotation_signature(path),
        ))
        if index == len(images) or index % 50 == 0:
            context.report(f"相似度分析：{index}/{len(images)}")

    removed: dict[Path, dict[str, Any]] = {}
    for candidates in sequences.values():
        _kept, sequence_removed = _select_sequence(candidates, threshold)
        removed.update(sequence_removed)
    return removed, unreadable


def _apply_plan(folder, images, removed, unreadable, cycles, settings, context):
    created = datetime.now()
    batch_name = f"{created.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    batch = _backup_root(folder) / batch_name
    files_root = batch / "files"
    files_root.mkdir(parents=True, exist_ok=False)
    moved_files: list[dict[str, str]] = []
    completed: list[tuple[Path, Path]] = []
    try:
        for image_path in sorted(removed, key=_natural_key):
            context.check_cancelled()
            related = [image_path]
            related.extend(image_path.with_suffix(suffix) for suffix in SIDECAR_SUFFIXES if image_path.with_suffix(suffix).is_file())
            for source in related:
                destination = files_root / source.name
                if destination.exists():
                    raise RuntimeError(f"备份中出现同名文件：{source.name}")
                source.replace(destination)
                completed.append((source, destination))
                moved_files.append({
                    "original": source.name,
                    "backup": str(destination.relative_to(batch)),
                })
    except Exception:
        for source, destination in reversed(completed):
            if destination.exists() and not source.exists():
                destination.replace(source)
        shutil.rmtree(batch, ignore_errors=True)
        raise

    manifest = {
        "version": 1,
        "status": "active",
        "createdAt": created.isoformat(timespec="seconds"),
        "sourceFolder": str(folder),
        "settings": settings,
        "cycles": cycles,
        "originalImageCount": len(images),
        "keptImageCount": len(images) - len(removed),
        "removedImageCount": len(removed),
        "unreadableImages": unreadable,
        "removedImages": [
            {"name": path.name, **removed[path]} for path in sorted(removed, key=_natural_key)
        ],
        "movedFiles": moved_files,
    }
    manifest_path = batch / "manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        for source, destination in reversed(completed):
            if destination.exists() and not source.exists():
                destination.replace(source)
        shutil.rmtree(batch, ignore_errors=True)
        raise
    context.report(
        f"去重完成：保留 {len(images) - len(removed)} 张，移出 {len(removed)} 张；"
        f"备份批次：{batch_name}。"
    )
    context.report(f"详细去重与回滚清单：{manifest_path}")
    if unreadable:
        context.report(f"警告：{len(unreadable)} 张图片无法读取，已原样保留。")
    return {
        "message": f"图片去重完成：保留 {len(images) - len(removed)} 张，移出 {len(removed)} 张，可随时回滚。",
        "imageCount": len(images),
        "keptCount": len(images) - len(removed),
        "removedCount": len(removed),
        "unreadableCount": len(unreadable),
        "backupManifest": str(manifest_path),
        "cycleDetails": cycles,
        "outputFiles": [str(manifest_path)],
        "outputFolders": [str(folder)],
    }


register_handler(HANDLER_ID, run_image_dedup)
