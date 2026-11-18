"""Extract VOC negative samples by moving images whose XML has no objects."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "dataset.extract_negative_samples"
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _has_voc_object(xml_file: Path) -> bool:
    root = ET.parse(xml_file).getroot()
    return any(element.tag.rsplit("}", 1)[-1].lower() == "object" for element in root.iter())


def _image_index(source: Path, recursive: bool, context: RunContext) -> dict[tuple[Path, str], list[Path]]:
    iterator = source.rglob("*") if recursive else source.iterdir()
    index: dict[tuple[Path, str], list[Path]] = {}
    for path in iterator:
        context.check_cancelled()
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parent = path.parent.relative_to(source)
        index.setdefault((relative_parent, path.stem.casefold()), []).append(path)
    return index


def run_extract_negative_samples(context: RunContext) -> dict[str, Any]:
    source = context.paths.get("source_folder")
    annotations = context.paths.get("xml_folder")
    destination = context.paths.get("negative_folder")
    if source is None or not source.expanduser().is_dir():
        raise ValueError("请选择包含 JPG/JPEG 图片的有效输入文件夹。")
    if annotations is not None and not annotations.expanduser().is_dir():
        raise ValueError("XML 标注文件夹不存在或不是有效文件夹；混放时请留空。")
    if destination is None:
        raise ValueError("请选择负样本图片输出文件夹。")
    source = source.expanduser().resolve()
    annotations = annotations.expanduser().resolve() if annotations is not None else source
    destination = destination.expanduser().resolve()
    if any(destination == folder or folder in destination.parents for folder in (source, annotations)):
        raise ValueError("负样本输出目录不能与图片或 XML 输入目录相同，也不能放在任一输入目录内部。")
    context.check_cancelled()
    destination.mkdir(parents=True, exist_ok=True)

    recursive = bool(context.parameters.get("recursive", False))
    xml_iterator = annotations.rglob("*") if recursive else annotations.iterdir()
    xml_files = []
    for path in xml_iterator:
        context.check_cancelled()
        if path.is_file() and path.suffix.lower() == ".xml":
            xml_files.append(path)
    xml_files.sort(key=lambda path: str(path).casefold())
    images = _image_index(source, recursive, context)
    by_stem: dict[str, list[Path]] = {}
    for (_, stem), paths in images.items():
        by_stem.setdefault(stem, []).extend(paths)
    # Build all associations before moving anything. Multiple XML files pointing
    # to one image must not let an empty XML remove a positively annotated image.
    matches: dict[Path, list[Path]] = {}
    for xml_file in xml_files:
        context.check_cancelled()
        relative_parent = xml_file.parent.relative_to(annotations)
        candidates = images.get((relative_parent, xml_file.stem.casefold()), [])
        if not candidates and annotations != source:
            candidates = by_stem.get(xml_file.stem.casefold(), [])
        matches[xml_file] = candidates
    uses = Counter(paths[0] for paths in matches.values() if len(paths) == 1)
    context.report(f"目录模式：{'图片与 XML 混放' if annotations == source else '图片与 XML 分开存放'}。")
    context.report(f"图片目录：{source}；XML 目录：{annotations}")
    context.report(f"找到 {len(xml_files)} 个 XML，正在检查是否包含目标标注。")

    moved: list[str] = []
    deleted_xml: list[str] = []
    skipped: list[dict[str, str]] = []
    labeled_count = 0

    for index, xml_file in enumerate(xml_files, 1):
        context.check_cancelled()
        candidates = matches[xml_file]
        if len(candidates) != 1 or uses[candidates[0]] > 1:
            reason = ("未找到同名 JPG/JPEG" if not candidates else
                      "找到多个同名 JPG/JPEG，无法确定对应图片" if len(candidates) != 1 else
                      "多个 XML 对应同一张图片，为避免误删已跳过")
            skipped.append({"xml": str(xml_file), "reason": reason})
            context.report(f"[{index}/{len(xml_files)}] 跳过 {xml_file.name}：{reason}")
            continue
        try:
            if _has_voc_object(xml_file):
                labeled_count += 1
                continue
        except (ET.ParseError, OSError) as exc:
            reason = f"XML 无法解析：{exc}"
            skipped.append({"xml": str(xml_file), "reason": reason})
            context.report(f"[{index}/{len(xml_files)}] 跳过 {xml_file.name}：{reason}")
            continue

        image = candidates[0]
        target_dir = destination / image.parent.relative_to(source) if recursive else destination
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / image.name
        if target.exists():
            reason = f"负样本目录已存在同名图片：{target}"
            skipped.append({"xml": str(xml_file), "reason": reason})
            context.report(f"[{index}/{len(xml_files)}] 跳过 {xml_file.name}：{reason}")
            continue

        try:
            shutil.move(str(image), str(target))
            try:
                xml_file.unlink()
            except OSError:
                shutil.move(str(target), str(image))
                raise
        except TaskCancelled:
            raise
        except OSError as exc:
            reason = f"移动图片或删除 XML 失败：{exc}"
            skipped.append({"xml": str(xml_file), "reason": reason})
            context.report(f"[{index}/{len(xml_files)}] 跳过 {xml_file.name}：{reason}")
            continue
        moved.append(str(target))
        deleted_xml.append(str(xml_file))
        context.report(f"[{index}/{len(xml_files)}] 已提取负样本：{image.name}，并删除 {xml_file.name}")

    message = f"负样本提取完成：移动 {len(moved)} 张图片，删除 {len(deleted_xml)} 个空标注 XML"
    if skipped:
        message += f"，跳过 {len(skipped)} 项"
    return {
        "message": message + "。",
        "xmlCount": len(xml_files),
        "negativeCount": len(moved),
        "labeledCount": labeled_count,
        "skippedCount": len(skipped),
        "movedImages": moved,
        "deletedXmlFiles": deleted_xml,
        "skipped": skipped,
        "outputFolders": [str(destination)],
    }


register_handler(HANDLER_ID, run_extract_negative_samples)
