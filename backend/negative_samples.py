"""Extract VOC negative samples by moving images whose XML has no objects."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "dataset.extract_negative_samples"
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _has_voc_object(xml_file: Path) -> bool:
    root = ET.parse(xml_file).getroot()
    return any(element.tag.rsplit("}", 1)[-1].lower() == "object" for element in root.iter())


def _image_index(source: Path, recursive: bool) -> dict[tuple[Path, str], list[Path]]:
    iterator = source.rglob("*") if recursive else source.iterdir()
    index: dict[tuple[Path, str], list[Path]] = {}
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parent = path.parent.relative_to(source)
        index.setdefault((relative_parent, path.stem.casefold()), []).append(path)
    return index


def run_extract_negative_samples(context: RunContext) -> dict[str, Any]:
    source = context.paths.get("source_folder")
    destination = context.paths.get("negative_folder")
    if source is None or not source.expanduser().is_dir():
        raise ValueError("请选择包含 JPG 和 XML 的有效输入文件夹。")
    if destination is None:
        raise ValueError("请选择负样本图片输出文件夹。")
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("负样本输出目录不能与输入目录相同，也不能放在输入目录内部。")
    destination.mkdir(parents=True, exist_ok=True)

    recursive = bool(context.parameters.get("recursive", False))
    xml_iterator = source.rglob("*") if recursive else source.iterdir()
    xml_files = sorted(
        (path for path in xml_iterator if path.is_file() and path.suffix.lower() == ".xml"),
        key=lambda path: str(path).casefold(),
    )
    images = _image_index(source, recursive)
    context.report(f"找到 {len(xml_files)} 个 XML，正在检查是否包含目标标注。")

    moved: list[str] = []
    deleted_xml: list[str] = []
    skipped: list[dict[str, str]] = []
    labeled_count = 0

    for index, xml_file in enumerate(xml_files, 1):
        context.check_cancelled()
        relative_parent = xml_file.parent.relative_to(source)
        candidates = images.get((relative_parent, xml_file.stem.casefold()), [])
        if len(candidates) != 1:
            reason = "未找到同名 JPG/JPEG" if not candidates else "找到多个同名 JPG/JPEG，无法确定对应图片"
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
        target_dir = destination / relative_parent if recursive else destination
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
