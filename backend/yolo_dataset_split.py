"""Safe, repeatable YOLO train/validation dataset preparation."""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler


HANDLER_ID = "yolo.split_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_TASK_TYPES = {"目标检测", "旋转目标检测", "分割", "分类"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None
    is_negative: bool = False


def _required_directory(context: RunContext, field_id: str, label: str, *, create: bool = False) -> Path:
    value = context.paths.get(field_id)
    if not value:
        raise ValueError(f"请选择{label}。")
    path = value.expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise ValueError(f"{label}不存在或不是文件夹：{path}")
    return path


def _optional_directory(context: RunContext, field_id: str, label: str) -> Path | None:
    value = context.paths.get(field_id)
    if not value:
        return None
    path = value.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label}不存在或不是文件夹：{path}")
    return path


def _float_parameter(value: Any, label: str, default: float) -> float:
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字。") from exc


def _int_parameter(value: Any, label: str, default: int) -> int:
    try:
        number = float(default if value in (None, "") else value)
        if not number.is_integer():
            raise ValueError
        return int(number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc


def _unique_output_folder(parent: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"yolo_dataset_{timestamp}"
    index = 2
    while candidate.exists():
        candidate = parent / f"yolo_dataset_{timestamp}_{index}"
        index += 1
    return candidate


def _images_in(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (path.stem.casefold(), path.name.casefold()),
    )


def _ensure_unique_stems(images: list[Path], label: str) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for image in images:
        key = image.stem.casefold()
        if key in seen:
            duplicates.append(f"{seen[key]} / {image.name}")
        else:
            seen[key] = image.name
    if duplicates:
        preview = "；".join(duplicates[:3])
        raise ValueError(f"{label}存在同名图片，无法唯一匹配标签：{preview}")


def _collect_samples(
    images: list[Path],
    labels_folder: Path | None,
    label_suffix: str | None,
    missing_policy: str,
    report: Any,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    missing: list[str] = []
    for image in images:
        label = labels_folder / f"{image.stem}{label_suffix}" if labels_folder and label_suffix else None
        if label is not None and not label.is_file():
            missing.append(image.name)
            if missing_policy == "跳过并报告":
                continue
            label = None
        samples.append(Sample(image=image, label=label))
    if missing:
        preview = "、".join(missing[:5])
        suffix = "……" if len(missing) > 5 else ""
        report(f"发现 {len(missing)} 张图片缺少标签：{preview}{suffix}")
    return samples, missing


def _xml_classes(samples: list[Sample]) -> list[str]:
    classes: list[str] = []
    for sample in samples:
        if not sample.label:
            continue
        try:
            root = ET.parse(sample.label).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"XML 标签解析失败：{sample.label.name}（{exc}）") from exc
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                raise ValueError(f"XML 标签缺少类别名称：{sample.label.name}")
            if name not in classes:
                classes.append(name)
    return classes


def _read_class_names(labels_folder: Path | None) -> list[str]:
    if not labels_folder:
        return []
    candidates = [labels_folder / "classes.txt", labels_folder.parent / "classes.txt"]
    class_file = next((path for path in candidates if path.is_file()), None)
    if not class_file:
        return []
    names: list[str] = []
    for raw_line in class_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        prefix, separator, remainder = line.partition(":")
        name = remainder.strip() if separator and prefix.strip().isdigit() else line
        if name:
            names.append(name)
    return names


def _xml_to_yolo_lines(xml_path: Path, classes: list[str]) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
        width = float(root.findtext("size/width") or 0)
        height = float(root.findtext("size/height") or 0)
    except (ET.ParseError, OSError, ValueError) as exc:
        raise ValueError(f"XML 标签解析失败：{xml_path.name}（{exc}）") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"XML 图片尺寸无效：{xml_path.name}")

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bbox = obj.find("bndbox")
        if name not in classes or bbox is None:
            raise ValueError(f"XML 目标信息不完整：{xml_path.name}")
        try:
            xmin = float(bbox.findtext("xmin") or 0)
            ymin = float(bbox.findtext("ymin") or 0)
            xmax = float(bbox.findtext("xmax") or 0)
            ymax = float(bbox.findtext("ymax") or 0)
        except ValueError as exc:
            raise ValueError(f"XML 坐标不是有效数字：{xml_path.name}") from exc
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"XML 标注框尺寸无效：{xml_path.name}")
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(
            f"{classes.index(name)} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return lines


def _write_sample(sample: Sample, split: str, output: Path, label_format: str, classes: list[str]) -> Path:
    image_target = output / "images" / split / sample.image.name
    shutil.copy2(sample.image, image_target)
    if label_format == "仅图片":
        return image_target

    label_target = output / "labels" / split / f"{sample.image.stem}.txt"
    if sample.is_negative or sample.label is None:
        label_target.write_text("", encoding="utf-8")
    elif label_format == "VOC XML":
        lines = _xml_to_yolo_lines(sample.label, classes)
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        shutil.copy2(sample.label, label_target)
    return image_target


def _write_index(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{image.resolve()}\n" for image in images), encoding="utf-8")


def _write_dataset_yaml(output: Path, classes: list[str]) -> None:
    lines = [f"path: {output}", "train: images/train", "val: images/val"]
    if classes:
        lines.append("names:")
        lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_yolo_dataset_split(context: RunContext) -> dict[str, Any]:
    task_type = str(context.parameters.get("dataset_task_type", "目标检测"))
    if task_type not in DATASET_TASK_TYPES:
        raise ValueError("请选择有效的数据集任务类型。")
    if task_type != "目标检测":
        raise ValueError(
            f"“{task_type}”模式已预留，但处理逻辑尚未接入；当前可使用“目标检测”模式。"
        )

    images_folder = _required_directory(context, "images_folder", "图片文件夹")
    output_parent = _required_directory(context, "output_folder", "输出文件夹", create=True)
    negative_folder = _optional_directory(context, "negative_images_folder", "负样本图片文件夹")

    label_format = str(context.parameters.get("label_format", "YOLO TXT"))
    if label_format not in {"YOLO TXT", "VOC XML", "仅图片"}:
        raise ValueError("标签格式只能选择 YOLO TXT、VOC XML 或仅图片。")
    train_ratio = _float_parameter(context.parameters.get("train_ratio"), "训练集比例", 0.7)
    if not 0 < train_ratio < 1:
        raise ValueError("训练集比例必须大于 0 且小于 1。")
    random_seed = _int_parameter(context.parameters.get("random_seed"), "随机种子", 42)
    missing_policy = str(context.parameters.get("missing_label_policy", "跳过并报告"))
    if missing_policy not in {"跳过并报告", "生成空标签"}:
        raise ValueError("缺少标签时只能选择“跳过并报告”或“生成空标签”。")

    labels_folder: Path | None = None
    label_suffix: str | None = None
    if label_format != "仅图片":
        labels_folder = _required_directory(context, "labels_folder", "标签文件夹")
        label_suffix = ".xml" if label_format == "VOC XML" else ".txt"

    images = _images_in(images_folder)
    if not images:
        raise ValueError("图片文件夹中没有找到 JPG、JPEG、PNG、BMP 或 WEBP 图片。")
    _ensure_unique_stems(images, "图片文件夹中")
    samples, missing = _collect_samples(
        images, labels_folder, label_suffix, missing_policy, context.report
    )

    negative_images = _images_in(negative_folder) if negative_folder else []
    _ensure_unique_stems(negative_images, "负样本文件夹中")
    existing_stems = {sample.image.stem.casefold() for sample in samples}
    duplicates = [image.name for image in negative_images if image.stem.casefold() in existing_stems]
    if duplicates:
        raise ValueError(f"负样本与普通样本存在同名图片：{'、'.join(duplicates[:5])}")
    samples.extend(Sample(image=image, label=None, is_negative=True) for image in negative_images)
    if not samples:
        raise ValueError("没有可分配的有效样本。请检查标签文件或更改缺少标签时的处理方式。")

    classes = _xml_classes(samples) if label_format == "VOC XML" else _read_class_names(labels_folder)
    shuffled = list(samples)
    random.Random(random_seed).shuffle(shuffled)
    if len(shuffled) == 1:
        train_count = 1
    else:
        train_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * train_ratio)))
    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count:]

    output = _unique_output_folder(output_parent)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        if label_format != "仅图片":
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    context.report(
        f"任务类型：{task_type}。找到 {len(images)} 张普通图片、"
        f"{len(negative_images)} 张负样本，使用随机种子 {random_seed}。"
    )
    train_targets = [_write_sample(sample, "train", output, label_format, classes) for sample in train_samples]
    val_targets = [_write_sample(sample, "val", output, label_format, classes) for sample in val_samples]
    _write_index(output / "train.txt", train_targets)
    _write_index(output / "val.txt", val_targets)
    if classes:
        (output / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        _write_dataset_yaml(output, classes)
    elif label_format != "仅图片":
        context.report("未找到 classes.txt，已完成分配，但未生成需要类别名称的 data.yaml。")

    context.report(f"训练集 {len(train_samples)} 张，验证集 {len(val_samples)} 张。")
    context.report("源图片和标签未被修改。")
    return {
        "message": (
            f"YOLO 数据集分配完成：训练集 {len(train_samples)} 张，"
            f"验证集 {len(val_samples)} 张，共 {len(shuffled)} 张。"
        ),
        "trainCount": len(train_samples),
        "valCount": len(val_samples),
        "sampleCount": len(shuffled),
        "negativeCount": len(negative_images),
        "missingLabelCount": len(missing),
        "taskType": task_type,
        "classes": classes,
        "outputFolders": [str(output)],
    }


register_handler(HANDLER_ID, run_yolo_dataset_split)
