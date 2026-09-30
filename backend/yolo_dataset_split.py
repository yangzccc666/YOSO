"""Safe, repeatable YOLO train/validation dataset preparation."""

from __future__ import annotations

import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .dataset_split_strategy import read_yolo_image_classes, split_with_class_coverage


HANDLER_ID = "yolo.split_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
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


def _dataset_folder_name(value: Any) -> str:
    name = str(value if value not in (None, "") else "").strip()
    if not name:
        name = f"yolo_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("数据集子文件夹名称只能填写单个文件夹名称，不能包含路径。")
    return name


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
    context: RunContext | None = None,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    missing: list[str] = []
    for image in images:
        if context:
            context.check_cancelled()
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


def _xml_classes(labels_folder: Path, context: RunContext | None = None) -> list[str]:
    """Match the reference script's class order: first appearance in XML scan order."""
    classes: list[str] = []
    for xml_path in labels_folder.glob("*.xml"):
        if context:
            context.check_cancelled()
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"XML 标签解析失败：{xml_path.name}（{exc}）") from exc
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                raise ValueError(f"XML 标签缺少类别名称：{xml_path.name}")
            if name not in classes:
                classes.append(name)
    return classes


def _read_class_names(labels_folder: Path | None, dataset_root: Path | None = None) -> list[str]:
    if not labels_folder:
        return []
    candidates = [labels_folder / "classes.txt", labels_folder.parent / "classes.txt"]
    if dataset_root:
        candidates.append(dataset_root / "classes.txt")
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
        width = int(float(root.findtext("size/width") or 0))
        height = int(float(root.findtext("size/height") or 0))
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
            xmin = int(float(bbox.findtext("xmin") or 0))
            ymin = int(float(bbox.findtext("ymin") or 0))
            xmax = int(float(bbox.findtext("xmax") or 0))
            ymax = int(float(bbox.findtext("ymax") or 0))
        except ValueError as exc:
            raise ValueError(f"XML 坐标不是有效数字：{xml_path.name}") from exc
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(f"XML 标注框无效或越界：{xml_path.name}")
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(
            f"{classes.index(name)} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return lines


def _write_sample(sample: Sample, split: str, output: Path, label_format: str) -> Path:
    split_folder = output / split
    image_target = split_folder / sample.image.name
    shutil.copy2(sample.image, image_target)
    if label_format == "仅图片":
        return image_target

    label_target = split_folder / f"{sample.image.stem}.txt"
    if sample.is_negative or sample.label is None:
        label_target.write_text("", encoding="utf-8")
    else:
        shutil.copy2(sample.label, label_target)
    return image_target


def _write_index(path: Path, images: list[Path]) -> None:
    path.write_text("".join(f"{image.resolve()}\n" for image in images), encoding="utf-8")


def _write_classes(path: Path, classes: list[str]) -> None:
    path.write_text(
        "".join(f"{index}: {name}\n" for index, name in enumerate(classes)),
        encoding="utf-8",
    )


def _data_yaml_parent(images_folder: Path, labels_folder: Path | None) -> Path:
    if labels_folder is None:
        return images_folder.parent
    try:
        common = Path(os.path.commonpath((str(images_folder), str(labels_folder))))
    except ValueError:
        return images_folder.parent
    if common in {images_folder, labels_folder}:
        return images_folder.parent
    return common


def _write_data_yaml(path: Path, output: Path, classes: list[str]) -> None:
    names = ",".join(f"'{name.replace(chr(39), chr(39) * 2)}'" for name in classes)
    content = (
        "# YOLOv5 🚀 by Ultralytics, GPL-3.0 license\n"
        "# COCO 2017 dataset http://cocodataset.org by Microsoft\n"
        "# Example usage: python train.py --data coco.yaml\n"
        "# parent\n"
        "# ├── yolov5\n"
        "# └── datasets\n"
        "#     └── coco  ← downloads here\n\n\n"
        "# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n"
        "  # dataset root dir\n"
        f"train: {(output / 'train.txt').resolve()}\n"
        f"val: {(output / 'val.txt').resolve()}\n"
        f"test: {(output / 'val.txt').resolve()}\n\n"
        "# Classes\n"
        f"nc: {len(classes)}  # number of classes\n"
        f"names: [{names}]  # class names\n\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"无法写入数据集配置文件：{path}（{exc}）") from exc


def _split_samples(samples: list[Sample], train_ratio: float, random_seed: int) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    random.Random(random_seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    return shuffled[:train_count], shuffled[train_count:]


def _prepare_output(output: Path, converted_labels_folder: Path | None = None) -> None:
    generated_folders = [output / "train", output / "val"]
    occupied = [folder for folder in generated_folders if folder.is_dir() and any(folder.iterdir())]
    if occupied:
        names = "、".join(folder.name for folder in occupied)
        raise ValueError(f"输出文件夹中的 {names} 已包含文件，请选择空输出文件夹，避免覆盖已有数据。")
    if converted_labels_folder and converted_labels_folder.is_dir() and any(converted_labels_folder.iterdir()):
        raise ValueError(
            f"转换标签文件夹已包含文件：{converted_labels_folder}。"
            "请清空或移走旧 TXT，避免覆盖已有标签。"
        )
    for folder in generated_folders:
        folder.mkdir(parents=True, exist_ok=True)
    if converted_labels_folder:
        try:
            converted_labels_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"无法创建转换标签文件夹：{converted_labels_folder}（{exc}）") from exc


def _convert_xml_labels(
    samples: list[Sample], converted_folder: Path, classes: list[str], context: RunContext | None = None
) -> list[Sample]:
    converted: list[Sample] = []
    for sample in samples:
        if context:
            context.check_cancelled()
        if sample.label is None:
            converted.append(sample)
            continue
        target = converted_folder / f"{sample.image.stem}.txt"
        lines = _xml_to_yolo_lines(sample.label, classes)
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted.append(Sample(image=sample.image, label=target, is_negative=sample.is_negative))
    return converted


def _run_local_yolo_dataset_split(context: RunContext) -> dict[str, Any]:
    task_type = str(context.parameters.get("dataset_task_type", "目标检测"))
    if task_type not in DATASET_TASK_TYPES:
        raise ValueError("请选择有效的数据集任务类型。")
    if task_type != "目标检测":
        raise ValueError(
            f"“{task_type}”模式已预留，但处理逻辑尚未接入；当前可使用“目标检测”模式。"
        )

    images_folder = _required_directory(context, "images_folder", "图片文件夹")
    output_parent = _required_directory(context, "output_folder", "输出位置", create=True)
    dataset_folder_name = _dataset_folder_name(context.parameters.get("dataset_folder_name"))
    output = output_parent / dataset_folder_name
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
        raise ValueError("图片文件夹中没有找到 JPG、JPEG 或 PNG 图片。")
    _ensure_unique_stems(images, "图片文件夹中")
    samples, missing = _collect_samples(
        images, labels_folder, label_suffix, missing_policy, context.report, context
    )

    negative_images = _images_in(negative_folder) if negative_folder else []
    _ensure_unique_stems(negative_images, "负样本文件夹中")
    existing_stems = {sample.image.stem.casefold() for sample in samples}
    duplicates = [image.name for image in negative_images if image.stem.casefold() in existing_stems]
    if duplicates:
        raise ValueError(f"负样本与普通样本存在同名图片：{'、'.join(duplicates[:5])}")
    if not samples:
        raise ValueError("没有可分配的有效样本。请检查标签文件或更改缺少标签时的处理方式。")

    dataset_root = _data_yaml_parent(images_folder, labels_folder)
    classes = (
        _xml_classes(labels_folder, context)
        if label_format == "VOC XML" and labels_folder
        else _read_class_names(labels_folder, dataset_root)
    )
    if label_format != "仅图片" and not classes:
        raise ValueError(
            "未找到类别名称，无法生成 data.yaml。"
            "使用 YOLO TXT 时，请在标签文件夹或其上级目录放置 classes.txt。"
        )
    context.check_cancelled()
    converted_labels_folder = labels_folder.parent / "txts" if label_format == "VOC XML" and labels_folder else None
    _prepare_output(output, converted_labels_folder)
    if label_format == "VOC XML":
        context.report(f"XML 转换标签目录：{converted_labels_folder}")
        samples = _convert_xml_labels(samples, converted_labels_folder, classes, context)

    negative_samples = [Sample(image=image, label=None, is_negative=True) for image in negative_images]
    image_classes = [read_yolo_image_classes(sample.label, len(classes)) for sample in samples]
    train_indices, val_indices, class_distribution = split_with_class_coverage(
        image_classes, train_ratio, random_seed, len(classes)
    )
    train_samples = [samples[index] for index in train_indices]
    val_samples = [samples[index] for index in val_indices]
    negative_train, negative_val = _split_samples(negative_samples, train_ratio, random_seed)
    train_samples.extend(negative_train)
    val_samples.extend(negative_val)

    if not val_samples:
        raise ValueError("类别保护后验证集为空：请补充独立图片或负样本，再划分数据集。")
    target_train = max(1, min(len(samples) - 1, int(len(samples) * train_ratio))) if len(samples) > 1 else 1
    if len(train_indices) > target_train:
        context.report(f"为保证每个已出现类别进入训练集，普通图片训练数量由目标 {target_train} 调整为 {len(train_indices)}。")
    for class_name, counts in zip(classes, class_distribution):
        total = counts["total"]
        class_target = max(1, min(total - 1, int(total * train_ratio + 0.5))) if total > 1 else total
        context.report(
            f"类别 {class_name}：共 {total} 张图片，按比例训练目标约 {class_target} 张；"
            f"实际训练 {counts['train']} 张，验证 {counts['val']} 张。"
        )
        if counts["total"] == 0:
            context.report(f"警告：类别 {class_name} 在有效样本中没有图片，请检查标签。")
        elif counts["val"] == 0:
            reason = "仅有 1 张图片" if counts["total"] == 1 else "当前图片共现关系或训练比例限制"
            context.report(f"警告：类别 {class_name} 验证集未覆盖（{reason}），该类别的验证指标不可靠。")

    context.report(
        f"任务类型：{task_type}。找到 {len(images)} 张普通图片、"
        f"{len(negative_images)} 张负样本，使用随机种子 {random_seed}。"
    )
    context.report(f"数据集输出目录：{output}")
    train_targets = []
    for sample in train_samples:
        context.check_cancelled()
        train_targets.append(_write_sample(sample, "train", output, label_format))
    val_targets = []
    for sample in val_samples:
        context.check_cancelled()
        val_targets.append(_write_sample(sample, "val", output, label_format))
    _write_index(output / "train.txt", train_targets)
    _write_index(output / "val.txt", val_targets)
    if classes:
        classes_file = dataset_root / "classes.txt"
        _write_classes(classes_file, classes)
        context.report(f"类别文件：{classes_file}")

    data_yaml = dataset_root / "data.yaml"
    _write_data_yaml(data_yaml, output, classes)
    context.report(f"训练配置文件：{data_yaml}")

    context.report(f"训练集 {len(train_samples)} 张，验证集 {len(val_samples)} 张。")
    context.report("源图片和标签未被修改。")
    return {
        "message": (
            f"YOLO 数据集划分完成：训练集 {len(train_samples)} 张，"
            f"验证集 {len(val_samples)} 张，共 {len(train_samples) + len(val_samples)} 张。"
        ),
        "trainCount": len(train_samples),
        "valCount": len(val_samples),
        "sampleCount": len(train_samples) + len(val_samples),
        "negativeCount": len(negative_images),
        "missingLabelCount": len(missing),
        "taskType": task_type,
        "classes": classes,
        "classDistribution": [dict(name=name, **counts) for name, counts in zip(classes, class_distribution)],
        "dataYaml": str(data_yaml),
        "trainingDefaults": {
            "data": str(data_yaml),
            "project": str(data_yaml.parent),
        },
        "outputFolders": [str(output)] + ([str(converted_labels_folder)] if converted_labels_folder else []),
        "executionLocation": "本地",
    }


def run_yolo_dataset_split(context: RunContext) -> dict[str, Any]:
    from .remote_yolo_dataset import is_remote_yolo_parameters, run_remote_yolo_dataset

    if is_remote_yolo_parameters(context.parameters):
        return run_remote_yolo_dataset(context)
    return _run_local_yolo_dataset_split(context)


register_handler(HANDLER_ID, run_yolo_dataset_split)
