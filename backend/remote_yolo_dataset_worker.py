#!/usr/bin/env python3
"""Dependency-free YOLO dataset splitter uploaded to an SSH server."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from .dataset_split_strategy import read_yolo_image_classes, split_with_class_coverage
except ImportError:  # Uploaded script runs directly beside the helper on SSH.
    from dataset_split_strategy import read_yolo_image_classes, split_with_class_coverage


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None
    is_negative: bool = False


def log(message: str) -> None:
    print(message, flush=True)


def required_directory(value: str, label: str, create: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise ValueError(f"{label}不存在或不是文件夹：{path}")
    return path


def optional_directory(value: str, label: str) -> Path | None:
    if not value.strip():
        return None
    return required_directory(value, label)


def images_in(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (path.stem.casefold(), path.name.casefold()),
    )


def ensure_unique_stems(images: list[Path], label: str) -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for image in images:
        key = image.stem.casefold()
        if key in seen:
            duplicates.append(f"{seen[key]} / {image.name}")
        else:
            seen[key] = image.name
    if duplicates:
        raise ValueError(f"{label}存在同名图片，无法唯一匹配标签：{'；'.join(duplicates[:3])}")


def collect_samples(
    images: list[Path], labels_folder: Path | None, label_suffix: str | None, missing_policy: str
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    missing: list[str] = []
    for image in images:
        label = labels_folder / f"{image.stem}{label_suffix}" if labels_folder and label_suffix else None
        if label is not None and not label.is_file():
            missing.append(image.name)
            if missing_policy == "skip":
                continue
            label = None
        samples.append(Sample(image=image, label=label))
    if missing:
        preview = "、".join(missing[:5])
        log(f"发现 {len(missing)} 张图片缺少标签：{preview}{'……' if len(missing) > 5 else ''}")
    return samples, missing


def xml_classes(labels_folder: Path) -> list[str]:
    classes: list[str] = []
    for xml_path in labels_folder.glob("*.xml"):
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


def read_class_names(labels_folder: Path | None, dataset_root: Path | None = None) -> list[str]:
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


def xml_to_yolo_lines(xml_path: Path, classes: list[str]) -> list[str]:
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
        lines.append(
            f"{classes.index(name)} {((xmin + xmax) / 2) / width:.8f} "
            f"{((ymin + ymax) / 2) / height:.8f} {(xmax - xmin) / width:.8f} "
            f"{(ymax - ymin) / height:.8f}"
        )
    return lines


def prepare_output(output: Path, converted_labels_folder: Path | None = None) -> None:
    folders = [output / "train", output / "val"]
    occupied = [folder for folder in folders if folder.is_dir() and any(folder.iterdir())]
    if occupied:
        raise ValueError(
            f"输出文件夹中的 {'、'.join(folder.name for folder in occupied)} 已包含文件，"
            "请选择空输出位置或修改数据集子文件夹名称。"
        )
    if converted_labels_folder and converted_labels_folder.is_dir() and any(converted_labels_folder.iterdir()):
        raise ValueError(
            f"转换标签文件夹已包含文件：{converted_labels_folder}。"
            "请清空或移走旧 TXT，避免覆盖已有标签。"
        )
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    if converted_labels_folder:
        try:
            converted_labels_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"无法创建转换标签文件夹：{converted_labels_folder}（{exc}）") from exc


def convert_xml_labels(samples: list[Sample], converted_labels_folder: Path, classes: list[str]) -> list[Sample]:
    converted: list[Sample] = []
    for sample in samples:
        if sample.label is None:
            converted.append(sample)
            continue
        target = converted_labels_folder / f"{sample.image.stem}.txt"
        lines = xml_to_yolo_lines(sample.label, classes)
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted.append(Sample(sample.image, target, sample.is_negative))
    return converted


def split_samples(samples: list[Sample], ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * ratio)
    return shuffled[:train_count], shuffled[train_count:]


def write_sample(sample: Sample, split: str, output: Path, label_format: str) -> Path:
    target_folder = output / split
    image_target = target_folder / sample.image.name
    shutil.copy2(sample.image, image_target)
    if label_format != "images_only":
        label_target = target_folder / f"{sample.image.stem}.txt"
        if sample.is_negative or sample.label is None:
            label_target.write_text("", encoding="utf-8")
        else:
            shutil.copy2(sample.label, label_target)
    return image_target


def data_yaml_parent(images_folder: Path, labels_folder: Path | None) -> Path:
    if labels_folder is None:
        return images_folder.parent
    try:
        common = Path(os.path.commonpath((str(images_folder), str(labels_folder))))
    except ValueError:
        return images_folder.parent
    return images_folder.parent if common in {images_folder, labels_folder} else common


def write_data_yaml(path: Path, output: Path, classes: list[str]) -> None:
    names = ",".join(f"'{name.replace(chr(39), chr(39) * 2)}'" for name in classes)
    content = (
        "# YOLOv5 🚀 by Ultralytics, GPL-3.0 license\n"
        "# COCO 2017 dataset http://cocodataset.org by Microsoft\n"
        "# Example usage: python train.py --data coco.yaml\n"
        "# parent\n# ├── yolov5\n# └── datasets\n#     └── coco  ← downloads here\n\n\n"
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
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-folder", required=True)
    parser.add_argument("--labels-folder", default="")
    parser.add_argument("--negative-folder", default="")
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--dataset-folder-name", default="")
    parser.add_argument("--label-format", choices=["yolo_txt", "voc_xml", "images_only"], required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-policy", choices=["skip", "empty"], default="skip")
    parser.add_argument("--session-token", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0 < args.train_ratio < 1:
        raise ValueError("训练集比例必须大于 0 且小于 1。")
    name = args.dataset_folder_name.strip() or f"yolo_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("数据集子文件夹名称只能填写单个文件夹名称，不能包含路径。")

    images_folder = required_directory(args.images_folder, "图片文件夹")
    labels_folder = None if args.label_format == "images_only" else required_directory(args.labels_folder, "标签文件夹")
    negative_folder = optional_directory(args.negative_folder, "负样本图片文件夹")
    output_parent = required_directory(args.output_parent, "输出位置", create=True)
    output = output_parent / name

    images = images_in(images_folder)
    if not images:
        raise ValueError("图片文件夹中没有找到 JPG、JPEG 或 PNG 图片。")
    ensure_unique_stems(images, "图片文件夹中")
    suffix = ".xml" if args.label_format == "voc_xml" else ".txt" if args.label_format == "yolo_txt" else None
    samples, missing = collect_samples(images, labels_folder, suffix, args.missing_policy)
    if not samples:
        raise ValueError("没有可划分的有效样本。请检查标签文件。")

    negative_images = images_in(negative_folder) if negative_folder else []
    ensure_unique_stems(negative_images, "负样本文件夹中")
    existing_stems = {sample.image.stem.casefold() for sample in samples}
    duplicates = [image.name for image in negative_images if image.stem.casefold() in existing_stems]
    if duplicates:
        raise ValueError(f"负样本与普通样本存在同名图片：{'、'.join(duplicates[:5])}")

    dataset_root = data_yaml_parent(images_folder, labels_folder)
    classes = (
        xml_classes(labels_folder)
        if args.label_format == "voc_xml" and labels_folder
        else read_class_names(labels_folder, dataset_root)
    )
    if args.label_format != "images_only" and not classes:
        raise ValueError("未找到类别名称。YOLO TXT 模式请在标签文件夹或上级目录放置 classes.txt。")
    converted_labels_folder = labels_folder.parent / "txts" if args.label_format == "voc_xml" and labels_folder else None
    prepare_output(output, converted_labels_folder)
    if args.label_format == "voc_xml":
        log(f"XML 转换标签目录：{converted_labels_folder}")
        samples = convert_xml_labels(samples, converted_labels_folder, classes)

    image_classes = [read_yolo_image_classes(sample.label, len(classes)) for sample in samples]
    train_indices, val_indices, class_distribution = split_with_class_coverage(
        image_classes, args.train_ratio, args.seed, len(classes)
    )
    train_samples = [samples[index] for index in train_indices]
    val_samples = [samples[index] for index in val_indices]
    negative_samples = [Sample(image=image, label=None, is_negative=True) for image in negative_images]
    negative_train, negative_val = split_samples(negative_samples, args.train_ratio, args.seed)
    train_samples.extend(negative_train)
    val_samples.extend(negative_val)

    if not val_samples:
        raise ValueError("类别保护后验证集为空：请补充独立图片或负样本，再划分数据集。")
    target_train = max(1, min(len(samples) - 1, int(len(samples) * args.train_ratio))) if len(samples) > 1 else 1
    if len(train_indices) > target_train:
        log(f"为保证每个已出现类别进入训练集，普通图片训练数量由目标 {target_train} 调整为 {len(train_indices)}。")
    for class_name, counts in zip(classes, class_distribution):
        total = counts["total"]
        class_target = max(1, min(total - 1, int(total * args.train_ratio + 0.5))) if total > 1 else total
        log(
            f"类别 {class_name}：共 {total} 张图片，按比例训练目标约 {class_target} 张；"
            f"实际训练 {counts['train']} 张，验证 {counts['val']} 张。"
        )
        if counts["total"] == 0:
            log(f"警告：类别 {class_name} 在有效样本中没有图片，请检查标签。")
        elif counts["val"] == 0:
            reason = "仅有 1 张图片" if counts["total"] == 1 else "当前图片共现关系或训练比例限制"
            log(f"警告：类别 {class_name} 验证集未覆盖（{reason}），该类别的验证指标不可靠。")

    log(f"找到 {len(images)} 张普通图片、{len(negative_images)} 张负样本。")
    log(f"数据集输出目录：{output}")
    train_targets = [write_sample(sample, "train", output, args.label_format) for sample in train_samples]
    val_targets = [write_sample(sample, "val", output, args.label_format) for sample in val_samples]
    (output / "train.txt").write_text("".join(f"{path.resolve()}\n" for path in train_targets), encoding="utf-8")
    (output / "val.txt").write_text("".join(f"{path.resolve()}\n" for path in val_targets), encoding="utf-8")
    if classes:
        classes_file = dataset_root / "classes.txt"
        classes_file.write_text(
            "".join(f"{index}: {class_name}\n" for index, class_name in enumerate(classes)), encoding="utf-8"
        )
        log(f"类别文件：{classes_file}")
    data_yaml = dataset_root / "data.yaml"
    write_data_yaml(data_yaml, output, classes)
    log(f"训练集 {len(train_samples)} 张，验证集 {len(val_samples)} 张。")
    log(f"训练配置文件：{data_yaml}")
    return {
        "message": f"远程 YOLO 数据集划分完成：训练集 {len(train_samples)} 张，验证集 {len(val_samples)} 张。",
        "trainCount": len(train_samples),
        "valCount": len(val_samples),
        "sampleCount": len(train_samples) + len(val_samples),
        "negativeCount": len(negative_images),
        "missingLabelCount": len(missing),
        "taskType": "目标检测",
        "classes": classes,
        "classDistribution": [dict(name=name, **counts) for name, counts in zip(classes, class_distribution)],
        "dataYaml": str(data_yaml),
        "trainingDefaults": {
            "data": str(data_yaml),
            "project": str(data_yaml.parent),
        },
        "outputFolders": [str(output)] + ([str(converted_labels_folder)] if converted_labels_folder else []),
        "executionLocation": "SSH远程服务器",
    }


def main() -> int:
    try:
        result = run(parse_args())
        print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {str(exc).strip() or exc.__class__.__name__}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
