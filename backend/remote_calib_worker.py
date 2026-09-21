"""Select a class- and brightness-diverse calibration image set on the SSH host."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dirs-json", required=True)
    parser.add_argument("--annotation-dirs-json", required=True)
    parser.add_argument("--negative-dirs-json", default="[]")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("auto", "xml", "json"), default="auto")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label}不存在或不是文件夹：{path}")
    return path


def _images(folder: Path) -> list[Path]:
    return sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), key=lambda path: path.name)


def _annotation_format(image_dirs: list[Path], annotation_dirs: list[Path], requested: str) -> str:
    if requested != "auto":
        return requested
    # Keep the reference script's preference for paired XML annotations and
    # LabelMe JSON next to images, while also accepting in-place XML labels.
    for folders, suffix, name in (
        (annotation_dirs, ".xml", "xml"),
        (image_dirs, ".json", "json"),
        (annotation_dirs, ".json", "json"),
        (image_dirs, ".xml", "xml"),
    ):
        if any(any(path.is_file() and path.suffix.lower() == suffix for path in folder.iterdir()) for folder in folders):
            return name
    raise ValueError("未检测到 XML 或 JSON 标注；请检查标注目录，或明确选择标注格式。")


def _labels(path: Path | None, fmt: str) -> set[str]:
    if path is None:
        return {"background"}
    if fmt == "xml":
        root = ET.parse(path).getroot()
        labels = {(element.findtext("name") or "").strip() for element in root.findall("object")}
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        labels = {str(shape.get("label", "")).strip() for shape in data.get("shapes", []) if isinstance(shape, dict)}
    labels.discard("")
    return labels or {"background"}


def _brightness(path: Path) -> float:
    with Image.open(path) as image:
        return float(ImageStat.Stat(image.convert("L")).mean[0])


def _bucket(brightness: float) -> str:
    return "dark" if brightness < 80 else "medium" if brightness < 170 else "bright"


def _select(records: list[dict], count: int, seed: int) -> list[dict]:
    """Proportional class allocation with brightness diversity and unique images."""
    rng = random.Random(seed)
    target = min(count, len(records))
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for label in record["labels"]:
            by_label[label].append(record)
    total_memberships = sum(len(items) for items in by_label.values())
    selected: list[dict] = []
    selected_paths: set[Path] = set()
    labels = sorted(by_label, key=lambda label: (len(by_label[label]), label))
    quotas = {label: max(1, round(target * len(by_label[label]) / total_memberships)) for label in labels}
    for label in labels:
        candidates = [record for record in by_label[label] if record["path"] not in selected_paths]
        groups = {bucket: [record for record in candidates if record["bucket"] == bucket] for bucket in ("dark", "medium", "bright")}
        chosen: list[dict] = []
        for bucket in groups:
            if groups[bucket] and len(chosen) < quotas[label]:
                chosen.append(rng.choice(groups[bucket]))
        remaining = [record for record in candidates if record not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:max(0, quotas[label] - len(chosen))])
        for record in chosen[:max(0, target - len(selected))]:
            if record["path"] not in selected_paths:
                selected.append(record)
                selected_paths.add(record["path"])
        if len(selected) >= target:
            break
    remaining = [record for record in records if record["path"] not in selected_paths]
    rng.shuffle(remaining)
    selected.extend(remaining[:target - len(selected)])
    return selected


def _supplement_counts(pools: list[list[dict]], count: int) -> list[int]:
    """Allocate remaining slots proportionally, reserving one per available pool."""
    allocations = [0] * len(pools)
    available = [index for index, pool in enumerate(pools) if pool]
    if count <= 0 or not available:
        return allocations
    if count < len(available):
        for index in sorted(available, key=lambda value: (-len(pools[value]), value))[:count]:
            allocations[index] = 1
        return allocations
    for index in available:
        allocations[index] = 1
    remaining = count - len(available)
    while remaining:
        candidates = [index for index in available if allocations[index] < len(pools[index])]
        if not candidates:
            break
        index = max(candidates, key=lambda value: (len(pools[value]) / allocations[value], len(pools[value]), -value))
        allocations[index] += 1
        remaining -= 1
    return allocations


def _select_prioritized(records: list[dict], count: int, seed: int) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Prefer every non-copy source image, then add copy and negative images."""
    target = min(count, len(records))
    pools = {
        "original": [record for record in records if record["source"] == "original"],
        "copy": [record for record in records if record["source"] == "copy"],
        "negative": [record for record in records if record["source"] == "negative"],
    }
    available = {name: len(pool) for name, pool in pools.items()}
    if len(pools["original"]) >= target:
        selected = _select(pools["original"], target, seed)
    else:
        # When the genuine dataset is smaller than the target, keeping it in full
        # is more valuable than sampling it again. Augmented copies/backgrounds
        # only fill the remaining slots.
        selected = list(pools["original"])
        remaining = target - len(selected)
        copy_count, negative_count = _supplement_counts([pools["copy"], pools["negative"]], remaining)
        selected.extend(_select(pools["copy"], copy_count, seed + 1))
        selected.extend(_select(pools["negative"], negative_count, seed + 2))
    selected_counts = Counter(record["source"] for record in selected)
    return selected, available, {name: selected_counts.get(name, 0) for name in pools}


def run(args: argparse.Namespace) -> dict:
    image_values = json.loads(args.image_dirs_json)
    annotation_values = json.loads(args.annotation_dirs_json)
    negative_values = json.loads(getattr(args, "negative_dirs_json", "[]"))
    if not isinstance(image_values, list) or not image_values or not all(isinstance(value, str) and value.strip() for value in image_values):
        raise ValueError("请至少填写一个图片目录。")
    if not isinstance(annotation_values, list) or not all(isinstance(value, str) and value.strip() for value in annotation_values):
        raise ValueError("标注目录格式不正确。")
    if not isinstance(negative_values, list) or not all(isinstance(value, str) and value.strip() for value in negative_values):
        raise ValueError("负样本图片目录格式不正确。")
    if len(annotation_values) > len(image_values):
        raise ValueError("标注目录数量不能多于图片目录数量。")
    if not 1 <= args.count <= 100_000:
        raise ValueError("校准图片数量必须在 1 到 100000 之间。")
    image_dirs = [_directory(value, "图片目录") for value in image_values]
    annotation_dirs = [_directory(value, "标注目录") for value in annotation_values]
    negative_dirs = [_directory(value, "负样本图片目录") for value in negative_values]
    fmt = _annotation_format(image_dirs, annotation_dirs, args.format)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"输出目录已包含文件，请选择空文件夹，避免覆盖已有校准集：{output}")
    records: list[dict] = []
    seen_paths: set[Path] = set()
    for index, folder in enumerate(image_dirs):
        ann_folder = annotation_dirs[index] if index < len(annotation_dirs) else folder
        ann_index = {path.stem: path for path in ann_folder.iterdir() if path.is_file() and path.suffix.lower() == f".{fmt}"}
        images = _images(folder)
        print(f"图片目录 {index + 1}/{len(image_dirs)}：{folder}，找到 {len(images)} 张图片。", flush=True)
        for path in images:
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                annotation = ann_index.get(path.stem)
                records.append({
                    "path": resolved,
                    "labels": _labels(annotation, fmt),
                    "bucket": _bucket(_brightness(path)),
                    # Source priority is based on the directory and filename,
                    # not annotation presence: an unlabeled genuine image must
                    # still be protected from supplementary sampling.
                    "source": "copy" if "copy" in path.stem.casefold() else "original",
                })
            except (OSError, ValueError, ET.ParseError, KeyError, TypeError) as exc:
                print(f"警告：跳过无法读取的图片或标注 {path.name}：{exc}", flush=True)
    for index, folder in enumerate(negative_dirs):
        images = _images(folder)
        print(f"负样本目录 {index + 1}/{len(negative_dirs)}：{folder}，找到 {len(images)} 张图片。", flush=True)
        for path in images:
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                records.append({
                    "path": resolved,
                    "labels": {"background"},
                    "bucket": _bucket(_brightness(path)),
                    "source": "negative",
                })
            except (OSError, ValueError) as exc:
                print(f"警告：跳过无法读取的负样本 {path.name}：{exc}", flush=True)
    if not records:
        raise ValueError("没有可用图片，请检查远程图片目录和文件格式。")
    print(f"标注格式：{fmt}。可用图片 {len(records)} 张，目标抽取 {args.count} 张。", flush=True)
    selected, available_sources, selected_sources = _select_prioritized(records, args.count, args.seed)
    print(
        "分层统计："
        f"原始图片 {available_sources['original']} 张（选中 {selected_sources['original']}），"
        f"copy 扩充图片 {available_sources['copy']} 张（选中 {selected_sources['copy']}），"
        f"负样本 {available_sources['negative']} 张（选中 {selected_sources['negative']}）。",
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for index, record in enumerate(selected, 1):
        source = record["path"]
        base = source.stem
        name = f"{base}.jpg"
        suffix = 2
        while name.casefold() in used_names:
            name = f"{base}_{suffix}.jpg"
            suffix += 1
        used_names.add(name.casefold())
        destination = output / name
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(source, destination)
        else:
            with Image.open(source) as image:
                image.convert("RGB").save(destination, "JPEG", quality=95)
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"已生成 {index}/{len(selected)} 张校准图片。", flush=True)
    distribution = Counter(label for record in selected for label in record["labels"])
    for label, amount in sorted(distribution.items(), key=lambda item: (-item[1], item[0])):
        print(f"类别 {label}：{amount} 张", flush=True)
    return {
        "message": f"量化校准数据集制作完成：{len(selected)} 张 JPG 图片。",
        "sampleCount": len(selected),
        "availableCount": len(records),
        "annotationFormat": fmt,
        "classCounts": dict(distribution),
        "availableSourceCounts": available_sources,
        "selectedSourceCounts": selected_sources,
        "outputFolders": [str(output)],
    }


if __name__ == "__main__":
    try:
        result = run(parse_args())
        print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"ERROR: {str(exc).strip() or exc.__class__.__name__}", file=sys.stderr, flush=True)
        raise SystemExit(1)
