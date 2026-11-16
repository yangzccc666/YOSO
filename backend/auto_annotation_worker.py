"""Batch prediction and VOC XML writing; also runs in a separate YOLO environment."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def write_xml(image, shape, detections, target):
    height, width = shape[:2]
    root = ET.Element('annotation')
    for key, value in [('folder', image.parent.name), ('filename', image.name), ('path', str(image))]:
        ET.SubElement(root, key).text = value
    size = ET.SubElement(root, 'size')
    for key, value in [('width', width), ('height', height), ('depth', 3)]:
        ET.SubElement(size, key).text = str(value)
    ET.SubElement(root, 'segmented').text = '0'
    count = 0
    for name, coordinates in detections:
        if not all(math.isfinite(float(v)) for v in coordinates):
            continue
        x1, y1, x2, y2 = coordinates
        # Pixel coordinates, clamped to the image, matching common VOC editors.
        box = [max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
               max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        obj = ET.SubElement(root, 'object')
        for key, value in [('name', name), ('pose', 'Unspecified'), ('truncated', '0'), ('difficult', '0')]:
            ET.SubElement(obj, key).text = str(value)
        bounds = ET.SubElement(obj, 'bndbox')
        for key, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), box):
            ET.SubElement(bounds, key).text = str(value)
        count += 1
    ET.indent(root)
    fd, temporary = tempfile.mkstemp(prefix='.annotation-', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            ET.ElementTree(root).write(stream, encoding='utf-8', xml_declaration=True)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return count


def collect_images(config):
    folder = Path(config['folder'])
    images = sorted(p for p in (folder.rglob('*') if config['recursive'] else folder.iterdir())
                    if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not images:
        raise ValueError('图片目录中没有支持的图片。')
    targets = set()
    for image in images:
        target = image.with_suffix('.xml')
        if target in targets:
            raise ValueError(f'同目录存在不同扩展名的同名图片，会产生同名 XML：{target}')
        targets.add(target)
    return images


def run(config, model, report=print, check_cancelled=lambda: None, images=None):
    folder = Path(config['folder'])
    report(f'正在扫描图片目录：{folder}')
    images = collect_images(config) if images is None else images
    if getattr(model, 'task', None) != 'detect':
        raise ValueError('当前自动标注仅支持目标检测 PT 模型。')
    names = model.names
    valid_ids = set(names) if isinstance(names, dict) else set(range(len(names)))
    if config['classes'] and not set(config['classes']).issubset(valid_ids):
        raise ValueError('类别编号超出模型的类别范围。')
    backup = folder / '.annotation_backups' / uuid.uuid4().hex
    result = dict(written=0, skipped=0, failed=0, objects=0, empty=0, total=len(images), backup='')
    report(f'找到 {len(images)} 张图片；模型类别：{names}')
    for index, image in enumerate(images, 1):
        check_cancelled()
        target = image.with_suffix('.xml')
        if target.exists() and config['existing'] == '跳过':
            result['skipped'] += 1
            report(f'[{index}/{len(images)}] 已有标注，跳过：{image.name}')
            continue
        try:
            started = time.monotonic()
            report(f'[{index}/{len(images)}] 开始检测：{image}；首次检测可能需要初始化推理环境。' if index == 1
                   else f'[{index}/{len(images)}] 开始检测：{image}')
            prediction = model.predict(source=str(image), conf=config['conf'], iou=config['iou'],
                                       imgsz=config['imgsz'], device=config['device'] or None,
                                       classes=config['classes'] or None, verbose=False, save=False, show=False)[0]
            report(f'[{index}/{len(images)}] 检测完成，用时 {time.monotonic()-started:.2f} 秒，正在生成 XML：{target.name}')
            boxes = prediction.boxes
            detections = [(names[int(cls)], xyxy) for cls, xyxy in
                          zip(boxes.cls.cpu().tolist(), boxes.xyxy.cpu().tolist())]
            check_cancelled()
            if not detections and not config['empty_xml']:
                result['skipped'] += 1
                report(f'[{index}/{len(images)}] 无目标，保留原标注不变：{image.name}')
                continue
            if target.exists():
                saved = backup / target.relative_to(folder)
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
                result['backup'] = str(backup)
            count = write_xml(image, prediction.orig_shape, detections, target)
            result['written'] += 1
            result['objects'] += count
            result['empty'] += int(count == 0)
            report(f'[{index}/{len(images)}] {image.name}：{count} 个目标 → {target.name}')
        except Exception as exc:
            check_cancelled()
            result['failed'] += 1
            report(f'[{index}/{len(images)}] 失败：{image.name}：{exc}')
    return result


if __name__ == '__main__':
    try:
        print('正在加载 Ultralytics / PyTorch…', flush=True)
        from ultralytics import YOLO
        config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        print(f"正在加载 PT 模型：{config['weights']}", flush=True)
        model = YOLO(config['weights'])
        print('PT 模型加载完成。', flush=True)
        result = run(config, model, lambda text: print(text, flush=True))
        print('RESULT_JSON:' + json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f'自动标注失败：{exc}', flush=True)
        sys.exit(1)
