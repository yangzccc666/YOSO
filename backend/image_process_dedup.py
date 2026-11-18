"""Workstation-scoped visual representatives, not pixel-identical frames.

The user explicitly identifies the relevant workstation. Outside pixels never
participate. This is a visual heuristic, NOT a learned process/defect classifier.
"""
from __future__ import annotations

import base64
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .image_quality_dedup import annotation_key, descriptor, read_original

PROCESS_MODE = '工位工序去重（忽略背景）'
PROCESS_RULE = 'workstation-visual-representatives-v1'
# (global structural similarity, worst local tile, mean absolute difference)
PROFILES = {'保守': (.95, .74, .045), '较保守': (.92, .67, .055),
            '平衡（推荐）': (.88, .57, .075), '较强': (.84, .47, .09), '强力': (.80, .38, .11)}


def load_regions(raw) -> dict:
    try:
        config = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
        if not isinstance(config, dict) or config.get('version') != 1 or not isinstance(config.get('regions'), dict):
            raise ValueError()
        root = Path(config['root']).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise ValueError()
        regions = {}
        for key, coords in config['regions'].items():
            if not isinstance(key, str) or Path(key).is_absolute() or '..' in Path(key).parts:
                raise ValueError()
            if not isinstance(coords, list) or len(coords) != 4 or any(isinstance(x, bool) for x in coords):
                raise ValueError()
            coords = [float(x) for x in coords]
            if not all(math.isfinite(x) for x in coords) or not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
                raise ValueError()
            regions[str(Path(key))] = coords
        return {'version': 1, 'root': str(root.resolve()), 'regions': regions}
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError('请先读取真实图片目录，为要处理的工位框选并保存关注区域。') from exc


def region_for(config: dict, folder: Path) -> list[float]:
    try:
        key = str(folder.relative_to(Path(config['root'])))
    except ValueError as exc:
        raise ValueError('图片路径已改变，请重新读取目录并框选工位，避免套用其他场景的区域。') from exc
    if key not in config['regions']:
        raise ValueError(f'目录 {key} 尚未框选工位区域，请先保存该目录的关注区域。')
    return config['regions'][key]


def _crop(image, region):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('工位模式只处理可读取的 8 位彩色图片')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = region
    patch = image[math.floor(y1 * h):math.ceil(y2 * h), math.floor(x1 * w):math.ceil(x2 * w)]
    if min(patch.shape[:2]) < 16:
        raise ValueError('工位区域过小，请至少框选 16×16 像素且覆盖完整作业区')
    return patch


def preview_regions(payload):
    """Read-only UI preview, restricted to a directory actually in this scan."""
    from .handlers import RunContext
    from .image_dedup import IMAGE_SUFFIXES, _natural_key
    from .image_dedup_batch import _folders
    root = Path(str(payload.get('imagesFolder', ''))).expanduser().resolve()
    if not str(payload.get('imagesFolder', '')).strip() or not root.is_dir():
        raise ValueError('请先填写存在的图片文件夹。')
    recursive = payload.get('recursive', False)
    if not isinstance(recursive, bool):
        raise ValueError('包含子文件夹参数不正确。')
    context = RunContext('image_dedup', {}, {}, lambda _: None)
    folders = _folders(root, context) if recursive else [root]
    if len(folders) > 2000:
        raise ValueError('图片目录超过 2000 个，请缩小输入范围后再框选。')
    keys = [str(folder.relative_to(root)) for folder in folders]
    key = str(payload.get('folderKey', keys[0] if keys else '.'))
    if key not in keys:
        raise ValueError('所选目录不在当前扫描范围中，请重新读取目录。')
    folder = folders[keys.index(key)]
    images = sorted((p for p in folder.iterdir() if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
    sample = str(payload.get('sampleName', ''))
    if sample:
        images = [p for p in images if p.name == sample]  # No arbitrary path traversal.
    for path in images:
        image = read_original(path)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            continue
        h, w = image.shape[:2]
        thumb = cv2.resize(image, (max(1, round(w * min(1, 960 / w))), max(1, round(h * min(1, 960 / w)))))
        ok, encoded = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            return {'root': str(root), 'folders': keys, 'folderKey': key, 'sampleName': path.name,
                    'width': w, 'height': h, 'image': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')}
    raise ValueError(f'目录 {key} 没有可预览的彩色图片，请检查文件或更换示例文件名。')


def _annotations(path, shape, region):
    # Validate every sidecar first; bad/unsupported annotations protect the image.
    annotation_key(path)
    h, w = shape[:2]
    result, flags = {}, {}
    for suffix in ('.txt', '.xml', '.json'):
        sidecar = path.with_suffix(suffix)
        if not sidecar.exists():
            continue
        objects = []
        if suffix == '.txt':
            for line in sidecar.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError('工位模式暂只支持检测框 TXT，其他标注请使用严格模式')
                cx, cy, bw, bh = map(float, parts[1:])
                objects.append((parts[0], [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]))
        elif suffix == '.xml':
            root = ET.parse(sidecar).getroot()
            for node in root.findall('object'):
                box = node.find('bndbox')
                if box is None:
                    raise ValueError('工位模式需要 VOC 矩形框')
                label = (node.findtext('name'), node.findtext('difficult', '0'), node.findtext('truncated', '0'))
                objects.append((str(label), [float(box.findtext(k)) / scale for k, scale in
                                [('xmin', w), ('ymin', h), ('xmax', w), ('ymax', h)]]))
        else:
            data = json.loads(sidecar.read_text(encoding='utf-8-sig'))
            flags[suffix] = json.dumps(data.get('flags', {}), sort_keys=True)
            for obj in data['shapes']:
                if obj.get('shape_type') != 'rectangle' or len(obj.get('points', [])) != 2:
                    raise ValueError('工位模式暂只支持 LabelMe 矩形框')
                (x1, y1), (x2, y2) = obj['points']
                label = str((obj['label'], json.dumps(obj.get('flags', {}), sort_keys=True)))
                objects.append((label, [min(x1, x2)/w, min(y1, y2)/h, max(x1, x2)/w, max(y1, y2)/h]))
        selected = []
        for label, box in objects:
            if not all(math.isfinite(x) for x in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('标注框坐标无效')
            if min(box[2], region[2]) > max(box[0], region[0]) and min(box[3], region[3]) > max(box[1], region[1]):
                selected.append((label, box))
        result[suffix] = selected
    signature = json.dumps({k: sorted(Counter(label for label, _ in v).items()) for k, v in result.items()}, sort_keys=True)
    return (signature, json.dumps(flags, sort_keys=True)), result


def _same_annotations(left, right):
    for suffix, objects in left.items():
        remaining = list(right.get(suffix, []))
        for label, a in objects:
            def iou(item):
                name, b = item
                if name != label:
                    return -1
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                return intersection / max(1e-10, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)
            if not remaining:
                return False
            index = max(range(len(remaining)), key=lambda i: iou(remaining[i]))
            if iou(remaining[index]) < .6:
                return False
            remaining.pop(index)
        if remaining:
            return False
    return True


def _patch(image):
    return cv2.resize(cv2.GaussianBlur(image, (3, 3), 0), (192, 192), interpolation=cv2.INTER_AREA)


def verify_process(left, right, strength):
    a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    bias = float(np.median(a - b))
    if abs(bias) > .08:
        return False, {'reason': '工位区域亮度变化较大'}
    b = np.clip(b + bias, 0, 1)
    def blur(value):
        return cv2.GaussianBlur(value, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb = np.maximum(0, blur(a*a)-ma*ma), np.maximum(0, blur(b*b)-mb*mb)
    cov = blur(a*b) - ma*mb
    structural = ((2*ma*mb + .01**2)*(2*cov + .03**2)) / ((ma*ma + mb*mb + .01**2)*(va+vb+.03**2))
    score = float(structural.mean())
    local = float(cv2.resize(structural, (6, 6), interpolation=cv2.INTER_AREA).min())
    difference = float(np.abs(a-b).mean())
    # Check color too: equal-gray but differently colored parts must not collapse.
    color_delta = np.abs(left.astype(np.float32)/255 - np.clip(right.astype(np.float32)/255 + bias, 0, 1)).mean(axis=2)
    color = float(color_delta.mean())
    local_color = float(cv2.resize(color_delta, (6, 6), interpolation=cv2.INTER_AREA).max())
    global_limit, local_limit, difference_limit = PROFILES[strength]
    duplicate = (score >= global_limit and local >= local_limit and difference <= difference_limit
                 and color <= difference_limit and local_color <= difference_limit * 1.8)
    return duplicate, {'structuralSimilarity': round(score, 6), 'worstLocalSimilarity': round(local, 6),
                       'meanRoiDifference': round(difference, 6), 'meanRoiColorDifference': round(color, 6),
                       'maximumLocalColorDifference': round(local_color, 6)}


def plan_process(images, folder, context, strength):
    config = load_regions(context.parameters.get('process_regions'))
    region = region_for(config, folder)
    context.report(f'工位工序模式：只比较框选区域 {region}，忽略区域外所有变化；力度 {strength}。')
    context.report('按工位动作画面保留代表样本，不要求重复轮次时长相同，不按比例凑删除量；这是视觉相似判断，不是工序/异常语义识别。')
    representatives, vectors, annotations = [], [], []
    buckets = OrderedDict()
    removed, unreadable = {}, []
    stats = Counter()
    @lru_cache(maxsize=8)
    def feature(path):
        return _patch(_crop(read_original(path), region))
    last = time.monotonic()
    for index, path in enumerate(images, 1):
        context.check_cancelled()
        try:
            image = read_original(path)
            crop = _crop(image, region)
            signature, labels = _annotations(path, image.shape, region)
            patch = _patch(crop)
        except Exception as exc:
            unreadable.append(path.name)
            if len(unreadable) <= 10:
                context.report(f'保留 {path.name}：{exc}')
            continue
        vector, keys = descriptor(patch)
        identity = (image.shape, signature)
        candidates = set(i for key in keys for i in buckets.get((identity, key), ()))
        candidates = sorted(candidates, key=lambda i: (float(np.abs(vectors[i].astype(np.int16)-vector).mean()), i))[:16]
        for candidate in candidates:
            context.check_cancelled()
            if not _same_annotations(annotations[candidate], labels):
                stats['annotationProtected'] += 1
                continue
            duplicate, metrics = verify_process(feature(representatives[candidate]), patch, strength)
            stats['comparedPairs'] += 1
            if duplicate:
                reference = representatives[candidate]
                removed[path] = {'reference': reference.name, 'reason': '目标工位动作画面相近，区域外变化已忽略',
                                 'ruleVersion': PROCESS_RULE, 'region': region, 'matchType': 'process', **metrics}
                if len(removed) <= 20:
                    context.report(f'建议移除 {path.name} → {reference.name}（工位结构相似 {metrics["structuralSimilarity"]:.3f}）')
                break
        else:
            identifier = len(representatives)
            representatives.append(path)
            vectors.append(vector)
            annotations.append(labels)
            for key in keys:
                bucket = (identity, key)
                buckets.setdefault(bucket, deque(maxlen=128)).append(identifier)
                buckets.move_to_end(bucket)
                if len(buckets) > 8192:
                    buckets.popitem(last=False)
            if not candidates:
                stats['noCandidate'] += 1
        if index % 50 == 0 or index == len(images) or time.monotonic() - last >= 5:
            context.report(f'工位分析 {index}/{len(images)}；保留 {len(representatives)}，拟移除 {len(removed)}。')
            last = time.monotonic()
    context.report(f'工位分析统计：比较 {stats["comparedPairs"]} 对；标注差异拦截 {stats["annotationProtected"]} 对；无法核验 {len(unreadable)} 张。')
    if not removed:
        context.report('没有找到满足当前工位规则的重复样本。请检查框选是否仍包含其他工位、是否框得过大；可先用“平衡（推荐）”或“较强”重新分析，不要盲目扩大删除力度。')
    feature.cache_clear()
    return removed, unreadable, {'region': region, 'strength': strength, 'rule': PROCESS_RULE, 'diagnostics': dict(stats)}
