"""Quota-free representative selection with original-resolution verification.

Near-duplicate decisions are visual heuristics, not proof of semantic equality.
Coarse features only retrieve candidates; they can never authorize a move.
"""
from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

AUTO_MODE = '差异优先去重（自动）'
EXACT_MODE = '仅精确重复'
RULE_VERSION = 'quality-original-resolution-v1'


def read_original(path):
    if path.is_symlink():
        return None
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    except (OSError, cv2.error):
        return None


def annotation_key(path):
    """Compare all sidecars, including object count/geometry, not only classes."""
    from .image_cycles import _annotation
    for suffix in ('.txt', '.xml', '.json'):
        if path.with_suffix(suffix).is_symlink():
            raise ValueError('标注是符号链接')
    txt = path.with_suffix('.txt')
    if txt.exists():
        if txt.is_symlink():
            raise ValueError('标注是符号链接')
        for line in txt.read_text(encoding='utf-8-sig').splitlines():
            parts = line.split()
            if not parts:
                continue
            values = np.asarray([float(v) for v in parts])
            if len(parts) < 5 or len(parts) % 2 != 1 or not np.isfinite(values).all() or values[0] < 0 or not values[0].is_integer():
                raise ValueError('无法安全识别 YOLO 标注格式')
    xml = path.with_suffix('.xml')
    if xml.exists():
        root = ET.parse(xml).getroot()
        if root.tag != 'annotation' or any(not (obj.findtext('name') or '').strip() for obj in root.findall('object')):
            raise ValueError('无法安全识别 VOC 标注格式')
    label_json = path.with_suffix('.json')
    if label_json.exists():
        raw = json.loads(label_json.read_text(encoding='utf-8-sig'))
        if not isinstance(raw, dict) or not isinstance(raw.get('shapes'), list) or any(
                not isinstance(shape, dict) or not str(shape.get('label', '')).strip() for shape in raw['shapes']):
            raise ValueError('无法安全识别 JSON 标注格式')
    value = _annotation(path)
    if label_json.exists():
        value = [value, raw.get('flags', {})]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def verify_pair(left, right, exact_only=False):
    if left is None or right is None or left.shape != right.shape or left.dtype != right.dtype:
        return False, '图像尺寸或位深不同', {}
    if np.array_equal(left, right):
        return True, '完整像素一致', {'matchType': 'exact'}
    if exact_only or left.dtype != np.uint8 or left.ndim != 3 or left.shape[2] != 3:
        return False, '像素不同，或不是受支持的 8 位彩色近重复核验输入', {}
    # Small sensor/compression noise is tolerated, but global lighting changes,
    # sparse local marks and coherent motion must not disappear in a thumbnail.
    sample = left[::4, ::4].astype(np.float32) - right[::4, ::4].astype(np.float32)
    bias = np.median(sample.reshape(-1, 3), axis=0)
    if float(np.max(np.abs(bias))) > 6:
        return False, '光照或颜色变化', {}
    noise = float(np.median(np.abs(sample - bias))) * 1.4826
    tolerance = float(np.clip(3 + 3 * noise, 4, 10))
    corrected = np.clip(right.astype(np.int16) + np.clip(np.rint(bias), -2, 2).astype(np.int16), 0, 255).astype(np.uint8)
    delta = cv2.absdiff(cv2.GaussianBlur(left, (3, 3), 0), cv2.GaussianBlur(corrected, (3, 3), 0)).max(axis=2)
    mean = float(delta.mean())
    changed = (delta > tolerance).astype(np.float32)
    fraction = float(changed.mean())
    tiles = (max(1, (delta.shape[1] + 31) // 32), max(1, (delta.shape[0] + 31) // 32))
    local_fraction = float(cv2.resize(changed, tiles, interpolation=cv2.INTER_AREA).max())
    local_mean = float(cv2.resize(delta.astype(np.float32), tiles, interpolation=cv2.INTER_AREA).max())
    metrics = {'matchType': 'near', 'meanPixelDifference': round(mean, 5),
               'changedPixelRatio': round(fraction, 7), 'maximumLocalChangeRatio': round(local_fraction, 7),
               'maximumLocalMeanDifference': round(local_mean, 5), 'estimatedNoise': round(noise, 5)}
    if mean > 2.5 or fraction > .002 or local_fraction > .02 or local_mean > 5:
        return False, '整体或局部变化超出噪声解释范围', metrics
    strong = (delta > max(14, tolerance * 1.8)).astype(np.uint8)
    if strong.any():
        _, _, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
        if len(stats) > 1 and int(stats[1:, cv2.CC_STAT_AREA].max()) >= 6:
            return False, '存在局部明显变化，可能是标记、动作或缺陷', metrics
    return True, '整体与局部原图复核通过，仅有很小的画面差异', metrics


def descriptor(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        return None, []
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    vector = small.reshape(-1)
    gray = cv2.cvtColor(cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bits = np.packbits(gray[:, 1:] > gray[:, :-1]).tolist()
    keys = [('edge', i, v) for i, v in enumerate(bits)]
    mean = small.mean(axis=(0, 1))
    keys += [('color', offset, *(np.floor((mean + offset) / 16).astype(int).tolist())) for offset in (0, 8)]
    return vector, keys


def plan_quality(images, context, exact_only=False):
    context.report(f'找到 {len(images)} 张图片；按差异保留代表样本，不设保留比例或删除数量配额。' +
                   ('只处理完整像素一致的图片。' if exact_only else '全文件夹检索重复画面，候选必须再通过原图整体与局部复核；不确定就保留。'))
    # Bounded candidate buckets limit work on very large datasets. Missing a
    # candidate keeps extra data; it never makes deletion more aggressive.
    buckets = OrderedDict()
    exact = {}
    representatives = []
    vectors = []
    removed, unreadable = {}, []
    last_progress = time.monotonic()
    @lru_cache(maxsize=2)
    def original(path): return read_original(path)
    for index, path in enumerate(images, 1):
        context.check_cancelled()
        if time.monotonic() - last_progress >= 5:
            context.report(f'原图差异核验仍在运行：{index}/{len(images)}；保留代表 {len(representatives)}，拟移除 {len(removed)}。')
            last_progress = time.monotonic()
        image = read_original(path)
        try:
            signature = annotation_key(path)
        except Exception as exc:
            context.report(f'保留 {path.name}：标注无法安全核验（{exc}）。')
            unreadable.append(path.name)
            continue
        if image is None or not image.size:
            unreadable.append(path.name)
            continue
        shape = (image.shape, image.dtype.str, signature)
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest()
        exact_key = (shape, digest)
        candidates = []
        if exact_key in exact:
            candidates.append(exact[exact_key])
        vector, keys = descriptor(image) if not exact_only else (None, [])
        if vector is not None:
            possible = set(i for key in keys for i in buckets.get((shape, key), ()))
            ranked = sorted(possible, key=lambda i: (float(np.mean(np.abs(vectors[i].astype(np.int16) - vector))), i))
            candidates.extend(i for i in ranked[:8] if i not in candidates)
        matched = False
        for candidate in candidates:
            context.check_cancelled()
            reference = representatives[candidate]
            duplicate, reason, metrics = verify_pair(original(reference), image, exact_only)
            if duplicate:
                removed[path] = {'reference': reference.name, 'reason': reason, 'ruleVersion': RULE_VERSION, **metrics}
                matched = True
                if len(removed) <= 20:
                    context.report(f'建议移除 {path.name} → 对照 {reference.name}：{reason}')
                break
        if not matched:
            identifier = len(representatives)
            representatives.append(path)
            vectors.append(vector)
            exact[exact_key] = identifier
            for key in keys:
                identity = (shape, key)
                if identity not in buckets:
                    buckets[identity] = deque(maxlen=64)
                buckets[identity].append(identifier)
                buckets.move_to_end(identity)
                if len(buckets) > 4096:
                    buckets.popitem(last=False)
        if index % 50 == 0 or index == len(images):
            context.report(f'差异核验：{index}/{len(images)}；已保留代表 {len(representatives)}，拟移除 {len(removed)}。')
    original.cache_clear()
    return removed, unreadable
