"""Conservative image-sequence matching with variable-speed cycle alignment.

Subsequence DTW proposes complete template matches. A banded full-frame DTW
and spatial patch checks then verify EVERY image before a match can be moved.
No downloaded models, video timestamps, or fixed cycle duration are required.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .handlers import RunContext
from .image_dedup import _number, _read_image, _sequence_key

CYCLE_MODE = "完整工序循环去重"


def _reference_path(folder: Path, raw: object, images: list[Path], label: str) -> int:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"请选择参考循环的{label}。")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else folder / path
    if path.is_symlink() or path.resolve().parent != folder:
        raise ValueError(f"{label}必须是当前图片文件夹中的图片，不能使用符号链接或其他目录。")
    try:
        return images.index(path.resolve())
    except ValueError:
        raise ValueError(f"找不到{label}：{path.name}") from None


def _annotation(path: Path) -> str | None:
    """Preserve class, count AND geometry changes; unreadable labels forbid removal."""
    signatures = []
    for suffix in (".txt", ".xml", ".json"):
        label = path.with_suffix(suffix)
        if not label.exists():
            continue
        if label.is_symlink():
            raise ValueError("标注是符号链接")
        if suffix == ".txt":
            signatures.append(sorted(" ".join(line.split()) for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()))
        elif suffix == ".xml":
            signatures.append(sorted(ET.tostring(node, encoding="unicode") for node in ET.parse(label).getroot().findall("object")))
        else:
            raw = json.loads(label.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or not isinstance(raw.get("shapes"), list):
                raise ValueError("未知 JSON 标注格式")
            signatures.append(raw["shapes"])
    return json.dumps(signatures, sort_keys=True, ensure_ascii=False) if signatures else None


def _proposals(cost: np.ndarray, minimum: int, maximum: int, check) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Open-begin subsequence DTW, bounded working memory per search window."""
    rows, columns = cost.shape
    scores = np.full((rows + 1, columns + 1), np.inf, dtype=np.float32)
    scores[0, :] = 0
    moves = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    starts = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    starts[0, :] = np.arange(columns + 1)
    for j in range(1, columns + 1):
        if j % 64 == 1:
            check()
        for i in range(1, rows + 1):
            choices = (scores[i - 1, j - 1], scores[i, j - 1], scores[i - 1, j])
            move = min(range(3), key=choices.__getitem__)
            pi, pj = ((i - 1, j - 1), (i, j - 1), (i - 1, j))[move]
            start = starts[pi, pj]
            if j - start <= maximum:
                scores[i, j] = cost[i - 1, j - 1] + choices[move]
                starts[i, j], moves[i, j] = start, move
    spans = np.arange(columns + 1) - starts[-1]
    eligible = [j for j in range(1, columns + 1)
                if minimum <= spans[j] <= maximum and np.isfinite(scores[-1, j])
                and cost[-1, j - 1] <= .12]
    # Best non-overlapping candidates. Equal scores prefer the earliest end.
    eligible.sort(key=lambda j: (float(scores[-1, j]) / max(1, spans[j]), j))
    candidates = []
    occupied = np.zeros(columns, dtype=bool)
    for j in eligible:
        start = int(starts[-1, j])
        if occupied[start:j].any():
            continue
        i, end, path = rows, j, []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            move = moves[i, j]
            if move == 0:
                i, j = i - 1, j - 1
            elif move == 1:
                j -= 1
            else:
                i -= 1
        path.reverse()
        if path and path[0][0] == 0 and path[-1][0] == rows - 1:
            # Open-begin DTW trims repeated first/last poses. Include these
            # dwell frames only when the two endpoint poses are distinguishable;
            # otherwise the boundary between two cycles is inherently ambiguous.
            if cost[-1, start] > .02 and cost[0, end - 1] > .02:
                while start > 0 and end - start < maximum and not occupied[start - 1] \
                        and cost[0, start - 1] <= .01 \
                        and cost[0, start - 1] <= float(cost[:, start - 1].min()) + 1e-6:
                    start -= 1
                    path.insert(0, (0, start))
                while end < columns and end - start < maximum and not occupied[end] \
                        and cost[-1, end] <= .01 \
                        and cost[-1, end] <= float(cost[:, end].min()) + 1e-6:
                    path.append((rows - 1, end))
                    end += 1
            candidates.append((start, end, path))
            occupied[start:end] = True
    return sorted(candidates, key=lambda item: item[0])


def _refine(reference: np.ndarray, target: np.ndarray, anchors: np.ndarray,
            coarse_path: list[tuple[int, int]], offset: int, check) -> list[tuple[int, int]]:
    """All original frames participate; compressed reference anchors only guide a band."""
    n, m = len(reference), len(target)
    centers = np.zeros(m, dtype=np.int32)
    for i, j in coarse_path:
        centers[j - offset] = anchors[i]
    band = max(6, math.ceil(n / max(1, len(anchors) - 1)) * 3)
    # Coarse DTW can advance many template anchors within one target frame
    # (especially at static dwell poses). Include that entire vertical segment,
    # not just its last anchor, or the fine path can become disconnected.
    ranges = [(max(1, int(min(center, centers[j - 1] if j else 0)) + 1 - band),
               min(n, int(center) + 1 + band)) for j, center in enumerate(centers)]
    ranges[0] = (1, ranges[0][1])
    ranges[-1] = (ranges[-1][0], n)
    traces: list[np.ndarray] = []
    previous = np.full(n + 1, np.inf)
    previous[0] = 0
    for j, (low, high) in enumerate(ranges):
        if j % 64 == 0:
            check()
        current = np.full(n + 1, np.inf)
        trace = np.zeros(high - low + 1, dtype=np.uint8)
        costs = np.mean(np.abs(reference[low - 1:high] - target[j]), axis=1)
        for i in range(low, high + 1):
            choices = (previous[i - 1], previous[i], current[i - 1])
            move = min(range(3), key=choices.__getitem__)
            current[i] = costs[i - low] + choices[move]
            trace[i - low] = move
        traces.append(trace)
        previous = current
    if not np.isfinite(previous[n]):
        return []
    i, j, path = n, m, []
    while i > 0 and j > 0:
        low, high = ranges[j - 1]
        if not low <= i <= high:
            return []
        path.append((i - 1, j - 1))
        move = traces[j - 1][i - low]
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
    return list(reversed(path)) if i == 0 and j == 0 else []


def plan_cycles(images: list[Path], folder: Path, context: RunContext, threshold: float):
    from .image_quality_dedup import read_original, verify_pair
    @lru_cache(maxsize=2)
    def original(path): return read_original(path)
    first = _reference_path(folder, context.paths.get("cycle_start_image"), images, "起始图片")
    last = _reference_path(folder, context.paths.get("cycle_end_image"), images, "结束图片")
    if last - first < 3:
        raise ValueError("结束图片必须位于起始图片之后，参考循环至少包含 4 张图片（包含两端）。")
    minimum_ratio = _number(context.parameters.get("cycle_min_ratio"), "最短循环张数倍数", .4, .1, 1)
    maximum_ratio = _number(context.parameters.get("cycle_max_ratio"), "最长循环张数倍数", 2.5, 1, 10)
    if last - first + 1 > 10000:
        raise ValueError("单个参考循环超过 10000 张，请降低抽帧密度或缩小到一个完整工序循环。")
    key = _sequence_key(images[first])
    if any(_sequence_key(path) != key for path in images[first:last + 1]):
        raise ValueError("参考循环包含不同来源的图片，请把同一个视频的图片放在一起，并重新选择起止图片。")
    # Never join a cycle across an unreadable frame, another source or a symlink.
    selected = images[first:last + 1]
    reference = []
    coarse: dict[Path, np.ndarray | None] = {}
    shapes: dict[Path, tuple[int, ...]] = {}
    signatures: dict[Path, str | None] = {}
    unreadable = []
    for index, path in enumerate(images):
        context.check_cancelled()
        if _sequence_key(path) != key:
            continue
        image = None if path.is_symlink() else _read_image(path)
        try:
            signatures[path] = _annotation(path)
        except (OSError, ValueError, ET.ParseError, UnicodeError):
            image = None
        if image is None:
            coarse[path] = None
            unreadable.append(path.name)
            if first <= index <= last:
                raise ValueError(f"参考循环中的图片或标注无法读取：{path.name}；请先检查文件。")
            continue
        shapes[path] = image.shape
        feature = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255
        coarse[path] = feature
        if first <= index <= last:
            reference.append(feature)
        if index % 100 == 0:
            context.report(f"读取循环特征：{index + 1}/{len(images)}")
    reference_array = np.stack(reference)
    # A nearly stationary template cannot establish a complete operation cycle.
    motion = reference_array.std(axis=0)
    if float(np.percentile(motion, 95)) < .025:
        raise ValueError("参考循环的动作变化不足，无法可靠识别轮次。请选择包含完整动作变化的起止图片，或使用连续相似帧去重。")
    weights = .15 + .85 * np.minimum(1, motion / max(float(motion.max()), 1e-6))
    anchors = np.unique(np.linspace(0, len(reference) - 1, min(96, len(reference))).round().astype(int))
    minimum = max(4, math.ceil(len(reference) * minimum_ratio))
    maximum = max(minimum, math.ceil(len(reference) * maximum_ratio))
    # Bound DP matrices regardless of the number of images in the whole folder.
    if maximum > 30000:
        raise ValueError("参考循环张数与最长倍数过大（搜索上限 30000 张），请调小最长循环张数倍数。")

    @lru_cache(maxsize=128)
    def detailed(path: Path) -> np.ndarray:
        image = _read_image(path)
        if image is None:
            raise ValueError(f"分析中图片无法读取：{path.name}")
        return cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    removed, rows = {}, []
    def add_row(start: int, end: int, action: str, reason: str, similarity=None):
        if start >= end:
            return
        rows.append({"startImage": images[start].name, "endImage": images[end - 1].name,
                     "imageCount": end - start, "action": action, "reason": reason,
                     "minimumSimilarity": similarity})

    add_row(0, first, "保留", "参考循环之前的图片")
    add_row(first, last + 1, "保留", "指定的完整参考循环，全部保留")
    cursor = last + 1
    context.report(f"参考循环：{selected[0].name} → {selected[-1].name}，共 {len(reference)} 张；后续允许 {minimum}～{maximum} 张/轮。")
    while cursor < len(images):
        context.check_cancelled()
        # Discontinuities are explicit protected boundaries, never filtered out.
        if coarse.get(images[cursor]) is None:
            end = cursor + 1
            while end < len(images) and coarse.get(images[end]) is None:
                end += 1
            add_row(cursor, end, "保留", "不同来源或图片/标注无法读取")
            cursor = end
            continue
        end = cursor
        while end < len(images) and end - cursor < maximum * 2 and coarse.get(images[end]) is not None:
            end += 1
        targets = np.stack([coarse[path] for path in images[cursor:end]])
        costs = np.stack([np.average(np.abs(targets - reference_array[i]), axis=1, weights=weights) for i in anchors])
        proposals = _proposals(costs, minimum, maximum, context.check_cancelled)
        consumed = cursor
        for start_offset, end_offset, path in proposals:
            start, stop = cursor + start_offset, cursor + end_offset
            add_row(consumed, start, "保留", "未确认属于完整重复循环")
            aligned = _refine(reference_array, targets[start_offset:end_offset], anchors, path,
                              start_offset, context.check_cancelled)
            reason, worst = "动作顺序或完整性未通过复核", 0.0
            if aligned:
                reason = "与参考循环完整匹配"
                for pair_index, (i, j) in enumerate(aligned):
                    if pair_index % 20 == 0:
                        context.check_cancelled()
                    left, right = selected[i], images[start + j]
                    if shapes[left] != shapes[right] or signatures[left] != signatures[right]:
                        reason = "图像尺寸或标注类别/数量/位置存在差异"
                        break
                    diff = np.mean(np.abs(detailed(left) - detailed(right)), axis=2)
                    patch = float(diff.reshape(10, 16, 10, 16).mean(axis=(1, 3)).max())
                    distance = max(float(diff.mean()), .75 * patch)
                    worst = max(worst, distance)
                    if distance > 1 - threshold:
                        reason = "存在不同动作或局部画面差异，整段保留"
                        break
                    verified, _why, _metrics = verify_pair(original(left), original(right))
                    if not verified:
                        reason = '原图局部复核存在变化或无法确认，整段保留'
                        break
                    if pair_index and pair_index % 200 == 0:
                        context.report(f'正在核验完整循环的原图局部差异：{pair_index}/{len(aligned)}')
            duplicate = bool(aligned) and reason == "与参考循环完整匹配"
            add_row(start, stop, "移除" if duplicate else "保留", reason, round(max(0., 1 - worst), 6) if aligned else None)
            if duplicate:
                for image in images[start:stop]:
                    removed[image] = {"reason": "完整重复工序循环", "reference": selected[0].name,
                                      "referenceEnd": selected[-1].name, "similarity": round(1 - worst, 6),
                                      "cycleStart": images[start].name, "cycleEnd": images[stop - 1].name}
            context.report(f"{'建议移除' if duplicate else '保留'} {stop - start} 张：{images[start].name} → {images[stop - 1].name}；{reason}")
            consumed = stop
        if consumed == cursor:
            # Overlap windows so a full cycle crossing a window edge is considered.
            consumed = end if end == len(images) or coarse.get(images[end]) is None else max(cursor + 1, end - maximum)
            add_row(cursor, consumed, "保留", "未找到可信的完整重复循环")
        cursor = consumed
    return removed, rows, unreadable
