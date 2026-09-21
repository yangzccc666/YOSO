"""Deterministic image-level multilabel train/validation allocation.

Each image contributes at most one count to each class, regardless of the
number of boxes in that image. No image is copied into both splits.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence


def split_with_class_coverage(
    image_classes: Sequence[set[int]], train_ratio: float, seed: int, class_count: int
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Cover every class in train and approximate its requested image ratio.

    A multi-label image counts once for each class it contains, so independent
    per-class quotas are not always simultaneously attainable.
    """
    count = len(image_classes)
    if count == 0:
        return [], [], [{"total": 0, "train": 0, "val": 0} for _ in range(class_count)]
    if class_count < 0 or any(not 0 <= category < class_count for classes in image_classes for category in classes):
        raise ValueError("标签类别 ID 超出 classes.txt 或 XML 类别列表的范围。")

    order = list(range(count))
    random.Random(seed).shuffle(order)
    rank = {index: position for position, index in enumerate(order)}
    totals = Counter(category for classes in image_classes for category in classes)
    targets = {
        category: max(1, min(total - 1, int(total * train_ratio + 0.5))) if total > 1 else 1
        for category, total in totals.items()
    }

    def penalty(category: int, train_count: int) -> float:
        target = targets[category]
        # Normalize error so a class with 3 images is not drowned out by one
        # with 60 images. The ideal proportion has zero penalty.
        weight = max(1, target * (totals[category] - target))
        return ((train_count - target) ** 2) / weight

    def change_cost(categories: set[int], counts: Counter[int], delta: int) -> float:
        return sum(penalty(category, counts[category] + delta) - penalty(category, counts[category])
                   for category in categories)

    target_train = max(1, min(count - 1, int(count * train_ratio))) if count > 1 else 1

    train: set[int] = set()
    uncovered = set(totals)
    while uncovered:
        # Weighted set cover: images containing rare unseen classes go first.
        selected = max(
            (index for index in order if index not in train),
            key=lambda index: (
                sum(1 / totals[category] for category in image_classes[index] & uncovered),
                len(image_classes[index] & uncovered),
                -rank[index],
            ),
        )
        train.add(selected)
        uncovered.difference_update(image_classes[selected])

    val = set(range(count)) - train
    train_counts = Counter(category for index in train for category in image_classes[index])
    val_counts = Counter(category for index in val for category in image_classes[index])
    while len(train) < target_train and val:
        # Fill the global quota while reducing normalized per-class error.
        selected = min(
            val,
            key=lambda index: (
                change_cost(image_classes[index], train_counts, 1),
                sum(val_counts[category] == 1 and totals[category] >= 2 for category in image_classes[index]),
                rank[index],
            ),
        )
        val.remove(selected)
        train.add(selected)
        train_counts.update(image_classes[selected])
        val_counts.subtract(image_classes[selected])

    # A one-for-one exchange can recover validation coverage missed by the
    # greedy fill without changing the split size or losing train coverage.
    eligible = {category for category, total in totals.items() if total >= 2}
    while val:
        covered = sum(val_counts[category] > 0 for category in eligible)
        missing = {category for category in eligible if val_counts[category] == 0}
        if not missing:
            break
        improvement = None
        for train_index in order:
            if train_index not in train or not (image_classes[train_index] & missing):
                continue
            for val_index in order:
                if val_index not in val:
                    continue
                if any(
                    train_counts[category] == 1 and category not in image_classes[val_index]
                    for category in image_classes[train_index]
                ):
                    continue
                changed = image_classes[train_index] | image_classes[val_index]
                gain = sum(
                    (val_counts[category] + (category in image_classes[train_index])
                     - (category in image_classes[val_index]) > 0)
                    - (val_counts[category] > 0)
                    for category in changed & eligible
                )
                if gain > 0 and (improvement is None or gain > improvement[0]):
                    improvement = (gain, train_index, val_index)
        if improvement is None:
            break
        _gain, train_index, val_index = improvement
        train.remove(train_index)
        train.add(val_index)
        val.remove(val_index)
        val.add(train_index)
        train_counts.subtract(image_classes[train_index])
        train_counts.update(image_classes[val_index])
        val_counts.subtract(image_classes[val_index])
        val_counts.update(image_classes[train_index])
        assert sum(val_counts[category] > 0 for category in eligible) > covered

    # Improve the class ratios with one-for-one exchanges. The total train
    # image count stays fixed, and already covered classes remain in both sets.
    for _ in range(min(count * 2, 200)):
        train_candidates = [index for index in order if index in train]
        val_candidates = [index for index in order if index in val]
        if not train_candidates or not val_candidates:
            break
        if len(train_candidates) * len(val_candidates) > 50000:
            train_candidates.sort(key=lambda index: (change_cost(image_classes[index], train_counts, -1), rank[index]))
            val_candidates.sort(key=lambda index: (change_cost(image_classes[index], train_counts, 1), rank[index]))
            train_candidates = train_candidates[:128]
            val_candidates = val_candidates[:128]
        best: tuple[float, int, int] | None = None
        for train_index in train_candidates:
            for val_index in val_candidates:
                changed = image_classes[train_index] | image_classes[val_index]
                new_counts = {
                    category: train_counts[category]
                    - (category in image_classes[train_index])
                    + (category in image_classes[val_index])
                    for category in changed
                }
                if any(
                    new_count < 1 or (totals[category] >= 2 and val_counts[category] > 0
                                      and new_count == totals[category])
                    for category, new_count in new_counts.items()
                ):
                    continue
                cost = sum(penalty(category, new_count) - penalty(category, train_counts[category])
                           for category, new_count in new_counts.items())
                if cost < -1e-12 and (best is None or cost < best[0] - 1e-12):
                    best = (cost, train_index, val_index)
        if best is None:
            break
        _, train_index, val_index = best
        train.remove(train_index)
        train.add(val_index)
        val.remove(val_index)
        val.add(train_index)
        train_counts.subtract(image_classes[train_index])
        train_counts.update(image_classes[val_index])
        val_counts.subtract(image_classes[val_index])
        val_counts.update(image_classes[train_index])

    train_indices = [index for index in order if index in train]
    val_indices = [index for index in order if index in val]
    distribution = [
        {"total": totals[category], "train": train_counts[category], "val": val_counts[category]}
        for category in range(class_count)
    ]
    return train_indices, val_indices, distribution


def read_yolo_image_classes(label_path, class_count: int) -> set[int]:
    """Read class IDs from a YOLO detection TXT label without counting boxes."""
    if label_path is None:
        return set()
    classes: set[int] = set()
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            category = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"标签类别 ID 无效：{label_path.name} 第 {line_number} 行。") from exc
        if not 0 <= category < class_count:
            raise ValueError(f"标签类别 ID {category} 超出类别列表范围：{label_path.name} 第 {line_number} 行。")
        classes.add(category)
    return classes
