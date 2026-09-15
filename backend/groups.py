"""Persistent grouping and ordering for functions in the local workspace."""

from __future__ import annotations

import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_GROUPS = [
    {"id": "video_processing", "name": "视频处理", "order": 0},
    {"id": "dataset_processing", "name": "数据集处理", "order": 1},
    {"id": "ai_inference", "name": "AI 推理", "order": 2},
]

DEFAULT_ASSIGNMENTS = {
    "video_frames": {"groupId": "video_processing", "order": 0},
    "video_clip": {"groupId": "video_processing", "order": 1},
    "yolo_dataset_split": {"groupId": "dataset_processing", "order": 0},
    "remote_star_inference": {"groupId": "ai_inference", "order": 0},
}


class GroupCatalog:
    def __init__(self, storage_file: Path) -> None:
        self.storage_file = storage_file
        self._lock = threading.RLock()
        self._groups, self._assignments = self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _default_state(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        groups = [
            {**deepcopy(group), "updatedAt": self._now()}
            for group in DEFAULT_GROUPS
        ]
        return groups, deepcopy(DEFAULT_ASSIGNMENTS)

    def _load(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if not self.storage_file.exists():
            return self._default_state()
        try:
            raw = json.loads(self.storage_file.read_text(encoding="utf-8"))
            raw_groups = raw.get("groups", []) if isinstance(raw, dict) else []
            raw_assignments = raw.get("assignments", {}) if isinstance(raw, dict) else {}
            if not isinstance(raw_groups, list) or not isinstance(raw_assignments, dict):
                raise ValueError("invalid group data")

            seen: set[str] = set()
            groups: list[dict[str, Any]] = []
            for index, entry in enumerate(raw_groups):
                if not isinstance(entry, dict):
                    continue
                group_id = str(entry.get("id", "")).strip()
                name = str(entry.get("name", "")).strip()
                if not group_id or not name or group_id in seen:
                    continue
                seen.add(group_id)
                groups.append({
                    "id": group_id,
                    "name": name,
                    "order": int(entry.get("order", index)),
                    "updatedAt": str(entry.get("updatedAt") or self._now()),
                })
            groups.sort(key=lambda item: (item["order"], item["name"]))
            for index, group in enumerate(groups):
                group["order"] = index

            assignments: dict[str, dict[str, Any]] = {}
            valid_group_ids = {group["id"] for group in groups}
            for function_id, entry in raw_assignments.items():
                if not isinstance(entry, dict):
                    continue
                group_id = entry.get("groupId")
                if group_id not in valid_group_ids:
                    group_id = None
                assignments[str(function_id)] = {
                    "groupId": group_id,
                    "order": max(0, int(entry.get("order", 0))),
                }
            return groups, assignments
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return self._default_state()

    def _save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "groups": self._groups,
            "assignments": self._assignments,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_file)

    def list_groups(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(sorted(self._groups, key=lambda item: item["order"]))

    def decorate(self, functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._lock:
            valid_group_ids = {group["id"] for group in self._groups}
            group_positions = {group["id"]: group["order"] for group in self._groups}
            decorated = []
            for fallback_order, item in enumerate(functions):
                assignment = self._assignments.get(str(item["id"]), {})
                group_id = assignment.get("groupId")
                if group_id not in valid_group_ids:
                    group_id = None
                decorated.append({
                    **deepcopy(item),
                    "groupId": group_id,
                    "order": int(assignment.get("order", fallback_order)),
                })
            return sorted(decorated, key=lambda item: (
                group_positions.get(item["groupId"], len(group_positions)),
                item["order"],
                item["name"],
            ))

    def create(self, name: str) -> dict[str, Any]:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("分组名称不能为空。")
        with self._lock:
            group = {
                "id": f"group_{uuid.uuid4().hex[:10]}",
                "name": clean_name,
                "order": len(self._groups),
                "updatedAt": self._now(),
            }
            self._groups.append(group)
            self._save()
            return deepcopy(group)

    def rename(self, group_id: str, name: str) -> dict[str, Any]:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("分组名称不能为空。")
        with self._lock:
            for group in self._groups:
                if group["id"] == group_id:
                    group["name"] = clean_name
                    group["updatedAt"] = self._now()
                    self._save()
                    return deepcopy(group)
        raise ValueError("分组不存在。")

    def delete(self, group_id: str, functions: list[dict[str, Any]]) -> None:
        with self._lock:
            if group_id not in {group["id"] for group in self._groups}:
                raise ValueError("分组不存在。")
            decorated = self.decorate(functions)
            ungrouped = sorted(
                [item for item in decorated if item["groupId"] is None],
                key=lambda item: item["order"],
            )
            moved = sorted(
                [item for item in decorated if item["groupId"] == group_id],
                key=lambda item: item["order"],
            )
            self._groups = [group for group in self._groups if group["id"] != group_id]
            for index, group in enumerate(sorted(self._groups, key=lambda item: item["order"])):
                group["order"] = index
            for index, item in enumerate([*ungrouped, *moved]):
                self._assignments[item["id"]] = {"groupId": None, "order": index}
            self._save()

    def reorder_groups(self, group_ids: list[str]) -> list[dict[str, Any]]:
        with self._lock:
            current_ids = {group["id"] for group in self._groups}
            if len(group_ids) != len(current_ids) or set(group_ids) != current_ids:
                raise ValueError("分组排序数据不完整。")
            positions = {group_id: index for index, group_id in enumerate(group_ids)}
            self._groups.sort(key=lambda group: positions[group["id"]])
            for index, group in enumerate(self._groups):
                group["order"] = index
            self._save()
            return self.list_groups()

    def move_function(
        self,
        function_id: str,
        target_group_id: str | None,
        functions: list[dict[str, Any]],
        position: int | None = None,
    ) -> None:
        with self._lock:
            function_ids = {item["id"] for item in functions}
            if function_id not in function_ids:
                raise ValueError("功能不存在。")
            if target_group_id is not None and target_group_id not in {group["id"] for group in self._groups}:
                raise ValueError("目标分组不存在。")

            decorated = self.decorate(functions)
            moved = next(item for item in decorated if item["id"] == function_id)
            source_group_id = moved["groupId"]
            source_items = sorted(
                [item for item in decorated if item["groupId"] == source_group_id and item["id"] != function_id],
                key=lambda item: item["order"],
            )
            target_items = sorted(
                [item for item in decorated if item["groupId"] == target_group_id and item["id"] != function_id],
                key=lambda item: item["order"],
            )
            insert_at = len(target_items) if position is None else max(0, min(int(position), len(target_items)))
            target_items.insert(insert_at, moved)

            if source_group_id != target_group_id:
                for index, item in enumerate(source_items):
                    self._assignments[item["id"]] = {"groupId": source_group_id, "order": index}
            for index, item in enumerate(target_items):
                self._assignments[item["id"]] = {"groupId": target_group_id, "order": index}
            self._save()

    def remove_function(self, function_id: str) -> None:
        with self._lock:
            if function_id in self._assignments:
                del self._assignments[function_id]
                self._save()
