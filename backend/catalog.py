"""Editable function catalog persisted locally as JSON."""

from __future__ import annotations

import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


VIDEO_FUNCTION = {
    "id": "video_frames",
    "name": "视频切图",
    "description": "批量从 MP4 视频中按帧率或时间间隔提取 JPG 图片。",
    "handlerId": "video.extract_frames",
    "pathFields": [
        {"id": "video_folder", "label": "视频文件夹", "mode": "directory"},
        {"id": "video_file", "label": "单个视频（可选）", "mode": "file"},
        {"id": "output_folder", "label": "输出文件夹（可选）", "mode": "directory"},
    ],
    "parameters": [
        {
            "id": "extraction_mode",
            "label": "抽帧方式",
            "type": "select",
            "default": "每秒抽取张数",
            "options": ["每秒抽取张数", "按时间间隔"],
        },
        {"id": "target_fps", "label": "每秒抽取张数", "type": "number", "default": 5, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "每秒抽取张数"}},
        {"id": "interval_sec", "label": "抽帧时间间隔（秒）", "type": "number", "default": 1, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "按时间间隔"}},
        {"id": "max_images_limit", "label": "最大抽取数量（留空则不限）", "type": "number", "default": "", "options": []},
        {"id": "range_start_time", "label": "开始时间（如 01:30）", "type": "text", "default": "00:00:00", "options": []},
        {"id": "range_end_time", "label": "结束时间（如 02:40，留空到结尾）", "type": "text", "default": "", "options": []},
        {
            "id": "limit_strategy",
            "label": "上限策略（填写数量上限时生效）",
            "type": "select",
            "default": "全程均匀抽取",
            "options": ["全程均匀抽取", "从前往后抽取"],
        },
        {"id": "fallback_step", "label": "备用帧间隔", "type": "number", "default": 24, "options": []},
    ],
}

YOLO_DATASET_FUNCTION = {
    "id": "yolo_dataset_split",
    "name": "YOLO 数据集分配",
    "description": "将图片和标签按比例复制到标准 YOLO 训练集与验证集目录，源数据保持不变。",
    "handlerId": "yolo.split_dataset",
    "pathFields": [
        {"id": "images_folder", "label": "图片文件夹", "mode": "directory"},
        {"id": "labels_folder", "label": "标签文件夹（TXT/XML）", "mode": "directory"},
        {"id": "negative_images_folder", "label": "负样本图片文件夹（可选）", "mode": "directory"},
        {"id": "output_folder", "label": "输出文件夹", "mode": "directory"},
    ],
    "parameters": [
        {
            "id": "dataset_task_type",
            "label": "数据集任务类型",
            "type": "select",
            "default": "目标检测",
            "options": ["目标检测", "旋转目标检测", "分割", "分类"],
        },
        {
            "id": "label_format",
            "label": "目标检测标签格式",
            "type": "select",
            "default": "YOLO TXT",
            "options": ["YOLO TXT", "VOC XML", "仅图片"],
        },
        {"id": "train_ratio", "label": "训练集比例", "type": "number", "default": 0.7, "options": []},
        {"id": "random_seed", "label": "随机种子", "type": "number", "default": 42, "options": []},
        {
            "id": "missing_label_policy",
            "label": "缺少标签时",
            "type": "select",
            "default": "跳过并报告",
            "options": ["跳过并报告", "生成空标签"],
        },
    ],
}

VIDEO_CLIP_FUNCTION = {
    "id": "video_clip",
    "name": "视频裁剪",
    "description": "按指定时间范围批量裁剪视频，支持精确裁剪和快速裁剪，保留原视频不变。",
    "handlerId": "video.clip",
    "pathFields": [
        {"id": "video_folder", "label": "视频文件夹", "mode": "directory"},
        {"id": "video_file", "label": "单个视频（可选）", "mode": "file"},
        {"id": "output_folder", "label": "输出文件夹（可选）", "mode": "directory"},
    ],
    "parameters": [
        {
            "id": "clip_mode",
            "label": "裁剪时间方案",
            "type": "select",
            "default": "指定开始和结束时间",
            "options": ["指定开始和结束时间", "指定开始时间和持续时长", "跳过开头和结尾"],
        },
        {"id": "range_start_time", "label": "开始时间（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始和结束时间"}},
        {"id": "range_end_time", "label": "结束时间（秒或 HH:MM:SS）", "type": "text", "default": "00:01:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始和结束时间"}},
        {"id": "duration_start_time", "label": "开始时间（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始时间和持续时长"}},
        {"id": "clip_duration", "label": "持续时长（秒或 HH:MM:SS）", "type": "text", "default": "00:01:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始时间和持续时长"}},
        {"id": "head_skip_time", "label": "跳过开头（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "跳过开头和结尾"}},
        {"id": "tail_skip_time", "label": "忽略结尾（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "跳过开头和结尾"}},
        {
            "id": "encoding_mode",
            "label": "处理方式",
            "type": "select",
            "default": "精确裁剪（推荐）",
            "options": ["精确裁剪（推荐）", "快速裁剪（不重新编码）"],
        },
    ],
}

REMOTE_STAR_INFERENCE_FUNCTION = {
    "id": "remote_star_inference",
    "name": "远程实时 AI 推理",
    "description": "平台自动上传内置推理脚本、所选模型和视频到远端 AI 设备，并实时显示 STAR/TensorRT 带框画面。",
    "handlerId": "remote.star_inference",
    "pathFields": [
        {"id": "local_script_file", "label": "本地推理脚本（可选，默认使用内置版本）", "mode": "file"},
        {"id": "local_model_file", "label": "本地 STAR 模型文件", "mode": "file"},
        {"id": "local_video_file", "label": "本地视频文件", "mode": "file"},
    ],
    "parameters": [
        {"id": "host", "label": "设备 IP / 主机名", "type": "text", "default": "192.168.1.223", "options": []},
        {"id": "port", "label": "SSH 端口", "type": "number", "default": 22, "options": []},
        {"id": "username", "label": "SSH 用户名", "type": "text", "default": "wel", "options": []},
        {"id": "password", "label": "SSH 密码", "type": "password", "default": "", "options": []},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": False, "options": []},
        {"id": "python_path", "label": "远端 Python 解释器", "type": "text", "default": "python3", "options": []},
        {"id": "labels", "label": "类别名称（英文逗号分隔）", "type": "text", "default": "class0", "options": []},
        {"id": "conf", "label": "置信度阈值", "type": "number", "default": 0.5, "options": []},
        {"id": "iou", "label": "NMS IoU 阈值", "type": "number", "default": 0.45, "options": []},
        {"id": "max_det", "label": "每帧最大检测框数", "type": "number", "default": 300, "options": []},
        {"id": "save_path", "label": "远端保存路径（可选）", "type": "text", "default": "", "options": []},
        {"id": "realtime", "label": "按视频原帧率播放", "type": "boolean", "default": True, "options": []},
        {"id": "preview_fps", "label": "本地预览帧率", "type": "number", "default": 12, "options": []},
        {"id": "stream_width", "label": "本地预览宽度", "type": "number", "default": 1280, "options": []},
        {"id": "jpeg_quality", "label": "预览画质（20-100）", "type": "number", "default": 80, "options": []},
        {
            "id": "cuda_backend",
            "label": "CUDA 后端",
            "type": "select",
            "default": "auto",
            "options": ["auto", "cuda-driver", "cuda-python", "pycuda"],
        },
    ],
}

DEFAULT_FUNCTIONS = [VIDEO_FUNCTION, YOLO_DATASET_FUNCTION, VIDEO_CLIP_FUNCTION, REMOTE_STAR_INFERENCE_FUNCTION]


class FunctionCatalog:
    def __init__(self, storage_file: Path) -> None:
        self.storage_file = storage_file
        self._lock = threading.RLock()
        self._items = self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _field_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _normalize(self, data: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
        previous = existing or {}
        item_id = str(previous.get("id") or data.get("id") or uuid.uuid4().hex[:12])
        name = str(data.get("name", previous.get("name", "未命名功能"))).strip() or "未命名功能"
        description = str(data.get("description", previous.get("description", ""))).strip()

        paths = []
        for raw in data.get("pathFields", previous.get("pathFields", [])):
            label = str(raw.get("label", "数据路径")).strip() or "数据路径"
            paths.append({
                "id": str(raw.get("id") or self._field_id("path")),
                "label": label,
                "mode": str(raw.get("mode", "directory")) if raw.get("mode") in {"file", "directory"} else "directory",
            })

        parameters = []
        for raw in data.get("parameters", previous.get("parameters", [])):
            param_type = str(raw.get("type", "text"))
            if param_type not in {"text", "password", "number", "boolean", "select"}:
                param_type = "text"
            parameters.append({
                "id": str(raw.get("id") or self._field_id("param")),
                "label": str(raw.get("label", "新参数")).strip() or "新参数",
                "type": param_type,
                "default": raw.get("default", False if param_type == "boolean" else ""),
                "options": [str(value) for value in raw.get("options", []) if str(value).strip()],
                "visibleWhen": raw.get("visibleWhen") if isinstance(raw.get("visibleWhen"), dict) else None,
            })

        return {
            "id": item_id,
            "name": name,
            "description": description,
            "handlerId": previous.get("handlerId", data.get("handlerId")),
            "pathFields": paths,
            "parameters": parameters,
            "updatedAt": self._now(),
        }

    def _load(self) -> list[dict[str, Any]]:
        if not self.storage_file.exists():
            return [self._normalize(deepcopy(item)) for item in DEFAULT_FUNCTIONS]
        try:
            content = json.loads(self.storage_file.read_text(encoding="utf-8"))
            if isinstance(content, list):
                return [self._normalize(item) for item in content if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            pass
        return [self._normalize(deepcopy(item)) for item in DEFAULT_FUNCTIONS]

    def _save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_file)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._items)

    def get(self, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = next((item for item in self._items if item["id"] == item_id), None)
            return deepcopy(item) if item else None

    def create(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = data or {
            "name": "未命名功能",
            "description": "接入处理脚本后，在这里配置路径和运行参数。",
            "pathFields": [
                {"id": "input_path", "label": "输入路径", "mode": "directory"},
                {"id": "output_path", "label": "输出路径", "mode": "directory"},
            ],
            "parameters": [],
        }
        with self._lock:
            item = self._normalize(payload)
            self._items.append(item)
            self._save()
            return deepcopy(item)

    def update(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for index, item in enumerate(self._items):
                if item["id"] == item_id:
                    updated = self._normalize(data, existing=item)
                    self._items[index] = updated
                    self._save()
                    return deepcopy(updated)
        raise ValueError("功能不存在。")

    def delete(self, item_id: str) -> None:
        with self._lock:
            original_length = len(self._items)
            self._items = [item for item in self._items if item["id"] != item_id]
            if len(self._items) == original_length:
                raise ValueError("功能不存在。")
            self._save()
