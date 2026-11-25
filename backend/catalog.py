"""Editable function catalog persisted locally as JSON."""

from __future__ import annotations

import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .docker_onnx_quant import CUSTOM_EXCLUDE_PRESET, DEFAULT_EXCLUDE_PRESET, EXCLUDE_MODULE_PRESETS


VIDEO_FUNCTION = {
    "id": "video_frames",
    "name": "视频切图",
    "description": "支持多个视频文件夹，每行一个路径；递归合并、去重并按资源并行提取 JPG，可按固定密度、时间间隔或工序重点时间段抽帧。",
    "handlerId": "video.extract_frames",
    "pathFields": [
        {"id": "video_folder", "label": "视频文件夹（每行一个，递归扫描）", "mode": "directory"},
        {"id": "video_file", "label": "指定 MP4 视频（每行一个，可多选）", "mode": "file"},
        {"id": "output_folder", "label": "输出文件夹（可选）", "mode": "directory"},
    ],
    "parameters": [
        {"id": "output_layout", "label": "图片保存方式", "type": "select", "default": "每个视频单独保存", "options": ["每个视频单独保存", "按源文件夹分别保存", "全部视频合并保存"]},
        {"id": "project_code", "label": "项目英文标识（留空使用视频文件名）", "type": "text", "default": "", "options": []},
        {"id": "scene_code", "label": "场景英文标识（留空使用视频文件名）", "type": "text", "default": "", "options": []},
        {"id": "parallel_workers", "label": "同时处理的视频数（0 自动，1 串行，最多 16）", "type": "number", "default": 0, "options": []},
        {
            "id": "extraction_mode",
            "label": "抽帧方式",
            "type": "select",
            "default": "每秒抽取张数",
            "options": ["每秒抽取张数", "按时间间隔", "工序时间段高密度"],
        },
        {"id": "target_fps", "label": "每秒抽取张数", "type": "number", "default": 5, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "每秒抽取张数"}},
        {"id": "interval_sec", "label": "抽帧时间间隔（秒）", "type": "number", "default": 1, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "按时间间隔"}},
        {"id": "priority_time_ranges", "label": "工序时间段（用分号分隔，如 00:10-00:20; 00:40-00:55）", "type": "text", "default": "", "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "工序时间段高密度"}},
        {"id": "priority_fps", "label": "工序时间段每秒抽取张数", "type": "number", "default": 25, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "工序时间段高密度"}},
        {"id": "background_fps", "label": "其余片段每秒抽取张数", "type": "number", "default": 5, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "工序时间段高密度"}},
        {"id": "priority_max_images", "label": "总图片数量上限", "type": "number", "default": 500, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "equals": "工序时间段高密度"}},
        {"id": "max_images_limit", "label": "最大抽取数量（留空则不限）", "type": "number", "default": "", "options": [], "visibleWhen": {"fieldId": "extraction_mode", "notEquals": "工序时间段高密度"}},
        {"id": "range_start_time", "label": "开始时间（如 01:30）", "type": "text", "default": "00:00:00", "options": []},
        {"id": "range_end_time", "label": "结束时间（如 02:40，留空到结尾）", "type": "text", "default": "", "options": []},
        {
            "id": "limit_strategy",
            "label": "上限策略（填写数量上限时生效）",
            "type": "select",
            "default": "全程均匀抽取",
            "options": ["全程均匀抽取", "从前往后抽取"],
            "visibleWhen": {"fieldId": "extraction_mode", "notEquals": "工序时间段高密度"},
        },
        {"id": "fallback_step", "label": "备用帧间隔", "type": "number", "default": 24, "options": [], "visibleWhen": {"fieldId": "extraction_mode", "notEquals": "工序时间段高密度"}},
    ],
}

IMAGE_DEDUP_FUNCTION = {
    "id": "image_dedup",
    "name": "图片智能去重",
    "description": "按实际画面差异保留代表样本，不设保留比例或删除配额。可包含子文件夹，各工位目录独立分析，不跨目录比较，支持批量报告与回滚。自动模式核验原图整体、局部差异及标注，不确定就保留；近重复判断不保证语义等价或识别所有微小缺陷。完整工序循环模式需单目录指定参考轮。",
    "handlerId": "image.smart_dedup",
    "pathFields": [
        {"id": "images_folder", "label": "需要去重的图片文件夹", "mode": "directory"},
        {"id": "cycle_start_image", "label": "参考循环起始图片（包含此图）", "mode": "file"},
        {"id": "cycle_end_image", "label": "参考循环结束图片（包含此图）", "mode": "file"},
    ],
    "parameters": [
        {
            "id": "operation",
            "label": "操作方式",
            "type": "select",
            "default": "分析并去重",
            "options": ["仅分析（不移动文件）", "分析并去重", "回滚最近一次"],
        },
        {"id": "dedup_mode", "label": "去重模式", "type": "select", "default": "差异优先去重（自动）",
         "options": ["差异优先去重（自动）", "工位工序去重（忽略背景）", "仅精确重复", "连续相似帧去重", "完整工序循环去重"], "visibleWhen": {"fieldId": "operation", "notEquals": "回滚最近一次"}},
        {"id": "recursive", "label": "包含子文件夹（各目录独立去重）", "type": "boolean", "default": False, "options": []},
        {"id": "process_regions", "label": "工位关注区域配置", "type": "text", "default": "", "options": [], "visibleWhen": {"fieldId": "dedup_mode", "equals": "工位工序去重（忽略背景）"}},
        {"id": "process_strength", "label": "工位去重力度", "type": "select", "default": "平衡（推荐）", "options": ["保守", "较保守", "平衡（推荐）", "较强", "强力"], "visibleWhen": {"fieldId": "dedup_mode", "equals": "工位工序去重（忽略背景）"}},
        {
            "id": "dedup_strength",
            "label": "去重力度",
            "type": "select",
            "default": "保守",
            "options": ["保守", "较保守", "平衡（推荐）", "较强", "强力"],
            "visibleWhen": {"fieldId": "operation", "notEquals": "回滚最近一次"},
        },
        {"id": "cycle_min_ratio", "label": "最短循环张数倍数（默认参考轮的 0.4 倍）", "type": "number", "default": .4, "options": [], "visibleWhen": {"fieldId": "dedup_mode", "equals": "完整工序循环去重"}},
        {"id": "cycle_max_ratio", "label": "最长循环张数倍数（默认参考轮的 2.5 倍）", "type": "number", "default": 2.5, "options": [], "visibleWhen": {"fieldId": "dedup_mode", "equals": "完整工序循环去重"}},
    ],
}

YOLO_DATASET_FUNCTION = {
    "id": "yolo_dataset_split",
    "name": "YOLO 数据集划分",
    "description": "支持本地或 SSH 远程服务器完成数据集划分与 YOLO 模型训练，并可保存复用训练场景。",
    "handlerId": "yolo.split_dataset",
    "pathFields": [
        {"id": "images_folder", "label": "图片文件夹", "mode": "directory"},
        {"id": "labels_folder", "label": "标签文件夹（XML/TXT）", "mode": "directory"},
        {"id": "negative_images_folder", "label": "负样本图片文件夹（可选）", "mode": "directory"},
        {"id": "output_folder", "label": "输出位置（将在其中新建数据集文件夹）", "mode": "directory"},
    ],
    "parameters": [
        {
            "id": "dataset_task_type",
            "label": "数据集任务类型",
            "type": "select",
            "default": "目标检测",
            "options": ["目标检测", "旋转目标检测", "分割", "分类"],
        },
        {"id": "remote_host", "label": "远程服务器 IP / 主机名（留空则本地运行）", "type": "text", "default": "", "options": []},
        {"id": "remote_port", "label": "SSH 端口", "type": "number", "default": 22, "options": []},
        {"id": "remote_username", "label": "SSH 用户名", "type": "text", "default": "", "options": []},
        {"id": "remote_password", "label": "SSH 密码", "type": "password", "default": "", "options": []},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": True, "options": []},
        {"id": "remote_python", "label": "远程 Python 解释器（可填绝对路径）", "type": "text", "default": "python3", "options": []},
        {
            "id": "label_format",
            "label": "目标检测标签格式",
            "type": "select",
            "default": "VOC XML",
            "options": ["YOLO TXT", "VOC XML", "仅图片"],
        },
        {"id": "dataset_folder_name", "label": "数据集子文件夹名称（自动按日期时间生成）", "type": "text", "default": "", "options": []},
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

YOLO_TRAINING_FUNCTION = {
    "id": "yolo_model_training",
    "name": "YOLO 模型训练",
    "description": "使用远程 Ultralytics 环境训练，可选自定义源码目录和模型；支持多任务、后台 tmux、训练日志和任务管理。",
    "handlerId": "yolo.train",
    "pathFields": [],
    "parameters": [
        {"id": "remote_host", "label": "远程服务器 IP / 主机名", "type": "text", "default": "", "options": []},
        {"id": "remote_port", "label": "SSH 端口", "type": "number", "default": 22, "options": []},
        {"id": "remote_username", "label": "SSH 用户名", "type": "text", "default": "", "options": []},
        {"id": "remote_password", "label": "SSH 密码", "type": "password", "default": "", "options": []},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": True, "options": []},
    ],
}

NEGATIVE_SAMPLE_FUNCTION = {
    "id": "extract_negative_samples",
    "name": "提取负样本",
    "description": "支持图片与 XML 混放或分开存放。XML 文件夹留空时在图片目录查找；按同名 VOC XML 提取无 object 的 JPG/JPEG，移动图片并删除对应空 XML。递归时优先匹配相同子目录，否则仅匹配唯一同名图片；输出保留图片子目录。",
    "handlerId": "dataset.extract_negative_samples",
    "pathFields": [
        {"id": "source_folder", "label": "图片输入文件夹（可与 XML 混放）", "mode": "directory"},
        {"id": "xml_folder", "label": "XML 标注文件夹（可选，混放时留空）", "mode": "directory"},
        {"id": "negative_folder", "label": "负样本图片输出文件夹", "mode": "directory"},
    ],
    "parameters": [
        {"id": "recursive", "label": "包含子文件夹", "type": "boolean", "default": False, "options": []},
    ],
}

VIDEO_CLIP_FUNCTION = {
    "id": "video_clip",
    "name": "视频裁剪",
    "description": "按时间精确裁剪、拆分训练/测试集或删除片段并拼接；默认另存，也可确认后备份并替换原视频。",
    "handlerId": "video.clip",
    "pathFields": [
        {"id": "video_folder", "label": "视频文件夹", "mode": "directory"},
        {"id": "video_file", "label": "单个视频（可选）", "mode": "file"},
        {"id": "output_folder", "label": "输出文件夹（可选）", "mode": "directory"},
    ],
    "parameters": [
        {"id": "clip_save_mode", "label": "裁剪保存方式", "type": "select", "default": "另存为新视频", "options": ["另存为新视频", "替换原视频（自动备份）"]},
        {"id": "clip_exact_time", "label": "按帧精确裁剪时间", "type": "boolean", "default": True, "options": []},
        {"id": "clip_output_name", "label": "保存文件名（留空自动命名）", "type": "text", "default": "", "options": []},
        {
            "id": "clip_mode",
            "label": "裁剪时间方案",
            "type": "select",
            "default": "指定开始和结束时间",
            "options": ["指定开始和结束时间", "指定开始时间和持续时长", "跳过开头和结尾", "删除多个时间段并拼接剩余"],
        },
        {"id": "remove_time_ranges", "label": "删除时间段（原视频时间轴，每行一个：开始-结束）", "type": "text", "default": "", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "删除多个时间段并拼接剩余"}},
        {"id": "range_start_time", "label": "开始时间（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始和结束时间"}},
        {"id": "range_end_time", "label": "结束时间（留空则到视频结尾）", "type": "text", "default": "", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始和结束时间"}},
        {"id": "duration_start_time", "label": "开始时间（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始时间和持续时长"}},
        {"id": "clip_duration", "label": "持续时长（秒或 HH:MM:SS）", "type": "text", "default": "00:01:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "指定开始时间和持续时长"}},
        {"id": "head_skip_time", "label": "跳过开头（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "跳过开头和结尾"}},
        {"id": "tail_skip_time", "label": "忽略结尾（秒或 HH:MM:SS）", "type": "text", "default": "00:00:00", "options": [], "visibleWhen": {"fieldId": "clip_mode", "equals": "跳过开头和结尾"}},
        {
            "id": "encoding_mode",
            "label": "处理方式",
            "type": "select",
            "default": "原画质裁剪（推荐，不重新编码）",
            "options": ["原画质裁剪（推荐，不重新编码）", "精确裁剪（重新编码）"],
        },
        {
            "id": "split_train_test",
            "label": "拆分训练集和测试集（指定片段为训练集，其余为测试集）",
            "type": "boolean",
            "default": False,
            "options": [],
            "visibleWhen": {"fieldId": "clip_mode", "notEquals": "删除多个时间段并拼接剩余"},
        },
    ],
}

REMOTE_STAR_INFERENCE_FUNCTION = {
    "id": "remote_star_inference",
    "name": "远程实时 AI 推理",
    "description": "平台自动上传模型和视频到远端 AI 设备，按原视频时间轴显示 STAR/TensorRT 带框画面；推理不足时自动跳过落后帧。",
    "handlerId": "remote.star_inference",
    "pathFields": [
        {"id": "local_script_file", "label": "本地推理脚本（可选，默认使用内置版本）", "mode": "file"},
        {"id": "local_model_file", "label": "本地 STAR 模型文件", "mode": "file"},
        {"id": "local_video_file", "label": "本地视频文件", "mode": "file"},
    ],
    "parameters": [
        {"id": "host", "label": "设备 IP / 主机名", "type": "text", "default": "", "options": []},
        {"id": "port", "label": "SSH 端口", "type": "number", "default": 22, "options": []},
        {"id": "username", "label": "SSH 用户名", "type": "text", "default": "", "options": []},
        {"id": "password", "label": "SSH 密码", "type": "password", "default": "", "options": []},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": True, "options": []},
        {"id": "python_path", "label": "远端 Python 解释器", "type": "text", "default": "python3", "options": []},
        {"id": "labels", "label": "类别名称（英文逗号分隔）", "type": "text", "default": "", "options": []},
        {"id": "conf", "label": "置信度阈值", "type": "number", "default": 0.5, "options": []},
        {"id": "iou", "label": "NMS IoU 阈值", "type": "number", "default": 0.45, "options": []},
        {"id": "max_det", "label": "每帧最大检测框数", "type": "number", "default": 300, "options": []},
        {"id": "save_path", "label": "远端保存路径（可选）", "type": "text", "default": "", "options": []},
        {"id": "realtime", "label": "保持原视频速度（性能不足时自动跳帧）", "type": "boolean", "default": True, "options": []},
        {"id": "detection_fps", "label": "目标检测帧率（按源视频每秒取帧，0 表示不限）", "type": "number", "default": 20, "options": []},
        {"id": "preview_fps", "label": "本地预览帧率", "type": "number", "default": 20, "options": []},
        {"id": "stream_width", "label": "本地预览宽度", "type": "number", "default": 960, "options": []},
        {"id": "jpeg_quality", "label": "预览画质（20-100）", "type": "number", "default": 70, "options": []},
        {"id": "replay_grace_seconds", "label": "结束后可回看时间（秒，0 为关闭）", "type": "number", "default": 120, "options": []},
        {
            "id": "cuda_backend",
            "label": "CUDA 后端",
            "type": "select",
            "default": "auto",
            "options": ["auto", "cuda-driver", "cuda-python", "pycuda"],
        },
    ],
}

LOCAL_PT_INFERENCE_FUNCTION = {
    "id": "local_pt_inference",
    "name": "本地 PT / ONNX 推理检测",
    "description": "支持视频或图片文件夹，使用 Ultralytics PT 和标准 YOLO ONNX 自动读取类别、检测目标。图片可递归检测并保存带框结果，不修改原图。不支持 Jetson 裁剪头 ONNX。",
    "handlerId": "local.pt_inference",
    "pathFields": [
        {"id": "model_file", "label": "本地 PT / ONNX 模型文件", "mode": "file"},
        {"id": "video_file", "label": "本地视频文件", "mode": "file"},
        {"id": "output_folder", "label": "检测结果输出文件夹", "mode": "directory"},
        {"id": "image_folder", "label": "本地图片文件夹", "mode": "directory"},
    ],
    "parameters": [
        {"id": "input_mode", "label": "输入类型", "type": "select", "default": "视频检测", "options": ["视频检测", "图片文件夹检测"]},
        {"id": "recursive_images", "label": "包含图片子文件夹", "type": "boolean", "default": True, "options": []},
        {"id": "conf", "label": "置信度阈值", "type": "number", "default": 0.5, "options": []},
        {"id": "iou", "label": "NMS IoU 阈值", "type": "number", "default": 0.45, "options": []},
        {"id": "max_det", "label": "每帧最大检测框数", "type": "number", "default": 300, "options": []},
        {"id": "device", "label": "推理设备（auto、cpu、0 等）", "type": "text", "default": "auto", "options": []},
        {"id": "realtime", "label": "保持原视频速度（性能不足时自动跳帧）", "type": "boolean", "default": True, "options": []},
        {"id": "preview_fps", "label": "预览帧率（0 表示不限）", "type": "number", "default": 20, "options": []},
        {"id": "stream_width", "label": "预览宽度", "type": "number", "default": 960, "options": []},
        {"id": "jpeg_quality", "label": "预览画质（20-100）", "type": "number", "default": 70, "options": []},
        {"id": "replay_grace_seconds", "label": "结束后可回看时间（秒）", "type": "number", "default": 120, "options": []},
    ],
}

CALIB_DATASET_FUNCTION = {
    "id": "quant_calibration_dataset",
    "name": "制作量化数据集",
    "description": "支持本机或 SSH 服务器制作量化数据集；优先保留全部非 copy 原始图片，不足时再补充 copy 图片与负样本。",
    "handlerId": "calib.select_dataset",
    "pathFields": [],
    "parameters": [
        {"id": "execution_location", "label": "运行位置", "type": "select", "default": "本地运行", "options": ["本地运行", "SSH 远程服务器"]},
        {"id": "image_dirs", "label": "图片目录（每行一个）", "type": "text", "default": "", "options": []},
        {"id": "annotation_dirs", "label": "标注目录（每行一个，可选）", "type": "text", "default": "", "options": []},
        {"id": "negative_dirs", "label": "负样本图片目录（每行一个，可选）", "type": "text", "default": "", "options": []},
        {"id": "output_dir", "label": "输出目录（留空默认 calib_dataset）", "type": "text", "default": "", "options": []},
        {"id": "annotation_format", "label": "标注格式", "type": "select", "default": "auto", "options": ["auto", "xml", "json"]},
        {"id": "num_samples", "label": "抽取图片数量", "type": "number", "default": 128, "options": []},
        {"id": "random_seed", "label": "随机种子", "type": "number", "default": 42, "options": []},
        {"id": "remote_host", "label": "服务器 IP / 主机名", "type": "text", "default": "", "options": []},
        {"id": "remote_port", "label": "SSH 端口", "type": "number", "default": 22, "options": []},
        {"id": "remote_username", "label": "SSH 用户名", "type": "text", "default": "", "options": []},
        {"id": "remote_password", "label": "SSH 密码", "type": "password", "default": "", "options": []},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": True, "options": []},
        {"id": "remote_python", "label": "远程 Python 解释器", "type": "text", "default": "python3", "options": []},
    ],
}

DOCKER_ONNX_QUANT_FUNCTION = {
    "id": "docker_onnx_quant",
    "name": "Docker ONNX 量化",
    "description": "在本机 onnx2quant Docker 容器的 onnxquant Conda 环境中，使用量化图片导出 INT8 ONNX。",
    "handlerId": "docker.quantize_onnx",
    "pathFields": [
        {"id": "input_onnx", "label": "本机输入 ONNX 模型", "mode": "file"},
        {"id": "calibration_images", "label": "本机量化图片文件夹", "mode": "directory"},
        {"id": "output_folder", "label": "本机输出文件夹（留空则与模型同级）", "mode": "directory"},
    ],
    "parameters": [
        {"id": "operation", "label": "操作方式", "type": "select", "default": "导出量化 ONNX", "options": ["导出量化 ONNX", "测试 Docker 环境"]},
        {"id": "container_name", "label": "Docker 容器名称", "type": "text", "default": "onnx2quant", "options": []},
        {"id": "conda_environment", "label": "容器内 Conda 环境", "type": "text", "default": "onnxquant", "options": []},
        {"id": "conda_path", "label": "容器内 Conda 路径", "type": "text", "default": "/root/anaconda3/bin/conda", "options": []},
        {"id": "model_type", "label": "模型类型 type", "type": "select", "default": "yolo", "options": ["yolo", "mobilenet", "custom"], "visibleWhen": {"fieldId": "operation", "equals": "导出量化 ONNX"}},
        {"id": "limit", "label": "量化图片上限 limit", "type": "number", "default": 128, "options": [], "visibleWhen": {"fieldId": "operation", "equals": "导出量化 ONNX"}},
        {"id": "output_filename", "label": "输出文件名（留空自动生成）", "type": "text", "default": "", "options": [], "visibleWhen": {"fieldId": "operation", "equals": "导出量化 ONNX"}},
        {"id": "method", "label": "量化方法", "type": "select", "default": "默认", "options": ["默认", "minmax", "entropy"], "visibleWhen": {"fieldId": "operation", "equals": "导出量化 ONNX"}},
        {"id": "exclude_modules", "label": "排除模块", "type": "select", "default": DEFAULT_EXCLUDE_PRESET,
         "options": [*EXCLUDE_MODULE_PRESETS, CUSTOM_EXCLUDE_PRESET], "visibleWhen": {"fieldId": "operation", "equals": "导出量化 ONNX"}},
        {"id": "nodes_to_exclude", "label": "自定义排除节点（每行一个正则，留空则不排除）", "type": "text", "default": "", "options": [],
         "visibleWhen": {"fieldId": "exclude_modules", "equals": CUSTOM_EXCLUDE_PRESET}},
    ],
}

JETSON_ONNX_EXPORT_FUNCTION = {
    "id": "jetson_onnx_export",
    "name": "Jetson ONNX 导出",
    "description": "支持批量选择多个 PT，依次完成 YOLO26 切头或 YOLOv5 官方导出，生成 Jetson/TensorRT 可用的静态 ONNX。",
    "handlerId": "model.export_jetson_onnx",
    "pathFields": [
        {"id": "weights_file", "label": "本机 PT 权重文件（每行一个，可批量选择）", "mode": "file"},
        {"id": "output_folder", "label": "ONNX 输出文件夹（留空则与 PT 同级）", "mode": "directory"},
    ],
    "parameters": [
        {"id": "model_family", "label": "模型版本", "type": "select", "default": "YOLO26", "options": ["YOLO26", "YOLOv5"]},
        {"id": "deployment_precision", "label": "后续部署精度", "type": "select", "default": "INT8", "options": ["INT8", "FP16", "FP32"], "visibleWhen": {"fieldId": "target_platform", "equals": "Jetson / TensorRT（NCHW）"}},
        {"id": "task_type", "label": "模型任务类型", "type": "select", "default": "目标检测", "options": ["目标检测", "目标分割", "旋转目标检测", "姿态估计", "图像分类"]},
        {"id": "target_platform", "label": "目标平台", "type": "select", "default": "Jetson / TensorRT（NCHW）", "options": ["Jetson / TensorRT（NCHW）", "RKNN（NHWC）"]},
        {"id": "imgsz", "label": "输入尺寸 imgsz", "type": "number", "default": 640, "options": []},
        {"id": "opset", "label": "ONNX opset", "type": "number", "default": 12, "options": []},
        {"id": "output_filename", "label": "单模型输出文件名（批量时留空）", "type": "text", "default": "", "options": []},
        {"id": "python_path", "label": "YOLO Python（YOLO26 留空可自动查找）", "type": "text", "default": "", "options": []},
        {"id": "yolov5_repo", "label": "YOLOv5 仓库目录（应包含 export.py 和 models）", "type": "text", "default": "", "options": [], "visibleWhen": {"fieldId": "model_family", "equals": "YOLOv5"}},
    ],
}

REMOTE_TENSORRT_BUILD_FUNCTION = {
    "id": "remote_tensorrt_build",
    "name": "远程 TensorRT 引擎构建",
    "description": "通过 SSH 顺序构建 TensorRT .plan。支持未量化 FP32 ONNX 构建 FP16/FP32 引擎，也支持已有量化 ONNX 的部署流程。切换构建精度不会恢复量化损失；FP16/FP32 会检查并拒绝量化图。",
    "handlerId": "remote.build_tensorrt",
    "pathFields": [],
    "parameters": [
        {"id": "onnx_files", "label": "远端 ONNX 路径（每行一个）", "type": "text", "default": "", "options": []},
        {"id": "operation", "label": "操作方式", "type": "select", "default": "构建 TensorRT 引擎", "options": ["构建 TensorRT 引擎", "检查远程环境"]},
        {"id": "trtexec_path", "label": "远端 trtexec 路径", "type": "text", "default": "/usr/src/tensorrt/bin/trtexec", "options": []},
        {"id": "remote_workspace", "label": "远端任务根目录", "type": "text", "default": "~/.local/state/yolo-processing/tensorrt", "options": []},
        {"id": "build_mode", "label": "构建精度", "type": "select", "default": "自动最佳（--best）", "options": ["自动最佳（--best）", "FP16", "FP32"]},
        {"id": "run_benchmark", "label": "构建后执行性能测试", "type": "boolean", "default": True, "options": []},
        {"id": "keep_remote_files", "label": "保留远端后台任务状态目录", "type": "boolean", "default": True, "options": []},
        {"id": "overwrite_output", "label": "允许覆盖远端同名结果", "type": "boolean", "default": False, "options": []},
        {"id": "host", "label": "AI 推理盒子 IP / 主机名", "type": "text", "default": "", "options": []},
        {"id": "port", "label": "SSH 端口", "type": "number", "default": 22, "options": []},
        {"id": "username", "label": "SSH 用户名", "type": "text", "default": "", "options": []},
        {"id": "password", "label": "SSH 密码", "type": "password", "default": "", "options": []},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": True, "options": []},
    ],
}

LOCAL_STAR_PACKAGE_FUNCTION = {
    "id": "local_star_package",
    "name": "本地 STAR 模型打包",
    "description": "使用本机 package_tool 和 trt.toml 模板，将 TensorRT .plan 打包为同级目录下的 .star 模型。",
    "handlerId": "model.package_star",
    "pathFields": [
        {"id": "model_file", "label": "本机 TensorRT PLAN 模型", "mode": "file"},
    ],
    "parameters": [
        {"id": "title", "label": "打包名称 title", "type": "text", "default": "", "options": []},
        {"id": "labels", "label": "类别名称 labels", "type": "text", "default": "", "options": []},
        {"id": "package_tool_path", "label": "package_tool 路径", "type": "text", "default": "", "options": []},
        {"id": "template_file", "label": "trt.toml 模板路径", "type": "text", "default": "", "options": []},
        {"id": "hardware_name", "label": "硬件平台", "type": "text", "default": "TensorRT", "options": []},
        {"id": "architecture", "label": "硬件架构", "type": "text", "default": "Turing", "options": []},
        {"id": "driver", "label": "驱动信息", "type": "text", "default": "CUDA12.2", "options": []},
        {"id": "model_name", "label": "模型架构名称", "type": "text", "default": "yolo26", "options": []},
        {"id": "category", "label": "模型类别", "type": "select", "default": "generic", "options": ["generic", "seg", "classify"]},
        {"id": "version", "label": "版本信息", "type": "text", "default": "1", "options": []},
        {"id": "precision", "label": "引擎精度声明（不转换模型）", "type": "select", "default": "INT8 (1)", "options": ["INT4 (0)", "INT8 (1)", "FP8 (2)", "FP16 (3)", "FP32 (4)"]},
        {"id": "input_n", "label": "输入张量 N", "type": "number", "default": 1, "options": []},
        {"id": "input_c", "label": "输入张量 C", "type": "number", "default": 3, "options": []},
        {"id": "input_h", "label": "输入张量 H", "type": "number", "default": 640, "options": []},
        {"id": "input_w", "label": "输入张量 W", "type": "number", "default": 640, "options": []},
        {"id": "color_format", "label": "颜色格式", "type": "select", "default": "RGB (2)", "options": ["NV12 (0)", "NV21 (1)", "RGB (2)", "BGR (3)", "GRAY (4)"]},
    ],
}

AUTO_ANNOTATE_FUNCTION = {
    "id": "auto_annotation", "name": "模型自动标注", "handlerId": "dataset.auto_annotate",
    "description": "本地 PT 或远程 STAR 模型自动标注。本机图片旁生成同名 VOC XML，原图不变。STAR 模式逐张上传到 SSH 盒子，复用实时推理预处理与解码；任务结束清理临时文件。自动标注及未检出目标的空标注都需要人工复核。",
    "pathFields": [
        {"id": "image_folder", "label": "待标注图片文件夹", "mode": "directory"},
        {"id": "weights_file", "label": "训练后的 PT 检测模型", "mode": "file"},
    ],
    "parameters": [
        {"id": "annotation_mode", "label": "自动标注模式", "type": "select", "default": "本地 PT 标注", "options": ["本地 PT 标注", "远程 STAR 标注"]},
        {"id": "conf", "label": "置信度阈值（0–1）", "type": "number", "default": 0.5},
        {"id": "iou", "label": "重叠框抑制 IoU 阈值（0–1）", "type": "number", "default": 0.7},
        {"id": "imgsz", "label": "推理尺寸（32 的倍数）", "type": "number", "default": 640, "visibleWhen": {"fieldId": "annotation_mode", "equals": "本地 PT 标注"}},
        {"id": "device", "label": "设备（留空自动；cpu 或 GPU 编号如 0）", "type": "text", "default": "", "visibleWhen": {"fieldId": "annotation_mode", "equals": "本地 PT 标注"}},
        {"id": "classes", "label": "仅标注类别编号（逗号分隔；留空全部）", "type": "text", "default": ""},
        {"id": "recursive", "label": "包含子文件夹", "type": "boolean", "default": True},
        {"id": "existing", "label": "已有 XML", "type": "select", "default": "跳过", "options": ["跳过", "备份后覆盖"]},
        {"id": "empty_xml", "label": "没有目标时生成空 XML（待人工确认的负样本）", "type": "boolean", "default": True},
        {"id": "python_path", "label": "Python 路径（留空自动使用本机 yolo26 环境）", "type": "text", "default": "", "visibleWhen": {"fieldId": "annotation_mode", "equals": "本地 PT 标注"}},
        {"id": "star_labels", "label": "STAR 类别名称（每行一个；留空读取模型）", "type": "text", "default": "", "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "remote_host", "label": "标注盒子 IP / 主机名", "type": "text", "default": "", "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "remote_port", "label": "SSH 端口", "type": "number", "default": 22, "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "remote_username", "label": "SSH 用户名", "type": "text", "default": "", "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "remote_password", "label": "SSH 密码", "type": "password", "default": "", "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "remember_password", "label": "记住 SSH 密码", "type": "boolean", "default": True, "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "remote_python", "label": "盒子 Python 解释器（命令或绝对路径）", "type": "text", "default": "python3", "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
        {"id": "cuda_backend", "label": "CUDA 后端", "type": "select", "default": "auto", "options": ["auto", "cuda-driver-ctypes", "cuda-driver", "cuda-python", "pycuda"], "visibleWhen": {"fieldId": "annotation_mode", "equals": "远程 STAR 标注"}},
    ],
}

DEFAULT_FUNCTIONS = [VIDEO_FUNCTION, YOLO_DATASET_FUNCTION, VIDEO_CLIP_FUNCTION, REMOTE_STAR_INFERENCE_FUNCTION, CALIB_DATASET_FUNCTION, LOCAL_PT_INFERENCE_FUNCTION, IMAGE_DEDUP_FUNCTION, JETSON_ONNX_EXPORT_FUNCTION, DOCKER_ONNX_QUANT_FUNCTION, REMOTE_TENSORRT_BUILD_FUNCTION, LOCAL_STAR_PACKAGE_FUNCTION, AUTO_ANNOTATE_FUNCTION, YOLO_TRAINING_FUNCTION, NEGATIVE_SAMPLE_FUNCTION]


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
                migration_marker = self.storage_file.with_name(self.storage_file.name + ".local-pt-v1")
                local_images_marker = self.storage_file.with_name(self.storage_file.name + ".local-images-v1")
                clip_migration_marker = self.storage_file.with_name(self.storage_file.name + ".video-clip-copy-v1")
                clip_split_marker = self.storage_file.with_name(self.storage_file.name + ".video-clip-train-test-v2")
                clip_remove_marker = self.storage_file.with_name(self.storage_file.name + ".video-clip-remove-v3")
                dedup_migration_marker = self.storage_file.with_name(self.storage_file.name + ".image-dedup-v1")
                cycle_dedup_marker = self.storage_file.with_name(self.storage_file.name + ".image-cycle-dedup-v2")
                quality_dedup_marker = self.storage_file.with_name(self.storage_file.name + ".image-quality-dedup-v3")
                recursive_dedup_marker = self.storage_file.with_name(self.storage_file.name + ".image-recursive-dedup-v4")
                process_dedup_marker = self.storage_file.with_name(self.storage_file.name + ".image-process-dedup-v5")
                quant_migration_marker = self.storage_file.with_name(self.storage_file.name + ".docker-onnx-quant-v1")
                export_migration_marker = self.storage_file.with_name(self.storage_file.name + ".jetson-onnx-export-v1")
                export_family_marker = self.storage_file.with_name(self.storage_file.name + ".jetson-onnx-family-v2")
                export_batch_marker = self.storage_file.with_name(self.storage_file.name + ".jetson-onnx-batch-v3")
                calib_priority_marker = self.storage_file.with_name(self.storage_file.name + ".calib-priority-v1")
                calib_local_marker = self.storage_file.with_name(self.storage_file.name + ".calib-local-v2")
                remote_trt_marker = self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-build-v1")
                remote_trt_launcher_marker = self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-launcher-v2")
                remote_trt_batch_marker = self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-batch-v3")
                remote_trt_paths_marker = self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-paths-v4")
                remote_trt_remote_paths_marker = self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-remote-paths-v5")
                local_star_package_marker = self.storage_file.with_name(self.storage_file.name + ".local-star-package-v1")
                local_star_only_marker = self.storage_file.with_name(self.storage_file.name + ".local-star-package-local-v3")
                video_frame_naming_marker = self.storage_file.with_name(self.storage_file.name + ".video-frame-naming-v1")
                video_filename_fallback_marker = self.storage_file.with_name(self.storage_file.name + ".video-frame-filename-fallback-v2")
                video_frame_priority_marker = self.storage_file.with_name(self.storage_file.name + ".video-frame-priority-v2")
                video_folder_recursive_marker = self.storage_file.with_name(self.storage_file.name + ".video-folder-recursive-v1")
                video_multi_folders_marker = self.storage_file.with_name(self.storage_file.name + ".video-multi-folders-v2")
                video_output_layout_marker = self.storage_file.with_name(self.storage_file.name + ".video-output-layout-v2")
                video_parallel_marker = self.storage_file.with_name(self.storage_file.name + ".video-frame-parallel-v1")
                realtime_playback_marker = self.storage_file.with_name(self.storage_file.name + ".realtime-playback-v1")
                yolo_dataset_timestamp_marker = self.storage_file.with_name(self.storage_file.name + ".yolo-dataset-timestamp-v1")
                yolo_training_function_marker = self.storage_file.with_name(self.storage_file.name + ".yolo-training-function-v1")
                yolo_training_remote_source_marker = self.storage_file.with_name(self.storage_file.name + ".yolo-training-remote-source-v2")
                negative_sample_marker = self.storage_file.with_name(self.storage_file.name + ".negative-sample-extractor-v1")
                negative_separate_marker = self.storage_file.with_name(self.storage_file.name + ".negative-sample-separate-folders-v2")
                changed = False
                annotation_marker = self.storage_file.with_name(self.storage_file.name + ".auto-annotation-v1")
                annotation_star_marker = self.storage_file.with_name(self.storage_file.name + ".auto-annotation-star-v2")
                precision_options_marker = self.storage_file.with_name(self.storage_file.name + ".precision-options-v1")
                if not yolo_training_function_marker.exists():
                    if not any(isinstance(item, dict) and item.get("id") == YOLO_TRAINING_FUNCTION["id"] for item in content):
                        content.append(deepcopy(YOLO_TRAINING_FUNCTION))
                        changed = True
                if not negative_sample_marker.exists():
                    if not any(isinstance(item, dict) and item.get("id") == NEGATIVE_SAMPLE_FUNCTION["id"] for item in content):
                        content.append(deepcopy(NEGATIVE_SAMPLE_FUNCTION))
                        changed = True
                if not annotation_marker.exists():
                    if not any(isinstance(item, dict) and item.get("id") == AUTO_ANNOTATE_FUNCTION["id"] for item in content):
                        content.append(deepcopy(AUTO_ANNOTATE_FUNCTION))
                        changed = True
                if not migration_marker.exists() and not any(isinstance(item, dict) and item.get("id") == LOCAL_PT_INFERENCE_FUNCTION["id"] for item in content):
                    content.append(deepcopy(LOCAL_PT_INFERENCE_FUNCTION))
                    changed = True
                if not dedup_migration_marker.exists() and not any(isinstance(item, dict) and item.get("id") == IMAGE_DEDUP_FUNCTION["id"] for item in content):
                    content.append(deepcopy(IMAGE_DEDUP_FUNCTION))
                    changed = True
                if not quant_migration_marker.exists() and not any(isinstance(item, dict) and item.get("id") == DOCKER_ONNX_QUANT_FUNCTION["id"] for item in content):
                    content.append(deepcopy(DOCKER_ONNX_QUANT_FUNCTION))
                    changed = True
                if not export_migration_marker.exists() and not any(isinstance(item, dict) and item.get("id") == JETSON_ONNX_EXPORT_FUNCTION["id"] for item in content):
                    content.append(deepcopy(JETSON_ONNX_EXPORT_FUNCTION))
                    changed = True
                if not remote_trt_marker.exists() and not any(isinstance(item, dict) and item.get("id") == REMOTE_TENSORRT_BUILD_FUNCTION["id"] for item in content):
                    content.append(deepcopy(REMOTE_TENSORRT_BUILD_FUNCTION))
                    changed = True
                if not local_star_package_marker.exists() and not any(isinstance(item, dict) and item.get("id") == LOCAL_STAR_PACKAGE_FUNCTION["id"] for item in content):
                    content.append(deepcopy(LOCAL_STAR_PACKAGE_FUNCTION))
                    changed = True
                for item in content:
                    if isinstance(item, dict) and item.get("id") == DOCKER_ONNX_QUANT_FUNCTION["id"]:
                        parameters = item.setdefault("parameters", [])
                        if not any(isinstance(p, dict) and p.get("id") == "exclude_modules" for p in parameters):
                            preset, custom = deepcopy(DOCKER_ONNX_QUANT_FUNCTION["parameters"][-2:])
                            for index, field in enumerate(parameters):
                                if isinstance(field, dict) and field.get("id") == "nodes_to_exclude":
                                    custom["default"] = field.get("default", "")
                                    parameters[index:index + 1] = [preset, custom]
                                    break
                            else:
                                parameters.extend([preset, custom])
                            changed = True
                    if (not process_dedup_marker.exists() and isinstance(item, dict)
                            and item.get('handlerId') == IMAGE_DEDUP_FUNCTION['handlerId']):
                        params = item.setdefault('parameters', [])
                        for source in IMAGE_DEDUP_FUNCTION['parameters']:
                            if source['id'] not in {'dedup_mode', 'process_regions', 'process_strength'}:
                                continue
                            existing = next((p for p in params if p.get('id') == source['id']), None)
                            if existing is None:
                                params.append(deepcopy(source))
                            elif source['id'] == 'dedup_mode':
                                existing['options'] = deepcopy(source['options'])
                        changed = True
                    if (not recursive_dedup_marker.exists() and isinstance(item, dict)
                            and item.get('handlerId') == IMAGE_DEDUP_FUNCTION['handlerId']):
                        item['description'] = IMAGE_DEDUP_FUNCTION['description']
                        params = item.setdefault('parameters', [])
                        if not any(p.get('id') == 'recursive' for p in params):
                            params.insert(2, deepcopy(next(p for p in IMAGE_DEDUP_FUNCTION['parameters'] if p['id'] == 'recursive')))
                        changed = True
                    if (not quality_dedup_marker.exists() and isinstance(item, dict)
                            and item.get('handlerId') == IMAGE_DEDUP_FUNCTION['handlerId']):
                        item['description'] = IMAGE_DEDUP_FUNCTION['description']
                        item['parameters'] = [p for p in item.get('parameters', []) if p.get('id') not in {'minimum_keep_ratio', 'max_consecutive_skips'}]
                        mode_field = next((p for p in item['parameters'] if p.get('id') == 'dedup_mode'), None)
                        source = next(p for p in IMAGE_DEDUP_FUNCTION['parameters'] if p['id'] == 'dedup_mode')
                        if mode_field is None:
                            item['parameters'].insert(1, deepcopy(source))
                        else:
                            mode_field['options'] = deepcopy(source['options'])
                            if mode_field.get('default') == '连续相似帧去重':
                                mode_field['default'] = source['default']
                        changed = True
                    if (not video_output_layout_marker.exists() and isinstance(item, dict)
                            and item.get('handlerId') == VIDEO_FUNCTION['handlerId']):
                        params = item.setdefault('parameters', [])
                        if not any(isinstance(p, dict) and p.get('id') == 'output_layout' for p in params):
                            params.insert(0, deepcopy(VIDEO_FUNCTION['parameters'][0]))
                            changed = True
                        else:
                            field = next(p for p in params if isinstance(p, dict) and p.get('id') == 'output_layout')
                            field['options'] = list(VIDEO_FUNCTION['parameters'][0]['options'])
                            if field.get('default') == '按源文件夹分别保存':
                                field['default'] = VIDEO_FUNCTION['parameters'][0]['default']
                            changed = True
                    if not precision_options_marker.exists() and isinstance(item, dict):
                        handler = item.get('handlerId')
                        if handler == JETSON_ONNX_EXPORT_FUNCTION['handlerId']:
                            params = item.setdefault('parameters', [])
                            if not any(isinstance(p, dict) and p.get('id') == 'deployment_precision' for p in params):
                                params.insert(1, deepcopy(next(p for p in JETSON_ONNX_EXPORT_FUNCTION['parameters'] if p['id'] == 'deployment_precision')))
                                changed = True
                        if handler == REMOTE_TENSORRT_BUILD_FUNCTION['handlerId']:
                            item['description'] = REMOTE_TENSORRT_BUILD_FUNCTION['description']
                            for parameter in item.get('parameters', []):
                                if parameter.get('id') == 'onnx_files' and parameter.get('label') == '量化 ONNX 路径（每行一个）':
                                    parameter['label'] = '远端 ONNX 路径（每行一个）'
                            changed = True
                        if handler == LOCAL_STAR_PACKAGE_FUNCTION['handlerId']:
                            for parameter in item.get('parameters', []):
                                if parameter.get('id') == 'precision' and parameter.get('label') == '模型精度':
                                    parameter['label'] = '引擎精度声明（不转换模型）'
                                    changed = True
                    if (not annotation_star_marker.exists() and isinstance(item, dict)
                            and item.get("handlerId") == AUTO_ANNOTATE_FUNCTION["handlerId"]):
                        parameters = item.setdefault("parameters", [])
                        existing = {field.get("id"): field for field in parameters if isinstance(field, dict)}
                        for field in AUTO_ANNOTATE_FUNCTION["parameters"]:
                            if field["id"] not in existing:
                                parameters.append(deepcopy(field))
                            elif field.get("visibleWhen"):
                                existing[field["id"]]["visibleWhen"] = deepcopy(field["visibleWhen"])
                        item["description"] = AUTO_ANNOTATE_FUNCTION["description"]
                        changed = True
                    if (not negative_separate_marker.exists() and isinstance(item, dict)
                            and item.get("handlerId") == NEGATIVE_SAMPLE_FUNCTION["handlerId"]):
                        fields = item.setdefault("pathFields", [])
                        if not any(isinstance(field, dict) and field.get("id") == "xml_folder" for field in fields):
                            insert_at = next((index for index, field in enumerate(fields)
                                              if isinstance(field, dict) and field.get("id") == "source_folder"), -1) + 1
                            fields.insert(insert_at, deepcopy(NEGATIVE_SAMPLE_FUNCTION["pathFields"][1]))
                        for field in fields:
                            if isinstance(field, dict) and field.get("id") == "source_folder" and field.get("label") == "图片和 XML 输入文件夹":
                                field["label"] = NEGATIVE_SAMPLE_FUNCTION["pathFields"][0]["label"]
                        if item.get("description") == "检查同名 VOC XML，将没有 object 标注的 JPG/JPEG 移动到负样本目录，并删除对应空 XML。":
                            item["description"] = NEGATIVE_SAMPLE_FUNCTION["description"]
                        changed = True
                    if (not video_parallel_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_FUNCTION["id"]):
                        parameters = item.setdefault("parameters", [])
                        if not any(isinstance(parameter, dict) and parameter.get("id") == "parallel_workers" for parameter in parameters):
                            parameters.insert(2, deepcopy(next(parameter for parameter in VIDEO_FUNCTION["parameters"] if parameter["id"] == "parallel_workers")))
                            changed = True
                    if (not video_folder_recursive_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_FUNCTION["id"]):
                        item["description"] = VIDEO_FUNCTION["description"]
                        item["pathFields"] = [
                            deepcopy(VIDEO_FUNCTION["pathFields"][0])
                            if isinstance(path, dict) and path.get("id") == "video_folder"
                            else path
                            for path in item.get("pathFields", [])
                        ]
                        changed = True
                    if isinstance(item, dict) and item.get("id") == VIDEO_FUNCTION["id"]:
                        for field in item.get("pathFields", []):
                            if isinstance(field, dict) and field.get("id") == "video_file" and field.get("label") == "单个视频（可选）":
                                field["label"] = VIDEO_FUNCTION["pathFields"][1]["label"]
                                changed = True
                    if (not video_multi_folders_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_FUNCTION["id"]):
                        item["description"] = VIDEO_FUNCTION["description"]
                        for field in item.get("pathFields", []):
                            if isinstance(field, dict) and field.get("id") == "video_folder" and field.get("label") in {"视频文件夹", "视频文件夹（递归扫描子文件夹）"}:
                                field["label"] = VIDEO_FUNCTION["pathFields"][0]["label"]
                                changed = True
                        changed = True
                    if (not yolo_training_remote_source_marker.exists() and isinstance(item, dict)
                            and item.get("id") == YOLO_TRAINING_FUNCTION["id"]):
                        item["description"] = YOLO_TRAINING_FUNCTION["description"]
                        existing_parameters = {
                            parameter.get("id"): parameter
                            for parameter in item.get("parameters", [])
                            if isinstance(parameter, dict)
                        }
                        item["parameters"] = [
                            deepcopy(existing_parameters.get(parameter["id"], parameter))
                            for parameter in YOLO_TRAINING_FUNCTION["parameters"]
                        ]
                        for parameter in item["parameters"]:
                            source_parameter = next(
                                source for source in YOLO_TRAINING_FUNCTION["parameters"]
                                if source["id"] == parameter["id"]
                            )
                            if parameter["id"] in {"remote_host", "remote_username"} and not str(parameter.get("default", "")).strip():
                                parameter.update(deepcopy(source_parameter))
                        changed = True
                    if (not yolo_dataset_timestamp_marker.exists() and isinstance(item, dict)
                            and item.get("id") == YOLO_DATASET_FUNCTION["id"]):
                        source_parameter = next(
                            parameter for parameter in YOLO_DATASET_FUNCTION["parameters"]
                            if parameter["id"] == "dataset_folder_name"
                        )
                        for parameter in item.get("parameters", []):
                            if isinstance(parameter, dict) and parameter.get("id") == "dataset_folder_name":
                                parameter.update(deepcopy(source_parameter))
                                changed = True
                                break
                    if (not video_frame_naming_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_FUNCTION["id"]):
                        item["description"] = VIDEO_FUNCTION["description"]
                        parameters = item.setdefault("parameters", [])
                        existing_ids = {
                            parameter.get("id") for parameter in parameters
                            if isinstance(parameter, dict)
                        }
                        new_parameters = [
                            deepcopy(parameter) for parameter in VIDEO_FUNCTION["parameters"]
                            if parameter['id'] in {'project_code', 'scene_code'} and parameter["id"] not in existing_ids
                        ]
                        item["parameters"] = [*new_parameters, *parameters]
                        changed = True
                    if (not video_filename_fallback_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_FUNCTION["id"]):
                        labels = {field["id"]: field["label"] for field in VIDEO_FUNCTION["parameters"] if field['id'] in {'project_code', 'scene_code'}}
                        old_labels = {"project_code": "项目英文标识（如 tcl）", "scene_code": "场景英文标识（如 line2_station3）"}
                        for parameter in item.get("parameters", []):
                            if isinstance(parameter, dict) and parameter.get("id") in labels:
                                field_id = parameter["id"]
                                if parameter.get("label") == old_labels[field_id]:
                                    parameter["label"] = labels[field_id]
                                    changed = True
                    if (not video_frame_priority_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_FUNCTION["id"]):
                        item["description"] = VIDEO_FUNCTION["description"]
                        existing_parameters = {
                            parameter.get("id"): parameter
                            for parameter in item.get("parameters", [])
                            if isinstance(parameter, dict)
                        }
                        item["parameters"] = [
                            deepcopy(existing_parameters.get(parameter["id"], parameter))
                            for parameter in VIDEO_FUNCTION["parameters"]
                        ]
                        for parameter in item["parameters"]:
                            source_parameter = next(
                                source for source in VIDEO_FUNCTION["parameters"]
                                if source["id"] == parameter["id"]
                            )
                            if parameter["id"] in {"extraction_mode", "max_images_limit"}:
                                parameter.update(deepcopy(source_parameter))
                        changed = True
                    if isinstance(item, dict) and item.get("id") == REMOTE_STAR_INFERENCE_FUNCTION["id"]:
                        parameters = item.setdefault("parameters", [])
                        if not any(isinstance(p, dict) and p.get("id") == "detection_fps" for p in parameters):
                            field = next(p for p in REMOTE_STAR_INFERENCE_FUNCTION["parameters"] if p["id"] == "detection_fps")
                            index = next((i for i, p in enumerate(parameters) if isinstance(p, dict) and p.get("id") == "preview_fps"), len(parameters))
                            parameters.insert(index, deepcopy(field))
                            changed = True
                    if (not realtime_playback_marker.exists() and isinstance(item, dict)
                            and item.get("id") in {REMOTE_STAR_INFERENCE_FUNCTION["id"], LOCAL_PT_INFERENCE_FUNCTION["id"]}):
                        source = REMOTE_STAR_INFERENCE_FUNCTION if item.get("id") == REMOTE_STAR_INFERENCE_FUNCTION["id"] else LOCAL_PT_INFERENCE_FUNCTION
                        item["description"] = source["description"]
                        source_realtime = next(parameter for parameter in source["parameters"] if parameter["id"] == "realtime")
                        for parameter in item.get("parameters", []):
                            if isinstance(parameter, dict) and parameter.get("id") == "realtime":
                                parameter.update(deepcopy(source_realtime))
                                break
                        changed = True
                    if isinstance(item, dict) and item.get("id") == LOCAL_PT_INFERENCE_FUNCTION["id"]:
                        if item.get("name") in {"本地 PT 视频检测", "本地 PT / ONNX 视频检测"}:
                            item["name"] = LOCAL_PT_INFERENCE_FUNCTION["name"]
                            item["description"] = LOCAL_PT_INFERENCE_FUNCTION["description"]
                            changed = True
                        fields = item.get("pathFields", [])
                        if not local_images_marker.exists():
                            if not any(isinstance(field, dict) and field.get("id") == "image_folder" for field in fields):
                                item.setdefault("pathFields", []).append(deepcopy(LOCAL_PT_INFERENCE_FUNCTION["pathFields"][3]))
                            for field in fields:
                                if isinstance(field, dict) and field.get("id") == "output_folder" and field.get("label") == "保存带框视频的文件夹（可选）":
                                    field["label"] = LOCAL_PT_INFERENCE_FUNCTION["pathFields"][2]["label"]
                            parameters = item.setdefault("parameters", [])
                            for source in LOCAL_PT_INFERENCE_FUNCTION["parameters"][:2]:
                                if not any(isinstance(param, dict) and param.get("id") == source["id"] for param in parameters):
                                    parameters.append(deepcopy(source))
                            changed = True
                        for field in fields:
                            if isinstance(field, dict) and field.get("id") == "model_file" and field.get("label") == "本地 PT 权重文件":
                                field["label"] = LOCAL_PT_INFERENCE_FUNCTION["pathFields"][0]["label"]
                                changed = True
                        if any(isinstance(field, dict) and field.get("id") == "output_file" for field in fields):
                            item["pathFields"] = [deepcopy(LOCAL_PT_INFERENCE_FUNCTION["pathFields"][2]) if isinstance(field, dict) and field.get("id") == "output_file" else field for field in fields]
                            changed = True
                    if (not clip_migration_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_CLIP_FUNCTION["id"]):
                        item["description"] = VIDEO_CLIP_FUNCTION["description"]
                        source_encoding = next(
                            parameter for parameter in VIDEO_CLIP_FUNCTION["parameters"]
                            if parameter["id"] == "encoding_mode"
                        )
                        for parameter in item.get("parameters", []):
                            if isinstance(parameter, dict) and parameter.get("id") == "encoding_mode":
                                parameter.update(deepcopy(source_encoding))
                                changed = True
                    if (not clip_split_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_CLIP_FUNCTION["id"]):
                        item["description"] = VIDEO_CLIP_FUNCTION["description"]
                        parameters = item.setdefault("parameters", [])
                        if not any(isinstance(parameter, dict) and parameter.get("id") == "split_train_test" for parameter in parameters):
                            source_split = next(
                                parameter for parameter in VIDEO_CLIP_FUNCTION["parameters"]
                                if parameter["id"] == "split_train_test"
                            )
                            parameters.append(deepcopy(source_split))
                            changed = True
                    if isinstance(item, dict) and item.get("id") == VIDEO_CLIP_FUNCTION["id"]:
                        parameters = item.setdefault("parameters", [])
                        if item.get('description') == '按时间裁剪、拆分训练/测试集，或删除多个不要的时间段后自动拼接剩余内容；原视频不修改，默认不重新编码。':
                            item['description'] = VIDEO_CLIP_FUNCTION['description']
                            changed = True
                        for field_id in ('clip_save_mode', 'clip_exact_time'):
                            if not any(isinstance(parameter, dict) and parameter.get('id') == field_id for parameter in parameters):
                                parameters.append(deepcopy(next(parameter for parameter in VIDEO_CLIP_FUNCTION['parameters'] if parameter['id'] == field_id)))
                                changed = True
                        if not any(isinstance(parameter, dict) and parameter.get("id") == "clip_output_name" for parameter in parameters):
                            source_name = next(parameter for parameter in VIDEO_CLIP_FUNCTION["parameters"] if parameter["id"] == "clip_output_name")
                            parameters.insert(0, deepcopy(source_name))
                            changed = True
                    if (not clip_remove_marker.exists() and isinstance(item, dict)
                            and item.get("id") == VIDEO_CLIP_FUNCTION["id"]):
                        item["description"] = VIDEO_CLIP_FUNCTION["description"]
                        parameters = item.setdefault("parameters", [])
                        for source in VIDEO_CLIP_FUNCTION["parameters"]:
                            parameter = next((entry for entry in parameters if isinstance(entry, dict) and entry.get("id") == source["id"]), None)
                            if source["id"] == "clip_mode":
                                if parameter is None:
                                    parameters.insert(0, deepcopy(source))
                                else:
                                    parameter["options"] = deepcopy(source["options"])
                            elif source["id"] == "remove_time_ranges" and parameter is None:
                                parameters.insert(1, deepcopy(source))
                            elif source["id"] == "split_train_test" and parameter is not None:
                                parameter["visibleWhen"] = deepcopy(source["visibleWhen"])
                        changed = True
                    if (not calib_priority_marker.exists() and isinstance(item, dict)
                            and item.get("id") == CALIB_DATASET_FUNCTION["id"]):
                        item["description"] = CALIB_DATASET_FUNCTION["description"]
                        parameters = item.setdefault("parameters", [])
                        if not any(isinstance(parameter, dict) and parameter.get("id") == "negative_dirs" for parameter in parameters):
                            negative_parameter = next(
                                parameter for parameter in CALIB_DATASET_FUNCTION["parameters"]
                                if parameter["id"] == "negative_dirs"
                            )
                            insert_at = next(
                                (index + 1 for index, parameter in enumerate(parameters)
                                 if isinstance(parameter, dict) and parameter.get("id") == "annotation_dirs"),
                                len(parameters),
                            )
                            parameters.insert(insert_at, deepcopy(negative_parameter))
                        changed = True
                    if (not calib_local_marker.exists() and isinstance(item, dict)
                            and item.get("id") == CALIB_DATASET_FUNCTION["id"]):
                        item["description"] = CALIB_DATASET_FUNCTION["description"]
                        existing_parameters = {
                            parameter.get("id"): parameter
                            for parameter in item.get("parameters", [])
                            if isinstance(parameter, dict)
                        }
                        item["parameters"] = [
                            deepcopy(existing_parameters.get(parameter["id"], parameter))
                            for parameter in CALIB_DATASET_FUNCTION["parameters"]
                        ]
                        changed = True
                    if (not export_family_marker.exists() and isinstance(item, dict)
                            and item.get("id") == JETSON_ONNX_EXPORT_FUNCTION["id"]):
                        item["description"] = JETSON_ONNX_EXPORT_FUNCTION["description"]
                        existing_parameters = {
                            parameter.get("id"): parameter
                            for parameter in item.get("parameters", [])
                            if isinstance(parameter, dict)
                        }
                        item["parameters"] = [
                            deepcopy(existing_parameters.get(parameter["id"], parameter))
                            for parameter in JETSON_ONNX_EXPORT_FUNCTION["parameters"]
                        ]
                        changed = True
                    if (not export_batch_marker.exists() and isinstance(item, dict)
                            and item.get("id") == JETSON_ONNX_EXPORT_FUNCTION["id"]):
                        item["description"] = JETSON_ONNX_EXPORT_FUNCTION["description"]
                        existing_paths = {
                            path.get("id"): path
                            for path in item.get("pathFields", [])
                            if isinstance(path, dict)
                        }
                        item["pathFields"] = [
                            deepcopy(existing_paths.get(path["id"], path))
                            for path in JETSON_ONNX_EXPORT_FUNCTION["pathFields"]
                        ]
                        for path in item["pathFields"]:
                            source_path = next(
                                source for source in JETSON_ONNX_EXPORT_FUNCTION["pathFields"]
                                if source["id"] == path["id"]
                            )
                            path.update(deepcopy(source_path))
                        for parameter in item.get("parameters", []):
                            if isinstance(parameter, dict) and parameter.get("id") == "output_filename":
                                source_parameter = next(
                                    source for source in JETSON_ONNX_EXPORT_FUNCTION["parameters"]
                                    if source["id"] == "output_filename"
                                )
                                parameter["label"] = source_parameter["label"]
                                break
                        changed = True
                    if (not remote_trt_launcher_marker.exists() and isinstance(item, dict)
                            and item.get("id") == REMOTE_TENSORRT_BUILD_FUNCTION["id"]):
                        item["description"] = REMOTE_TENSORRT_BUILD_FUNCTION["description"]
                        changed = True
                    if (not remote_trt_batch_marker.exists() and isinstance(item, dict)
                            and item.get("id") == REMOTE_TENSORRT_BUILD_FUNCTION["id"]):
                        item["description"] = REMOTE_TENSORRT_BUILD_FUNCTION["description"]
                        parameters = item.setdefault("parameters", [])
                        if not any(isinstance(parameter, dict) and parameter.get("id") == "onnx_files" for parameter in parameters):
                            parameters.insert(0, deepcopy(REMOTE_TENSORRT_BUILD_FUNCTION["parameters"][0]))
                        changed = True
                    if (not remote_trt_paths_marker.exists() and isinstance(item, dict)
                            and item.get("id") == REMOTE_TENSORRT_BUILD_FUNCTION["id"]):
                        parameters = item.setdefault("parameters", [])
                        current_parameter = next(
                            (parameter for parameter in parameters
                             if isinstance(parameter, dict) and parameter.get("id") == "onnx_files"),
                            None,
                        )
                        if current_parameter is None:
                            parameters.insert(0, deepcopy(REMOTE_TENSORRT_BUILD_FUNCTION["parameters"][0]))
                        else:
                            current_parameter.update(deepcopy(REMOTE_TENSORRT_BUILD_FUNCTION["parameters"][0]))
                        changed = True
                    if (not remote_trt_remote_paths_marker.exists() and isinstance(item, dict)
                            and item.get("id") == REMOTE_TENSORRT_BUILD_FUNCTION["id"]):
                        item["description"] = REMOTE_TENSORRT_BUILD_FUNCTION["description"]
                        item["pathFields"] = []
                        item["parameters"] = deepcopy(REMOTE_TENSORRT_BUILD_FUNCTION["parameters"])
                        changed = True
                    if (not local_star_only_marker.exists() and isinstance(item, dict)
                            and item.get("id") == LOCAL_STAR_PACKAGE_FUNCTION["id"]):
                        item["name"] = LOCAL_STAR_PACKAGE_FUNCTION["name"]
                        item["description"] = LOCAL_STAR_PACKAGE_FUNCTION["description"]
                        item["pathFields"] = deepcopy(LOCAL_STAR_PACKAGE_FUNCTION["pathFields"])
                        existing_parameters = {
                            parameter.get("id"): parameter
                            for parameter in item.get("parameters", [])
                            if isinstance(parameter, dict)
                        }
                        item["parameters"] = [
                            deepcopy(existing_parameters.get(parameter["id"], parameter))
                            for parameter in LOCAL_STAR_PACKAGE_FUNCTION["parameters"]
                        ]
                        changed = True
                if not cycle_dedup_marker.exists():
                    for item in content:
                        if not isinstance(item, dict) or item.get("handlerId") != "image.smart_dedup":
                            continue
                        item["description"] = IMAGE_DEDUP_FUNCTION["description"]
                        for key in ("pathFields", "parameters"):
                            existing = {field.get("id"): field for field in item.get(key, []) if isinstance(field, dict)}
                            for field in IMAGE_DEDUP_FUNCTION[key]:
                                if field["id"] not in existing:
                                    item.setdefault(key, []).append(deepcopy(field))
                                elif field["id"] in {"operation", "dedup_strength", "minimum_keep_ratio", "max_consecutive_skips"}:
                                    existing[field["id"]]["options"] = deepcopy(field["options"])
                                    existing[field["id"]]["visibleWhen"] = deepcopy(field.get("visibleWhen"))
                        changed = True
                if changed:
                    temporary = self.storage_file.with_suffix(".tmp")
                    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(self.storage_file)
                if not migration_marker.exists():
                    migration_marker.write_text("migrated\n", encoding="utf-8")
                if not clip_migration_marker.exists():
                    clip_migration_marker.write_text("migrated\n", encoding="utf-8")
                if not clip_split_marker.exists():
                    clip_split_marker.write_text("migrated\n", encoding="utf-8")
                if not clip_remove_marker.exists():
                    clip_remove_marker.write_text("migrated\n", encoding="utf-8")
                if not dedup_migration_marker.exists():
                    dedup_migration_marker.write_text("migrated\n", encoding="utf-8")
                if not cycle_dedup_marker.exists():
                    cycle_dedup_marker.write_text("migrated\n", encoding="utf-8")
                quality_dedup_marker.write_text("migrated\n", encoding="utf-8")
                recursive_dedup_marker.write_text("migrated\n", encoding="utf-8")
                process_dedup_marker.write_text("migrated\n", encoding="utf-8")
                if not quant_migration_marker.exists():
                    quant_migration_marker.write_text("migrated\n", encoding="utf-8")
                if not export_migration_marker.exists():
                    export_migration_marker.write_text("migrated\n", encoding="utf-8")
                if not export_family_marker.exists():
                    export_family_marker.write_text("migrated\n", encoding="utf-8")
                if not export_batch_marker.exists():
                    export_batch_marker.write_text("migrated\n", encoding="utf-8")
                if not calib_priority_marker.exists():
                    calib_priority_marker.write_text("migrated\n", encoding="utf-8")
                if not calib_local_marker.exists():
                    calib_local_marker.write_text("migrated\n", encoding="utf-8")
                if not remote_trt_marker.exists():
                    remote_trt_marker.write_text("migrated\n", encoding="utf-8")
                if not remote_trt_launcher_marker.exists():
                    remote_trt_launcher_marker.write_text("migrated\n", encoding="utf-8")
                if not remote_trt_batch_marker.exists():
                    remote_trt_batch_marker.write_text("migrated\n", encoding="utf-8")
                if not remote_trt_paths_marker.exists():
                    remote_trt_paths_marker.write_text("migrated\n", encoding="utf-8")
                if not remote_trt_remote_paths_marker.exists():
                    remote_trt_remote_paths_marker.write_text("migrated\n", encoding="utf-8")
                if not local_star_package_marker.exists():
                    local_star_package_marker.write_text("migrated\n", encoding="utf-8")
                if not local_star_only_marker.exists():
                    local_star_only_marker.write_text("migrated\n", encoding="utf-8")
                if not video_frame_naming_marker.exists():
                    video_frame_naming_marker.write_text("migrated\n", encoding="utf-8")
                if not video_filename_fallback_marker.exists():
                    video_filename_fallback_marker.write_text("migrated\n", encoding="utf-8")
                if not video_frame_priority_marker.exists():
                    video_frame_priority_marker.write_text("migrated\n", encoding="utf-8")
                if not video_folder_recursive_marker.exists():
                    video_folder_recursive_marker.write_text("migrated\n", encoding="utf-8")
                if not video_multi_folders_marker.exists():
                    video_multi_folders_marker.write_text("migrated\n", encoding="utf-8")
                video_output_layout_marker.write_text("migrated\n", encoding="utf-8")
                if not video_parallel_marker.exists():
                    video_parallel_marker.write_text("migrated\n", encoding="utf-8")
                if not realtime_playback_marker.exists():
                    realtime_playback_marker.write_text("migrated\n", encoding="utf-8")
                if not yolo_dataset_timestamp_marker.exists():
                    yolo_dataset_timestamp_marker.write_text("migrated\n", encoding="utf-8")
                if not yolo_training_function_marker.exists():
                    yolo_training_function_marker.write_text("migrated\n", encoding="utf-8")
                if not yolo_training_remote_source_marker.exists():
                    yolo_training_remote_source_marker.write_text("migrated\n", encoding="utf-8")
                if not local_images_marker.exists():
                    local_images_marker.write_text("migrated\n", encoding="utf-8")
                if not negative_sample_marker.exists():
                    negative_sample_marker.write_text("migrated\n", encoding="utf-8")
                if not negative_separate_marker.exists():
                    negative_separate_marker.write_text("migrated\n", encoding="utf-8")
                items = [self._normalize(item) for item in content if isinstance(item, dict)]
                annotation_marker.write_text("migrated\n", encoding="utf-8")
                annotation_star_marker.write_text("migrated\n", encoding="utf-8")
                precision_options_marker.write_text("migrated\n", encoding="utf-8")
                return items
        except (OSError, json.JSONDecodeError):
            pass
        return [self._normalize(deepcopy(item)) for item in DEFAULT_FUNCTIONS]

    def _save(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_file)
        self.storage_file.with_name(self.storage_file.name + ".auto-annotation-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".local-pt-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".local-images-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-clip-copy-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-clip-train-test-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-clip-remove-v3").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".yolo-training-function-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".yolo-training-remote-source-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".negative-sample-extractor-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".negative-sample-separate-folders-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".image-dedup-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".image-cycle-dedup-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".docker-onnx-quant-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".jetson-onnx-export-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".jetson-onnx-batch-v3").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".calib-priority-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-build-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-launcher-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-batch-v3").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-paths-v4").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".remote-tensorrt-remote-paths-v5").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".local-star-package-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".local-star-package-local-v3").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-frame-naming-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-frame-filename-fallback-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-folder-recursive-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-multi-folders-v2").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".video-frame-parallel-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".realtime-playback-v1").write_text("migrated\n", encoding="utf-8")
        self.storage_file.with_name(self.storage_file.name + ".yolo-dataset-timestamp-v1").write_text("migrated\n", encoding="utf-8")

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
