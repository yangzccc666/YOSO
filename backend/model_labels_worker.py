"""Read class names from trusted PT checkpoints in the configured AI environment."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .local_pt_worker import normalize_names, send
else:
    from local_pt_worker import normalize_names, send


def checkpoint_names(checkpoint: Any) -> list[str]:
    """YOLOv5 checkpoints may carry names on EMA, model/module, or the dict."""
    candidates = [checkpoint]
    if isinstance(checkpoint, dict):
        candidates = [checkpoint.get("ema"), checkpoint.get("model"), checkpoint]
    for candidate in candidates:
        if candidate is None:
            continue
        for owner in (candidate, getattr(candidate, "module", None)):
            names = owner.get("names") if isinstance(owner, dict) else getattr(owner, "names", None)
            if names is not None:
                return normalize_names(names)
    raise ValueError("PT 权重中没有保存 names 类别名称，不能从权重数值推断语义标签。")


def read_pt_labels(path: str, family: str, repository: str = "") -> dict[str, Any]:
    if family == "YOLOv5":
        repo = Path(repository).expanduser().resolve()
        if not repository or not (repo / "export.py").is_file() or not (repo / "models").is_dir():
            raise ValueError("读取 YOLOv5 类别需要与权重匹配的源码仓库（包含 export.py 和 models）。")
        sys.path.insert(0, str(repo))
        import torch
        # PT unpickling requires a trusted file and its original model classes.
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            checkpoint = torch.load(path, map_location="cpu")
        labels = checkpoint_names(checkpoint)
        task = "detect"
    elif family == "YOLO26":
        from ultralytics import YOLO
        model = YOLO(path)
        labels = normalize_names(model.names)
        task = model.task
    else:
        raise ValueError("请选择 YOLO26 或 YOLOv5 模型版本。")
    return {"labels": labels, "count": len(labels), "found": True, "modelFamily": family,
            "task": task, "source": "PT 权重内嵌 names",
            "message": f"已从 PT 权重读取 {len(labels)} 个类别，按类别 ID 顺序排列。"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--family", choices=("YOLO26", "YOLOv5"), default="YOLO26")
    parser.add_argument("--repository", default="")
    args = parser.parse_args()
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        metadata = read_pt_labels(args.weights, args.family, args.repository)
        send(output, {"type": "metadata", **metadata})
        return 0
    except Exception as exc:
        send(output, {"type": "error", "message": str(exc) or exc.__class__.__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
