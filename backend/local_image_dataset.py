"""Image-folder input and non-overwriting output for local YOLO inference."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Event

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_image_paths(image_text: str, output_text: str) -> tuple[Path, Path]:
    images = Path(image_text.strip()).expanduser()
    if not image_text.strip() or not images.is_dir():
        raise ValueError("请选择存在的图片文件夹。")
    if not output_text.strip():
        raise ValueError("图片文件夹检测必须指定检测结果输出文件夹。")
    images, output = images.resolve(), Path(output_text.strip()).expanduser().resolve()
    if images == output or output in images.parents:
        raise ValueError("输出目录不能与图片目录相同，也不能是它的上级目录；请选择独立目录或专用输出子目录，避免修改原图。")
    if output.exists() and not output.is_dir():
        raise ValueError("检测结果输出路径已存在，但不是文件夹。")
    return images, output


def collect_images(root: Path, output: Path, recursive: bool, stop_event: Event) -> list[Path]:
    files: list[Path] = []
    def on_error(error: OSError) -> None:
        raise ValueError(f"无法读取图片目录：{error}") from error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=on_error):
        if stop_event.is_set():
            break
        parent = Path(folder)
        # Do not recurse into previous outputs or symbolic-link directories.
        dirs[:] = sorted(name for name in dirs if not (parent / name).is_symlink()
                         and (parent / name).resolve() != output) if recursive else []
        files.extend(parent / name for name in names if Path(name).suffix.lower() in IMAGE_SUFFIXES
                     and (parent / name).is_file())
    return sorted(files, key=lambda path: (str(path.relative_to(root)).casefold(), str(path)))


def read_image(path: Path) -> np.ndarray:
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("无法解码图片，文件可能已损坏或格式不受支持。")
    return frame


def save_image(target: Path, frame: np.ndarray, output_root: Path) -> Path:
    """Encode once and exclusively create a new name; never overwrite any file."""
    ok, encoded = cv2.imencode(target.suffix.lower(), frame)
    if not ok:
        raise RuntimeError("带框图片编码失败。")
    if not target.parent.resolve().is_relative_to(output_root.resolve()):
        raise RuntimeError("输出子目录包含指向输出目录外的链接，已停止以保护其他文件。")
    target.parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = target if index == 0 else target.with_name(f"{target.stem}_{index}{target.suffix}")
        try:
            stream = candidate.open("xb")
        except FileExistsError:
            index += 1
            continue
        try:
            with stream:
                stream.write(encoded.tobytes())
        except BaseException:
            # Only our own newly created, incomplete result is removed.
            candidate.unlink(missing_ok=True)
            raise
        return candidate
