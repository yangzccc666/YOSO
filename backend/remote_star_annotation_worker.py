"""Uploaded to the inference box; stdout is a versioned JSON-lines protocol."""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path


def emit(payload, stream=None):
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), file=stream or sys.stdout, flush=True)


def main(protocol=None):
    root = Path(__file__).resolve().parent
    (root / 'pid').write_text(str(os.getpid()), encoding='ascii')
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    runner = None
    try:
        # Runtime diagnostics must never contaminate the JSON response stream.
        with contextlib.redirect_stdout(sys.stderr):
            import cv2
            import numpy as np
            from bundled_run_star_video import TensorRTRunner, read_engine_bytes, preprocess, decode_yolo26, pair_yolo26_outputs
            engine, _ = read_engine_bytes(root / 'model.star')
            runner = TensorRTRunner(engine, cuda_backend=config['cuda_backend'])
        emit({'type': 'ready', 'version': 1, 'input_shape': list(runner.input_shape)}, protocol)
        validated = False
        for line in sys.stdin:
            request = json.loads(line)
            identity = request['id']
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    frame = cv2.imdecode(np.fromfile(str(root / 'image.bin'), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError('无法解码图片，请检查文件内容。')
            except Exception as exc:
                emit({'type': 'result', 'id': identity, 'error': str(exc)}, protocol)
                continue
            with contextlib.redirect_stdout(sys.stderr):
                tensor, gain, pad = preprocess(frame, runner.input_shape)
                outputs = runner.infer(tensor)
                if not validated:
                    pairs = pair_yolo26_outputs(outputs, num_classes=config['num_classes'])
                    if not pairs or any(cls.shape[-1] != config['num_classes'] for _, cls in pairs):
                        raise ValueError('STAR 实际类别数量与填写的类别名称数量不一致，请核对。')
                    validated = True
                boxes, scores, ids = decode_yolo26(
                    outputs, runner.input_shape[2:4], frame.shape[:2], gain, pad,
                    config['conf'], config['iou'], 300, num_classes=config['num_classes'])
            emit({'type': 'result', 'id': identity, 'shape': list(frame.shape[:2]),
                      'detections': [{'class_id': int(cls), 'box': box.tolist(), 'score': float(score)}
                                     for box, score, cls in zip(boxes, scores, ids)]}, protocol)
    finally:
        if runner is not None:
            with contextlib.redirect_stdout(sys.stderr):
                runner.close()


if __name__ == '__main__':
    # TensorRT/CUDA also print from C/C++ directly to fd 1. Route fd 1 to
    # stderr for the entire lifetime (including native destructors); reserve a
    # duplicated descriptor exclusively for protocol messages.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        main(protocol)
    except Exception as exc:
        emit({'type': 'fatal', 'error': str(exc)}, protocol)
        sys.exit(1)
