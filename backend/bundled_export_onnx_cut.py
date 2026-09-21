"""YOLO26 raw-head ONNX export used by Jetson/TensorRT inference."""

from __future__ import annotations

import argparse
import shutil
import types
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO


def get_custom_forward(task_type: str, is_rknn_mode: bool):
    def forward2(self, x):
        y = []
        proto = None
        if task_type == "seg":
            box_head, cls_head, mask_head = self.one2one_cv2, self.one2one_cv3, self.one2one_cv4
            proto = self.proto(x)
        elif task_type == "obb":
            box_head, cls_head, angle_head = self.one2one_cv2, self.one2one_cv3, self.one2one_cv4
        elif task_type == "pose":
            box_head, cls_head, pose_head, kpts_head = (
                self.one2one_cv2, self.one2one_cv3, self.one2one_cv4, self.one2one_cv4_kpts
            )
        else:
            box_head, cls_head = self.one2one_cv2, self.one2one_cv3

        for index in range(self.nl):
            if task_type == "seg":
                outputs = [box_head[index](x[index]), cls_head[index](x[index]), mask_head[index](x[index])]
            elif task_type == "obb":
                outputs = [box_head[index](x[index]), cls_head[index](x[index]), angle_head[index](x[index])]
            elif task_type == "pose":
                outputs = [
                    box_head[index](x[index]), cls_head[index](x[index]),
                    kpts_head[index](pose_head[index](x[index])),
                ]
            else:
                outputs = [box_head[index](x[index]), cls_head[index](x[index])]
            if not is_rknn_mode:
                outputs = [output.permute(0, 2, 3, 1) for output in outputs]
            if task_type == "generic":
                y.extend([outputs[0], torch.sigmoid(outputs[1])])
            else:
                outputs[1] = torch.sigmoid(outputs[1])
                y.extend(outputs)
        return (y, proto) if task_type == "seg" else y

    return forward2


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26 Jetson-compatible ONNX export")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--task", choices=("detect", "segment", "obb", "pose", "classify"), default="detect")
    parser.add_argument("--platform", choices=("jetson", "rknn"), default="jetson")
    args = parser.parse_args()

    print(f"正在加载模型：{args.weights}", flush=True)
    model = YOLO(args.weights)
    task_type = {
        "detect": "generic", "segment": "seg", "obb": "obb",
        "pose": "pose", "classify": "cls",
    }[args.task]
    if task_type != "cls":
        head = model.model.model[-1]
        required = ("one2one_cv2", "one2one_cv3", "nl")
        missing = [name for name in required if not hasattr(head, name)]
        if missing:
            raise RuntimeError(
                "当前权重的检测头不兼容 YOLO26 切头导出，缺少：" + "、".join(missing)
            )
        head.forward = types.MethodType(
            get_custom_forward(task_type=task_type, is_rknn_mode=args.platform == "rknn"), head
        )

    print(
        f"导出配置：task={args.task}，platform={args.platform}，imgsz={args.imgsz}，"
        f"opset={args.opset}，batch=1，dynamic=False，FP32，NMS=False",
        flush=True,
    )
    exported = Path(model.export(
        format="onnx",
        imgsz=(args.imgsz, args.imgsz),
        keras=False,
        optimize=False,
        half=False,
        int8=False,
        dynamic=False,
        simplify=True,
        opset=args.opset,
        nms=False,
        batch=1,
        device="cpu",
    )).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported != output:
        shutil.move(str(exported), str(output))
    model_onnx = onnx.load(str(output))
    onnx.checker.check_model(model_onnx)
    outputs = [value.name for value in model_onnx.graph.output]
    print(f"ONNX 检查通过，输出节点 {len(outputs)} 个：{outputs}", flush=True)
    print(f"导出完成：{output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
