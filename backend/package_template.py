"""Portable default template; never read the developer's trt.toml."""

DEFAULT_TRT_TEMPLATE = '''title = "model"

[hardware]
name = "TensorRT"
architecture = "Turing"
driver = "CUDA12.2"

[model]
name = "yolo26"
category = "generic"
version = "1"
model_file = "model.plan"
labels = []
precision = 1

[[model.input_tensor]]
n = 1
c = 3
h = 640
w = 640
color_fmt = 2
'''
