#!/usr/bin/env python3
"""Run a YOLO26 TensorRT .star model on video and display detections.

The StarLink .star file used by this workspace is a FlatBuffers container whose
model_file field stores the TensorRT serialized engine as a byte vector.  This
script locates that vector, deserializes the engine in memory, and implements
the six-output YOLO26 decoder used by export_onnx_cut.py.

Runtime dependencies on the NVIDIA machine:
    python3 -m pip install numpy opencv-python cuda-python
    # TensorRT Python must match the TensorRT version used to build the engine.

Examples:
    python3 run_star_video.py --model model.star --source test.mp4 \
        --labels forgings_ct --conf 0.10 --no-show

    python3 run_star_video.py --model model.star --source 0 \
        --labels forgings_ct --conf 0.10

    python3 run_star_video.py --model model.star --source rtsp://... \
        --labels forgings_ct --save result.mp4

    python3 run_star_video.py --model model.star \
        --extract-only extracted.plan
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


TRT_ENGINE_MAGIC = b"ftrt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display real-time YOLO26 detections from a TensorRT .star model."
    )
    parser.add_argument("--model", required=True, help="Path to .star, .plan, or .engine file")
    parser.add_argument(
        "--source",
        default=None,
        help="Video path, camera index such as 0, or RTSP/HTTP URL",
    )
    parser.add_argument(
        "--labels", "--label", default="class0", help="Comma-separated class names"
    )
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum boxes per frame")
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output video file or directory (directory creates result.mp4)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an OpenCV window")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Keep the original video timeline; skip late frames when inference is slower than the source FPS",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Resize only the display window to this width; 0 keeps original size",
    )
    parser.add_argument(
        "--extract-only",
        default=None,
        metavar="OUTPUT.PLAN",
        help="Extract the embedded TensorRT engine and exit",
    )
    parser.add_argument(
        "--cuda-backend",
        choices=("auto", "cuda-driver", "cuda-python", "pycuda"),
        default="auto",
        help="CUDA memory backend; auto prefers the driver API on Jetson",
    )
    return parser.parse_args()


def read_engine_bytes(model_path: str | Path) -> tuple[bytes, int]:
    """Return serialized TensorRT engine bytes and their offset in the source file."""
    path = Path(model_path)
    data = path.read_bytes()
    if data.startswith(TRT_ENGINE_MAGIC):
        return data, 0

    # In this project's FlatBuffers .star format the engine byte-vector length
    # is the uint32 immediately before the TensorRT engine bytes. Search only
    # near the beginning so an accidental "ftrt" inside an engine is ignored.
    search_end = min(len(data), 1024 * 1024)
    cursor = 0
    while True:
        offset = data.find(TRT_ENGINE_MAGIC, cursor, search_end)
        if offset < 0:
            break
        if offset >= 4:
            engine_size = struct.unpack_from("<I", data, offset - 4)[0]
            end = offset + engine_size
            if engine_size >= 1024 and end <= len(data):
                return data[offset:end], offset
        cursor = offset + 1

    raise ValueError(
        f"No embedded TensorRT engine was found in {path}. "
        "Expected a FlatBuffers byte vector followed by the 'ftrt' engine magic."
    )


def import_cuda_runtime():
    """Import either the legacy cuda-python API or the current cuda.bindings API."""
    try:
        from cuda import cudart  # type: ignore

        return cudart
    except (ImportError, AttributeError):
        try:
            from cuda.bindings import runtime as cudart  # type: ignore

            return cudart
        except ImportError as exc:
            raise RuntimeError(
                "cuda-python is not installed. Run: python3 -m pip install cuda-python"
            ) from exc


def import_cuda_driver():
    """Import either the current or legacy cuda-python driver API."""
    try:
        from cuda.bindings import driver as cuda_driver  # type: ignore

        return cuda_driver
    except (ImportError, AttributeError):
        try:
            from cuda import cuda as cuda_driver  # type: ignore

            return cuda_driver
        except ImportError as exc:
            raise RuntimeError(
                "cuda-python driver bindings are not installed in this Python environment"
            ) from exc


def cuda_check(result, operation: str):
    """Check cuda-python's tuple return and return payload values."""
    if not isinstance(result, tuple):
        return result
    error = result[0]
    code = int(error.value) if hasattr(error, "value") else int(error)
    if code != 0:
        raise RuntimeError(f"CUDA call failed during {operation}: error code {code}")
    payload = result[1:]
    if not payload:
        return None
    return payload[0] if len(payload) == 1 else payload


class CtypesCudaDriverBackend:
    """CUDA driver backend using the device's libcuda without Python packages."""

    name = "cuda-driver-ctypes"

    def __init__(self):
        library_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.cuda = ctypes.CDLL(library_name)
        except OSError as exc:
            raise RuntimeError(f"Could not load {library_name}: {exc}") from exc

        self._cu_init = self._bind(("cuInit",), [ctypes.c_uint])
        self._cu_device_get = self._bind(
            ("cuDeviceGet",), [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        )
        self._cu_primary_context_retain = self._bind(
            ("cuDevicePrimaryCtxRetain",),
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        )
        self._cu_context_set_current = self._bind(
            ("cuCtxSetCurrent",), [ctypes.c_void_p]
        )
        self._cu_stream_create = self._bind(
            ("cuStreamCreate",), [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        )
        self._cu_stream_destroy = self._bind(
            ("cuStreamDestroy_v2", "cuStreamDestroy"), [ctypes.c_void_p]
        )
        self._cu_stream_synchronize = self._bind(
            ("cuStreamSynchronize",), [ctypes.c_void_p]
        )
        self._cu_mem_alloc = self._bind(
            ("cuMemAlloc_v2", "cuMemAlloc"),
            [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
        )
        self._cu_mem_free = self._bind(
            ("cuMemFree_v2", "cuMemFree"), [ctypes.c_uint64]
        )
        self._cu_copy_h2d = self._bind(
            ("cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync"),
            [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
        )
        self._cu_copy_d2h = self._bind(
            ("cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync"),
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p],
        )

        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self._check(self._cu_init(0), "cuInit")
        self._check(self._cu_device_get(ctypes.byref(self.device), 0), "cuDeviceGet")
        self._check(
            self._cu_primary_context_retain(ctypes.byref(self.context), self.device.value),
            "cuDevicePrimaryCtxRetain",
        )
        self._check(self._cu_context_set_current(self.context), "cuCtxSetCurrent")
        self._check(self._cu_stream_create(ctypes.byref(self.stream), 0), "cuStreamCreate")

    def _bind(self, names, argument_types):
        for name in names:
            function = getattr(self.cuda, name, None)
            if function is not None:
                function.argtypes = argument_types
                function.restype = ctypes.c_int
                return function
        raise RuntimeError(f"CUDA driver symbol is missing: {' or '.join(names)}")

    def _error_details(self, result: int) -> str:
        for function_name in ("cuGetErrorString", "cuGetErrorName"):
            function = getattr(self.cuda, function_name, None)
            if function is None:
                continue
            value = ctypes.c_char_p()
            function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            function.restype = ctypes.c_int
            if function(result, ctypes.byref(value)) == 0 and value.value:
                return value.value.decode("utf-8", errors="replace")
        return f"error code {result}"

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(
                f"CUDA driver call failed during {operation}: {self._error_details(result)}"
            )

    @property
    def stream_handle(self) -> int:
        return int(self.stream.value or 0)

    @staticmethod
    def pointer(allocation) -> int:
        return int(allocation)

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(self._cu_mem_alloc(ctypes.byref(pointer), size), "cuMemAlloc")
        return int(pointer.value)

    def free(self, allocation) -> None:
        self._check(self._cu_mem_free(int(allocation)), "cuMemFree")

    def copy_h2d(self, allocation, host: np.ndarray) -> None:
        self._check(
            self._cu_copy_h2d(
                int(allocation), ctypes.c_void_p(host.ctypes.data), host.nbytes, self.stream
            ),
            "cuMemcpyHtoDAsync",
        )

    def copy_d2h(self, host: np.ndarray, allocation) -> None:
        self._check(
            self._cu_copy_d2h(
                ctypes.c_void_p(host.ctypes.data), int(allocation), host.nbytes, self.stream
            ),
            "cuMemcpyDtoHAsync",
        )

    def synchronize(self) -> None:
        self._check(self._cu_stream_synchronize(self.stream), "cuStreamSynchronize")

    def close(self) -> None:
        if self.stream.value:
            self._check(self._cu_stream_destroy(self.stream), "cuStreamDestroy")
            self.stream = ctypes.c_void_p()


class CudaPythonBackend:
    """Device memory and stream operations using cuda-python."""

    name = "cuda-python"

    def __init__(self):
        self.cudart = import_cuda_runtime()
        try:
            self.stream = cuda_check(self.cudart.cudaStreamCreate(), "cudaStreamCreate")
        except Exception as exc:
            details = self.version_details()
            raise RuntimeError(f"{exc}. {details}") from exc

    def version_details(self) -> str:
        values = []
        for function_name, label in (
            ("cudaDriverGetVersion", "driver"),
            ("cudaRuntimeGetVersion", "runtime"),
        ):
            function = getattr(self.cudart, function_name, None)
            if function is None:
                continue
            try:
                result = function()
                error = result[0]
                code = int(error.value) if hasattr(error, "value") else int(error)
                value = result[1] if len(result) > 1 else None
                values.append(f"{label}={value if code == 0 else 'unavailable'}")
            except Exception:
                pass
        return "CUDA versions: " + ", ".join(values) if values else "CUDA versions unavailable"

    @property
    def stream_handle(self) -> int:
        return int(self.stream)

    @staticmethod
    def pointer(allocation) -> int:
        return int(allocation)

    def malloc(self, size: int):
        return cuda_check(self.cudart.cudaMalloc(size), "cudaMalloc")

    def free(self, allocation) -> None:
        cuda_check(self.cudart.cudaFree(allocation), "cudaFree")

    def copy_h2d(self, allocation, host: np.ndarray) -> None:
        cuda_check(
            self.cudart.cudaMemcpyAsync(
                allocation,
                host.ctypes.data,
                host.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def copy_d2h(self, host: np.ndarray, allocation) -> None:
        cuda_check(
            self.cudart.cudaMemcpyAsync(
                host.ctypes.data,
                allocation,
                host.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )

    def synchronize(self) -> None:
        cuda_check(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def close(self) -> None:
        if getattr(self, "stream", None) is not None:
            cuda_check(self.cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            self.stream = None


class CudaDriverBackend:
    """CUDA driver-API backend that does not load the CUDA Runtime library.

    This is useful on Jetson when a pip environment accidentally contains a
    CUDA Runtime newer than the BSP driver (for example runtime 13.3 with a
    driver supporting CUDA 12.6). The stable driver API can still use the
    installed Jetson driver directly.
    """

    name = "cuda-driver"

    def __init__(self):
        self.cuda = import_cuda_driver()
        self.stream = None
        self.device = None
        self.context = None
        cuda_check(self.cuda.cuInit(0), "cuInit")
        self.device = cuda_check(self.cuda.cuDeviceGet(0), "cuDeviceGet")
        self.context = cuda_check(
            self.cuda.cuDevicePrimaryCtxRetain(self.device),
            "cuDevicePrimaryCtxRetain",
        )
        cuda_check(self.cuda.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")
        self.stream = cuda_check(self.cuda.cuStreamCreate(0), "cuStreamCreate")

    @property
    def stream_handle(self) -> int:
        return int(self.stream)

    @staticmethod
    def pointer(allocation) -> int:
        return int(allocation)

    def malloc(self, size: int):
        return cuda_check(self.cuda.cuMemAlloc(size), "cuMemAlloc")

    def free(self, allocation) -> None:
        cuda_check(self.cuda.cuMemFree(allocation), "cuMemFree")

    def copy_h2d(self, allocation, host: np.ndarray) -> None:
        cuda_check(
            self.cuda.cuMemcpyHtoDAsync(
                allocation, host.ctypes.data, host.nbytes, self.stream
            ),
            "cuMemcpyHtoDAsync",
        )

    def copy_d2h(self, host: np.ndarray, allocation) -> None:
        cuda_check(
            self.cuda.cuMemcpyDtoHAsync(
                host.ctypes.data, allocation, host.nbytes, self.stream
            ),
            "cuMemcpyDtoHAsync",
        )

    def synchronize(self) -> None:
        cuda_check(self.cuda.cuStreamSynchronize(self.stream), "cuStreamSynchronize")

    def close(self) -> None:
        if self.stream is not None:
            cuda_check(self.cuda.cuStreamDestroy(self.stream), "cuStreamDestroy")
            self.stream = None
        # Keep the retained primary context alive until process shutdown. It is
        # also owned by TensorRT and releasing it here could invalidate TensorRT
        # objects during Python interpreter teardown.


class PyCudaBackend:
    """Device memory and stream operations using PyCUDA's driver API."""

    name = "pycuda"

    def __init__(self):
        try:
            import pycuda.autoinit  # noqa: F401  # type: ignore
            import pycuda.driver as cuda  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyCUDA is not installed. On Jetson run: sudo apt install python3-pycuda"
            ) from exc
        self.cuda = cuda
        self.stream = cuda.Stream()

    @property
    def stream_handle(self) -> int:
        return int(self.stream.handle)

    @staticmethod
    def pointer(allocation) -> int:
        return int(allocation)

    def malloc(self, size: int):
        return self.cuda.mem_alloc(size)

    @staticmethod
    def free(allocation) -> None:
        allocation.free()

    def copy_h2d(self, allocation, host: np.ndarray) -> None:
        self.cuda.memcpy_htod_async(allocation, host, self.stream)

    def copy_d2h(self, host: np.ndarray, allocation) -> None:
        self.cuda.memcpy_dtoh_async(host, allocation, self.stream)

    def synchronize(self) -> None:
        self.stream.synchronize()

    def close(self) -> None:
        self.stream = None


def create_cuda_backend(preference: str):
    """Create a CUDA backend and automatically fall back on Jetson."""
    choices = {
        "cuda-python": (CudaPythonBackend,),
        "cuda-driver": (CtypesCudaDriverBackend, CudaDriverBackend),
        "pycuda": (PyCudaBackend,),
        "auto": (
            CtypesCudaDriverBackend,
            CudaDriverBackend,
            CudaPythonBackend,
            PyCudaBackend,
        ),
    }[preference]
    errors = []
    for backend_type in choices:
        try:
            backend = backend_type()
            print(f"CUDA backend: {backend.name}")
            return backend
        except Exception as exc:
            errors.append(f"{backend_type.name}: {exc}")
            if preference == "auto":
                print(f"WARNING: {errors[-1]}", file=sys.stderr)
    raise RuntimeError("No usable CUDA backend. " + " | ".join(errors))


class TensorRTRunner:
    """Small TensorRT 8/10 runner backed by cuda-python device allocations."""

    def __init__(self, engine_bytes: bytes, cuda_backend: str = "auto"):
        # Initialize cleanup attributes before any operation that can fail.
        self.device_buffers: dict[str, object] = {}
        self.host_buffers: dict[str, np.ndarray] = {}
        self.backend = None
        self.runtime = None
        self.engine = None
        self.context = None
        try:
            import tensorrt as trt  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python is not installed. Install the Python package that "
                "matches the TensorRT runtime on this NVIDIA machine."
            ) from exc

        self.trt = trt
        # PyCUDA must make its CUDA context current before TensorRT creates the
        # runtime/engine. This also makes partial-initialization cleanup safe.
        self.backend = create_cuda_backend(cuda_backend)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "TensorRT could not deserialize this engine. The usual cause is a "
                "GPU architecture, TensorRT, CUDA, or JetPack version mismatch."
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        self.is_trt10 = hasattr(self.engine, "num_io_tensors")
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype] = {}
        self.bindings: list[int] | None = None
        self._prepare_bindings()

    def _tensor_names(self) -> list[str]:
        if self.is_trt10:
            return [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        return [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]

    def _is_input(self, name: str) -> bool:
        if self.is_trt10:
            return self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
        return self.engine.binding_is_input(self.engine.get_binding_index(name))

    def _engine_shape(self, name: str) -> tuple[int, ...]:
        if self.is_trt10:
            return tuple(self.engine.get_tensor_shape(name))
        return tuple(self.engine.get_binding_shape(self.engine.get_binding_index(name)))

    def _context_shape(self, name: str) -> tuple[int, ...]:
        if self.is_trt10:
            return tuple(self.context.get_tensor_shape(name))
        return tuple(self.context.get_binding_shape(self.engine.get_binding_index(name)))

    def _dtype(self, name: str) -> np.dtype:
        if self.is_trt10:
            dtype = self.engine.get_tensor_dtype(name)
        else:
            dtype = self.engine.get_binding_dtype(self.engine.get_binding_index(name))
        return np.dtype(self.trt.nptype(dtype))

    def _set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        if self.is_trt10:
            if not self.context.set_input_shape(name, shape):
                raise RuntimeError(f"TensorRT rejected input shape {shape} for {name}")
        else:
            index = self.engine.get_binding_index(name)
            if not self.context.set_binding_shape(index, shape):
                raise RuntimeError(f"TensorRT rejected input shape {shape} for {name}")

    def _prepare_bindings(self) -> None:
        names = self._tensor_names()
        self.input_names = [name for name in names if self._is_input(name)]
        self.output_names = [name for name in names if not self._is_input(name)]
        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected one image input, found {self.input_names}")

        input_name = self.input_names[0]
        input_shape = self._engine_shape(input_name)
        if any(dim < 0 for dim in input_shape):
            # The packaged models in this project use 1x3x640x640. Keep this as
            # a safe dynamic-shape default and report the actual binding below.
            input_shape = (1, 3, 640, 640)
            self._set_input_shape(input_name, input_shape)

        if not self.is_trt10:
            self.bindings = [0] * self.engine.num_bindings

        for name in names:
            shape = self._context_shape(name)
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved dynamic shape for tensor {name}: {shape}")
            dtype = self._dtype(name)
            host = np.empty(shape, dtype=dtype)
            device = self.backend.malloc(host.nbytes)
            device_ptr = self.backend.pointer(device)
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            self.host_buffers[name] = host
            # Keep the allocation object, not only its integer address. PyCUDA
            # needs the DeviceAllocation object for copies and explicit free.
            self.device_buffers[name] = device
            if self.is_trt10:
                if not self.context.set_tensor_address(name, device_ptr):
                    raise RuntimeError(f"Could not bind tensor address for {name}")
            else:
                assert self.bindings is not None
                self.bindings[self.engine.get_binding_index(name)] = device_ptr

        print(f"TensorRT version: {self.trt.__version__}")
        for name in names:
            role = "INPUT " if name in self.input_names else "OUTPUT"
            print(f"  {role} {name}: shape={self.shapes[name]}, dtype={self.dtypes[name]}")

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.shapes[self.input_names[0]]

    def infer(self, image_tensor: np.ndarray) -> dict[str, np.ndarray]:
        name = self.input_names[0]
        expected = self.shapes[name]
        if tuple(image_tensor.shape) != expected:
            raise ValueError(f"Input has shape {image_tensor.shape}; engine expects {expected}")
        np.copyto(self.host_buffers[name], image_tensor.astype(self.dtypes[name], copy=False))

        self.backend.copy_h2d(self.device_buffers[name], self.host_buffers[name])

        if self.is_trt10:
            ok = self.context.execute_async_v3(self.backend.stream_handle)
        else:
            assert self.bindings is not None
            ok = self.context.execute_async_v2(self.bindings, self.backend.stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async failed")

        for output_name in self.output_names:
            host = self.host_buffers[output_name]
            self.backend.copy_d2h(host, self.device_buffers[output_name])
        self.backend.synchronize()
        return {name: self.host_buffers[name].copy() for name in self.output_names}

    def close(self) -> None:
        for pointer in self.device_buffers.values():
            try:
                if self.backend is not None:
                    self.backend.free(pointer)
            except Exception:
                pass
        self.device_buffers.clear()
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
            self.backend = None

    def __del__(self):
        self.close()


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Ultralytics-style centered letterbox with padding value 114."""
    target_h, target_w = size
    height, width = image.shape[:2]
    gain = min(target_h / height, target_w / width)
    resized_w, resized_h = round(width * gain), round(height * gain)
    pad_w = (target_w - resized_w) / 2
    pad_h = (target_h - resized_h) / 2
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, gain, (left, top)


def preprocess(frame: np.ndarray, input_shape: tuple[int, ...]):
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(f"Expected NCHW input 1x3xHxW, got {input_shape}")
    input_h, input_w = input_shape[2], input_shape[3]
    padded, gain, pad = letterbox(frame, (input_h, input_w))
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    return tensor[None], gain, pad


def to_nhwc(array: np.ndarray, channels: int) -> np.ndarray | None:
    """Convert a batch-1 NCHW/NHWC TensorRT output to HWC."""
    value = np.asarray(array)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        return None
    if value.shape[-1] == channels:
        return value
    if value.shape[0] == channels:
        return value.transpose(1, 2, 0)
    return None


def pair_yolo26_outputs(
    outputs: dict[str, np.ndarray], num_classes: int | None = None
):
    """Pair YOLO26 box and class outputs by their feature-map resolution.

    A four-class model is ambiguous by shape because both LTRB boxes and class
    scores have four channels. The custom exporter writes each scale as
    ``box, sigmoid(class)``; preserve that order and use the score range as an
    additional check in this special case.
    """
    if num_classes == 4:
        by_grid: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
        for name, output in outputs.items():
            value = to_nhwc(output, 4)
            if value is not None:
                value = value.astype(np.float32, copy=False)
                by_grid.setdefault(value.shape[:2], []).append((name, value))

        pairs = []
        for grid in sorted(by_grid, reverse=True):
            candidates = by_grid[grid]
            if len(candidates) != 2:
                names = [name for name, _ in candidates]
                raise RuntimeError(
                    f"Expected one box and one class output at grid {grid}, "
                    f"found {names}"
                )

            # Sigmoid class scores are bounded to [0, 1]. If exactly one output
            # has that range, use it as classes; otherwise fall back to the
            # exporter's deterministic box-then-class output order.
            score_like = [
                index
                for index, (_, value) in enumerate(candidates)
                if float(value.min()) >= -1e-6 and float(value.max()) <= 1.0 + 1e-6
            ]
            class_index = score_like[0] if len(score_like) == 1 else 1
            box_index = 1 - class_index
            pairs.append((candidates[box_index][1], candidates[class_index][1]))
        if pairs:
            return pairs

    boxes: dict[tuple[int, int], np.ndarray] = {}
    classes: dict[tuple[int, int], np.ndarray] = {}
    for name, output in outputs.items():
        box = to_nhwc(output, 4)
        if box is not None:
            boxes[box.shape[:2]] = box.astype(np.float32, copy=False)
            continue

        value = np.asarray(output)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            continue
        # Classification output is either HWC or CHW. Grid dimensions are the
        # repeated dimensions (80x80, 40x40, or 20x20).
        if value.shape[0] == value.shape[1]:
            cls = value
        elif value.shape[1] == value.shape[2]:
            cls = value.transpose(1, 2, 0)
        else:
            continue
        classes[cls.shape[:2]] = cls.astype(np.float32, copy=False)

    grids = sorted(set(boxes) & set(classes), reverse=True)
    if not grids:
        summary = {name: tuple(value.shape) for name, value in outputs.items()}
        raise RuntimeError(
            "Could not identify YOLO26 box/class output pairs. " f"TensorRT outputs: {summary}"
        )
    return [(boxes[grid], classes[grid]) for grid in grids]


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(area_a + area_b - intersection, 1e-7)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_det: int,
) -> np.ndarray:
    keep: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size and len(keep) < max_det:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            ious = box_iou_one_to_many(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= iou_threshold]
    keep.sort(key=lambda index: float(scores[index]), reverse=True)
    return np.asarray(keep[:max_det], dtype=np.int64)


def decode_yolo26(
    outputs: dict[str, np.ndarray],
    input_hw: tuple[int, int],
    original_hw: tuple[int, int],
    gain: float,
    pad: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    num_classes: int | None = None,
):
    """Decode direct LTRB distances from the three YOLO26 feature maps."""
    input_h, input_w = input_hw
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for box_map, cls_map in pair_yolo26_outputs(outputs, num_classes=num_classes):
        grid_h, grid_w = box_map.shape[:2]
        stride_x = input_w / grid_w
        stride_y = input_h / grid_h
        class_ids = np.argmax(cls_map, axis=2)
        scores = np.max(cls_map, axis=2)
        ys, xs = np.where(scores >= conf_threshold)
        if not len(xs):
            continue
        distances = box_map[ys, xs]
        decoded = np.column_stack(
            (
                (xs + 0.5 - distances[:, 0]) * stride_x,
                (ys + 0.5 - distances[:, 1]) * stride_y,
                (xs + 0.5 + distances[:, 2]) * stride_x,
                (ys + 0.5 + distances[:, 3]) * stride_y,
            )
        ).astype(np.float32)
        all_boxes.append(decoded)
        all_scores.append(scores[ys, xs].astype(np.float32))
        all_classes.append(class_ids[ys, xs].astype(np.int32))

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_classes)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / gain
    original_h, original_w = original_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h - 1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold, max_det)
    return boxes[keep], scores[keep], class_ids[keep]


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (66, 135, 245),
        (76, 175, 80),
        (244, 67, 54),
        (255, 193, 7),
        (156, 39, 176),
        (0, 188, 212),
    )
    return palette[class_id % len(palette)]


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    labels: list[str],
) -> None:
    line_width = max(2, round((frame.shape[0] + frame.shape[1]) / 900))
    font_scale = max(0.5, line_width / 3)
    for box, score, class_id in zip(boxes, scores, class_ids):
        class_id = int(class_id)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = class_color(class_id)
        label = labels[class_id] if class_id < len(labels) else f"class{class_id}"
        text = f"{label} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, line_width - 1)
        )
        text_y = max(text_h + baseline, y1)
        cv2.rectangle(
            frame,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, line_width - 1),
            cv2.LINE_AA,
        )


def open_source(source: str):
    source_value: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def create_writer(path: str, capture: cv2.VideoCapture, frame: np.ndarray):
    output = Path(path)
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if output.suffix.lower() not in video_suffixes:
        output.mkdir(parents=True, exist_ok=True)
        output = output / "result.mp4"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")
    return writer, output


def capture_frame_index(capture: cv2.VideoCapture, *, after_read: bool = False) -> int:
    value = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if not np.isfinite(value) or value < 0:
        return 0
    index = int(round(value))
    return max(0, index - 1) if after_read else max(0, index)


def skip_late_video_frames(capture: cv2.VideoCapture, source_fps: float,
                           source_frames: float, anchor_source_seconds: float,
                           anchor_wall_time: float) -> int:
    if source_fps <= 0 or not np.isfinite(source_frames) or source_frames <= 0:
        return 0
    next_index = capture_frame_index(capture)
    elapsed = max(0.0, time.monotonic() - anchor_wall_time)
    target_index = min(
        max(0, int((anchor_source_seconds + elapsed) * source_fps)),
        max(0, int(source_frames) - 1),
    )
    skipped = 0
    while next_index < target_index:
        if not capture.grab():
            break
        next_index += 1
        skipped += 1
    return skipped


def write_timeline_frame(writer: cv2.VideoWriter, frame: np.ndarray, source_index: int,
                         previous_frame: np.ndarray | None,
                         previous_index: int | None) -> tuple[np.ndarray, int, int]:
    written = 0
    if previous_frame is not None and previous_index is not None:
        for _ in range(max(0, source_index - previous_index - 1)):
            writer.write(previous_frame)
            written += 1
    writer.write(frame)
    return frame.copy(), source_index, written + 1


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")

    engine_bytes, offset = read_engine_bytes(args.model)
    print(
        f"Embedded TensorRT engine: offset={offset}, "
        f"size={len(engine_bytes) / 1024 / 1024:.2f} MiB"
    )
    if args.extract_only:
        output = Path(args.extract_only)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(engine_bytes)
        print(f"Extracted engine: {output}")
        return 0
    if args.source is None:
        raise ValueError("--source is required unless --extract-only is used")

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    runner = TensorRTRunner(engine_bytes, cuda_backend=args.cuda_backend)
    capture = open_source(args.source)
    writer = None
    saved_video_path = None
    fps_ema = 0.0
    frame_index = 0
    skipped_frames = 0
    saved_frame_count = 0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_period = 1.0 / source_fps if np.isfinite(source_fps) and source_fps > 0 else 0.0
    source_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    playback_anchor_source = capture_frame_index(capture) / source_fps if frame_period > 0 else 0.0
    playback_anchor_wall = time.monotonic()
    previous_saved_frame = None
    previous_saved_index = None

    if args.realtime and frame_period > 0:
        print("Real-time playback enabled: late source frames will be skipped when inference is slower than the video FPS.")

    try:
        while True:
            if args.realtime and frame_period > 0:
                skipped_frames += skip_late_video_frames(
                    capture, source_fps, source_frames,
                    playback_anchor_source, playback_anchor_wall,
                )
            ok, frame = capture.read()
            if not ok:
                break
            source_index = capture_frame_index(capture, after_read=True)
            source_position = (source_index + 1) / source_fps if frame_period > 0 else 0.0
            start = time.perf_counter()
            tensor, gain, pad = preprocess(frame, runner.input_shape)
            outputs = runner.infer(tensor)
            boxes, scores, class_ids = decode_yolo26(
                outputs=outputs,
                input_hw=(runner.input_shape[2], runner.input_shape[3]),
                original_hw=frame.shape[:2],
                gain=gain,
                pad=pad,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                max_det=args.max_det,
                num_classes=len(labels),
            )
            elapsed = max(time.perf_counter() - start, 1e-9)
            current_fps = 1.0 / elapsed
            fps_ema = current_fps if frame_index == 0 else 0.9 * fps_ema + 0.1 * current_fps
            frame_index += 1

            draw_detections(frame, boxes, scores, class_ids, labels)
            cv2.putText(
                frame,
                f"Infer {fps_ema:.1f} FPS  detections {len(boxes)}  skipped {skipped_frames}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.save:
                if writer is None:
                    writer, saved_video_path = create_writer(args.save, capture, frame)
                previous_saved_frame, previous_saved_index, written = write_timeline_frame(
                    writer, frame, source_index, previous_saved_frame, previous_saved_index,
                )
                saved_frame_count += written

            delay_ms = 1
            if args.realtime and frame_period > 0:
                deadline = playback_anchor_wall + max(0.0, source_position - playback_anchor_source)
                delay_ms = max(1, round((deadline - time.monotonic()) * 1000))

            if not args.no_show:
                display = frame
                if args.display_width > 0 and frame.shape[1] > args.display_width:
                    scale = args.display_width / frame.shape[1]
                    display = cv2.resize(
                        frame,
                        (args.display_width, round(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("STAR TensorRT Detection - Q/Esc to quit", display)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            elif args.realtime and delay_ms > 1:
                time.sleep(delay_ms / 1000.0)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        runner.close()

    print(f"Processed {frame_index} frames; skipped {skipped_frames} late source frames")
    if saved_frame_count:
        print(f"Saved timeline frames: {saved_frame_count}")
    if saved_video_path is not None:
        print(f"Saved annotated video: {saved_video_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
