"""Read ONNX operator/type declarations without onnx/NumPy or loading weights.

Wire field numbers follow https://github.com/onnx/onnx/blob/main/onnx/onnx.proto.
This is a conservative preflight, not a certificate of final engine precision.
The same standard-library-only script is executed on the SSH device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


CHILDREN = {
    'model': {7: 'graph', 25: 'function'},
    'graph': {1: 'node', 5: 'tensor', 11: 'value', 12: 'value', 13: 'value'},
    'function': {7: 'node', 11: 'attribute'},
    'node': {5: 'attribute'},
    'attribute': {5: 'tensor', 10: 'tensor', 6: 'graph', 11: 'graph'},
    'value': {2: 'type'}, 'type': {1: 'tensor_type'},
}
REDUCED_FLOAT_TYPES = {10, 16, 17, 18, 19, 20, 23, 24}


def inspect_onnx(path):
    quantized_ops, reduced_types = set(), set()
    nodes, graphs, fields = 0, 0, 0
    with Path(path).open('rb') as stream:
        end = stream.seek(0, 2)
        stream.seek(0)

        def varint(bound):
            value = 0
            for shift in range(0, 70, 7):
                if stream.tell() >= bound:
                    raise ValueError('ONNX protobuf 字段不完整。')
                raw = stream.read(1)
                if not raw:
                    raise ValueError('ONNX 文件意外结束。')
                byte = raw[0]
                value |= (byte & 127) << shift
                if byte < 128:
                    return value
            raise ValueError('ONNX protobuf 整数过长。')

        def visit(kind, bound, depth=0):
            nonlocal nodes, graphs, fields
            if depth > 64:
                raise ValueError('ONNX 嵌套层级超出检查限制。')
            nodes += kind == 'node'
            graphs += kind == 'graph'
            while stream.tell() < bound:
                fields += 1
                if fields > 2_000_000:
                    raise ValueError('ONNX 字段数量超出检查限制。')
                tag = varint(bound)
                field, wire = tag >> 3, tag & 7
                if not field:
                    raise ValueError('ONNX protobuf 字段编号无效。')
                if wire == 0:
                    value = varint(bound)
                    if (kind == 'tensor' and field == 2 or kind == 'tensor_type' and field == 1) and value in REDUCED_FLOAT_TYPES:
                        reduced_types.add(value)
                    continue
                if wire == 2:
                    size = varint(bound)
                elif wire in (1, 5):
                    size = 8 if wire == 1 else 4
                else:
                    raise ValueError('不支持的 ONNX protobuf 编码。')
                finish = stream.tell() + size
                if finish > bound:
                    raise ValueError('ONNX protobuf 字段长度越界。')
                child = CHILDREN.get(kind, {}).get(field) if wire == 2 else None
                if kind == 'node' and field == 4 and wire == 2:
                    if not 0 < size <= 4096:
                        raise ValueError('ONNX 算子类型长度无效。')
                    op = stream.read(size).decode('utf-8')
                    if 'quantiz' in op.lower() or op.startswith('QLinear') or op in {'ConvInteger', 'MatMulInteger'}:
                        quantized_ops.add(op)
                elif child:
                    visit(child, finish, depth + 1)
                # Skip raw tensor data and metadata without allocating it.
                stream.seek(finish)
            if stream.tell() != bound:
                raise ValueError('ONNX protobuf 结构无效。')

        visit('model', end)
        if not graphs or not nodes:
            raise ValueError('未识别到有效的 ONNX 计算图。')
    return {'quantized': bool(quantized_ops), 'quantizedOps': sorted(quantized_ops),
            'reducedFloatTypes': sorted(reduced_types), 'nodes': nodes}


if __name__ == '__main__':
    try:
        print(json.dumps(inspect_onnx(sys.argv[1])))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
