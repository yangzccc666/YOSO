import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.catalog import FunctionCatalog, JETSON_ONNX_EXPORT_FUNCTION
from backend.handlers import RunContext
from backend.jetson_onnx_export import run_jetson_onnx_export
from backend.onnx_precision_probe import inspect_onnx
from backend.remote_tensorrt_build import _configuration, _trtexec_arguments, _validate_remote_jobs
from backend.yolo_training import build_training_command


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def message(field, content):
    return varint(field * 8 + 2) + varint(len(content)) + content


def node(op, attributes=b''):
    return message(4, op.encode()) + (message(5, attributes) if attributes else b'')


def graph(op='Conv', *, extra=b''):
    return message(7, message(1, node(op)) + extra)


class PrecisionProbeTests(unittest.TestCase):
    def inspect(self, raw):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.onnx'
            path.write_bytes(raw)
            return inspect_onnx(path)

    def test_fp32_graph_and_quantization_nodes(self):
        self.assertEqual(self.inspect(graph())['quantized'], False)
        for op in ['QuantizeLinear', 'DequantizeLinear', 'QLinearConv', 'MatMulInteger', 'DynamicQuantizeLinear']:
            with self.subTest(op=op):
                self.assertEqual(self.inspect(graph(op))['quantizedOps'], [op])

    def test_nested_graph_and_local_functions(self):
        subgraph = message(1, node('DequantizeLinear'))
        raw = message(7, message(1, node('If', message(6, subgraph))))
        self.assertTrue(self.inspect(raw)['quantized'])
        self.assertTrue(self.inspect(graph() + message(25, message(7, node('QuantizeLinear'))))['quantized'])

    def test_metadata_and_weight_strings_do_not_cause_false_positives(self):
        tensor = varint(2 * 8) + varint(1) + message(9, b'QuantizeLinear DequantizeLinear')
        self.assertFalse(self.inspect(graph(extra=message(5, tensor)) + message(14, b'quantized'))['quantized'])

    def test_detects_half_precision_tensor_and_input_types(self):
        tensor = varint(2 * 8) + varint(10) + message(9, b'weights')
        self.assertEqual(self.inspect(graph(extra=message(5, tensor)))['reducedFloatTypes'], [10])
        value = message(2, message(1, varint(1 * 8) + varint(16)))
        self.assertEqual(self.inspect(graph(extra=message(11, value)))['reducedFloatTypes'], [16])

    def test_invalid_graphs_fail_closed(self):
        for raw in [b'', b'not an onnx', b'\x3a\xff\xff', message(7, b''), message(7, b'\x00'), message(7, b'\x0a\x7f')]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.inspect(raw)

    def test_large_raw_tensor_is_skipped_without_reading_into_memory(self):
        size = 32 * 1024 * 1024
        tensor_head = varint(2 * 8) + varint(1) + varint(9 * 8 + 2) + varint(size)
        graph_head = message(1, node('Conv')) + varint(5 * 8 + 2) + varint(len(tensor_head) + size) + tensor_head
        model_head = varint(7 * 8 + 2) + varint(len(graph_head) + size) + graph_head
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'big.onnx'
            with path.open('wb') as stream:
                stream.write(model_head)
                stream.seek(size - 1, 1)
                stream.write(b'\0')
            self.assertFalse(inspect_onnx(path)['quantized'])


class PrecisionIntegrationTests(unittest.TestCase):
    def context(self, mode, **parameters):
        return RunContext('remote_tensorrt_build', {}, dict(onnx_files='/models/model.onnx', build_mode=mode, **parameters), lambda _: None)

    def test_training_amp_and_fp32_reach_real_command(self):
        for raw, expected in [(True, 'amp=True'), (False, 'amp=False'), ('False', 'amp=False')]:
            with self.subTest(amp=raw):
                command = build_training_command({'device': '0', 'amp': raw})
                self.assertIn(expected, command)
                self.assertFalse(any('int8' in v or 'half=' in v for v in command))

    def test_build_flags_do_not_enable_low_precision_for_fp32(self):
        for mode, flag in [('FP16', '--fp16'), ('FP32', '--noTF32'), ('自动最佳（--best）', '--best')]:
            config = _configuration(self.context(mode), require_paths=True)
            args = _trtexec_arguments(config, '/a.onnx', '/a.plan')
            self.assertIn(flag, args)
            if mode == 'FP32':
                self.assertFalse(set(args) & {'--fp16', '--int8', '--best'})

    def test_quantized_onnx_is_rejected_for_both_high_precision_modes(self):
        client = MagicMock()
        client.open_sftp.return_value.stat.return_value.st_size = 100
        for mode in ['FP16', 'FP32']:
            config = _configuration(self.context(mode, overwrite_output=True), require_paths=True)
            response = dict(quantized=True, quantizedOps=['QuantizeLinear'], reducedFloatTypes=[], nodes=2)
            with patch('backend.remote_tensorrt_build._remote_capture', return_value=(0, json.dumps(response), '')), \
                 self.assertRaisesRegex(ValueError, '跳过 Docker'):
                _validate_remote_jobs(client, config, lambda _: None)

    def test_fp32_input_is_accepted_and_half_input_cannot_pretend_fp32(self):
        client = MagicMock()
        client.open_sftp.return_value.stat.return_value.st_size = 100
        config = _configuration(self.context('FP32', overwrite_output=True), require_paths=True)
        response = dict(quantized=False, reducedFloatTypes=[], nodes=1)
        with patch('backend.remote_tensorrt_build._remote_capture', return_value=(0, json.dumps(response), '')) as probe:
            _validate_remote_jobs(client, config, lambda _: None)
            self.assertIn('python3', probe.call_args.args[1])
            self.assertIn('/models/model.onnx', probe.call_args.args[1])
        response['reducedFloatTypes'] = [10]
        with patch('backend.remote_tensorrt_build._remote_capture', return_value=(0, json.dumps(response), '')), \
             self.assertRaisesRegex(ValueError, '重新导出 FP32'):
            _validate_remote_jobs(client, config, lambda _: None)

    def test_failed_probe_never_silently_allows_high_precision_build(self):
        client = MagicMock()
        client.open_sftp.return_value.stat.return_value.st_size = 100
        config = _configuration(self.context('FP16', overwrite_output=True), require_paths=True)
        for response in [(1, '', 'python3 missing'), (0, 'bad-json', ''), (0, '{}', '')]:
            with self.subTest(response=response), patch('backend.remote_tensorrt_build._remote_capture', return_value=response), self.assertRaises(ValueError):
                _validate_remote_jobs(client, config, lambda _: None)

    def test_export_fp32_and_fp16_intent_do_not_create_quantization_todos(self):
        for precision in ['INT8', 'FP16', 'FP32']:
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model = root / 'model.pt'
                model.write_bytes(b'pt')
                python = root / 'python'
                python.touch()
                context = RunContext('jetson_onnx_export', {'weights_file': model},
                                     dict(python_path=str(python), deployment_precision=precision), lambda _: None)
                def fake_export(config, _): config['output'].write_bytes(graph())
                with patch('backend.jetson_onnx_export._export_one', side_effect=fake_export):
                    result = run_jetson_onnx_export(context)
                self.assertEqual(result['deploymentPrecision'], precision)
                self.assertEqual(len(result['onnxQuantDefaultsList']), 1 if precision == 'INT8' else 0)
                self.assertEqual(model.read_bytes(), b'pt')
                if precision != 'INT8': self.assertIsNone(result['onnxQuantDefaults'])

    def test_precision_setting_migrates_once_without_resetting_user_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'functions.json'
            item = deepcopy(JETSON_ONNX_EXPORT_FUNCTION)
            item['parameters'] = [p for p in item['parameters'] if p['id'] != 'deployment_precision']
            next(p for p in item['parameters'] if p['id'] == 'imgsz')['default'] = 960
            path.write_text(json.dumps([item]))
            for _ in range(2):
                result = FunctionCatalog(path).get(item['id'])
                fields = {p['id']: p for p in result['parameters']}
                self.assertEqual(fields['imgsz']['default'], 960)
                self.assertEqual(fields['deployment_precision']['options'], ['INT8', 'FP16', 'FP32'])
                self.assertEqual(len(fields), len(result['parameters']))
