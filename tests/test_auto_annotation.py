import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from backend.auto_annotation_worker import run, write_xml
from backend.catalog import FunctionCatalog


class Tensor:
    def __init__(self, data): self.data = data
    def cpu(self): return self
    def tolist(self): return self.data


class FakeModel:
    task = 'detect'
    names = {0: '工件 & 合格'}
    def predict(self, **kwargs):
        return [SimpleNamespace(orig_shape=(20, 30), boxes=SimpleNamespace(
            cls=Tensor([0]), xyxy=Tensor([[-2.5, 3.2, 40.1, 18.3]])))]


class AnnotationTests(unittest.TestCase):
    def test_xml_bounds_escaping_and_degenerate_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / '中文.jpg'
            target = image.with_suffix('.xml')
            count = write_xml(image, (20, 30), [('工件 & 合格', [-2.5, 3.2, 40.1, 18.3]),
                ('invalid', [5, 5, 1, 1])], target)
            root = ET.parse(target).getroot()
            self.assertEqual(count, 1)
            self.assertEqual(root.findtext('object/name'), '工件 & 合格')
            self.assertEqual([int(root.findtext('object/bndbox/' + k)) for k in
                              ['xmin', 'ymin', 'xmax', 'ymax']], [0, 3, 30, 19])

    def test_skip_backup_and_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / 'part.jpg'
            image.touch()
            xml = image.with_suffix('.xml')
            xml.write_text('original')
            config = dict(folder=temp, recursive=True, classes=[], existing='跳过',
                          conf=.5, iou=.7, imgsz=640, device='', empty_xml=True)
            self.assertEqual(run(config, FakeModel(), lambda _: None)['skipped'], 1)
            self.assertEqual(xml.read_text(), 'original')
            config['existing'] = '备份后覆盖'
            result = run(config, FakeModel(), lambda _: None)
            self.assertEqual(result['written'], 1)
            self.assertEqual((Path(result['backup']) / 'part.xml').read_text(), 'original')
            (folder / 'part.png').touch()
            with self.assertRaisesRegex(ValueError, '同名 XML'):
                run(config, FakeModel())

    def test_empty_and_failed_images(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / 'empty.png'
            image.touch()
            model = FakeModel()
            model.predict = lambda **kw: [SimpleNamespace(orig_shape=(20, 30), boxes=SimpleNamespace(cls=Tensor([]), xyxy=Tensor([])))]
            config = dict(folder=temp, recursive=False, classes=[], existing='跳过',
                          conf=.5, iou=.7, imgsz=640, device='', empty_xml=True)
            result = run(config, model, lambda _: None)
            self.assertEqual(result['empty'], 1)
            self.assertEqual(ET.parse(image.with_suffix('.xml')).findall('object'), [])

    def test_existing_catalog_receives_new_function_once(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'functions.json'
            path.write_text('[]')
            for _ in range(2):
                catalog = FunctionCatalog(path)
                self.assertEqual(sum(i['id'] == 'auto_annotation' for i in catalog.list()), 1)
