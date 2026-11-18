from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.package_naming import PackageNamingStore, parse_title
from tests import test_local_star_package as packaging_fixtures
from backend import local_star_package
from backend.task_manager import TaskCancelled


class PackageNamingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'versions.json'
        self.store = PackageNamingStore(self.path)

    def remember(self, title):
        return self.store.remember(title, '/models/test.plan', '/models/test.star', ['MARK','OKP'], '增加夜班样本')

    def test_title_parts_and_valid_dates_and_scope_normalization(self):
        result = parse_title('ct2-2#1.1.0.1117')
        self.assertEqual(result, {'line':'CT2','station':2,'version':[1,1,0],'day':'1117'})
        self.assertIsNotNone(parse_title('CT2-2#1.0.0.0229'))
        for title in ['CT2-0#1.1.0.1117','CT2-2#0.0.0.1117','CT2-2#1.0.0.0230',
                      'CT2-2#1.0.0.1301','CT2-2#1.0.0.1117\n','../CT2-2#1.0.0.1117',
                      'CT2-2#1.01.0.1117','CT2-2#1.0.0.1117/other']:
            self.assertIsNone(parse_title(title), title)

    def test_highest_version_does_not_regress_with_newer_lower_version_and_is_per_station(self):
        for title in ['CT2-2#1.9.0.1116','CT2-2#1.10.0.1117','CT2-2#1.2.0.1117','CT2-3#9.0.0.1117']:
            self.remember(title)
        payload = self.store.payload('CT2',2)
        series = next(item for item in payload['series'] if item['key']=='CT2-2')
        self.assertEqual(series['version'], [1,10,0])
        self.assertEqual(series['count'], 3)
        self.assertEqual(payload['count'], 3)
        self.assertTrue(all('CT2-2#' in record['title'] for record in payload['records']))

    def test_history_survives_restart_and_does_not_have_five_record_limit(self):
        for number in range(35): self.remember(f'CT2-2#1.0.{number}.1117')
        payload = PackageNamingStore(self.path).payload('ct2','2')
        self.assertEqual(payload['count'],35)
        self.assertEqual(len(payload['records']),30)
        self.assertEqual(len(self.store._read()),35)
        self.assertEqual(payload['records'][0]['note'],'增加夜班样本')

    def test_old_manual_names_do_not_consume_versions_or_touch_file(self):
        self.assertFalse(self.remember('old-model-1117'))
        self.assertFalse(self.path.exists())
        self.assertEqual(self.store.payload()['series'], [])

    def test_concurrent_remembers_are_atomic(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda i: self.remember(f'CT2-2#1.0.{i}.1117'), range(20)))
        self.assertEqual(self.store.payload('CT2',2)['count'],20)

    def test_corrupt_ledger_is_not_silently_overwritten(self):
        for content in ['broken', '[]', 'null', '{}']:
            self.path.write_text(content)
            with self.assertRaisesRegex(ValueError,'原文件未覆盖'): self.remember('CT2-2#1.0.0.1117')
            self.assertEqual(self.path.read_text(),content)

    def test_equal_timestamps_list_newest_inserted_first(self):
        self.remember('CT2-2#1.0.0.1117')
        self.remember('CT2-2#1.0.1.1117')
        import json
        data = json.loads(self.path.read_text())
        for record in data['records']: record['createdAt'] = '2026-11-17T10:00:00+08:00'
        self.path.write_text(json.dumps(data))
        self.assertEqual(self.store.payload('CT2',2)['records'][0]['title'], 'CT2-2#1.0.1.1117')

    def test_successful_real_packaging_records_but_failure_does_not(self):
        fixture = packaging_fixtures.LocalStarPackageTests()
        fixture.setUp()
        try:
            with patch.object(local_star_package, 'package_naming_store', self.store):
                result = local_star_package.run_local_star_package(fixture.context(title='CT2-2#1.1.0.1117', package_change_note='补数据'))
                self.assertEqual(self.store.payload('CT2',2)['count'],1)
                self.assertEqual(self.store.payload('CT2',2)['records'][0]['output'], result['outputFiles'][0])
                with patch.object(local_star_package, '_run_process', side_effect=RuntimeError('failed')):
                    with self.assertRaises(RuntimeError): local_star_package.run_local_star_package(fixture.context(title='CT2-2#1.1.1.1117'))
                with patch.object(local_star_package, '_run_process', side_effect=TaskCancelled('cancelled')):
                    with self.assertRaises(TaskCancelled): local_star_package.run_local_star_package(fixture.context(title='CT2-2#1.1.1.1117'))
                self.assertEqual(self.store.payload('CT2',2)['count'],1)
        finally: fixture.tearDown()

    def test_api_requires_activation_and_filters_station_and_reports_corrupt_ledger(self):
        import io
        import json
        from types import SimpleNamespace
        from backend import server
        self.remember('CT2-2#1.0.0.1117')
        self.remember('CT2-3#2.0.0.1117')
        def request(activated):
            raw = b'GET /api/local-star-package/naming?line=ct2&station=2 HTTP/1.0\r\n\r\n'
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            with patch.object(server,'package_naming_store',self.store), patch.object(server.license_store,'status',return_value={'activated':activated,'role':'user','message':'activation required'}):
                server.AppHandler(connection,('127.0.0.1',1),SimpleNamespace())
            head,data=bytes(output).split(b'\r\n\r\n',1)
            return int(head.split()[1]),json.loads(data)
        self.assertEqual(request(False)[0],403)
        status,payload=request(True)
        self.assertEqual(status,200)
        self.assertEqual(payload['count'],1)
        self.assertEqual(len(payload['series']),2)
        self.path.write_text('broken')
        status,payload=request(True)
        self.assertEqual(status,400)
        self.assertIn('原文件未覆盖',payload['error'])

    def test_ledger_write_failure_does_not_turn_successful_model_into_failed_task(self):
        fixture = packaging_fixtures.LocalStarPackageTests()
        fixture.setUp()
        try:
            with patch.object(local_star_package, 'package_naming_store', self.store), \
                 patch.object(self.store, 'remember', side_effect=OSError('disk full')):
                result = local_star_package.run_local_star_package(fixture.context(title='CT2-2#1.0.0.1117'))
                self.assertTrue(Path(result['outputFiles'][0]).is_file())
                self.assertIn('版本记录保存失败', result['namingWarning'])
        finally: fixture.tearDown()
