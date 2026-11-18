from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.inference_video_history import InferenceVideoHistoryStore


class InferenceVideoHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / '测试视频.mp4'
        self.video.write_bytes(b'video-not-copied')
        self.store = InferenceVideoHistoryStore(self.root / 'config' / 'videos.json')

    def save(self, path=None, **kwargs):
        return self.store.remember(str(path or self.video), **kwargs)

    def test_path_only_references_and_metadata_survive_restart_with_private_permissions(self):
        self.save(scene='磁探3号线', station='工位2')
        record = InferenceVideoHistoryStore(self.store.path).list()[0]
        self.assertEqual((record['scene'], record['station']), ('磁探3号线','工位2'))
        self.assertEqual(record['path'], str(self.video))
        self.assertTrue(record['available'])
        self.assertIsNone(record['lastUsedAt'])
        self.assertEqual(self.video.read_bytes(), b'video-not-copied')
        self.assertEqual(list(self.store.path.parent.iterdir()), [self.store.path])
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)

    def test_reuse_deduplicates_normalized_path_keeps_metadata_and_moves_to_top(self):
        original = self.save(scene='产线A', station='2')
        other = self.root / 'other.mp4'; other.write_bytes(b'other')
        self.save(other)
        self.store.remember_payload({'paths':{'video_file':str(self.root / '.' / self.video.name)},'parameters':{}})
        records = self.store.list()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['id'], original['id'])
        self.assertEqual(records[0]['createdAt'], original['createdAt'])
        self.assertEqual((records[0]['scene'],records[0]['station']), ('产线A','2'))
        self.assertEqual(records[0]['useCount'], 1)
        self.assertIsNotNone(records[0]['lastUsedAt'])

    def test_manual_update_and_blank_metadata_are_supported(self):
        self.save(scene='old',station='1')
        self.save(scene='新场景',station='工位三')
        self.assertEqual(self.store.list()[0]['scene'],'新场景')
        self.save(scene='',station='')
        self.assertEqual(self.store.list()[0]['station'],'')
        self.assertEqual(len(self.store.list()),1)

    def test_remote_and_local_share_path_without_credentials_models_or_upload_location(self):
        paths={'local_video_file':str(self.video),'video_file':str(self.video),'model_file':'secret.pt'}
        parameters={'test_video_scene':'TCL','test_video_station':'工位1','password':'secret-password','host':'private-host'}
        self.store.remember_payload({'paths':paths,'parameters':parameters},remote=True)
        self.store.remember_payload({'paths':paths,'parameters':parameters})
        self.assertEqual(len(self.store.list()),1)
        self.assertEqual(self.store.list()[0]['useCount'],2)
        saved=self.store.path.read_text()
        for secret in ['secret-password','private-host','secret.pt']: self.assertNotIn(secret,saved)

    def test_image_mode_or_empty_legacy_payload_does_not_add_video(self):
        self.store.remember_payload({'paths':{'video_file':str(self.video)},'parameters':{'input_mode':'图片文件夹检测'}})
        self.store.remember_payload({})
        self.assertFalse(self.store.path.exists())

    def test_more_than_five_records_and_explicit_date_order(self):
        for number in range(12):
            video=self.root/f'{number}.mp4';video.write_bytes(b'video');self.save(video)
        self.assertEqual(len(self.store.list()),12)
        data=json.loads(self.store.path.read_text())
        for i,row in enumerate(data['videos']): row['updatedAt']=f'2026-11-{i+1:02d}T08:00:00+00:00'
        self.store.path.write_text(json.dumps(data))
        self.assertEqual(self.store.list()[0]['updatedAt'],'2026-11-12T08:00:00+00:00')

    def test_missing_path_is_retained_and_delete_never_removes_video(self):
        record=self.save()
        self.video.rename(self.root/'moved.mp4')
        self.assertFalse(self.store.list()[0]['available'])
        self.store.delete(record['id'])
        self.assertEqual(self.store.list(),[])
        self.assertTrue((self.root/'moved.mp4').exists())
        with self.assertRaises(ValueError): self.store.delete('missing')

    def test_invalid_inputs_do_not_write_history(self):
        for value in ['',None,{},'bad\npath',str(self.root/'missing.mp4'),str(self.root)]:
            with self.subTest(value=value), self.assertRaises(ValueError): self.store.remember(value)
        for scene in ['bad\nscene','x'*101,42]:
            with self.subTest(scene=scene), self.assertRaises(ValueError): self.save(scene=scene)
        self.assertFalse(self.store.path.exists())

    def test_corrupt_file_is_never_silently_overwritten(self):
        self.store.path.parent.mkdir(parents=True)
        for content in ['broken','[]','null','{"version":1,"videos":[{}]}']:
            self.store.path.write_text(content)
            with self.assertRaisesRegex(ValueError,'原文件未覆盖'):self.save()
            self.assertEqual(self.store.path.read_text(),content)

    def test_concurrent_uses_keep_one_record_and_all_counts(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _:self.save(used=True),range(20)))
        self.assertEqual(len(self.store.list()),1)
        self.assertEqual(self.store.list()[0]['useCount'],20)

    def request(self, method, route, payload=None, activated=True):
        from backend import server
        body=json.dumps(payload or {}).encode()
        raw=f'{method} {route} HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body
        output=bytearray()
        connection=SimpleNamespace(makefile=lambda *_:io.BytesIO(raw),sendall=output.extend)
        with patch.object(server,'inference_video_history',self.store), patch.object(server.license_store,'status',return_value={'activated':activated,'role':'user','message':'activation required'}):
            server.AppHandler(connection,('127.0.0.1',1),SimpleNamespace())
        head,body=bytes(output).split(b'\r\n\r\n',1)
        return int(head.split()[1]),json.loads(body)

    def test_api_save_list_delete_license_and_invalid_file(self):
        endpoint='/api/inference-video-history'
        self.assertEqual(self.request('POST',endpoint,{'path':str(self.video)},False)[0],403)
        self.assertEqual(self.request('GET',endpoint,activated=False)[0],403)
        self.assertEqual(self.request('POST',endpoint,{'path':'missing.mp4'})[0],400)
        self.assertEqual(self.request('POST',endpoint,{'path':str(self.video),'scene':'车间A','station':'1'})[0],200)
        status,payload=self.request('GET',endpoint)
        self.assertEqual(status,200)
        record=payload['videos'][0]
        self.assertEqual(record['scene'],'车间A')
        route=endpoint+'/'+record['id']
        self.assertEqual(self.request('DELETE',route,activated=False)[0],403)
        self.assertEqual(self.request('DELETE',route)[0],200)
        self.assertTrue(self.video.exists())

    def test_accepted_starts_record_both_modes_and_write_failure_does_not_stop_task(self):
        from backend import server
        payload={'paths':{'local_video_file':str(self.video),'video_file':str(self.video)},'parameters':{'test_video_scene':'TCL','test_video_station':'工位2'}}
        with patch.object(server.platform_task_manager,'start',side_effect=lambda *_:SimpleNamespace(id='task')), patch.object(server.platform_task_manager,'set_stop_callback'), patch.object(server.star_model_history,'remember_payload'), patch.object(server.remote_inference_manager,'start',return_value={'id':'remote'}), patch.object(server.local_pt_inference_manager,'start',side_effect=lambda *a,**k:{'id':'local'}):
            self.assertEqual(self.request('POST','/api/remote-inference/start',payload)[0],201)
            self.assertEqual(self.request('POST','/api/local-pt-inference/start',payload)[0],201)
            self.assertEqual(self.store.list()[0]['useCount'],2)
            with patch.object(self.store,'remember_payload',side_effect=OSError('disk full')):
                status,response=self.request('POST','/api/local-pt-inference/start',payload)
            self.assertEqual(status,201)
            self.assertIn('disk full',response['videoHistoryWarning'])

    def test_rejected_start_does_not_create_record(self):
        from backend import server
        for endpoint,manager in [('/api/local-pt-inference/start',server.local_pt_inference_manager),('/api/remote-inference/start',server.remote_inference_manager)]:
            with patch.object(server.platform_task_manager,'start',return_value=SimpleNamespace(id='task')),patch.object(server.platform_task_manager,'finish'),patch.object(server,'record_finished_run'),patch.object(manager,'start',side_effect=ValueError('invalid model')):
                status,_=self.request('POST',endpoint,{'paths':{'video_file':str(self.video),'local_video_file':str(self.video)},'parameters':{}})
            self.assertEqual(status,400)
            self.assertEqual(self.store.list(),[])
