import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend import video_clip as clip
from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.task_manager import TaskCancelled


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg/ffprobe required')
class VideoClipReplaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / '中文_长关键帧视频.mp4'
        frames=[]
        for index in range(100):
            frame=np.zeros((64,128,3),dtype=np.uint8)
            for bit in range(8): frame[:,bit*16:(bit+1)*16,:]=255 if index & (1<<bit) else 0
            frames.append(frame)
        result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','128x64','-framerate','25','-i','pipe:0','-f','lavfi','-i','sine=frequency=500:duration=4','-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','ultrafast','-crf','0','-g','100','-keyint_min','100','-sc_threshold','0','-c:a','aac','-t','4',str(self.source)],input=b''.join(frame.tobytes() for frame in frames),capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.original=self.source.read_bytes()
        self.logs=[]

    def context(self, **changes):
        parameters={'clip_save_mode':clip.REPLACE_SOURCE,'clip_replace_confirmed':True,'clip_exact_time':True,
                    'clip_mode':'指定开始和结束时间','range_start_time':'1.131','range_end_time':'2.379'}
        parameters.update(changes)
        return RunContext('video_clip',{'video_file':self.source},parameters,self.logs.append)

    def frame_ids(self, source):
        capture=cv2.VideoCapture(str(source));ids=[]
        try:
            while True:
                ok,frame=capture.read()
                if not ok:break
                ids.append(sum(1<<bit for bit in range(8) if float(frame[32,bit*16+8].mean())>128))
        finally:capture.release()
        return ids

    def assert_source_unchanged(self):
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual(list(self.root.glob('.video_clip_staging_*')),[])

    def test_replace_preserves_exact_frame_ids_filename_original_backup_and_audio(self):
        result=clip.run_video_clip(self.context())
        self.assertEqual(result['outputFiles'],[str(self.source)])
        self.assertEqual(self.frame_ids(self.source),list(range(29,60)))
        backup=Path(result['backupFiles'][0])
        self.assertEqual(backup.read_bytes(),self.original)
        self.assertEqual(backup.name,self.source.name)
        self.assertIn('.video_clip_backups',backup.parts)
        self.assertAlmostEqual(result['details'][0]['actualDurationSec'],31/25,places=5)
        self.assertTrue(clip._has_audio('ffmpeg',self.source,self.context()))

    def test_save_as_exact_mode_overrides_saved_fast_setting_and_never_changes_source(self):
        context=self.context(clip_save_mode=clip.SAVE_AS,encoding_mode=clip.COPY_MODE)
        context.paths['output_folder']=self.root/'output'
        result=clip.run_video_clip(context)
        output=Path(result['outputFiles'][0])
        self.assertEqual(self.frame_ids(output),list(range(29,60)))
        self.assertEqual(result['details'][0]['encodingMode'],clip.PRECISE_MODE)
        self.assertEqual(result['backupFiles'],[])
        self.assert_source_unchanged()

    def test_end_boundary_excluded_and_duration_or_empty_end_schemes_work(self):
        for parameters,expected in [({'range_start_time':'1','range_end_time':'2'},list(range(25,50))),
                                    ({'range_start_time':'3','range_end_time':''},list(range(75,100))),
                                    ({'clip_mode':'指定开始时间和持续时长','duration_start_time':'1','clip_duration':'0.4'},list(range(25,35)))]:
            result=clip.run_video_clip(self.context(clip_save_mode=clip.SAVE_AS,**parameters))
            self.assertEqual(self.frame_ids(result['outputFiles'][0]),expected)

    def test_remove_intervals_replace_keeps_only_selected_remaining_frames(self):
        result=clip.run_video_clip(self.context(clip_mode=clip.REMOVE_MODE,remove_time_ranges='0.4-1\n1.4-2\n3-'))
        self.assertEqual(self.frame_ids(self.source),list(range(0,10))+list(range(25,35))+list(range(50,75)))
        self.assertEqual(Path(result['backupFiles'][0]).read_bytes(),self.original)

    def test_split_uses_precise_single_encode_for_test_complement(self):
        result=clip.run_video_clip(self.context(clip_save_mode=clip.SAVE_AS,range_start_time='1',range_end_time='2',split_train_test=True))
        self.assertEqual(self.frame_ids(result['details'][0]['trainOutput']),list(range(25,50)))
        self.assertEqual(self.frame_ids(result['details'][0]['testOutput']),list(range(25))+list(range(50,100)))
        self.assert_source_unchanged()

    def test_second_replace_creates_distinct_backup_and_never_overwrites_previous_backup(self):
        first=clip.run_video_clip(self.context(range_start_time='0',range_end_time='3'))
        intermediate=self.source.read_bytes()
        second=clip.run_video_clip(self.context(range_start_time='0',range_end_time='1'))
        self.assertNotEqual(first['backupFiles'][0],second['backupFiles'][0])
        self.assertEqual(Path(first['backupFiles'][0]).read_bytes(),self.original)
        self.assertEqual(Path(second['backupFiles'][0]).read_bytes(),intermediate)

    def test_confirmation_required_and_split_conflict_rejected_before_encoding(self):
        for changes in [{'clip_replace_confirmed':False},{'clip_replace_confirmed':'true'},{'split_train_test':True}]:
            with self.assertRaises(ValueError),patch.object(clip,'_run_process') as runner:
                clip.run_video_clip(self.context(**changes))
            runner.assert_not_called()
            self.assert_source_unchanged()

    def test_failed_encoding_or_validation_never_replaces_original(self):
        for target in ['_clip_one','_validate_precise_output']:
            with patch.object(clip,target,side_effect=RuntimeError('injected failure')),self.assertRaisesRegex(RuntimeError,'injected failure'):
                clip.run_video_clip(self.context())
            self.assert_source_unchanged()
            self.assertFalse((self.root/'.video_clip_backups').exists())

    def test_cancel_during_validation_preserves_original_and_cleans_staging(self):
        context=self.context();context.stop_event=threading.Event()
        validate=clip._validate_precise_output
        def cancel(*args,**kwargs):
            result=validate(*args,**kwargs);context.stop_event.set();return result
        with patch.object(clip,'_validate_precise_output',side_effect=cancel),self.assertRaises(TaskCancelled):
            clip.run_video_clip(context)
        self.assert_source_unchanged()

    def test_backup_failure_does_not_touch_original(self):
        with patch.object(clip.shutil,'copystat',side_effect=OSError('disk failure')),self.assertRaisesRegex(RuntimeError,'disk failure'):
            clip.run_video_clip(self.context())
        self.assert_source_unchanged()
        self.assertEqual(list((self.root/'.video_clip_backups').rglob('*.mp4')),[])

    def test_atomic_commit_failure_keeps_original_and_recoverable_backup(self):
        original_replace=Path.replace
        def replace(source,target):
            if Path(target)==self.source: raise PermissionError('file is in use')
            return original_replace(source,target)
        with patch.object(Path,'replace',replace),self.assertRaisesRegex(RuntimeError,'原视频替换失败'):
            clip.run_video_clip(self.context())
        self.assert_source_unchanged()
        backups=list((self.root/'.video_clip_backups').rglob('*.mp4'))
        self.assertEqual(len(backups),1)
        self.assertEqual(backups[0].read_bytes(),self.original)

    def test_source_change_by_other_program_cancels_commit(self):
        original_clip=clip._clip_one
        def changed(*args,**kwargs):
            output=original_clip(*args,**kwargs)
            with self.source.open('ab') as stream:stream.write(b'foreign change')
            return output
        with patch.object(clip,'_clip_one',side_effect=changed),self.assertRaisesRegex(RuntimeError,'其他程序修改'):
            clip.run_video_clip(self.context())
        self.assertEqual(self.source.read_bytes(),self.original+b'foreign change')

    def test_exact_mode_rejects_out_of_bounds_instead_of_silently_clamping(self):
        for changes in [{'range_end_time':'9'}, {'clip_mode':clip.REMOVE_MODE,'remove_time_ranges':'1-9'}]:
            with self.assertRaisesRegex(RuntimeError,'超出'):
                clip.run_video_clip(self.context(**changes))
            self.assert_source_unchanged()

    def test_mkv_replacement_keeps_actual_container_and_backup_name(self):
        mkv=self.root/'camera.mkv'
        completed=subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.source),'-c','copy',str(mkv)],capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        context=self.context(range_start_time='1',range_end_time='2');context.paths['video_file']=mkv
        result=clip.run_video_clip(context)
        self.assertEqual(self.frame_ids(mkv),list(range(25,50)))
        self.assertEqual(Path(result['backupFiles'][0]).name,'camera.mkv')
        self.assertEqual(mkv.read_bytes()[:4],b'\x1aE\xdf\xa3')

    def test_variable_frame_rate_keeps_original_frame_identity_without_duplication(self):
        vfr=self.root/'variable.mp4'
        completed=subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.source),'-vf',"select='not(mod(n,3))+not(mod(n,7))'",'-vsync','vfr','-c:v','libx264','-crf','0','-an',str(vfr)],capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(vfr)],capture_output=True,text=True)
        times=[float(frame['best_effort_timestamp_time']) for frame in json.loads(probe.stdout)['frames']]
        expected=[frame for frame,time in zip(self.frame_ids(vfr),times) if 0.31<=time<2.41]
        context=self.context(clip_save_mode=clip.SAVE_AS,range_start_time='0.31',range_end_time='2.41');context.paths['video_file']=vfr
        result=clip.run_video_clip(context)
        self.assertEqual(self.frame_ids(result['outputFiles'][0]),expected)

    def test_nonzero_container_start_time_does_not_shift_ui_timeline(self):
        shifted=self.root/'offset.mp4'
        completed=subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.source),'-vf','setpts=PTS+10/TB','-vsync','vfr','-c:v','libx264','-crf','0','-an',str(shifted)],capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        context=self.context(clip_save_mode=clip.SAVE_AS,range_start_time='3',range_end_time='');context.paths['video_file']=shifted
        result=clip.run_video_clip(context)
        self.assertEqual(self.frame_ids(result['outputFiles'][0]),list(range(75,100)))

    def test_symlink_and_unsupported_replace_container_are_rejected(self):
        link=self.root/'linked.mp4';link.symlink_to(self.source)
        context=self.context();context.paths['video_file']=link
        with self.assertRaisesRegex(ValueError,'符号链接'):clip.run_video_clip(context)
        avi=self.root/'camera.avi';avi.write_bytes(b'original-avi')
        context.paths['video_file']=avi
        with self.assertRaisesRegex(ValueError,'其他容器'):clip.run_video_clip(context)
        self.assert_source_unchanged()

    def test_mov_and_m4v_replacement_use_real_compatible_containers(self):
        for suffix in ['.mov','.m4v']:
            video=self.root/f'camera{suffix}'
            completed=subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.source),'-c','copy','-f','mov' if suffix=='.mov' else 'ipod',str(video)],capture_output=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            original=video.read_bytes()
            context=self.context(range_start_time='1',range_end_time='2');context.paths['video_file']=video
            result=clip.run_video_clip(context)
            self.assertEqual(self.frame_ids(video),list(range(25,50)))
            self.assertEqual(Path(result['backupFiles'][0]).read_bytes(),original)

    def test_batch_cancel_keeps_completed_replacement_and_preserves_unprocessed_video(self):
        second=self.root/'zzz_second.mp4';second.write_bytes(self.original)
        context=self.context(range_start_time='1',range_end_time='2')
        context.paths={'video_folder':self.root};context.stop_event=threading.Event()
        def report(message):
            self.logs.append(message)
            if '原视频已安全替换' in message:context.stop_event.set()
        context.report=report
        with self.assertRaises(TaskCancelled):clip.run_video_clip(context)
        # zzz sorts before the Chinese name; exactly one commit is retained.
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual(self.frame_ids(second),list(range(25,50)))
        backups=list((self.root/'.video_clip_backups').rglob('*.mp4'))
        self.assertEqual(len(backups),1)
        self.assertEqual(backups[0].read_bytes(),self.original)


class ClipSaveModeCatalogTests(unittest.TestCase):
    def test_old_catalog_gets_safe_options_once_without_resetting_existing_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            file=Path(temporary)/'functions.json'
            items=FunctionCatalog(file).list()
            item=next(item for item in items if item['id']=='video_clip')
            item['parameters']=[field for field in item['parameters'] if field['id'] not in {'clip_save_mode','clip_exact_time'}]
            next(field for field in item['parameters'] if field['id']=='range_start_time')['default']='1:30'
            file.write_text(json.dumps(items))
            for _ in range(2):
                item=next(item for item in FunctionCatalog(file).list() if item['id']=='video_clip')
                fields={field['id']:field for field in item['parameters']}
                self.assertEqual(fields['clip_save_mode']['default'],clip.SAVE_AS)
                self.assertIs(fields['clip_exact_time']['default'],True)
                self.assertEqual(fields['range_start_time']['default'],'1:30')
                self.assertEqual(len(fields),len(item['parameters']))
