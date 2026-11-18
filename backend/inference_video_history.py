"""Shared local-video references for local and remote inference; never copies video data."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid


class InferenceVideoHistoryStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    @staticmethod
    def _text(value, label, maximum=100):
        if not isinstance(value, str) or len(value) > maximum or any(char in value for char in '\x00\r\n'):
            raise ValueError(f'{label}应为不含换行的文本，最多{maximum}个字符。')
        return value.strip()

    def _read(self):
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('videos'), list):
                raise ValueError()
            for row in data['videos']:
                if not isinstance(row, dict): raise ValueError()
                for field in ('id', 'path', 'scene', 'station', 'createdAt', 'updatedAt'):
                    if not isinstance(row[field], str): raise ValueError()
                for field in ('createdAt', 'updatedAt', 'lastUsedAt'):
                    if field == 'lastUsedAt' and row[field] is None: continue
                    if datetime.fromisoformat(row[field]).tzinfo is None: raise ValueError()
                if not row['id'] or not row['path'] or type(row['useCount']) is not int or row['useCount'] < 0:
                    raise ValueError()
            return data['videos']
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError('测试视频记录读取失败，原文件未覆盖；请备份并检查 inference_video_history.json。') from exc

    def _write(self, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f'.{self.path.name}.{uuid.uuid4().hex}.tmp')
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump({'version': 1, 'videos': rows}, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def list(self):
        with self._lock:
            rows = sorted(self._read(), key=lambda row: datetime.fromisoformat(row['updatedAt']), reverse=True)
            return [{**row, 'available': Path(row['path']).is_file()} for row in rows]

    def remember(self, path, scene=None, station=None, *, used=False, require_file=True):
        path = self._text(path, '视频路径', 8192)
        if not path:
            raise ValueError('请先填写本地视频路径。')
        source = Path(path).expanduser().resolve()
        if require_file and not source.is_file():
            raise ValueError(f'视频文件不存在或不可访问：{source}')
        scene = None if scene is None else self._text(scene, '场景')
        station = None if station is None else self._text(station, '工位')
        normalized = os.path.normcase(str(source))
        with self._lock:
            rows = self._read()
            previous = next((row for row in rows if os.path.normcase(row['path']) == normalized), {})
            now = datetime.now(timezone.utc).isoformat(timespec='microseconds')
            record = {
                'id': previous.get('id', uuid.uuid4().hex), 'path': str(source),
                'scene': previous.get('scene', '') if scene is None else scene,
                'station': previous.get('station', '') if station is None else station,
                'createdAt': previous.get('createdAt', now), 'updatedAt': now,
                'lastUsedAt': now if used else previous.get('lastUsedAt'),
                'useCount': previous.get('useCount', 0) + int(used),
            }
            self._write([record, *(row for row in rows if row['id'] != record['id'])])
            return record

    def remember_payload(self, payload, *, remote=False):
        parameters = payload.get('parameters', {})
        if not remote and parameters.get('input_mode') == '图片文件夹检测':
            return
        path = payload.get('paths', {}).get('local_video_file' if remote else 'video_file', '')
        if not path: return
        # Missing metadata in older callers must not erase an existing classification.
        self.remember(path, parameters.get('test_video_scene'), parameters.get('test_video_station'), used=True)

    def delete(self, record_id):
        with self._lock:
            rows = self._read()
            remaining = [row for row in rows if row['id'] != record_id]
            if len(remaining) == len(rows):
                raise ValueError('视频记录不存在或已被删除。')
            self._write(remaining)
