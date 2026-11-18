"""Successful STAR naming ledger, independent of the five-item run history."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import threading
import uuid

from .platform_paths import platform_config_dir

TITLE = re.compile(r'(?P<line>[A-Za-z][A-Za-z0-9_]{0,31})-(?P<station>[1-9][0-9]{0,3})#(?P<major>[1-9][0-9]{0,3})\.(?P<minor>0|[1-9][0-9]{0,3})\.(?P<patch>0|[1-9][0-9]{0,3})\.(?P<day>[0-9]{4})\Z')


def parse_title(value):
    match = TITLE.fullmatch(str(value))
    if not match:
        return None
    parts = match.groupdict()
    try:
        datetime.strptime('2000' + parts['day'], '%Y%m%d')
    except ValueError:
        return None
    return {'line': parts['line'].upper(), 'station': int(parts['station']),
            'version': [int(parts[key]) for key in ('major', 'minor', 'patch')], 'day': parts['day']}


class PackageNamingStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()

    def _read(self):
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data['records'], list):
                raise ValueError()
            for record in data['records']:
                if not isinstance(record, dict) or not parse_title(record['title']):
                    raise ValueError()
                for key in ('id', 'createdAt', 'output', 'plan', 'note'):
                    if not isinstance(record[key], str): raise ValueError()
                if not isinstance(record['labels'], list) or not all(isinstance(label, str) for label in record['labels']):
                    raise ValueError()
            return data['records']
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError('模型命名台账读取失败，原文件未覆盖；请检查 star_package_versions.json。') from exc

    def payload(self, line='', station=''):
        with self.lock:
            records = sorted(reversed(self._read()), key=lambda r: r['createdAt'], reverse=True)
            grouped = {}
            selected = []
            for record in records:
                parsed = parse_title(record['title'])
                key = f"{parsed['line']}-{parsed['station']}"
                previous = grouped.get(key)
                if previous is None:
                    grouped[key] = {'key': key, 'line': parsed['line'], 'station': parsed['station'],
                                    'latest': record, 'version': parsed['version'], 'count': 1}
                else:
                    previous['count'] += 1
                    if tuple(parsed['version']) > tuple(previous['version']):
                        previous.update(latest=record, version=parsed['version'])
                if str(line).upper() == parsed['line'] and str(station) == str(parsed['station']):
                    selected.append(record)
            return {'series': list(grouped.values()), 'records': selected[:30], 'count': len(selected)}

    def remember(self, title, plan, output, labels, note=''):
        if not parse_title(title):
            return False  # Old/manual naming remains supported, without guessed versions.
        record = {'id': uuid.uuid4().hex, 'title': str(title), 'plan': str(plan), 'output': str(output),
                  'labels': list(labels), 'note': str(note).strip()[:500],
                  'createdAt': datetime.now().astimezone().isoformat(timespec='milliseconds')}
        with self.lock:
            records = self._read()
            records.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f'.{self.path.name}.{uuid.uuid4().hex}.tmp')
            try:
                temporary.write_text(json.dumps({'version': 1, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return True


package_naming_store = PackageNamingStore(platform_config_dir() / 'star_package_versions.json')
