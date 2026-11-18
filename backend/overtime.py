"""Local, reviewable overtime ledger. GET/polling NEVER count as user activity."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import threading
import uuid

from .work_clock import CHINA_CALENDAR, normalize_work_clock

SCHEDULE_KEYS = ('startTime', 'endTime', 'weekendMode', 'singleRestDay', 'followChinaCalendar')
MIN_OVERTIME_SECONDS = 3600


def _stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec='milliseconds')


def _date(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError()
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('请填写有效的开始、结束时间。') from exc


def _zone(offset):
    if type(offset) is not int or not -840 <= offset <= 840:
        raise ValueError('本机时区偏移不正确。')
    return timezone(timedelta(minutes=offset))


def _working_day(day, settings, calendar):
    key = day.isoformat()
    if settings['followChinaCalendar']:
        if key in calendar['workdays']:
            return True
        if any(item['start'] <= key <= item['end'] for item in calendar['holidays']):
            return False
    js_weekday = (day.weekday() + 1) % 7
    return js_weekday not in (0, 6) if settings['weekendMode'] == 'double' else js_weekday != settings['singleRestDay']


def _work_intervals(start, end, settings, offset, calendar):
    zone = _zone(offset)
    day = start.astimezone(zone).date() - timedelta(days=1)
    last = end.astimezone(zone).date()
    if (last - day).days > 370:
        raise ValueError('单段记录时间过长，请拆分为一年以内的记录。')
    while day <= last:
        if _working_day(day, settings, calendar):
            begin = datetime.combine(day, time.fromisoformat(settings['startTime']), zone)
            finish = datetime.combine(day, time.fromisoformat(settings['endTime']), zone)
            if finish <= begin:
                finish += timedelta(days=1)
            yield begin.astimezone(timezone.utc), finish.astimezone(timezone.utc)
        day += timedelta(days=1)


def overtime_seconds(start, end, settings, offset, calendar=CHINA_CALENDAR):
    seconds = max(0, (end - start).total_seconds())
    for begin, finish in _work_intervals(start, end, settings, offset, calendar):
        seconds -= max(0, (min(end, finish) - max(start, begin)).total_seconds())
    return max(0, int(seconds))


def _in_work(now, settings, offset, calendar):
    return any(begin <= now < finish for begin, finish in _work_intervals(now, now, settings, offset, calendar))


class OvertimeStore:
    def __init__(self, path: Path, settings, now=None, calendar=None):
        self.path, self.settings = path, settings
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.calendar = calendar if calendar is not None else CHINA_CALENDAR
        self.lock = threading.RLock()

    def _load(self):
        if not self.path.exists():
            return {'version': 1, 'revision': 0, 'records': [], 'offsetMinutes': 480, 'manualStopDate': ''}
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('version') != 1 or not isinstance(data.get('records'), list):
                raise ValueError()
            for record in data['records']:
                if record['status'] not in ('running', 'pending', 'confirmed'):
                    raise ValueError()
                _date(record['startAt']); _date(record['lastActivityAt'])
                if record['endAt'] is not None: _date(record['endAt'])
                _zone(record['offsetMinutes'])
            if sum(r['status'] == 'running' for r in data['records']) > 1:
                raise ValueError()
            return data
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ValueError('加班记录读取失败；原文件未覆盖，请先备份并检查 overtime.json。') from exc

    def _save(self, data):
        data['revision'] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)

    def _settle(self, data, settings, now):
        active = next((r for r in data['records'] if r['status'] == 'running'), None)
        if not active:
            return
        start, last = _date(active['startAt']), _date(active['lastActivityAt'])
        deadline = last + timedelta(minutes=settings['overtimeIdleMinutes'])
        # Earliest stop wins: never extend an abandoned evening to next morning.
        # Recovery after months offline only needs the original idle window,
        # not every intervening workday up to the reopening date.
        search_end = min(now, deadline)
        boundaries = [begin for begin, _ in _work_intervals(start, search_end, active['schedule'], active['offsetMinutes'], self.calendar) if start < begin <= search_end]
        boundary = min(boundaries) if boundaries else None
        end, reason = None, ''
        if now >= deadline and (boundary is None or deadline <= boundary):
            end, reason = last, 'idle'
        elif boundary is not None:
            end, reason = boundary, 'work-start'
        elif not settings['overtimeEnabled'] or active['schedule'] != {key: settings[key] for key in SCHEDULE_KEYS}:
            end, reason = min(now, last), 'settings-changed'
        if end is not None:
            active.update(status='pending', endAt=_stamp(max(start, end)), reason=reason)

    def _can_continue(self, data, record, settings, now):
        if record['status'] != 'pending' or not settings['overtimeEnabled']:
            return False
        # Do not join unrelated days/previous pending sessions or bridge a normal shift.
        pending = [r for r in data['records'] if r['status'] == 'pending']
        if max(pending, key=lambda r: r['startAt'])['id'] != record['id']:
            return False
        end = _date(record['endAt'])
        if not end <= now <= end + timedelta(hours=24):
            return False
        if record['schedule'] != {key: settings[key] for key in SCHEDULE_KEYS}:
            return False
        return not any(min(now, finish) > max(end, begin) or begin == now
                       for begin, finish in _work_intervals(end, now, record['schedule'], record['offsetMinutes'], self.calendar))

    def _payload(self, data, settings, now):
        records = []
        zone = _zone(data['offsetMinutes'])
        local = now.astimezone(zone)
        today = datetime.combine(local.date(), time(), zone).astimezone(timezone.utc)
        month = datetime.combine(local.date().replace(day=1), time(), zone).astimezone(timezone.utc)
        totals = {'todaySeconds': 0, 'monthSeconds': 0, 'totalSeconds': 0}
        for raw in sorted(data['records'], key=lambda r: r['startAt'], reverse=True):
            record = deepcopy(raw)
            start, end = _date(record['startAt']), _date(record['endAt']) if record['endAt'] else now
            record['seconds'] = overtime_seconds(start, end, record['schedule'], record['offsetMinutes'], self.calendar)
            record['canContinue'] = self._can_continue(data, record, settings, now)
            if record['status'] == 'confirmed':
                # Keep the raw ledger intact for overlap checks and compatibility,
                # but short sessions are neither history entries nor counted time.
                if record['seconds'] < MIN_OVERTIME_SECONDS:
                    continue
                totals['totalSeconds'] += record['seconds']
                if end > today:
                    totals['todaySeconds'] += overtime_seconds(max(start, today), end, record['schedule'], record['offsetMinutes'], self.calendar)
                if end > month:
                    totals['monthSeconds'] += overtime_seconds(max(start, month), end, record['schedule'], record['offsetMinutes'], self.calendar)
            records.append(record)
        return {'revision': data['revision'], 'serverTime': _stamp(now), 'enabled': settings['overtimeEnabled'],
                'idleMinutes': settings['overtimeIdleMinutes'], 'offsetMinutes': data['offsetMinutes'],
                'workingNow': _in_work(now, settings, data['offsetMinutes'], self.calendar),
                'manualStoppedToday': data['manualStopDate'] == local.date().isoformat(),
                'records': records, 'active': next((r for r in records if r['status'] == 'running'), None),
                'pendingCount': sum(r['status'] == 'pending' for r in records), **totals}

    def request(self, action='status', payload=None):
        if payload is not None and not isinstance(payload, dict):
            raise ValueError('加班记录请求格式不正确。')
        payload = payload or {}
        with self.lock:
            now = self.now().astimezone(timezone.utc)
            settings = normalize_work_clock(self.settings())
            data = self._load()
            before = deepcopy(data)
            self._settle(data, settings, now)
            # Persist timeout recovery even when a stale UI action later fails.
            if data != before:
                self._save(data)
            baseline = deepcopy(data)
            if action == 'activity' or action == 'start':
                offset = payload.get('offsetMinutes', data['offsetMinutes'])
                _zone(offset)
                data['offsetMinutes'] = offset
                observed = _date(payload['at']) if action == 'activity' and payload.get('at') else now
                if abs((observed - now).total_seconds()) > 120:
                    observed = now  # Device clock skew must not create future/ancient time.
                observed = min(now, observed)
                active = next((r for r in data['records'] if r['status'] == 'running'), None)
                if active:
                    active['lastActivityAt'] = _stamp(max(observed, _date(active['lastActivityAt'])))
                elif (settings['overtimeEnabled'] and not _in_work(observed, settings, offset, self.calendar)
                      and not _in_work(now, settings, offset, self.calendar)):
                    stopped = data['manualStopDate'] == now.astimezone(_zone(offset)).date().isoformat()
                    if action == 'start' or not stopped:
                        data['manualStopDate'] = ''
                        ends = [_date(r['endAt']) for r in data['records'] if r['endAt'] is not None]
                        if ends: observed = max(observed, max(ends))
                        data['records'].append({'id': uuid.uuid4().hex, 'status': 'running', 'startAt': _stamp(observed),
                            'lastActivityAt': _stamp(observed), 'endAt': None, 'reason': 'activity', 'offsetMinutes': offset,
                            'schedule': {key: settings[key] for key in SCHEDULE_KEYS}})
                elif action == 'start':
                    raise ValueError('当前是正常上班时间，或尚未开启加班记录。')
            elif action == 'finish':
                record = next((r for r in data['records'] if r['status'] == 'running' and r['id'] == payload.get('id')), None)
                if not record:
                    raise ValueError('该段已暂时结束或状态已更新，请刷新后确认待确认记录。')
                record.update(status='confirmed', endAt=_stamp(now), reason='manual')
                data['manualStopDate'] = now.astimezone(_zone(record['offsetMinutes'])).date().isoformat()
            elif action == 'continue':
                record = next((r for r in data['records'] if r['id'] == payload.get('id')), None)
                if not record or not self._can_continue(data, record, settings, now):
                    raise ValueError('无法直接衔接到现在（可能跨过正常班次或超过24小时），请修正并确认原结束时间。')
                others = [r for r in data['records'] if r['id'] != record['id'] and
                          _date(r['startAt']) <= now and (r['endAt'] is None or _date(r['endAt']) > _date(record['endAt']))]
                if any(r['status'] != 'running' or r['schedule'] != record['schedule'] for r in others):
                    raise ValueError('衔接时段存在其他已结束记录，请先修正时间，避免重复统计。')
                data['records'] = [r for r in data['records'] if r not in others]
                record.update(status='running', endAt=None, lastActivityAt=_stamp(now), reason='continued')
                data['manualStopDate'] = ''
            elif action == 'confirm':
                record = next((r for r in data['records'] if r['id'] == payload.get('id')), None)
                if not record or record['status'] == 'running':
                    raise ValueError('请先结束当前加班，再修正时间。')
                start = _date(payload.get('startAt', record['startAt']))
                end = _date(payload.get('endAt', record['endAt']))
                if end < start or end > now or start.year < 2000 or end - start > timedelta(days=31):
                    raise ValueError('结束不能早于开始、不能在未来；单段最多31天。')
                for other in data['records']:
                    if other['id'] == record['id']: continue
                    a, b = _date(other['startAt']), _date(other['endAt']) if other['endAt'] else now
                    if min(end, b) > max(start, a):
                        raise ValueError('与其他加班记录重叠，请修正时间，避免重复统计。')
                record.update(startAt=_stamp(start), endAt=_stamp(end), status='confirmed', reason='reviewed')
            elif action != 'status':
                raise ValueError('未知的加班记录操作。')
            if data != baseline:
                self._save(data)
            return self._payload(data, settings, now)
