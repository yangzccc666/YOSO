from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.overtime import OvertimeStore, overtime_seconds
from backend.work_clock import DEFAULT_WORK_CLOCK


class OvertimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'overtime.json'
        self.settings = deepcopy(DEFAULT_WORK_CLOCK)
        self.calendar = {'holidays': [], 'workdays': []}
        self.at('2026-11-12T18:30')
        self.store = OvertimeStore(self.path, lambda: self.settings, lambda: self.now, self.calendar)

    def at(self, value):
        self.now = datetime.fromisoformat(value + ':00+08:00')

    def activity(self):
        return self.store.request('activity', {'offsetMinutes': 480, 'at': self.now.isoformat()})

    def idle_record(self):
        self.activity(); self.at('2026-11-12T18:40'); self.activity(); self.at('2026-11-12T19:11')
        return self.store.request()['records'][0]

    def test_loading_and_polling_do_not_start_overtime(self):
        for _ in range(5): self.assertEqual(self.store.request()['records'], [])
        self.assertFalse(self.path.exists())
        self.at('2026-11-12T10:00'); self.assertIsNone(self.activity()['active'])

    def test_idle_stops_at_last_real_activity_and_does_not_add_grace_period_to_totals(self):
        record = self.idle_record()
        self.assertEqual((record['status'], record['seconds']), ('pending', 600))
        self.assertEqual(record['endAt'], '2026-11-12T10:40:00.000+00:00')
        payload = self.store.request()
        self.assertIsNone(payload['active']); self.assertEqual(payload['pendingCount'], 1)
        self.assertEqual((payload['todaySeconds'], payload['totalSeconds']), (0, 0))

    def test_deadline_is_exact_and_get_requests_do_not_extend_last_activity(self):
        initial = self.activity()['active']
        self.at('2026-11-12T18:59'); self.assertIsNotNone(self.store.request()['active'])
        self.at('2026-11-12T19:00'); result = self.store.request()
        self.assertIsNone(result['active']); self.assertEqual(result['records'][0]['endAt'], initial['startAt'])

    def test_restart_next_day_recovers_pending_without_counting_the_night(self):
        self.activity()
        for clock in ['18:50', '19:10', '19:30', '19:50', '20:10']:
            self.at(f'2026-11-12T{clock}'); self.activity()
        self.at('2026-11-13T09:30')
        reloaded = OvertimeStore(self.path, lambda: self.settings, lambda: self.now, self.calendar)
        result = reloaded.request('activity', {'offsetMinutes': 480})
        self.assertIsNone(result['active']); self.assertEqual(result['records'][0]['seconds'], 6000)
        self.assertEqual(result['records'][0]['status'], 'pending')

    def test_return_starts_a_new_segment_and_continue_merges_without_double_counting(self):
        old = self.idle_record()
        self.at('2026-11-12T19:20'); self.assertIsNotNone(self.activity()['active'])
        result = self.store.request('continue', {'id': old['id']})
        self.assertEqual(len(result['records']), 1)
        self.assertEqual((result['active']['id'], result['active']['seconds']), (old['id'], 3000))
        self.at('2026-11-12T19:25')
        end = self.store.request('finish', {'id': old['id']})
        self.assertEqual((end['todaySeconds'], end['pendingCount']), (0, 0))
        self.assertEqual(end['records'], [])

    def test_confirm_and_correction_are_persistent_and_only_confirmed_time_counts(self):
        old = self.idle_record()
        result = self.store.request('confirm', {'id': old['id']})
        self.assertEqual(result['totalSeconds'], 0)
        self.assertEqual(result['records'], [])
        result = self.store.request('confirm', {'id': old['id'], 'endAt': '2026-11-12T19:00:00+08:00'})
        self.assertEqual(result['totalSeconds'], 0)
        reloaded = OvertimeStore(self.path, lambda: self.settings, lambda: self.now, self.calendar)
        self.assertEqual(reloaded.request()['totalSeconds'], 0)

    def test_manual_finish_suppresses_auto_restart_until_tomorrow_or_explicit_start(self):
        record = self.activity()['active']; self.at('2026-11-12T18:50')
        result = self.store.request('finish', {'id': record['id']})
        self.assertEqual(result['todaySeconds'], 0)
        self.at('2026-11-12T19:00'); self.assertIsNone(self.activity()['active'])
        self.assertTrue(self.store.request()['manualStoppedToday'])
        result = self.store.request('start', {'offsetMinutes': 480})
        self.assertIsNotNone(result['active']); self.assertFalse(result['manualStoppedToday'])

    def test_cannot_silently_continue_through_a_normal_shift(self):
        record = self.idle_record(); self.at('2026-11-13T19:00')
        self.assertFalse(self.store.request()['records'][0]['canContinue'])
        with self.assertRaisesRegex(ValueError, '正常班次'):
            self.store.request('continue', {'id': record['id']})

    def test_regular_shift_boundary_caps_overtime_even_with_recent_activity(self):
        self.at('2026-11-12T08:50'); self.activity()
        self.at('2026-11-12T08:59'); self.activity()
        self.at('2026-11-12T09:00'); result = self.activity()
        self.assertIsNone(result['active'])
        self.assertEqual((result['records'][0]['seconds'], result['records'][0]['reason']), (600, 'work-start'))

    def test_night_shift_weekends_and_makeup_days_use_the_existing_calendar_policy(self):
        self.settings.update(startTime='22:00', endTime='06:00')
        self.at('2026-11-13T02:00'); self.assertIsNone(self.activity()['active'])
        self.at('2026-11-13T07:00'); self.assertIsNotNone(self.activity()['active'])
        self.calendar['workdays'] = ['2026-11-15']
        self.at('2026-11-15T23:00'); self.assertIsNone(self.activity()['active'])

    def test_holiday_daytime_is_overtime_but_makeup_day_is_normal_work(self):
        self.calendar['holidays'] = [{'start': '2026-11-12', 'end': '2026-11-12'}]
        self.at('2026-11-12T12:00'); self.assertIsNotNone(self.activity()['active'])
        self.at('2026-11-15T10:00'); self.calendar['workdays'] = ['2026-11-15']
        self.assertIsNone(self.activity()['active'])

    def test_confirmed_daily_and_monthly_totals_split_at_midnight(self):
        self.at('2026-07-31T23:00'); record = self.activity()['active']
        for clock in ['23:20', '23:40']:
            self.at(f'2026-07-31T{clock}'); self.activity()
        self.at('2026-08-01T00:00'); self.activity()
        self.at('2026-08-01T00:10'); self.activity()
        self.at('2026-08-01T00:20'); result = self.store.request('finish', {'id': record['id']})
        self.assertEqual((result['totalSeconds'], result['todaySeconds'], result['monthSeconds']), (4800, 1200, 1200))

    def test_under_one_hour_is_hidden_but_exactly_one_hour_counts_in_full(self):
        record = self.idle_record()
        self.at('2026-11-12T20:00')
        result = self.store.request('confirm', {'id': record['id'], 'endAt': '2026-11-12T19:29:59+08:00'})
        self.assertEqual((result['records'], result['totalSeconds']), ([], 0))
        result = self.store.request('confirm', {'id': record['id'], 'endAt': '2026-11-12T19:30:00+08:00'})
        self.assertEqual(len(result['records']), 1)
        self.assertEqual((result['totalSeconds'], result['todaySeconds'], result['monthSeconds']), (3600, 3600, 3600))
        reloaded = OvertimeStore(self.path, lambda: self.settings, lambda: self.now, self.calendar)
        self.assertEqual(reloaded.request()['totalSeconds'], 3600)

    def test_separate_short_sessions_do_not_accumulate_into_one_hour(self):
        first = self.idle_record()
        self.at('2026-11-12T19:20')
        self.store.request('confirm', {'id': first['id'], 'endAt': '2026-11-12T19:05:00+08:00'})
        second = self.store.request('start', {'offsetMinutes': 480})['active']
        for clock in ['19:40', '19:55']:
            self.at(f'2026-11-12T{clock}'); self.activity()
        result = self.store.request('finish', {'id': second['id']})
        self.assertEqual((result['records'], result['totalSeconds']), ([], 0))
        self.assertTrue(result['manualStoppedToday'])

    def test_edit_rejects_future_inverted_and_overlapping_ranges_without_overwriting(self):
        old = self.idle_record(); self.at('2026-11-12T19:20'); self.activity()
        before = self.path.read_bytes()
        for fields in [{'endAt': '2026-11-12T19:30:00+08:00'}, {'endAt': '2026-11-12T18:00:00+08:00'},
                       {'endAt': '2026-11-12T19:21:00+08:00'}, {'startAt': 'invalid'}, {'endAt': '2026-11-12T19:20:00'}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.store.request('confirm', {'id': old['id'], **fields})
            self.assertEqual(self.path.read_bytes(), before)
        self.at('2026-11-12T19:25')
        with self.assertRaisesRegex(ValueError, '重叠'):
            self.store.request('confirm', {'id': old['id'], 'endAt': '2026-11-12T19:24:00+08:00'})

    def test_disabling_or_changing_schedule_preserves_active_record_for_review(self):
        self.activity(); self.at('2026-11-12T18:40'); self.activity()
        self.settings['overtimeEnabled'] = False
        result = self.store.request()
        self.assertEqual(result['records'][0]['status'], 'pending')
        self.assertIsNone(self.activity()['active'])
        self.assertEqual(result['totalSeconds'], 0)

    def test_idle_timeout_can_be_changed_and_old_settings_load_new_defaults(self):
        self.settings.pop('overtimeIdleMinutes'); self.settings.pop('overtimeEnabled')
        self.assertEqual(self.store.request()['idleMinutes'], 30)
        self.activity(); self.settings['overtimeIdleMinutes'] = 10
        self.at('2026-11-12T18:40'); self.assertEqual(self.store.request()['pendingCount'], 1)

    def test_concurrent_windows_do_not_start_multiple_active_records(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: self.activity(), range(12)))
        self.assertEqual(len(self.store.request()['records']), 1)

    def test_corrupted_record_file_is_not_silently_reset(self):
        self.path.write_text('broken json')
        with self.assertRaisesRegex(ValueError, '原文件未覆盖'): self.store.request()
        self.assertEqual(self.path.read_text(), 'broken json')

    def test_reopen_years_later_still_recovers_the_original_idle_window(self):
        self.activity(); self.at('2026-11-12T18:40'); self.activity()
        self.at('2029-11-12T10:00')
        record = self.store.request()['records'][0]
        self.assertEqual((record['status'], record['seconds']), ('pending', 600))

    def test_delayed_activity_after_midnight_finish_does_not_restart_or_overlap(self):
        self.at('2026-11-14T23:59'); original = self.now.isoformat(); record = self.activity()['active']
        self.at('2026-11-15T00:01'); self.store.request('finish', {'id': record['id']})
        result = self.store.request('activity', {'offsetMinutes': 480, 'at': original})
        self.assertIsNone(result['active']); self.assertEqual(result['totalSeconds'], 0)

    def test_corrected_ranges_exclude_regular_work_hours(self):
        start = datetime.fromisoformat('2026-11-12T18:00:00+08:00')
        end = datetime.fromisoformat('2026-11-13T19:00:00+08:00')
        self.assertEqual(overtime_seconds(start, end, self.settings, 480, self.calendar), 16*3600)

    def test_api_requires_activation_and_does_not_count_get_as_activity(self):
        from backend import server
        def request(activated, method, route, payload=None):
            body = json.dumps(payload or {}).encode()
            raw = f'{method} {route} HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            with patch.object(server, 'overtime_store', self.store), patch.object(server.license_store, 'status', return_value={'activated': activated, 'role': 'user', 'message': 'activation required'}):
                server.AppHandler(connection, ('127.0.0.1', 1), SimpleNamespace())
            head, data = bytes(output).split(b'\r\n\r\n', 1)
            return int(head.split()[1]), json.loads(data)
        self.assertEqual(request(False, 'POST', '/api/overtime/activity')[0], 403)
        self.assertEqual(request(True, 'GET', '/api/overtime')[1]['records'], [])
        self.assertEqual(request(True, 'POST', '/api/overtime/activity', {'offsetMinutes': 480})[1]['active']['status'], 'running')
