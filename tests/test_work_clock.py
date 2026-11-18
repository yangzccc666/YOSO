import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from backend.app_settings import AppSettingsStore
from backend.work_clock import CHINA_CALENDAR, DEFAULT_WORK_CLOCK


class WorkClockTests(unittest.TestCase):
    def test_save_survives_restart_and_other_platform_settings_do_not_overwrite_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            store = AppSettingsStore(path)
            self.assertEqual(store.work_clock_payload()['settings'], DEFAULT_WORK_CLOCK)
            store.update({'externalApps': [{'id': 'antigravity', 'executable': '/opt/my-tool'}], 'aiPython': '/opt/python'})
            configured = {**DEFAULT_WORK_CLOCK, 'startTime': '22:00', 'endTime': '06:00', 'displayMode': 'both', 'weekendMode': 'single', 'singleRestDay': 3}
            store.update_work_clock(configured)
            self.assertEqual(store.payload()['aiPython'], '/opt/python')
            self.assertEqual(next(app for app in store.payload()['externalApps'] if app['id'] == 'antigravity')['executable'], '/opt/my-tool')
            settings = store.payload()
            settings['aiPython'] = '/opt/other-python'
            store.update(settings)
            self.assertEqual(AppSettingsStore(path).work_clock_payload()['settings'], configured)
            self.assertEqual(store.payload()['aiPython'], '/opt/other-python')

    def test_invalid_time_mode_and_flags_do_not_change_saved_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            store = AppSettingsStore(path)
            store.update_work_clock(DEFAULT_WORK_CLOCK)
            before = path.read_bytes()
            for key, value in [('startTime', '25:00'), ('endTime', '09:00'), ('displayMode', 'unknown'), ('singleRestDay', True), ('showWeekend', 'false')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    store.update_work_clock({**DEFAULT_WORK_CLOCK, key: value})
                self.assertEqual(before, path.read_bytes())

    def test_historical_external_app_settings_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'antigravity': {'executable': '/opt/editor'}}))
            store = AppSettingsStore(path)
            store.update_work_clock(DEFAULT_WORK_CLOCK)
            self.assertEqual([app['id'] for app in store.payload()['externalApps']], ['xanylabeling'])
            self.assertEqual(json.loads(path.read_text())['externalApps']['antigravity']['executable'], '/opt/editor')

    def test_calendar_has_no_overlapping_holidays_or_conflicting_makeup_days(self):
        holidays = set()
        for item in CHINA_CALENDAR['holidays']:
            day, end = date.fromisoformat(item['start']), date.fromisoformat(item['end'])
            self.assertLessEqual(day, end)
            self.assertIn(day.year, CHINA_CALENDAR['years'])
            while day <= end:
                self.assertNotIn(day.isoformat(), holidays)
                holidays.add(day.isoformat())
                day += timedelta(days=1)
        self.assertTrue(holidays.isdisjoint(CHINA_CALENDAR['workdays']))
        self.assertIn('2027-02-05', holidays)
        self.assertIn('2026-09-20', CHINA_CALENDAR['workdays'])


if __name__ == '__main__':
    unittest.main()
