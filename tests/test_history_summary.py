import unittest
from backend.history_summary import history_parameters


class HistorySummaryTests(unittest.TestCase):
    def test_parameter_allowlist_excludes_credentials_and_arbitrary_fields(self):
        result = history_parameters({'conf': 0.5, 'password': 'SECRET', 'remote_password': 'SECRET',
                                     'token': 'SECRET', 'labels': 'SECRET', 'private_key': 'SECRET'})
        self.assertEqual(result, {'关键参数': '置信度=0.5'})

    def test_only_active_frame_rate_is_summarized(self):
        result = history_parameters({'extraction_mode': '每秒抽取张数', 'target_fps': 5,
                                     'interval_sec': 1, 'priority_fps': 25, 'background_fps': 5})['关键参数']
        self.assertIn('张/秒=5', result)
        self.assertNotIn('间隔秒', result)
        self.assertNotIn('工序张', result)

    def test_clip_ignores_inactive_defaults(self):
        result = history_parameters({'clip_mode': '指定开始时间和持续时长', 'range_start_time': '00:00',
                                     'duration_start_time': '01:30', 'clip_duration': '00:40'})['关键参数']
        self.assertIn('开始=01:30', result)
        self.assertNotIn('00:00', result)

    def test_bounded_scalar_only_and_whitespace_normalized(self):
        self.assertEqual(history_parameters(None), {})
        result = history_parameters({'conf': [], 'amp': False, 'title': 'a\nb', 'device': 'x' * 500})['关键参数']
        self.assertIn('AMP=False', result)
        self.assertIn('打包名=a b', result)
        self.assertNotIn('\n', result)
        self.assertLess(len(result), 1000)
