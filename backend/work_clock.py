"""Offline work-clock preferences and published mainland China holiday dates.

Dates are factual transcriptions of the State Council notices linked below.
Keep each year's holiday ranges and compensatory workdays together when adding
a new year. Missing years must remain unknown rather than inferred schedules.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


DEFAULT_WORK_CLOCK = {
    "startTime": "09:00", "endTime": "18:00", "displayMode": "remaining",
    "weekendMode": "double", "singleRestDay": 0,
    "followChinaCalendar": True, "showWeekend": True, "showHolidays": True,
    "overtimeEnabled": True, "overtimeIdleMinutes": 30,
}

CHINA_CALENDAR = {
    "years": [2026, 2027],
    "sources": [
        {"year": 2026, "title": "国务院办公厅关于2026年部分节假日安排的通知",
         "url": "https://www.gov.cn/zhengce/zhengceku/202511/content_7047091.htm"},
        {"year": 2027, "title": "国务院办公厅关于2027年部分节假日安排的通知",
         "url": "https://www.gov.cn/zhengce/content/202611/content_7085101.htm"},
    ],
    "holidays": [
        {"name": "元旦", "start": "2026-01-01", "end": "2026-01-03"},
        {"name": "春节", "start": "2026-02-15", "end": "2026-02-23"},
        {"name": "清明节", "start": "2026-04-04", "end": "2026-04-06"},
        {"name": "劳动节", "start": "2026-05-01", "end": "2026-05-05"},
        {"name": "端午节", "start": "2026-06-19", "end": "2026-06-21"},
        {"name": "中秋节", "start": "2026-09-25", "end": "2026-09-27"},
        {"name": "国庆节", "start": "2026-10-01", "end": "2026-10-07"},
        {"name": "元旦", "start": "2027-01-01", "end": "2027-01-03"},
        {"name": "春节", "start": "2027-02-05", "end": "2027-02-13"},
        {"name": "清明节", "start": "2027-04-03", "end": "2027-04-05"},
        {"name": "劳动节", "start": "2027-05-01", "end": "2027-05-05"},
        {"name": "端午节", "start": "2027-06-09", "end": "2027-06-09"},
        {"name": "中秋节", "start": "2027-09-15", "end": "2027-09-15"},
        {"name": "国庆节", "start": "2027-10-01", "end": "2027-10-07"},
    ],
    "workdays": [
        "2026-01-04", "2026-02-14", "2026-02-28", "2026-05-09", "2026-09-20", "2026-10-10",
        "2027-01-31", "2027-02-14", "2027-05-08", "2027-09-26", "2027-10-09",
    ],
}


def normalize_work_clock(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("工作时间设置格式不正确。")
    settings = {**DEFAULT_WORK_CLOCK, **{key: raw[key] for key in DEFAULT_WORK_CLOCK if key in raw}}
    for key, label in (("startTime", "上班时间"), ("endTime", "下班时间")):
        if not isinstance(settings[key], str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", settings[key]):
            raise ValueError(f"{label}请使用有效的 HH:MM 格式。")
    if settings["startTime"] == settings["endTime"]:
        raise ValueError("上下班时间不能相同；下班时间早于上班时间时，按跨午夜夜班计算。")
    if settings["displayMode"] not in ("remaining", "elapsed", "both"):
        raise ValueError("请选择有效的计时显示方式。")
    if settings["weekendMode"] not in ("single", "double"):
        raise ValueError("请选择单休或双休。")
    if type(settings["singleRestDay"]) is not int or not 0 <= settings["singleRestDay"] <= 6:
        raise ValueError("请选择有效的单休日期。")
    for key in ("followChinaCalendar", "showWeekend", "showHolidays", "overtimeEnabled"):
        if type(settings[key]) is not bool:
            raise ValueError("计时开关设置必须为开启或关闭。")
    if type(settings['overtimeIdleMinutes']) is not int or settings['overtimeIdleMinutes'] not in (10, 20, 30, 60):
        raise ValueError('加班无操作等待时长请选择 10、20、30 或 60 分钟。')
    return settings


def calendar_payload() -> dict[str, Any]:
    return deepcopy(CHINA_CALENDAR)
