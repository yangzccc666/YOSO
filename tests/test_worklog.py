import csv
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openpyxl import load_workbook
from backend.project_workspace import ProjectWorkspace, ProjectConflict
from backend.worklog import WorkLog
from backend.worklog_export import export_report


def row(minutes=210, **changes):
    return dict(dict(public_affair="通用研发", task_category="评估", work_content="核对结果",
                     duration_minutes=minutes, deliverable_note=""), **changes)


class WorkLogTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.projects = ProjectWorkspace(Path(temp.name) / "projects.sqlite3")
        self.log = WorkLog(self.projects)

    def save(self, day, rows, revision=0):
        return self.log.save(dict(work_date=day, entries=rows, revision=revision))

    def test_previous_skips_empty_and_future_and_accepts_partial(self):
        self.save("2026-11-20", [row()])
        self.save("2026-11-23", [row(480)])
        self.save("2026-11-23", [], revision=1)
        self.save("2026-11-25", [row(480)])
        self.assertIsNone(self.log.day("2026-11-20")["previous"])
        source = self.log.day("2026-11-24")["previous"]
        self.assertEqual(source["work_date"], "2026-11-20")
        self.assertEqual(source["entries"][0]["duration_minutes"], 210)
        self.assertEqual(self.log.day("2026-11-25")["previous"], source)
        copied = self.save("2026-11-24", [row(30), {**source["entries"][0], "id": ""}])
        self.assertNotEqual(copied["entries"][1]["id"], source["entries"][0]["id"])
        self.assertEqual(self.log.day("2026-11-25")["previous"]["work_date"], "2026-11-24")

    def test_atomic_save_and_revision(self):
        d = "2026-11-20"
        initial = self.save(d, [row()])
        for invalid in [row(15), row(1470), row(30.5), row(True), row(public_affair=""),
                        row(work_content=" "), row(task_category=""), row(id="foreign-id")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.save(d, [row(), invalid], 1)
        with self.assertRaises(ValueError):
            self.save(d, [row(1440, task_category="未提交"), row(30)], 1)
        self.assertNotIn("未提交", self.log.options()["categories"])
        with self.assertRaises(ProjectConflict):
            self.save(d, [], 0)
        self.assertEqual(self.log.day(d), initial)
        next_day = self.save(d, [{**initial["entries"][0], "work_content": "已修订"}], 1)
        self.assertEqual(next_day["revision"], 2)
        self.assertEqual(initial["entries"][0]["id"], next_day["entries"][0]["id"])

    def test_line_ownership_and_project_deletion(self):
        p = self.projects.change(dict(action="createProject", name="龙城精锻"))["id"]
        line = self.projects.change(dict(action="createNode", project=p, name="渗透2号线"))["id"]
        station = self.projects.change(dict(action="createNode", project=p, node=line, name="2号工位"))["id"]
        other = self.projects.change(dict(action="createProject", name="独立项目"))["id"]
        with self.assertRaises(ValueError):
            self.save("2026-11-20", [row(project_id=other, scope_id=station, public_affair="")])
        original = self.save("2026-11-20", [row(project_id=p, scope_id=station, public_affair="")])
        self.projects.change(dict(action="renameNode",project=p,node=p,name="新项目名"))
        report = self.log.report("2026-11-20","2026-11-20","project:"+p)
        self.assertEqual(report["projects"][0]["name"], "新项目名")
        self.assertEqual(report["entries"][0]["scope_name"], "渗透2号线 / 2号工位")
        self.projects.change(dict(action="deleteNode",project=p))
        self.save("2026-11-20", original["entries"], 1)
        reopened = WorkLog(ProjectWorkspace(self.projects.path))
        self.assertEqual(reopened.options()["report_projects"], [{"id": p, "name": "龙城精锻"}])
        self.assertEqual(reopened.day("2026-11-20")["entries"][0]["project_name"], "龙城精锻")

    def test_reports_and_exports(self):
        self.save("2026-11-20", [row(210, work_content="=原始备注"), row(270, public_affair="内部管理")])
        self.save("2026-11-21", [row(60, task_category="现场排查")])
        rep = self.log.report("2026-11-20", "2026-11-21")
        self.assertEqual((rep["total_minutes"], rep["filled_days"]), (540, 2))
        self.assertEqual([g["minutes"] for g in rep["tasks"]], [480,60])
        filtered = self.log.report("2026-11-20", "2026-11-21", "affair:通用研发")
        self.assertEqual(filtered["total_minutes"],270)
        content, _ = export_report(filtered, "csv")
        csv_rows = list(csv.reader(StringIO(content.decode("utf-8-sig"))))
        self.assertEqual(len(csv_rows),3)
        self.assertEqual(csv_rows[1][4], "'=原始备注")
        content, _ = export_report(filtered, "xlsx")
        book = load_workbook(BytesIO(content))
        self.assertEqual(book.sheetnames, ["工作明细", "项目汇总", "任务类型汇总"])
        self.assertEqual(book.active["E2"].data_type,"s")
        self.assertEqual(book.active["F2"].value,3.5)
        self.assertEqual(book["项目汇总"]["B2"].value,4.5)
        self.assertEqual(book["项目汇总"]["C2"].value,1)
        empty = self.log.report("2026-11-22","2026-11-22")
        self.assertEqual(empty["filled_days"], 0)
        export_report(empty,"xlsx")
        with self.assertRaises(ValueError):
            self.log.report("2026-12-01","2026-11-30")

    def test_date_index_move_clear_remove_and_stale_writes(self):
        first, target = "2026-11-20", "2026-12-01"
        day = self.log.create_date(first)
        self.assertEqual(self.log.create_date(first)["revision"], day["revision"])
        next_day = self.save(first, [row(30), row(210)], day["revision"])
        rev = next_day["revision"]
        self.assertEqual(self.log.dates("2026-11")["dates"], [{"work_date": first, "revision": rev, "minutes": 240}])
        with self.assertRaises(ValueError):
            self.log.change_date(first, rev)
        self.log.create_date(target)
        with self.assertRaises(ValueError):
            self.log.change_date(first, rev, target=target)
        self.log.change_date(target, 1)
        self.log.change_date(first, rev, target=target)
        self.assertEqual(self.log.dates("2026-11")["dates"], [])
        moved = self.log.day(target)
        self.assertEqual([r["id"] for r in moved["entries"]], [r["id"] for r in next_day["entries"]])
        self.assertEqual(self.log.report(target, target)["total_minutes"], 240)
        empty = self.save(target, [], moved["revision"])
        self.assertIsNone(self.log.day("2026-12-03")["previous"])
        self.assertEqual(self.log.dates("2026-12")["dates"][0]["minutes"], 0)
        self.log.change_date(target, empty["revision"])
        self.log.create_date(target)
        with self.assertRaises(ProjectConflict):
            self.save(target, [], moved["revision"])
        with self.assertRaises(ProjectConflict):
            self.save(first, [], rev)
