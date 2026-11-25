import tempfile
import unittest
from pathlib import Path

from backend.project_workspace import ProjectConflict, ProjectWorkspace


class ProjectWorkspaceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = ProjectWorkspace(Path(temp.name) / "projects.sqlite3")
        self.project = self.store.change({"action": "createProject", "name": "龙城精锻"})["id"]

    def change(self, action, **kw):
        return self.store.change({"action": action, "project": self.project, **kw})

    def test_hierarchy_and_seeded_records(self):
        data = self.store.detail(self.project)
        self.assertEqual([r["kind"] for r in data["records"]].count("assessment"), 5)
        self.assertEqual([r["kind"] for r in data["records"]].count("asset"), 7)
        first = self.change("createNode", name="渗透2号线")["id"]
        second = self.change("createNode", name="渗透3号线")["id"]
        for line in [first, second]:
            self.change("createNode", name="2号工位", node=line)
        with self.assertRaises(ValueError):
            self.change("createNode", name="2号工位", node=first)
        self.change("deleteNode", node=first)
        self.assertEqual(len(self.store.detail(self.project)["nodes"]), 3)
        self.assertEqual(len(self.store.detail(self.project)["records"]), 12)

    def test_scoped_paths_and_revision(self):
        station = self.change("createNode", name="产线")["id"]
        data = dict(name="权重", type="计划文件", path=r"D:\中文\weights\v2.plan")
        r = self.change("saveRecord", node=station, kind="asset", data=data)
        second = ProjectWorkspace(self.store.path)
        found = second.detail(self.project)["records"][-1]
        self.assertEqual(found["node"], station)
        self.assertEqual(found["data"], data)
        with self.assertRaises(ValueError):
            self.change("saveRecord", **r, data={**data, "path": ""}, kind="asset")
        self.change("saveRecord", **r, data=data, kind="asset", node=self.project)
        with self.assertRaises(ProjectConflict):
            self.change("deleteRecord", **r)
        other = self.store.change(dict(action="createProject", name="其他"))["id"]
        with self.assertRaises(ValueError):
            self.change("saveRecord", project=other, node=station, data=data, kind="asset")
        with self.assertRaises(ProjectConflict):
            self.change("deleteRecord", project=other, id=r["id"], revision=2)

    def test_field_rename_and_delete(self):
        self.change("saveField", kind="assessment", name="负责人", format="select", options=["甲", "乙"])
        detail = self.store.detail(self.project)
        field = detail["fields"][0]
        r = detail["records"][0]
        self.change("saveRecord", **{**r, "data": {**r["data"], field["id"]: "甲"}})
        self.change("saveField", **{**field, "name": "对接人"})
        detail = self.store.detail(self.project)
        self.assertEqual(detail["records"][0]["data"][field["id"]], "甲")
        self.change("deleteField", id=field["id"])
        self.assertNotIn(field["id"], self.store.detail(self.project)["records"][0]["data"])
        with self.assertRaises(ProjectConflict):
            self.change("deleteRecord", id=r["id"], revision=2)
