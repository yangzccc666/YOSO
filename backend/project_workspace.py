"""Project hierarchy and research/asset ledgers, persisted independently of task runs."""
import json
import sqlite3
from time import time_ns
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ASSESSMENTS = {
    "业务痛点与判定标准": "检测哪些目标？如何判定 OK / NG？可接受多少误检和漏检？",
    "相机点位与光照环境": "安装视角和视野是否覆盖动作？反光、遮挡与光照变化有哪些？",
    "作业区域与 ROI 约束": "哪些固定区域需要检测？跨区域动作如何判定？",
    "数据与类别分布风险": "样本从哪里获取？负样本、稀有类别和标注难点有哪些？",
    "算力与推理性能约束": "部署在哪台设备？目标帧率、延迟和量化要求是什么？",
}
ASSETS = ["项目根目录", "原始视频路径", "抽帧图片路径", "标注文件路径", "训练数据集路径", "模型权重路径", "推理/部署路径"]
RISKS = ["待评估", "高", "中", "低"]
STATUSES = ["待确认", "已对齐", "方案已落地"]
KINDS = ("assessment", "asset")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
 id TEXT PRIMARY KEY, parent TEXT REFERENCES nodes(id) ON DELETE CASCADE,
 kind TEXT NOT NULL, name TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS sibling_name ON nodes(ifnull(parent,''),name);
CREATE TABLE IF NOT EXISTS records(
 id TEXT PRIMARY KEY, node TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
 kind TEXT NOT NULL, data TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS records_node ON records(node);
CREATE TABLE IF NOT EXISTS fields(
 id TEXT PRIMARY KEY, project TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
 kind TEXT NOT NULL, name TEXT NOT NULL, format TEXT NOT NULL, options TEXT NOT NULL,
 UNIQUE(project,kind,name));
CREATE TABLE IF NOT EXISTS project_order(
 project TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
 edited INTEGER NOT NULL DEFAULT 0, position INTEGER);
CREATE TABLE IF NOT EXISTS project_list_settings(
 id INTEGER PRIMARY KEY CHECK(id=1), mode TEXT NOT NULL);
"""


class ProjectConflict(ValueError):
    pass


def required(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"请填写{label}。")
    return value.strip()


class ProjectWorkspace:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self, *, write=False, extra_schema=""):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA + extra_schema)
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except sqlite3.IntegrityError as error:
            raise ValueError("同一层级或同一表单内已有该名称。") from error
        finally:
            db.close()

    def catalog(self):
        with self.connection() as db:
            setting = db.execute("SELECT mode FROM project_list_settings WHERE id=1").fetchone()
            mode = setting[0] if setting else "edited"
            return {"projects": self._projects(db, mode), "sortMode": mode,
                    "dimensions": list(ASSESSMENTS), "types": ASSETS}

    @staticmethod
    def _projects(db, mode):
        order = {"edited": "coalesce(o.edited,0) DESC, n.rowid DESC",
                 "created": "n.rowid DESC",
                 "manual": "o.position IS NULL, o.position, n.rowid"}[mode]
        return [dict(row) for row in db.execute(
            f"SELECT n.* FROM nodes n LEFT JOIN project_order o ON o.project=n.id "
            f"WHERE n.parent IS NULL ORDER BY {order}")]

    def _sort(self, db, payload):
        mode = payload.get("mode")
        if mode not in ("edited", "created", "manual"):
            raise ValueError("项目排序方式无效。")
        if "source" in payload:
            ids = [row["id"] for row in self._projects(db, "manual")]
            source, target = payload["source"], payload.get("target")
            if mode != "manual" or source not in ids or target not in ids:
                raise ValueError("项目列表已变化，请刷新后重试。")
            position = ids.index(target)
            ids.remove(source)
            ids.insert(position, source)
            db.executemany("INSERT INTO project_order(project,position) VALUES(?,?) "
                           "ON CONFLICT(project) DO UPDATE SET position=excluded.position",
                           ((key, index) for index, key in enumerate(ids)))
        db.execute("INSERT OR REPLACE INTO project_list_settings VALUES(1,?)", (mode,))
        return {}

    @staticmethod
    def _nodes(db, project):
        rows = db.execute("""
            WITH RECURSIVE tree(id,parent,kind,name) AS (
                SELECT * FROM nodes WHERE id=? AND kind='project'
                UNION ALL SELECT n.* FROM nodes n JOIN tree t ON n.parent=t.id)
            SELECT * FROM tree ORDER BY name
        """, (project,)).fetchall()
        if not rows:
            raise ValueError("项目不存在，请刷新项目列表。")
        return {row["id"]: dict(row) for row in rows}

    def detail(self, project):
        with self.connection() as db:
            nodes = self._nodes(db, project)
            marks = ",".join("?" for _ in nodes)
            records = [{**dict(row), "data": json.loads(row["data"])} for row in db.execute(
                f"SELECT * FROM records WHERE node IN ({marks}) ORDER BY rowid", tuple(nodes))]
            fields = [{**dict(row), "options": json.loads(row["options"])} for row in db.execute(
                "SELECT * FROM fields WHERE project=? ORDER BY rowid", (project,))]
            return {"nodes": list(nodes.values()), "records": records, "fields": fields}

    @staticmethod
    def _add_record(db, node, kind, data):
        key = uuid4().hex
        db.execute("INSERT INTO records(id,node,kind,data) VALUES(?,?,?,?)",
                   (key, node, kind, json.dumps(data, ensure_ascii=False)))
        return key

    def change(self, payload):
        action = payload.get("action")
        project = payload.get("project")
        key = payload.get("id")
        with self.connection(write=True) as db:
            if action == "sortProjects":
                return self._sort(db, payload)
            if action == "createProject":
                key = uuid4().hex
                db.execute("INSERT INTO nodes VALUES(?,NULL,'project',?)",
                           (key, required(payload.get("name"), "项目名称")))
                db.execute("INSERT INTO project_order(project,edited) VALUES(?,?)", (key, time_ns()))
                for name, question in ASSESSMENTS.items():
                    self._add_record(db, key, "assessment", dict(category=name, question=question,
                                     solution="", risk=RISKS[0], status=STATUSES[0]))
                for name in ASSETS:
                    self._add_record(db, key, "asset", dict(name=name, type=name, environment="",
                                     host="", path="", note=""))
                return {"id": key}

            nodes = self._nodes(db, project)
            db.execute("INSERT INTO project_order(project,edited) VALUES(?,?) "
                       "ON CONFLICT(project) DO UPDATE SET edited=excluded.edited", (project, time_ns()))
            node = payload.get("node", project)
            if node not in nodes:
                raise ValueError("所选归属不在当前项目中。")
            if action == "createNode":
                child = {"project": "line", "line": "station"}.get(nodes[node]["kind"])
                if not child:
                    raise ValueError("请在项目或产线下新增。")
                key = uuid4().hex
                db.execute("INSERT INTO nodes VALUES(?,?,?,?)",
                           (key, node, child, required(payload.get("name"), "名称")))
            elif action == "renameNode":
                db.execute("UPDATE nodes SET name=? WHERE id=?", (required(payload.get("name"), "名称"), node))
            elif action == "deleteNode":
                db.execute("DELETE FROM nodes WHERE id=?", (node,))
            elif action in ("saveField", "deleteField"):
                self._field(db, project, payload)
            elif action in ("saveRecord", "deleteRecord"):
                return self._record(db, nodes, project, node, payload)
            else:
                raise ValueError("未知的项目操作。")
            return {"id": key}

    @staticmethod
    def _field(db, project, p):
        key = p.get("id")
        existing = db.execute("SELECT * FROM fields WHERE id=? AND project=?", (key, project)).fetchone()
        if key and not existing:
            raise ValueError("字段不存在。")
        if p["action"] == "deleteField":
            if not existing:
                raise ValueError("请选择字段。")
            # Remove stored values as well as the definition. Increment revisions so an
            # editor with an old form cannot silently restore a deleted value.
            nodes = ProjectWorkspace._nodes(db, project)
            for node in nodes:
                for row in db.execute("SELECT id,data FROM records WHERE node=? AND kind=?",
                                      (node, existing["kind"])).fetchall():
                    data = json.loads(row["data"])
                    if key in data:
                        del data[key]
                        db.execute("UPDATE records SET data=?,revision=revision+1 WHERE id=?",
                                   (json.dumps(data, ensure_ascii=False), row["id"]))
            db.execute("DELETE FROM fields WHERE id=?", (key,))
            return
        name = required(p.get("name"), "字段名称")
        kind, fmt, options = p.get("kind"), p.get("format"), p.get("options", [])
        if kind not in KINDS or fmt not in ("text", "multiline", "select"):
            raise ValueError("字段类型无效。")
        if not isinstance(options, list):
            raise ValueError("选项格式无效。")
        options = list(dict.fromkeys(required(v, "选项") for v in options)) if fmt == "select" else []
        if fmt == "select" and not options:
            raise ValueError("下拉字段至少需要一个选项。")
        if existing and (kind != existing["kind"] or fmt != existing["format"]):
            raise ValueError("已有字段仅支持修改名称或选项。")
        db.execute("""INSERT INTO fields VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name,options=excluded.options""",
                   (key or uuid4().hex, project, kind, name, fmt, json.dumps(options, ensure_ascii=False)))

    def _record(self, db, nodes, project, node, p):
        key, kind = p.get("id"), p.get("kind")
        old = db.execute("SELECT * FROM records WHERE id=?", (key,)).fetchone()
        if key and (not old or old["node"] not in nodes):
            raise ProjectConflict("记录已被删除或已不可用。")
        if old and old["revision"] != p.get("revision"):
            raise ProjectConflict("记录已被其他窗口修改，请载入最新版本后再编辑。")
        if p["action"] == "deleteRecord":
            if not old:
                raise ValueError("请选择记录。")
            db.execute("DELETE FROM records WHERE id=?", (key,))
            return {}
        if kind not in KINDS or (old and old["kind"] != kind):
            raise ValueError("记录类型无效。")
        data = p.get("data")
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise ValueError("字段值必须为文本。")
        attrs = (["category", "question", "solution", "risk", "status"] if kind == "assessment"
                 else ["name", "type", "environment", "host", "path", "note"])
        fields = db.execute("SELECT * FROM fields WHERE project=? AND kind=?", (project, kind)).fetchall()
        if set(data) - set(attrs) - {field["id"] for field in fields}:
            raise ProjectConflict("表单字段已更改，请重新载入。")
        previous = json.loads(old["data"]) if old else {}
        for field in fields:
            value = data.get(field["id"], "")
            if field["format"] == "select" and value and value != previous.get(field["id"]) and value not in json.loads(field["options"]):
                raise ValueError("自定义字段选项已变更。")
        if kind == "assessment":
            required(data.get("category"), "评估维度")
            if data.get("risk") not in RISKS or data.get("status") not in STATUSES:
                raise ValueError("风险等级或对齐状态无效。")
        else:
            required(data.get("name"), "路径名称")
            required(data.get("type"), "路径类型")
            if not old or previous.get("path"):
                required(data.get("path"), "完整路径")
        if old:
            db.execute("UPDATE records SET data=?,node=?,revision=revision+1 WHERE id=?",
                       (json.dumps(data, ensure_ascii=False), node, key))
        else:
            key = self._add_record(db, node, kind, data)
        return {"id": key, "revision": old["revision"] + 1 if old else 1}
