"""Local, daily time sheets. Project references keep snapshots for archived work."""
from datetime import date
from uuid import uuid4

from .project_workspace import ProjectConflict, required

CATEGORIES = (
    "需求调研与现场方案对齐", "视频采集、抽帧与数据清洗", "标注任务分配与质量验收",
    "模型训练、调参与评估", "模型转换（ONNX/量化）与边缘端部署", "现场联调与问题排查",
    "文档沉淀与跨团队沟通",
)
AFFAIRS = ("通用研发", "内部管理", "技术调研")

SCHEMA = """
CREATE TABLE IF NOT EXISTS work_days (
    work_date TEXT PRIMARY KEY, revision INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS work_removed_dates (work_date TEXT PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS work_entries (
    id TEXT PRIMARY KEY, work_date TEXT NOT NULL REFERENCES work_days(work_date),
    position INTEGER NOT NULL, project_id TEXT NOT NULL, project_name TEXT NOT NULL,
    scope_id TEXT NOT NULL, scope_name TEXT NOT NULL, public_affair TEXT NOT NULL,
    task_category TEXT NOT NULL, work_content TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL CHECK(duration_minutes BETWEEN 30 AND 1440 AND duration_minutes % 30 = 0),
    deliverable_note TEXT NOT NULL,
    CHECK ((project_id != '') != (public_affair != '')),
    UNIQUE(work_date, position)
);
CREATE INDEX IF NOT EXISTS work_project_date ON work_entries(project_id, work_date);
CREATE INDEX IF NOT EXISTS work_category_date ON work_entries(task_category, work_date);
CREATE INDEX IF NOT EXISTS work_affair_date ON work_entries(public_affair, work_date);
CREATE TABLE IF NOT EXISTS work_categories (
    kind TEXT NOT NULL, name TEXT NOT NULL, PRIMARY KEY(kind,name)
);
"""


def valid_date(value):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return value
    except (ValueError, TypeError):
        pass
    raise ValueError("请使用有效的年月日。")


def _entries(db, day):
    return [dict(r) for r in db.execute(
        "SELECT * FROM work_entries WHERE work_date=? ORDER BY position", (day,))]


class WorkLog:
    def __init__(self, workspace):
        self.workspace = workspace

    def connection(self, write=False):
        return self.workspace.connection(write=write, extra_schema=SCHEMA)

    def dates(self, month):
        if not isinstance(month, str):
            raise ValueError("请选择月份。")
        valid_date(month + "-01")
        with self.connection() as db:
            rows = db.execute("""
                SELECT d.work_date,d.revision,COALESCE(SUM(e.duration_minutes),0) AS minutes
                FROM work_days d LEFT JOIN work_entries e ON e.work_date=d.work_date
                WHERE d.work_date BETWEEN ? AND ? GROUP BY d.work_date ORDER BY d.work_date DESC
            """, (month + "-01", month + "-31"))
            return {"dates": [dict(r) for r in rows]}

    @staticmethod
    def _revision(db, day):
        row = db.execute("""SELECT revision FROM work_days WHERE work_date=?
                            UNION ALL SELECT revision FROM work_removed_dates WHERE work_date=?""",
                         (day, day)).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _open_date(db, day):
        revision = WorkLog._revision(db, day) + 1
        db.execute("DELETE FROM work_removed_dates WHERE work_date=?", (day,))
        db.execute("""INSERT INTO work_days(work_date,revision) VALUES(?,?)
            ON CONFLICT(work_date) DO UPDATE SET revision=excluded.revision,updated_at=CURRENT_TIMESTAMP""",
                   (day, revision))

    def create_date(self, day):
        day = valid_date(day)
        with self.connection(write=True) as db:
            if not db.execute("SELECT 1 FROM work_days WHERE work_date=?", (day,)).fetchone():
                self._open_date(db, day)
            return self._day(db, day)

    def change_date(self, day, revision, *, target=None):
        day = valid_date(day)
        with self.connection(write=True) as db:
            row = db.execute("SELECT revision FROM work_days WHERE work_date=?", (day,)).fetchone()
            if not row or type(revision) is not int or revision != row[0]:
                raise ProjectConflict("日期记录已变化，请刷新后重试。")
            if target is not None:
                target = valid_date(target)
                if db.execute("SELECT 1 FROM work_days WHERE work_date=?", (target,)).fetchone():
                    raise ValueError("目标日期已有记录，请选择其他日期。")
                self._open_date(db, target)
                db.execute("UPDATE work_entries SET work_date=? WHERE work_date=?", (target, day))
            elif db.execute("SELECT 1 FROM work_entries WHERE work_date=? LIMIT 1", (day,)).fetchone():
                raise ValueError("请先清空明细，再移除日期。")
            db.execute("DELETE FROM work_days WHERE work_date=?", (day,))
            db.execute("INSERT INTO work_removed_dates VALUES(?,?)", (day, revision + 1))
            return {"work_date": target}

    @staticmethod
    def _day(db, day):
        version = WorkLog._revision(db, day)
        previous = db.execute("SELECT MAX(work_date) FROM work_entries WHERE work_date<?", (day,)).fetchone()[0]
        return {"work_date": day, "revision": version,
                "entries": _entries(db, day),
                "previous": {"work_date": previous, "entries": _entries(db, previous)} if previous else None}

    def day(self, day):
        with self.connection() as db:
            return self._day(db, valid_date(day))

    def options(self):
        with self.connection() as db:
            def choices(kind, defaults):
                names = db.execute("SELECT name FROM work_categories WHERE kind=? ORDER BY name", (kind,))
                return list(dict.fromkeys([*defaults, *(r[0] for r in names)]))
            projects = db.execute("""
                SELECT e.project_id AS id, COALESCE(n.name, e.project_name) AS name, MAX(e.work_date)
                FROM work_entries e LEFT JOIN nodes n ON n.id=e.project_id
                WHERE e.project_id!='' GROUP BY e.project_id ORDER BY name
            """)
            return {"nodes": [dict(r) for r in db.execute("SELECT * FROM nodes ORDER BY name")],
                    "report_projects": [{"id": r["id"], "name": r["name"]} for r in projects],
                    "categories": choices("task", CATEGORIES), "affairs": choices("affair", AFFAIRS)}

    @staticmethod
    def _scope(nodes, entry, old):
        project = entry.get("project_id", "")
        scope = entry.get("scope_id", "")
        if not isinstance(project, str) or not isinstance(scope, str):
            raise ValueError("项目或工位无效。")
        if not project:
            if scope:
                raise ValueError("公共事务不能关联工位。")
            return "", "", "", "", required(entry.get("public_affair"), "公共事务名称")
        if entry.get("public_affair"):
            raise ValueError("请只选择一个项目或公共事务。")
        # Historical rows remain editable after a project or station is removed.
        if old and (project, scope) == (old["project_id"], old["scope_id"]):
            return project, old["project_name"], scope, old["scope_name"], ""
        if project not in nodes or nodes[project]["kind"] != "project":
            raise ValueError("所选项目已移除，请重新选择。")
        names = []
        if scope:
            node = nodes.get(scope)
            while node and node["parent"]:
                names.append(node["name"])
                node = nodes.get(node["parent"])
            if not names or not node or node["id"] != project:
                raise ValueError("所选产线或工位不属于该项目，请重新选择。")
        return project, nodes[project]["name"], scope, " / ".join(reversed(names)), ""

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("工时表格式无效。")
        day = valid_date(payload.get("work_date"))
        rows = payload.get("entries")
        if not isinstance(rows, list) or len(rows) > 48:
            raise ValueError("每天最多填报 48 条半小时明细。")
        with self.connection(write=True) as db:
            previous_revision = self._revision(db, day)
            if type(payload.get("revision")) is not int or payload["revision"] != previous_revision:
                raise ProjectConflict("当天记录已更新，请重新载入后再保存。")
            old = {r["id"]: r for r in _entries(db, day)}
            nodes = {r["id"]: r for r in db.execute("SELECT * FROM nodes")}
            entries, seen, total = [], set(), 0
            for number, entry in enumerate(rows, 1):
                if not isinstance(entry, dict):
                    raise ValueError(f"第 {number} 条明细无效。")
                key = entry.get("id") or uuid4().hex
                if not isinstance(key, str) or key in seen or (entry.get("id") and key not in old):
                    raise ValueError(f"第 {number} 条记录归属或标识无效。")
                seen.add(key)
                minutes = entry.get("duration_minutes")
                if type(minutes) is not int or not 30 <= minutes <= 1440 or minutes % 30:
                    raise ValueError(f"第 {number} 条耗时须为 0.5～24 小时，步长 0.5。")
                total += minutes
                ownership = self._scope(nodes, entry, old.get(key))
                category = required(entry.get("task_category"), f"第 {number} 条的工作阶段")
                content = required(entry.get("work_content"), f"第 {number} 条的工作内容")
                note = entry.get("deliverable_note", "")
                if not isinstance(note, str):
                    raise ValueError("备注必须为文本。")
                entries.append((key, day, number, *ownership, category, content, minutes, note.strip()))
                db.execute("INSERT OR IGNORE INTO work_categories VALUES('task',?)", (category,))
                if ownership[-1]:
                    db.execute("INSERT OR IGNORE INTO work_categories VALUES('affair',?)", (ownership[-1],))
            if total > 1440:
                raise ValueError("当日总工时超过 24 小时，请核对明细。")
            self._open_date(db, day)
            db.execute("DELETE FROM work_entries WHERE work_date=?", (day,))
            db.executemany("""INSERT INTO work_entries(id,work_date,position,project_id,project_name,
                scope_id,scope_name,public_affair,task_category,work_content,duration_minutes,deliverable_note)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", entries)
            return self._day(db, day)

    def report(self, start, end, owner=""):
        start, end = valid_date(start), valid_date(end)
        if end < start:
            raise ValueError("结束日期应在开始日期之后或当天。")
        where = "e.work_date BETWEEN ? AND ?"
        args = [start, end]
        if owner:
            kind, sep, value = owner.partition(":")
            if not sep or not value or kind not in ("project", "affair"):
                raise ValueError("工时筛选条件无效。")
            where += " AND e." + ("project_id" if kind == "project" else "public_affair") + "=?"
            args.append(value)
        with self.connection() as db:
            source = f"FROM work_entries e LEFT JOIN nodes n ON n.id=e.project_id WHERE {where}"
            rows = [dict(r) for r in db.execute(f"SELECT e.* {source} ORDER BY work_date,position", args)]
            def grouped(key, name):
                return [dict(r) for r in db.execute(
                    f"SELECT {key} AS id,{name} AS name,SUM(duration_minutes) AS minutes,"
                    f"MAX(work_date) AS last_date {source} GROUP BY 1 ORDER BY minutes DESC,name", args)]
            projects = grouped("CASE WHEN project_id='' THEN 'affair:'||public_affair ELSE 'project:'||project_id END",
                               "CASE WHEN project_id='' THEN public_affair ELSE COALESCE(n.name,project_name) END")
            tasks = grouped("task_category", "task_category")
            days = {r["work_date"]: r["minutes"] for r in db.execute(
                f"SELECT work_date,SUM(duration_minutes) AS minutes {source} GROUP BY work_date", args)}
            total = sum(days.values())
            return {"start": start, "end": end, "entries": rows, "days": days, "total_minutes": total,
                    "filled_days": len(days), "projects": projects, "tasks": tasks}
