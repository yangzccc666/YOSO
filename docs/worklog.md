# 个人工时台账

顶部菜单的「工时记录」按本机维护一份台账。打开默认显示今天。选择日期后，可以在同一页新增、编辑、删除多项工作。每条选择已有项目（可选产线或工位），或选择「其他 / 公共事务」并填写事务名称。工作阶段和公共事务都可直接输入，成功保存后可再次选用。填报与平台自动记录的加班时长分别计算。

## 页面和统计口径

- 日填报实时累加编辑中的耗时。每条 0.5～24 小时、半小时递增。全天超过 8 小时提示超出量，超过 24 小时不能保存。工作内容、阶段、项目或公共事务必填，备注选填。
- 最近一次有效填报取所选日期**之前**最近的、有已保存明细的日期。不满 8 小时也有效。点击「复制到当天」追加独立草稿，来源和当天已有明细均保留。若来源项目或工位已删除，为新条目重新选择有效归属后即可保存。
- 「保存全部」按当前行序提交所有明细，上移或下移可调整顺序。右侧合计反映草稿，左侧徽标取已保存工时。切换日期、月份或模块时可选择「保存并切换」「放弃修改」「继续编辑」，校验失败会留在当前日期。
- 左侧月份索引按日期倒序，可通过「补录日期」创建空日期。同一天重复补录会打开已有记录。日期可跨月更改，有明细的日期可清空，空日期可移除。今天始终可直接进入填报。
- 在右上角打开「工时统计概览」，可按所选周、所选月或任意起止日期查看。周按周一至周日展示，月按自然月展示。点击日期进入日填报。漏填仅指过去的应工作日没有已保存明细，沿用平台单休、双休、节假日和调休设置。总工时包含休息日的实际填报，日均为总工时除以有填报的天数，空期间显示零。
- 报表可选择任意日期范围和项目或公共事务。CSV 为带 UTF-8 BOM 的明细，Excel 含明细、项目汇总和任务类型汇总。汇总占比的分母为当前筛选后的总分钟数。Excel 和 CSV 均只包含已保存记录。

## 数据结构

数据位于平台数据目录的 `projects.sqlite3`，与项目树共用 SQLite 事务。已有项目库首次使用工时时创建新表。Schema 的完整建表 SQL 见 `backend/worklog.py`。

| 表 | 字段与约束 |
| --- | --- |
| work_days | work_date 主键（本地日历 YYYY-MM-DD），revision 乐观锁，updated_at 更新时间 |
| work_entries | id 主键，work_date 外键，position 行序，project_id / project_name 与 scope_id / scope_name 为项目和范围标识、名称快照，public_affair 公共事务，task_category 阶段，work_content 工作内容，duration_minutes 整数分钟，deliverable_note 备注 |
| work_removed_dates | work_date 主键、revision。移除或改期时保留旧日期的修订号，防止过期草稿覆盖重建的记录 |
| work_categories | (kind, name) 联合主键，按工作阶段或公共事务保存已使用的自定义选项 |

同一天的 (work_date, position) 唯一。项目、任务类型、公共事务各有与日期组合的索引。项目或公共事务二选一，明细写入时在同一事务内验证项目归属。历史项目引用保留 ID 和名称快照，删除项目不会级联删除工时。删除当天全部明细后保留日期修订号以识别过期写入，同时该日自动退出最近有效记录和统计。

## 接口及 JSON

| 方法及路径 | 输入与结果 |
| --- | --- |
| GET /api/worklog/dates?month=2026-11 | 返回日期倒序的 dates，每项含 work_date、minutes 和 revision；已创建的空白日期也会返回 |
| POST /api/worklog/dates | 提交 work_date，创建或打开日期，返回当日内容 |
| PUT /api/worklog/dates/{date} | 提交 target、revision，整日原子改期，保留所有明细 ID 和顺序。目标日期已存在则拒绝 |
| DELETE /api/worklog/dates/{date}?revision=R | 仅可移除已保存的空日期；修订过期返回 409。清空明细使用既有日保存接口并传空数组 |
| GET /api/worklog/options | 项目节点、历史报表项目、任务类型、公共事务 |
| GET /api/worklog/day?date=2026-11-25 | 返回 work_date、revision、entries 及 previous。previous 为空或包含 work_date 和 entries |
| POST /api/worklog/day | 完整替换当天明细，返回已保存的当日对象。新行的 id 留空，既有行沿用原 id。版本冲突返回 409，校验失败返回 400，整笔回滚 |
| GET /api/worklog/report | start、end 为含首尾的日期，owner 可选。返回 entries、days、total_minutes、filled_days 和 projects、tasks 两组聚合 |
| GET /api/worklog/export | 同一组范围及筛选条件，format 为 xlsx 或 csv，返回可下载文件 |

首次保存公共事务示例：

```json
{
  "work_date": "2026-11-25",
  "revision": 0,
  "entries": [{
    "id": "",
    "project_id": "",
    "scope_id": "",
    "public_affair": "技术调研",
    "task_category": "模型训练、调参与评估",
    "work_content": "复核低照场景下的漏检样本",
    "duration_minutes": 210,
    "deliverable_note": "评估记录 v2"
  }]
}
```

关联项目时传项目树中的 project_id，public_affair 留空，scope_id 可选。报表筛选使用 `project:<项目ID>` 或 `affair:<公共事务名称>`。所有累加与 SQL 聚合使用整数分钟，展示与报表列再换算成小时。

## 代码入口

- `frontend/src/worklog/WorkLog.tsx` 管理月份索引、日期导航、统计窗口和离开保护。
- `WorkDay.tsx` 管理行内草稿、实时合计、复制和批量提交。`model.ts` 提供类型、日期与校验。
- `PeriodReport.tsx` 展示周月统计、按日回填及筛选导出。
- `backend/worklog.py` 实现日期事务、历史查找、归属校验与 SQL 聚合。`backend/worklog_export.py` 生成报表，Excel 使用 requirements.txt 中声明的 openpyxl。
