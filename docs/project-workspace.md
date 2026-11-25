# 项目评估与路径资产管理

从顶部进入「项目管理」。左栏固定 260px，可搜索、新建、切换项目，移入或聚焦列表项可重命名、删除。右栏显示选中项目，可用「工作范围」切换整个项目、产线或工位。两栏各自滚动。新建项目后可直接登记项目级信息。选中项目可新增产线，选中产线可新增工位。不同产线允许使用相同的工位名。新建项目时会创建五条评估模板和七条空路径模板，产线和工位按需添加自己的记录。

项目详情包含「接手评估与方案」和「路径与资产配置」。当前节点展示自身和下属记录，勾选「显示上级共享记录」可同时查看祖先节点的记录。每条记录显示完整归属，编辑共享记录会更新其原始归属下的同一条记录。可在编辑窗口中修改归属。

编辑停顿约 0.7 秒后自动保存，完成时会立即尝试保存。必填项缺失时会显示待填写内容；重载前请确认显示「已保存」。类别、环境可自行输入。每个项目的两张表单可以分别增加、重命名或删除扩展字段，支持单行、多行和下拉选项。删除字段会同时移除已有的对应值。删除项目、产线、工位时会同时移除其下属台账。路径是文本引用，复制时保留 Windows、Linux 和网络路径原文。

## 持久化结构

SQLite 文件位于平台数据目录内的 `projects.sqlite3`。数据目录遵循 `platform_data_dir()`，可通过 `PROCESSING_VIEW_DATA_DIR` 指定。项目资料保存在使用平台的这台机器上。

| 表 | 字段 | 约束与用途 |
| --- | --- | --- |
| nodes | id, parent, kind, name | 项目、产线、工位共用稳定 ID。parent 为父节点外键，同级名称唯一 |
| records | id, node, kind, data, revision | 每条记录只归属一个节点。data 为 JSON，revision 用于并发编辑检测 |
| fields | id, project, kind, name, format, options | 按项目和表单定义字段。options 为 JSON 数组，自定义字段的值以字段 ID 为键写入 records.data |

评估表的固定键为 category、question、solution、risk、status。资产表的固定键为 name、type、environment、host、path、note。模板允许暂时留空路径，新建路径和已有完整路径的记录要求路径非空。路径可以指向远端或未挂载的存储。

## 接口与代码入口

- `GET /api/projects` 提供项目列表和默认维度、路径类型。
- `GET /api/projects/{id}` 返回当前项目的 nodes、records 和 fields。
- `POST /api/projects/manage` 在单一事务中执行指定 action。createProject 需要 name，createNode、renameNode、deleteNode 通过 project 和 node 确定归属。saveField、deleteField 管理字段。saveRecord、deleteRecord 修改已有记录时须传 id 和 revision；过期的修改返回 HTTP 409。
- `backend/project_workspace.py` 定义实际 Schema、校验和事务。`frontend/src/projects/ProjectWorkspace.tsx` 负责层级导航和列表，`RecordEditor.tsx` 负责字段渲染、输入法合成与串行保存。

新增工位路径的请求示例（实际请求需使用接口返回的 ID）：

```json
{"action":"saveRecord","project":"<项目ID>","node":"<工位ID>","kind":"asset","data":{"name":"基线训练权重","type":"模型权重路径","environment":"远程训练服务器","host":"训练服务器 A","path":"/data/models/baseline/best.pt","note":"baseline"}}
```


## 双栏组件与联动

`frontend/src/workspaces/MasterDetailLayout.tsx` 统一渲染固定宽度索引、搜索框、新建按钮和独立滚动的右侧工作区。`MasterItem` 渲染选中状态、可聚焦的重命名和删除操作。两个模块均使用功能分组的颜色、间距和交互样式。项目侧通过稳定项目 ID 读取详情，在右栏使用节点 ID 过滤评估与资产。添加记录时自动带入当前范围。删除确认包含下级节点和台账记录数。

`App.tsx` 负责「功能分组」「项目管理」「工时记录」之间的导航。功能处理组件在切换期间保留挂载；切换回功能时可继续查看已有任务。工时页面在执行跨模块导航前检查编辑状态。
