# 扩展开发说明

界面只负责维护功能名称、路径、参数和一键运行入口。具体业务逻辑通过 `backend.handlers` 中的最小处理器接口接入。

## 新增处理逻辑

```python
from backend.handlers import RunContext, register_handler


def my_handler(context: RunContext) -> dict:
    input_dir = context.paths["input_path"]
    threshold = context.parameters.get("threshold", 0.5)
    context.report(f"正在处理：{input_dir}")
    # 在这里调用 tools 中已有脚本的核心函数
    return {"message": "处理完成", "threshold": threshold}


register_handler("tools.my_handler", my_handler)
```

之后把某个功能的 `handlerId` 设置为 `tools.my_handler`。前端不需要修改。

## 接入已有脚本

建议把旧脚本中的核心逻辑提取成函数，由处理器调用。路径、字段和开关通过 `context.params` 传入，不要再把路径写死在代码里。

## API 预留

- `GET /api/health`：运行状态
- `GET /api/workspace`：一次读取分组和带归属信息的功能列表
- `GET /api/groups`：读取分组
- `POST /api/groups`：新建分组
- `PUT /api/groups/{id}`：重命名分组
- `DELETE /api/groups/{id}`：删除分组，组内功能自动转入“未分组”
- `POST /api/groups/reorder`：调整分组顺序
- `GET /api/functions`：读取功能列表
- `POST /api/functions`：新建功能
- `PUT /api/functions/{id}`：修改名称、路径和参数定义
- `DELETE /api/functions/{id}`：删除功能
- `POST /api/functions/{id}/move`：移动功能到分组并设置组内位置
- `POST /api/functions/{id}/run`：统一运行入口
- `GET /api/tasks/active`：读取当前运行任务列表（`active` 数组）
- `POST /api/tasks/{id}/stop`：按任务 ID 终止指定任务，不影响其他任务
- `GET/POST /api/yolo-training/profiles`：读取或保存训练场景
- `PUT/DELETE /api/yolo-training/profiles/{id}`：更新或删除训练场景
- `POST /api/yolo-training/start`：启动本地或远程 YOLO 训练
- `GET /api/yolo-training/{id}/status`：读取训练日志和状态
- `POST /api/yolo-training/{id}/stop`：终止训练
- `POST /api/dialog`：打开本机文件/文件夹选择器
- `GET /api/open?path=...`：在系统文件管理器中打开结果

处理器应在文件循环或耗时步骤之间调用 `context.check_cancelled()`，这样用户点击“终止运行”后可以安全退出。后续仍可在不修改前端核心结构的情况下增加进度回调、运行历史和自然语言参数生成。

YOLO 数据集的 SSH 模式由 `backend/remote_yolo_dataset.py` 负责连接和任务控制，并临时上传独立的 `backend/remote_yolo_dataset_worker.py`。远程执行器必须保持仅依赖 Python 标准库，新增划分字段时应同步维护本地与远程两套实现及测试。

## 热更新

桌面程序会监控 `backend/**/*.py` 和 `frontend/dist/`：

- 后台处理代码变化后，会在同一端口替换本地服务并刷新界面；
- 前端执行 `npm run build` 后，新的 `dist` 文件会自动加载；
- 有任务运行时不会强制刷新，而是等待任务完成或终止；
- 窗口不关闭，浏览器存储的路径和参数不会丢失。

如需临时关闭，可在启动前设置 `PROCESSING_VIEW_HOT_RELOAD=0`。`desktop.py` 和 `run.py` 属于桌面外壳自身，修改这两个文件后仍需重新启动一次。

## 已接入示例：视频切图

`backend/video_frames.py` 是第一个实际适配器。它加载：

`tools/data_processing_toolkit/02-video_processing/batch_vedio2img.py`

界面负责收集视频路径和抽帧参数，适配器负责校验、调用原脚本并返回生成数量与输出目录。

## 已接入示例：YOLO 数据集划分

`backend/yolo_dataset_split.py` 对 `tools/data_processing_toolkit/04-dataset_preparation/deal_with_yolo_data.py` 的目标检测流程进行了安全封装。输出目录直接包含平铺的 `train/`、`val/`、XML 转换生成的 `txts/`、图片索引文件和类别文件；它不会删除源文件，检测到已有结果时会停止，避免误覆盖。
