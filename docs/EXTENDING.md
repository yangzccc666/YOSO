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
- `POST /api/dialog`：打开本机文件/文件夹选择器
- `GET /api/open?path=...`：在系统文件管理器中打开结果

后续可以在不修改前端核心结构的情况下增加进度回调、任务取消、运行历史和自然语言参数生成。

## 已接入示例：视频切图

`backend/video_frames.py` 是第一个实际适配器。它加载：

`tools/data_processing_toolkit/02-video_processing/batch_vedio2img.py`

界面负责收集视频路径和抽帧参数，适配器负责校验、调用原脚本并返回生成数量与输出目录。

## 已接入示例：YOLO 数据集分配

`backend/yolo_dataset_split.py` 对 `tools/data_processing_toolkit/04-dataset_preparation/deal_with_yolo_data.py` 的核心流程进行了安全封装。它不会删除源文件或覆盖已有结果，并修复了原脚本中 VOC XML 验证集被复制到训练集目录的问题。
