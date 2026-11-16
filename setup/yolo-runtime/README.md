# Windows YOLO 环境安装

前提：已安装 Anaconda 或 Miniconda，需要联网下载依赖。将整个目录解压后双击 `install_yolo_env.bat`，不要在压缩包内部直接运行。

脚本创建独立的 Python 3.11 `yolo26` 环境，安装 CPU PyTorch、Ultralytics、ONNX、ONNX Runtime 和 ONNXSlim。不会向 base 环境安装软件，也不设置系统环境变量。完成后复制输出的 `python.exe` 路径到平台“本机设置 → AI Python 路径”；旧版本可填自动标注页的 Python 路径。设备先选 `cpu`。环境路径同时写入同目录 `yolo_python_path.txt`。

若双击找不到 Anaconda，请打开 **Anaconda Prompt**，进入本目录，再运行：

```bat
install_yolo_env.bat
```

已有环境会先检查，通过后直接复用；检查不通过不会自动修改，可选择另外创建：

```bat
install_yolo_env.bat --env yolo26_platform
```

安装中断留下不完整环境时，确认可以更新该环境依赖后再修复：

```bat
install_yolo_env.bat --repair
```

修复会更新所选环境的依赖。已有项目依赖这个环境时，优先新建环境，避免影响其他项目。脚本不承诺任意历史依赖组合都能原地修好，也不会删除环境。

默认 CPU 版可以用于平台本地 PT 自动标注、PT/ONNX 推理和 YOLO26 ONNX 导出，不包含 YOLOv5 专用源码、TensorRT、STAR 第三方打包工具或 Docker 环境。自定义源码训练的 PT 还可能需要相同版本的模型代码。

GPU 加速不是默认选项：请按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 为目标显卡及驱动选择配套的 torch / torchvision，再安装本目录 requirements。requirements 是版本范围而非精确锁文件；用于运行环境，不是桌面程序打包依赖。

如果只需要手动安装依赖：先在 Anaconda Prompt 创建并激活独立环境，再安装 CPU PyTorch 和 requirements：

```bat
conda create -n yolo26 python=3.11 pip -y
conda activate yolo26
python -m pip install "torch>=2.2,<3" "torchvision>=0.17,<1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-yolo.txt
```

不需要另外安装完整 CUDA Toolkit 才能使用 CPU 模式。安装体积及耗时取决于下载源和依赖版本；网络不可用时无法完成安装。本安装器不会替你接受 Anaconda 仓库条款，也不会改变商业许可适用范围。
