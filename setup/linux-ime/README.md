# Linux 桌面中文输入法

这一步仅供维护者构建 Linux 的 Fcitx 原生插件。Windows、macOS 不使用此插件；普通用户不需要重装输入法。

## 为什么不能只设 xim

Qt6 的 Compose 插件也声明 `xim` 名称，但它处理的是键盘组合字符，不会替代搜狗/Fcitx 的原生输入通道。系统的 Fcitx Qt 插件还依赖 Qt 私有 ABI，不能直接复制不同版本的 `.so` 到平台中。

启动器会检查插件元数据和实际可加载性，优先使用平台专用、Qt 版本及架构匹配并通过 SHA-256 校验的插件，然后检查 Qt 已有搜索路径中的原生插件。不会加入系统整套 Qt 插件路径，不会设置全局环境变量、替换系统文件、安装软件包或重启输入法服务。

## Linux x86_64 发行包

- Qt/PySide6：6.9.2，Linux x86_64。
- Linux CI 使用固定源码和匹配的 Qt SDK 构建，打包后检查插件能加载。此插件面向 Fcitx4／搜狗，不替换系统输入法。
- 平台专用插件：`.desktop-ime/qt-6.9.2-x86_64/platforminputcontexts/libfcitxplatforminputcontextplugin-qt6.so`。
- 该目录属于本地构建产物，不提交到 Git；发布 Linux 包时由打包脚本复制进去。
- 更新安装包后，等任务空闲再退出旧版并打开新版。

## 重建插件

准备与平台 Python 的 Qt **完全匹配**的 Qt 开发包，必须包含 QtGui/QtCore 私有头文件、`moc` 和 `qdbusxml2cpp`。还需要 `g++`、XKBCommon 和 X11 开发头文件。将开发文件解压到临时目录即可，不需要更改现有 Python/Conda 环境。

从官方仓库获取 Fcitx 源码（本次使用的固定版本）：

```bash
git clone https://github.com/fcitx/fcitx-qt5.git /tmp/yoso-fcitx-source
git -C /tmp/yoso-fcitx-source checkout --detach c328b03f32a4276d9aeb3213d1f3f959ef6848f2
python setup/linux-ime/build_fcitx.py --source /tmp/yoso-fcitx-source --qt-sdk /path/to/extracted/qt-sdk
```

当前 Conda Qt 对应的 SDK 来自官方源 `qtbase-devel-6.9.2-hfd094ee_5.conda`；SHA-256 为 `7fcf42b85f60f89bd85f418a57a47189ebb0ddb12ea7f8ed4ac3332cd6ccd104`。SDK 只用于编译，不随平台分发，不替换现有 Qt。其他 Python 环境必须使用各自匹配的 SDK，不能套用此文件。

脚本只构建上游 Qt6 平台输入插件，不构建其他库，不执行 `make install`。输出包含插件、校验清单、BSD-3-Clause 许可证；工具子进程的库搜索路径仅在编译时临时设置。Fcitx5 用户应准备支持 Fcitx5 协议、匹配其 Qt 的原生插件；不能仅凭服务存在就认定 Fcitx4 插件可替代。

## 打包与检查

Linux 打包前先构建对应插件。设置 `PROCESSING_VIEW_REQUIRE_FCITX=1` 可让安装包构建在缺少匹配插件时明确失败，避免漏打包。构建结果会包括插件与许可证；其他操作系统忽略此项。

```bash
PROCESSING_VIEW_REQUIRE_FCITX=1 python -m PyInstaller packaging/yolo_data_platform.spec
```

升级 PySide6/Qt 或切换 CPU 架构后必须重新构建，不能继续加载旧产物。启动日志应显示 `fcitx（匹配当前 Qt 的原生插件已加载）`。此日志只证明插件加载成功；真实输入还需在桌面文本框中验证搜狗候选词、选词提交、退格和切换焦点。

若插件或会话服务不可用，会输出具体诊断，不会再错误回退到 `xim`。临时可在系统浏览器中打开**正在运行的平台**打印的 `http://127.0.0.1:端口/` 地址，保留桌面窗口，复用同一个后端；不要同时启动多个后端读写同一份配置。

`PROCESSING_VIEW_QT_IM_MODULE` 仍可作为进程级人工覆盖。不要为了本平台修改 `.profile`、搜狗/Fcitx 全局配置或系统 Qt 目录。

参考：[Fcitx 官方源码](https://github.com/fcitx/fcitx-qt5)、[Fcitx 关于应用自带 Qt 的说明](https://fcitx-im.org/wiki/FAQ/en)。
