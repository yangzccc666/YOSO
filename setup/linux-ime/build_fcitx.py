"""Build only Fcitx's BSD Qt6 input plugin; never install into Qt/system paths.

Usage: python setup/linux-ime/build_fcitx.py --source /path/to/fcitx-qt5 \
    --qt-sdk /path/to/extracted/matching/qtbase-devel
The SDK must match the Qt used by this Python. No pip/conda/apt installation
or input-method configuration is performed by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True, timeout=180)


def main() -> None:
    from PySide6.QtCore import QLibraryInfo, qVersion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qt-sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("This helper is Linux-only.")
    source = args.source.resolve() / "qt6" / "platforminputcontext"
    sdk = args.qt_sdk.resolve()
    version = qVersion()
    runtime_lib = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.LibrariesPath))
    includes = sdk / "include" / "qt6"
    tools = sdk / "lib" / "qt6"
    target = args.output or (Path(__file__).resolve().parents[2] / ".desktop-ime" /
                             f"qt-{version}-{platform.machine()}")
    target = target.resolve()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime_lib) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    actual = subprocess.check_output([str(tools / "moc"), "-v"], env=env, text=True, stderr=subprocess.STDOUT)
    if actual.strip() != f"moc {version}":
        parser.error(f"Qt SDK mismatch: {actual.strip()}, Python Qt={version}")
    if not (source / "main.cpp").is_file() or not (includes / "QtGui" / version / "QtGui" / "qpa").is_dir():
        parser.error("Missing Fcitx Qt6 source or matching Qt private headers.")

    with tempfile.TemporaryDirectory(prefix="yoso-fcitx-build-") as temporary:
        build = Path(temporary)
        flags = ["-I" + str(source), "-I" + str(build), "-I" + str(includes)]
        for module in ("QtCore", "QtGui", "QtDBus"):
            flags.extend(["-I" + str(includes / module),
                          "-I" + str(includes / module / version),
                          "-I" + str(includes / module / version / module)])
        cpp = [source / f"{name}.cpp" for name in
               ("main", "fcitxinputcontextproxy", "fcitxqtdbustypes", "fcitxwatcher", "qfcitxplatforminputcontext", "qtkey")]
        for interface, output in (("InputContext", "inputcontextproxy"), ("InputContext1", "inputcontext1proxy"),
                                  ("InputMethod", "inputmethodproxy"), ("InputMethod1", "inputmethod1proxy")):
            run([str(tools / "bin" / "qdbusxml2cpp"), "-i", "fcitxqtdbustypes.h", "-p", output,
                 str(source / f"org.fcitx.Fcitx.{interface}.xml")], env, build)
            cpp.append(build / f"{output}.cpp")
        headers = [source / f"{name}.h" for name in ("main", "fcitxinputcontextproxy", "fcitxwatcher", "qfcitxplatforminputcontext")]
        headers.extend(build / f"{name}.h" for name in ("inputcontextproxy", "inputcontext1proxy", "inputmethodproxy", "inputmethod1proxy"))
        for header in headers:
            output = build / f"moc_{header.stem}.cpp"
            run([str(tools / "moc"), "-DQT_NO_KEYWORDS", *flags, str(header), "-o", str(output)], env, build)
            cpp.append(output)
        output = build / "libfcitxplatforminputcontextplugin-qt6.so"
        run(["g++", "-std=c++17", "-O2", "-fPIC", "-fvisibility=hidden", "-shared", "-DQT_NO_KEYWORDS",
             *flags, *map(str, cpp), "-L" + str(runtime_lib), "-Wl,--no-undefined",
             "-l:libQt6Gui.so.6", "-l:libQt6DBus.so.6", "-l:libQt6Core.so.6", "-l:libxkbcommon.so.0",
             "-o", str(output)], env, build)
        # Validate against THIS Python's Qt, not just the development SDK.
        from PySide6 import QtGui, QtDBus  # noqa: F401
        from PySide6.QtCore import QPluginLoader
        loader = QPluginLoader(str(output))
        if not loader.load():
            raise RuntimeError(loader.errorString())
        loader.unload()
        destination = target / "platforminputcontexts" / output.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(f".{destination.name}.{os.getpid()}.new")
        try:
            shutil.copy2(output, pending)
            os.replace(pending, destination)
        finally:
            pending.unlink(missing_ok=True)
        shutil.copy2(args.source.resolve() / "COPYING.BSD", target / "COPYING.Fcitx.BSD")
        manifest = {"qt": version, "arch": platform.machine(), "plugin": output.name,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "source": "https://github.com/fcitx/fcitx-qt5", "license": "BSD-3-Clause"}
        manifest["source_files_sha256"] = {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.iterdir()) if path.is_file()
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"已构建平台专用 Fcitx 插件：{target}")
    print("未修改系统、Conda 环境、输入法服务或用户配置；平台空闲后重新打开即可生效。")


if __name__ == "__main__":
    main()
