# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
import os
import json
import hashlib
import importlib.util
from importlib import metadata

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata


ROOT = Path(SPECPATH).parent
APP_NAME = "YOLO-Data-Platform"
VERSION = os.environ.get("RELEASE_VERSION", "v26.1.6").removeprefix("v")
PROFILE = os.environ.get("YOLO_BUILD_PROFILE", "standard").strip()
AI_MODULES = ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "onnxslim")
if PROFILE not in {"standard", "ai"}:
    raise SystemExit("YOLO_BUILD_PROFILE must be standard or ai")
if PROFILE == "ai":
    if os.environ.get("YOLO_AI_LICENSE_ACKNOWLEDGED") != "1":
        raise SystemExit("Review AI dependency redistribution licenses, then set YOLO_AI_LICENSE_ACKNOWLEDGED=1.")
    missing = [name for name in AI_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("AI build cannot proceed with missing packages: " + ", ".join(missing))

extra_datas, extra_binaries, extra_hidden = collect_all("imageio_ffmpeg")
ime_packaged_relative = None
if sys.platform.startswith("linux"):
    from PySide6.QtCore import qVersion
    import platform
    import importlib.util

    ime_spec = importlib.util.spec_from_file_location("desktop_input_method", ROOT / "desktop_input_method.py")
    ime_module = importlib.util.module_from_spec(ime_spec)
    sys.modules[ime_spec.name] = ime_module
    ime_spec.loader.exec_module(ime_module)
    ime_root = ime_module.private_plugin_root(ROOT, qVersion(), platform.machine())
    extra_hidden.append("PySide6.QtDBus")
    if ime_root.exists():
        ime_plugin = ime_module.validate_private_plugin(ime_root, qVersion())
        ok, error, _ = ime_module.load_fcitx_plugin(ime_plugin)
        if not ok:
            raise SystemExit("Fcitx plugin is incompatible with build Qt: " + error)
        destination = str(ime_root.relative_to(ROOT))
        extra_binaries.append((str(ime_plugin), destination + "/platforminputcontexts"))
        ime_packaged_relative = ime_root.relative_to(ROOT)
        extra_datas.extend((str(ime_root / name), destination)
                           for name in ("manifest.json", "COPYING.Fcitx.BSD"))
    elif os.environ.get("PROCESSING_VIEW_REQUIRE_FCITX") == "1":
        raise SystemExit("Build the matching Linux IME plugin first; see setup/linux-ime/README.md")
    else:
        print("WARNING: Linux build has no private Fcitx plugin; see setup/linux-ime/README.md")
if PROFILE == "ai":
    for module in AI_MODULES:
        module_datas, module_binaries, module_hidden = collect_all(module)
        extra_datas.extend(module_datas)
        extra_binaries.extend(module_binaries)
        extra_hidden.extend(module_hidden)
        extra_datas.extend(copy_metadata(module))

manifest_dir = ROOT / "build" / "distribution"
manifest_dir.mkdir(parents=True, exist_ok=True)
manifest = {"profile": PROFILE, "version": VERSION, "dependencies": {}}
for module in (*(AI_MODULES if PROFILE == "ai" else ()), "PySide6", "imageio-ffmpeg"):
    try:
        manifest["dependencies"][module] = metadata.version(module)
    except metadata.PackageNotFoundError:
        pass
manifest_file = manifest_dir / "distribution-manifest.json"
manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

datas = [
    *[(str(ROOT / "setup" / "yolo-runtime" / filename), "AI-Environment-Setup")
      for filename in ("install_yolo_env.bat", "install_yolo_env.py", "requirements-yolo.txt", "README.md")],
    (str(ROOT / "frontend" / "dist"), "frontend/dist"),
    (str(ROOT / "backend" / "*.py"), "backend"),
    (str(ROOT / "backend" / "license_public.json"), "backend"),
    (str(ROOT / "README.md"), "."),
    (str(manifest_file), "."),
    *extra_datas,
]

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=extra_binaries,
    datas=datas,
    hiddenimports=[
        *collect_submodules("backend"),
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineCore",
        *extra_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        *(list(AI_MODULES) if PROFILE == "standard" else []),
        "tensorrt",
        "pycuda",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if ime_packaged_relative is not None:
    # PyInstaller can relocate ELF RUNPATHs even when strip/upx are disabled.
    # Hash the FINAL artifact, without changing the source build's manifest.
    packaged_ime = Path(coll.name) / (exe.contents_directory or "") / ime_packaged_relative
    packaged_manifest = packaged_ime / "manifest.json"
    plugin_manifest = json.loads(packaged_manifest.read_text(encoding="utf-8"))
    packaged_plugin = packaged_ime / "platforminputcontexts" / plugin_manifest["plugin"]
    plugin_manifest["sha256"] = hashlib.sha256(packaged_plugin.read_bytes()).hexdigest()
    packaged_manifest.write_text(json.dumps(plugin_manifest, indent=2) + "\n", encoding="utf-8")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="com.yoso.yolodataplatform",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "YOLO数据处理平台",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
