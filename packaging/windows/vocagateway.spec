# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows VocaGateway build.

One-directory layout: faster startup than one-file and the Inno Setup script
(packaging/windows/setup.iss) packages the directory verbatim. Build from the
repository root:

    uv run --with pyinstaller pyinstaller packaging/windows/vocagateway.spec

The Windows installer build also drops a bundled ``ffmpeg.exe`` into the
resulting ``dist/vocagateway/`` directory; ``app.audio`` finds it there via
``sys.executable`` when the app is frozen.
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir, os.pardir))

# The app resolves its non-Python assets relative to app/__file__, which under
# a frozen build lands inside the collected dist directory. Bundling them under
# app/ keeps `Path(__file__).parent / "templates"` (and friends) working.
datas = [
    (os.path.join(REPO_ROOT, "app", "templates"), "app/templates"),
    (os.path.join(REPO_ROOT, "app", "webui"), "app/webui"),
    (os.path.join(REPO_ROOT, "app", "model_pins.json"), "app"),
    (os.path.join(REPO_ROOT, "app", "cleanup_model_pins.json"), "app"),
]
binaries = []
hiddenimports = collect_submodules("pysbd")

# Speech engines are loaded lazily behind importlib.util.find_spec, so import
# analysis never sees them. Collect each installed package wholesale —
# binaries (native DLLs), data files, and submodules — and silently skip any
# engine extra that is not in the environment.
for package in (
    "sherpa_onnx",
    "sherpa_onnx_bin",
    "faster_whisper",
    "ctranslate2",
    "moonshine_voice",
    "onnxruntime",
    "tokenizers",
    "huggingface_hub",
    "av",
    "numpy",
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:  # noqa: BLE001 — package absent from the environment
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    [os.path.join(SPECPATH, "entry.py")],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Apple-silicon engines are guarded behind a Darwin sys_platform
        # marker and can never exist in a Windows environment.
        "mlx_audio",
        "uvloop",
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
    name="vocagateway",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
    upx_exclude=[],
    name="vocagateway",
)
