# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec：单体 exe 打包 visionHelper GUI。

构建命令::

    pyinstaller --noconfirm gui_main.spec

构建产物::

    dist/
    └── gui_main(.exe)

发布时把 ``scripts/`` 目录原样拷贝到 exe 同级目录即可。
"""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None


hidden_imports = collect_submodules("gui")

_STDLIB_FOR_SCRIPTS = [
    "json", "csv", "shutil", "subprocess", "hashlib", "argparse",
    "datetime", "logging", "tempfile", "shlex", "time", "math",
    "random", "re", "typing", "enum", "pathlib", "collections",
    "concurrent", "concurrent.futures", "threading", "queue", "io",
    "os", "sys", "traceback", "functools", "itertools", "glob",
    "fnmatch", "pickle", "copy", "string", "warnings", "platform",
    "uuid", "base64", "zipfile", "tarfile", "gzip", "struct", "ctypes",
]
hidden_imports += _STDLIB_FOR_SCRIPTS

excluded_modules = [
    "torch", "torchvision", "torchaudio", "ultralytics", "transformers",
    "tokenizers", "huggingface_hub", "safetensors", "cv2", "numpy",
    "scipy", "pandas", "matplotlib", "sklearn", "PIL", "tqdm", "yaml",
    "scripts",
]

a = Analysis(
    ["gui_main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="visionHelper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
