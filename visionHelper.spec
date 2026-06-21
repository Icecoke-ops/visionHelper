# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec：仅打包 visionHelper 的 GUI（gui/）。

设计要点
--------

1. 只把 ``gui/`` 与最薄的入口 ``gui_main.py`` 编进 exe；``scripts/`` **不**
   随包发布——发布时把 ``scripts/`` 目录原样拷贝到 exe 同级目录即可，
   GUI 运行期会通过 ``gui.config.app_root()`` 找到它。

2. 显式 ``excludes`` 掉所有重型依赖（torch / ultralytics / transformers /
   opencv 等），避免 PyInstaller 误以为 GUI 需要它们而把整个深度学习栈
   塞进 exe，导致体积膨胀到 GB 级别。

3. ``console=False`` 走窗口模式；如需排查启动崩溃，可临时改为
   ``console=True`` 或在终端里直接运行可执行文件查看错误输出。

构建命令::

    pyinstaller --noconfirm visionHelper.spec

构建产物::

    dist/visionHelper/
    ├── visionHelper(.exe)
    └── _internal/...

发布前在 ``dist/visionHelper/`` 下补上 ``scripts/`` 目录即可：

    cp -r scripts dist/visionHelper/scripts
"""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None


# 仅把 gui 包的子模块显式纳入隐式导入，确保动态导入也能找到。
hidden_imports = collect_submodules("gui")

# 运行期 GUI 会把同级 ``scripts/`` 源码目录加入 ``sys.path``，并 import 其中
# 的轻量子模块（如 ``scripts._common`` / ``scripts.core.annotation_type``）。
# 这些源码会用到一批标准库（json / csv / shutil / subprocess / hashlib /
# argparse 等），但 GUI 自身代码并未触发这些 import，导致 PyInstaller
# 默认不会把它们收进 PYZ，运行期就会出现 ``ModuleNotFoundError: No module
# named 'json'`` 这类报错。
#
# 因此显式声明一组 scripts 端可能用到的标准库为 hidden import，确保它们
# 进入打包产物。
_STDLIB_FOR_SCRIPTS = [
    "json",
    "csv",
    "shutil",
    "subprocess",
    "hashlib",
    "argparse",
    "datetime",
    "logging",
    "tempfile",
    "shlex",
    "time",
    "math",
    "random",
    "re",
    "typing",
    "enum",
    "pathlib",
    "collections",
    "concurrent",
    "concurrent.futures",
    "threading",
    "queue",
    "io",
    "os",
    "sys",
    "traceback",
    "functools",
    "itertools",
    "glob",
    "fnmatch",
    "pickle",
    "copy",
    "string",
    "warnings",
    "platform",
    "uuid",
    "base64",
    "zipfile",
    "tarfile",
    "gzip",
    "struct",
    "ctypes",
]
hidden_imports += _STDLIB_FOR_SCRIPTS



# 重型依赖：GUI 完全不需要，必须排除，否则 PyInstaller 会顺着
# requirements.txt 里的提示把它们全部收进来。
excluded_modules = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "cv2",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "sklearn",
    "PIL",
    "tqdm",
    "yaml",
    # 同时排除 scripts 包本身：scripts 不打进 exe，运行期由用户的 Python
    # 环境从同级 scripts/ 源码目录加载。
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
    [],
    exclude_binaries=True,
    name="visionHelper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="visionHelper",
)
