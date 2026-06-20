#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyInstaller 打包入口脚本。

PyInstaller 直接打包 ``gui/app.py`` 时会因为入口位于子包内，导致冻结后
``import gui.xxx`` 形态变得别扭；因此提供一个最薄的顶层入口文件，仅负
责调用 :func:`gui.app.main`。

开发期同样可以用：``python gui_main.py`` 启动 GUI（与 ``python -m gui.app``
等价）。

打包态特别处理
--------------

发布物形如::

    dist/visionHelper/
    ├── visionHelper(.exe)        ← 入口可执行文件
    ├── _internal/...             ← PyInstaller 运行时
    └── scripts/                  ← 与 exe 同级的 scripts 源码目录

由于 ``scripts/`` 不进 exe（重型依赖运行期由用户解释器加载），冻结进程
启动后 ``sys.path`` 中并不会自动包含它。但 GUI 端的若干页面在交互时仍
需要 ``import scripts._common`` / ``import scripts.annotation_stats``
等"轻量子模块"做本地工作，因此入口处需要把 exe 同级目录显式加入
``sys.path``，让 ``import scripts.xxx`` 能够定位到源码。
"""

import os
import sys
from pathlib import Path


def _ensure_scripts_on_syspath() -> None:
    """在打包态下把 exe 同级目录加入 ``sys.path``，便于 import 同级 ``scripts/``。

    - 开发态：项目根目录天然在 ``sys.path`` 中（``python gui_main.py`` 或
      ``python -m gui.app`` 启动），无需处理。
    - 冻结态（``sys.frozen``）：``sys.executable`` 指向打包后的可执行文件，
      其同级目录就是 ``scripts/`` 的父目录，把它放到 ``sys.path`` 头部即可。
    - 同时支持 ``VISIONHELPER_APP_ROOT`` 环境变量覆盖，便于自定义部署或调试。
    """
    candidates = []

    override = os.environ.get("VISIONHELPER_APP_ROOT", "").strip()
    if override:
        candidates.append(Path(override))

    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent)

    for root in candidates:
        try:
            root_str = str(root.resolve())
        except OSError:
            continue
        if root_str and root_str not in sys.path:
            sys.path.insert(0, root_str)


_ensure_scripts_on_syspath()


from gui.app import main


if __name__ == "__main__":
    main()
