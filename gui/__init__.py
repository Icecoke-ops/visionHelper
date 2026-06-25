#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui 包：存放 visionHelper 的图形界面相关模块。

入口脚本：
    - app.py: 启动 GUI 主窗口（导出 :func:`main`）

约定：
    - GUI 通过子进程调用 ``python scripts/vh.py <subcommand> <action> ...``
      执行后台任务，**不要**在 gui 模块内 ``import scripts.api`` 或直接
      调用 scripts 内的耗时函数，避免把 torch / ultralytics 等重型依赖
      拉进 GUI 主进程。
"""

from gui.app import main

__all__ = ["main"]
