#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts.common 子包：存放 visionHelper 后端共享的轻量工具与常量。

设计约束
--------

- **零副作用 import**：本包在 import 时不得触发任何重型依赖（torch / ultralytics /
  cv2 / transformers 等），确保 GUI 打包进程能安全 import ``scripts.common``；
- **零横向依赖**：公共模块只依赖标准库或更底层的公共模块，不依赖 ``scripts.images`` /
  ``scripts.datasets`` 等业务子包。

主要模块
--------

- :mod:`scripts.common.config`：纯常量配置（图片扩展名、任务类型、进度日志参数等）。
- :mod:`scripts.common.logging`：统一日志输出 ``log()`` 与 ``ProgressLogger``。
- :mod:`scripts.common.annotation_type`：X-AnyLabeling 标注类型判定（MANUAL / AUTO /
  AUTO_CORRECTED）。
- :mod:`scripts.common.utils`：轻量 IO 工具（图片 / 标注文件判定、目录遍历、模型发现等）。
"""

__all__: list[str] = []
