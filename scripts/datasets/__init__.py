#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集处理子包（``scripts.datasets``）。

本包包含与标注数据集相关的工具：统计、清理、自动标注、YOLO 导出等。

零副作用约定
------------

模块顶层不执行任何 I/O，也不 import ``torch`` / ``ultralytics`` / ``cv2`` 等
重型依赖；这些依赖仅在具体函数被调用时按需加载。
"""

__all__: list[str] = []
