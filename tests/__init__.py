#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visionHelper 测试包。

测试统一使用 ``pytest`` 运行，从仓库根目录执行：

    python -m pytest tests -q

测试仅依赖项目内的轻量模块，必要时通过 ``unittest.mock`` 替换重型依赖
（如 ``ultralytics.YOLO``）以保证在无 GPU / 离线环境下可运行。
"""
