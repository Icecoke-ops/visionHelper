#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``scripts.core``：后端各功能的 **真实实现** 所在地。

设计目标
--------

为了让 ``scripts/`` 的外层文件只承担两个职责：

1. **CLI 门面**（argparse + ``main()``）；
2. **对外稳定 API 的 re-export**，

所有具体业务实现都被收纳进 ``scripts/core/<feature>.py``。

例如：

- :mod:`scripts.core.auto_annotate` —— YOLO 自动标注实现
- :mod:`scripts.core.deduplicate_images` —— 图片去重实现
- :mod:`scripts.core.extract_video_frames` —— 视频抽帧实现
- :mod:`scripts.core.export_yolo_dataset` —— YOLO 数据集导出
- :mod:`scripts.core.train_model` —— Ultralytics YOLO 训练
- :mod:`scripts.core.annotation_stats` —— 标注统计
- :mod:`scripts.core.clear_annotations` —— 按类型清除标注

本包 **不暴露聚合 API**，请直接 ``from scripts.core.<feature> import ...``，
便于 IDE 跳转与 PyInstaller 的依赖分析。

本 ``__init__.py`` 保持 "零副作用"——不 import 任何子模块（与 ``scripts/__init__.py``
的策略一致），以避免在 GUI 进程中误触发 ``torch`` / ``ultralytics`` / ``cv2`` /
``transformers`` 等重依赖。
"""

__all__: list = []
