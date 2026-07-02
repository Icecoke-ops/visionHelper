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

from scripts.datasets.auto import auto_annotate
from scripts.datasets.clear import clear_annotations
from scripts.datasets.export import export_yolo_dataset
from scripts.datasets.stats import (
    collect_all_stats,
    collect_annotation_label_stats,
    collect_annotation_stats,
    emit_machine_block,
    parse_machine_block,
    print_label_stats_human,
    print_stats_human,
)

__all__ = [
    "auto_annotate",
    "clear_annotations",
    "export_yolo_dataset",
    "collect_all_stats",
    "collect_annotation_label_stats",
    "collect_annotation_stats",
    "emit_machine_block",
    "parse_machine_block",
    "print_label_stats_human",
    "print_stats_human",
]
