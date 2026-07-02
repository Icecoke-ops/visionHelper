#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端公共配置（``scripts`` 包专用）。

本模块集中存放所有 **与具体业务无关、但被多处复用** 的常量参数，例如支持的
任务类型、图片扩展名、CLI 输出协议标记、进度日志默认参数等。

设计原则
--------

1. **纯常量、零副作用**：仅放 ``Final`` 字面量，不写函数、不做 I/O；
2. **零依赖**：只依赖标准库，不 import ``scripts`` 其它任何子模块，也不
   依赖 ``torch`` / ``ultralytics`` / ``cv2`` / ``transformers`` 等重依赖；
3. **与 GUI 完全解耦**：``gui/`` 拥有自己独立的 ``gui/config.py``，本文件
   绝不会被 GUI 引用，也不要 import ``gui``。这样 GUI 能独立打包，
   ``scripts`` 也能在没有 PySide6 的环境下被调用。

希望从这里读取常量的代码应使用形如 ``from scripts.common.config import IMAGE_EXTENSIONS``
的明确路径，便于一眼定位来源。
"""

from typing import Dict, FrozenSet, Tuple

# --------------------------------------------------------------------------- #
# 通用常量
# --------------------------------------------------------------------------- #

#: 后端支持的图片扩展名（统一小写、带点号）。
IMAGE_EXTENSIONS: FrozenSet[str] = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".gif",
})

#: 支持的视觉任务类型。``detect`` / ``obb`` / ``segment`` / ``classify``。
SUPPORTED_TASKS: FrozenSet[str] = frozenset({"detect", "obb", "segment", "classify"})

# --------------------------------------------------------------------------- #
# annotation_stats CLI 输出协议
# --------------------------------------------------------------------------- #

#: ``python scripts/vh.py datasets stats`` 在 stdout 中输出结果 JSON 块时使用
#: 的起始 / 结束标记，GUI 通过这两个标记定位并提取 JSON。
STATS_RESULT_BEGIN_MARKER: str = "===VH_STATS_BEGIN==="
STATS_RESULT_END_MARKER: str = "===VH_STATS_END==="

# --------------------------------------------------------------------------- #
# annotation_type
# --------------------------------------------------------------------------- #

#: X-AnyLabeling JSON 中标记自动标注时间的键名。
AUTO_ANNOTATED_TIME_KEY: str = "auto_annotated_time"

#: 区分 "自动标注" 与 "自动标注后人工矫正" 的默认时间容差（秒）。
DEFAULT_TOLERANCE_SECONDS: float = 2.0

# --------------------------------------------------------------------------- #
# auto_annotate
# --------------------------------------------------------------------------- #

#: 自动标注批量推理的默认 batch size。
AUTO_ANNOTATE_DEFAULT_BATCH_SIZE: int = 8

#: 自动标注流程中识别的图片现有标注状态。
STATUS_UNANNOTATED: str = "unannotated"
STATUS_MANUAL: str = "manual"
STATUS_AUTO: str = "auto"
STATUS_AUTO_CORRECTED: str = "auto_corrected"

# --------------------------------------------------------------------------- #
# deduplicate_images
# --------------------------------------------------------------------------- #

#: 图片去重支持的特征后端。
SUPPORTED_DEDUP_BACKENDS: Tuple[str, ...] = ("vit", "phash")

#: ViT 去重的默认模型名称（HuggingFace 仓库标识）。
DEFAULT_VIT_MODEL: str = "google/vit-base-patch16-224"

#: ViT 去重的默认推理 batch size。
DEFAULT_VIT_BATCH_SIZE: int = 8

#: pHash 去重的默认哈希尺寸，向量维度 = ``hash_size ** 2``。
DEFAULT_PHASH_SIZE: int = 16

#: 图片去重网格分块大小（1 表示不分块，2 表示 2×2 网格，以此类推）。
DEFAULT_GRID_SIZE: int = 1

# --------------------------------------------------------------------------- #
# export_yolo_dataset
# --------------------------------------------------------------------------- #

#: YOLO 数据集图片子目录名称。
IMAGES_FOLDER: str = "images"

#: YOLO 数据集标签子目录名称。
LABELS_FOLDER: str = "labels"

#: 导出 YOLO 数据集时支持的图片落盘模式。
COPY_MODES: FrozenSet[str] = frozenset({"copy", "link", "symlink"})

#: detect 任务可接受的 X-AnyLabeling shape_type。
DETECT_SHAPE_TYPES: FrozenSet[str] = frozenset({"rectangle", "polygon", "rotation"})

#: obb 任务可接受的 X-AnyLabeling shape_type。
OBB_SHAPE_TYPES: FrozenSet[str] = frozenset({"rectangle", "rotation"})

#: segment 任务可接受的 X-AnyLabeling shape_type。
SEGMENT_SHAPE_TYPES: FrozenSet[str] = frozenset({"polygon"})

# --------------------------------------------------------------------------- #
# extract_video_frames
# --------------------------------------------------------------------------- #

#: 视频抽帧支持的输出图片格式。
SUPPORTED_VIDEO_FRAME_EXTENSIONS: FrozenSet[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)

#: 视频抽帧支持的跳帧策略。
SUPPORTED_SEEK_MODES: FrozenSet[str] = frozenset({"decode_all", "seek"})

# --------------------------------------------------------------------------- #
# train_model
# --------------------------------------------------------------------------- #

#: 训练支持的优化器名称。
SUPPORTED_OPTIMIZERS: FrozenSet[str] = frozenset(
    {"auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"}
)

#: 各任务对应的 Ultralytics 预训练权重后缀。
TASK_MODEL_SUFFIX: Dict[str, str] = {
    "detect": "",
    "obb": "-obb",
    "segment": "-seg",
    "classify": "-cls",
}

# --------------------------------------------------------------------------- #
# 进度日志
# --------------------------------------------------------------------------- #

#: 项目版本号。
VERSION: str = "1.0.1"

#: 通过该环境变量可整体关闭进度输出（保留首尾汇总）。
PROGRESS_DISABLE_ENV: str = "VH_NO_PROGRESS"

#: 进度日志默认百分比里程碑间隔（百分点）。
PROGRESS_DEFAULT_STEP_PERCENT: float = 5.0

#: 进度日志默认最小输出间隔（秒），用于节流刷屏。
PROGRESS_DEFAULT_MIN_INTERVAL: float = 1.0
