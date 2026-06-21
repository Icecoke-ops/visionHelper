#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts 包：存放 visionHelper 的核心工具模块与统一对外 API。

当前包含以下模块：
    - api: 统一对外 API 接口（VideoAPI / ImageAPI / AnnotationAPI / TrainingAPI）
    - annotation_stats: 标注统计 CLI 门面（实现位于 ``scripts.core.annotation_stats``）
    - extract_video_frames: 视频抽帧 CLI 门面（实现位于 ``scripts.core.extract_video_frames``）
    - deduplicate_images: 图片去重 CLI 门面（实现位于 ``scripts.core.deduplicate_images``）
    - auto_annotate: YOLO 自动标注 CLI 门面（实现位于 ``scripts.core.auto_annotate``）
    - clear_annotations: 标注清除 CLI 门面（实现位于 ``scripts.core.clear_annotations``）
    - export_yolo_dataset: 导出 YOLO 数据集 CLI 门面（实现位于 ``scripts.core.export_yolo_dataset``）
    - train_model: 模型训练 CLI 门面（实现位于 ``scripts.core.train_model``）

    具体业务实现统一收纳于 ``scripts.core`` 子包，包括标注类型判断
    (``scripts.core.annotation_type``) 等核心模块。

推荐通过 `scripts.api` 调用功能：
    from scripts.api import VideoAPI, ImageAPI, AnnotationAPI, TrainingAPI

注意（打包态）
--------------

GUI 进程中只会 import 极少量轻量子模块（如 ``scripts._common``、
``scripts.core.annotation_type``、``scripts.annotation_stats``）做本地工作，
真正涉及 ``torch`` / ``ultralytics`` / ``cv2`` / ``transformers`` 等重依赖
的模块（``scripts.auto_annotate`` / ``scripts.train_model`` /
``scripts.deduplicate_images`` / ``scripts.extract_video_frames`` /
``scripts.export_yolo_dataset`` / ``scripts.api``）都是通过用户在 GUI
里选择的外部 Python 解释器以子进程方式运行的。

因此 ``scripts/__init__.py`` 必须保持"零副作用"——在 import 时**不要**
顺带 import 任何带重依赖的子模块，否则在 PyInstaller 打包后的 GUI 进程
里只要触碰 ``scripts._common`` 就会被 ``__init__.py`` 牵连，去 import
``scripts.api`` 进而 import ``torch`` / ``cv2`` 等，立即闪退。

如果上层代码确实想从 ``scripts`` 顶层访问 ``api`` / ``AnnotationType`` 等
名字，请使用更明确的导入路径，例如::

    from scripts.api import VideoAPI
    from scripts.core.annotation_type import AnnotationType
"""

__all__ = [
    "AnnotationAPI",
    "AnnotationType",
    "AnnotationTypeChecker",
    "ImageAPI",
    "TrainingAPI",
    "VideoAPI",
    "deduplicate",
    "extract_video_frames",
]


def __getattr__(name):
    """按需懒加载子模块中的常用符号。

    这样既可以保持 ``from scripts import VideoAPI`` 这种便捷写法在外部
    Python 环境（已安装 torch / ultralytics 等）下继续可用，又能让 GUI
    打包进程仅 import ``scripts._common`` / ``scripts.core.annotation_type``
    时不会触发任何重依赖。
    """
    if name in {"AnnotationType", "AnnotationTypeChecker"}:
        from scripts.core.annotation_type import AnnotationType, AnnotationTypeChecker

        return {
            "AnnotationType": AnnotationType,
            "AnnotationTypeChecker": AnnotationTypeChecker,
        }[name]

    if name in {"AnnotationAPI", "ImageAPI", "TrainingAPI", "VideoAPI"}:
        from scripts import api as _api

        return getattr(_api, name)

    if name == "deduplicate":
        from scripts.deduplicate_images import deduplicate

        return deduplicate

    if name == "extract_video_frames":
        from scripts.extract_video_frames import extract_video_frames

        return extract_video_frames

    raise AttributeError(f"module 'scripts' has no attribute {name!r}")
