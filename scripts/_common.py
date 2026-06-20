#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts 包内部共享工具函数。

集中管理图片扩展名、LabelMe JSON 标注文件的读取与图片解析等
跨模块复用的辅助逻辑，避免在 ``annotation_stats``、``auto_annotate``、
``export_yolo_dataset`` 等模块中重复实现。

仅供 ``scripts`` 包内部使用，不对外暴露（模块名以下划线开头）。
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

# 常见图片扩展名（统一小写、带点号）
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".gif",
)


def is_image_file(path: Path) -> bool:
    """判断文件是否为支持的图片文件。"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_annotation_file(path: Path) -> bool:
    """判断文件是否为 LabelMe JSON 标注文件。"""
    return path.is_file() and path.suffix.lower() == ".json"


def load_annotation(annotation_path: Path) -> Optional[dict]:
    """安全加载 LabelMe JSON 标注文件，失败时返回 ``None``。"""
    try:
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def resolve_image_stem(annotation_path: Path, data: dict) -> Optional[str]:
    """
    根据标注文件内容解析其对应的图片去后缀文件名（stem）。

    优先使用 JSON 中的 ``imagePath`` 字段，回退为标注文件自身的 stem。
    """
    image_path = data.get("imagePath")
    if isinstance(image_path, str) and image_path:
        return Path(image_path).stem
    return annotation_path.stem


def resolve_image_path(annotation_path: Path, data: dict) -> Optional[Path]:
    """
    根据标注文件解析对应的图片绝对路径。

    优先使用 ``imagePath`` 字段；若不存在或对应文件不存在，则退化为
    在标注文件所在目录下按支持的扩展名查找同 stem 的图片文件。
    """
    root = annotation_path.parent
    image_path_value = data.get("imagePath")
    if isinstance(image_path_value, str) and image_path_value:
        candidate = root / image_path_value
        if candidate.is_file() and is_image_file(candidate):
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = root / f"{annotation_path.stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
    """
    扫描 runs 目录下的训练模型。

    Ultralytics 默认结构为 ``runs/<train_name>/weights/<name>.pt``，本函数
    枚举该结构并返回 ``(显示名称, 权重绝对路径)`` 列表，显示名称形如
    ``训练名称-模型权重名称``（例如 ``first-best``）。

    本函数只依赖标准库，便于在轻量 GUI 进程中直接调用，避免引入
    ``ultralytics`` / ``torch`` / ``PIL`` 等重型依赖。

    参数:
        runs_dir: 训练结果根目录。

    返回:
        模型显示名称与模型文件路径的列表，按显示名升序排序。
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return []

    models: List[Tuple[str, str]] = []
    for train_dir in runs_path.iterdir():
        if not train_dir.is_dir():
            continue
        weights_dir = train_dir / "weights"
        if not weights_dir.is_dir():
            continue
        for weight_file in weights_dir.iterdir():
            if weight_file.is_file() and weight_file.suffix.lower() == ".pt":
                display_name = f"{train_dir.name}-{weight_file.stem}"
                models.append((display_name, str(weight_file)))

    models.sort(key=lambda item: item[0])
    return models
