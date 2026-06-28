#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``scripts`` 包内部共享的 **轻量** IO 工具集。

集中管理跨模块复用的辅助逻辑：图片 / 标注文件判定、X-AnyLabeling JSON
读取、图片解析、目录遍历与已训练模型扫描等。所有逻辑只依赖标准库
（不引入 ``torch`` / ``ultralytics`` / ``cv2`` / ``transformers`` 等重依赖），
确保打包后的 GUI 进程也能在不触发 ``scripts.api`` 的情况下直接 import。

- 常量 :data:`IMAGE_EXTENSIONS` 与 :class:`ProgressLogger` 在此处仅作 **向后兼容
  的 re-export**：实现已分别迁移到 :mod:`scripts.common.config` 与 :mod:`scripts.common.logging`。
"""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from scripts.common.config import IMAGE_EXTENSIONS  # re-export，保持外部 import 兼容
from scripts.common.logging import ProgressLogger  # re-export，保持外部 import 兼容


__all__ = [
    "IMAGE_EXTENSIONS",
    "ProgressLogger",
    "discover_trained_models",
    "find_model_class_names",
    "is_annotation_file",
    "is_image_file",
    "iter_annotations",
    "iter_images",
    "iter_matched_pairs",
    "load_annotation",
    "resolve_image_path",
    "resolve_image_stem",
]


def find_model_class_names(model) -> list:
    """从 Ultralytics YOLO 模型中提取类别名称列表。"""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


def is_image_file(path: Path) -> bool:
    """判断文件是否为支持的图片文件。"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_annotation_file(path: Path) -> bool:
    """判断文件是否为 X-AnyLabeling JSON 标注文件。"""
    return path.is_file() and path.suffix.lower() == ".json"


def load_annotation(annotation_path: Path) -> Optional[dict]:
    """安全加载 X-AnyLabeling JSON 标注文件，失败时返回 ``None``。"""
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


def iter_images(folder: Path) -> Iterator[Path]:
    """按文件名排序遍历目录顶层的所有图片文件。"""
    if not folder.is_dir():
        return iter(())
    images = [p for p in folder.iterdir() if is_image_file(p)]
    images.sort(key=lambda p: p.name)
    return iter(images)


def iter_annotations(folder: Path) -> Iterator[Path]:
    """按文件名排序遍历目录顶层的所有 JSON 标注文件。"""
    if not folder.is_dir():
        return iter(())
    anns = [p for p in folder.iterdir() if is_annotation_file(p)]
    anns.sort(key=lambda p: p.name)
    return iter(anns)


def iter_matched_pairs(
        folder: Path,
        require_shapes: bool = False,
) -> Iterator[Tuple[Path, Path, dict]]:
    """
    遍历目录下与图片相匹配的 ``(image_path, annotation_path, data)`` 三元组。

    匹配规则：

    1. 标注文件 JSON 能被成功解析为 dict；
    2. 通过 :func:`resolve_image_path` 解析到的图片实际存在；
    3. 同一张图片（按 stem）只产出一次（标注文件按名称排序后取首个）。

    参数:
        folder: 待扫描的目录。
        require_shapes: 是否要求 ``shapes`` 非空（默认 ``False``）。

    产出顺序按图片文件名升序。
    """
    if not folder.is_dir():
        return

    # 先收集所有候选标注，按名称排序保证稳定性
    annotations = sorted(
        (p for p in folder.iterdir() if is_annotation_file(p)),
        key=lambda p: p.name,
    )

    pairs: List[Tuple[Path, Path, dict]] = []
    seen_stems: set = set()
    for ann_path in annotations:
        data = load_annotation(ann_path)
        if data is None:
            continue
        if require_shapes and not data.get("shapes"):
            continue
        image_path = resolve_image_path(ann_path, data)
        if image_path is None:
            continue
        if image_path.stem in seen_stems:
            continue
        seen_stems.add(image_path.stem)
        pairs.append((image_path, ann_path, data))

    pairs.sort(key=lambda item: item[0].name)
    for item in pairs:
        yield item


def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
    """
    扫描 ``runs`` 目录下的训练模型。

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
