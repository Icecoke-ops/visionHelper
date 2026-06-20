#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据标注统计模块。

遍历指定目录，统计图片数量及其对应的 LabelMe JSON 标注文件情况：
- 图片总数
- 已标注图片数量
- 包含目标检测框（rectangle）的图片数量
- 包含 OBB（rotation）的图片数量
- 包含多边形（polygon）的图片数量
- 手动标注、自动标注、自动标注并手动矫正的图片数量

标注文件与图片的匹配规则：
1. 优先读取 JSON 文件中的 ``imagePath`` 字段，取去掉后缀的文件名，
   与目录下图片去掉后缀的文件名进行匹配。
2. 若 ``imagePath`` 字段不存在或对应的图片不存在，则退化为用 JSON
   文件自身去掉后缀的文件名与图片名进行匹配。
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from scripts._common import (
    is_annotation_file,
    is_image_file,
    load_annotation,
    resolve_image_stem,
)
from scripts.annotation_type import AnnotationType, AnnotationTypeChecker


def _detect_shape_types(shapes: List[dict]) -> Set[str]:
    """提取 shapes 中出现的 shape_type 集合。"""
    types: Set[str] = set()
    for shape in shapes:
        if isinstance(shape, dict):
            shape_type = shape.get("shape_type")
            if isinstance(shape_type, str):
                types.add(shape_type)
    return types


def collect_annotation_stats(folder: str) -> Dict[str, int]:
    """
    统计目录下的图片与标注信息。

    参数:
        folder: 待统计的目录路径。

    返回:
        包含以下键的字典：
            total_images: 图片总数
            annotated_images: 已标注图片数（存在对应 JSON 且 shapes 非空）
            unannotated_images: 未标注图片数（不存在对应 JSON 或 JSON 中 shapes 为空）
            detection_images: 包含 rectangle 的图片数
            obb_images: 包含 rotation 的图片数
            polygon_images: 包含 polygon 的图片数
            manual_images: 手动标注图片数
            auto_images: 自动标注图片数
            auto_corrected_images: 自动标注并手动矫正图片数
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {folder}")

    # 收集目录下所有图片（按去掉后缀的文件名索引）
    image_paths_by_stem: Dict[str, Path] = {}
    for path in root.iterdir():
        if is_image_file(path):
            image_paths_by_stem.setdefault(path.stem, path)

    total_images = len(image_paths_by_stem)
    annotated_images = 0
    detection_images = 0
    obb_images = 0
    polygon_images = 0
    manual_images = 0
    auto_images = 0
    auto_corrected_images = 0

    # 记录已匹配的图片 stem，避免重复统计
    matched_stems: Set[str] = set()
    type_checker = AnnotationTypeChecker()

    for path in root.iterdir():
        if not is_annotation_file(path):
            continue

        data = load_annotation(path)
        if data is None:
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        image_stem = resolve_image_stem(path, data)
        if image_stem is None or image_stem not in image_paths_by_stem:
            continue
        if image_stem in matched_stems:
            continue

        matched_stems.add(image_stem)
        annotated_images += 1
        shape_types = _detect_shape_types(shapes)

        if "rectangle" in shape_types:
            detection_images += 1
        if "rotation" in shape_types:
            obb_images += 1
        if "polygon" in shape_types:
            polygon_images += 1

        ann_type = type_checker.check(data, json_mtime=path.stat().st_mtime)
        if ann_type == AnnotationType.MANUAL:
            manual_images += 1
        elif ann_type == AnnotationType.AUTO:
            auto_images += 1
        elif ann_type == AnnotationType.AUTO_CORRECTED:
            auto_corrected_images += 1

    unannotated_images = total_images - annotated_images

    return {
        "total_images": total_images,
        "annotated_images": annotated_images,
        "unannotated_images": unannotated_images,
        "detection_images": detection_images,
        "obb_images": obb_images,
        "polygon_images": polygon_images,
        "manual_images": manual_images,
        "auto_images": auto_images,
        "auto_corrected_images": auto_corrected_images,
    }


def collect_annotation_label_stats(folder: str) -> List[Dict[str, int]]:
    """
    按标签统计目录下的标注实例数量。

    遍历与图片匹配的 LabelMe JSON 文件，根据 ``label`` 与 ``shape_type``
    聚合统计每个标签的实例数量。

    参数:
        folder: 待统计的目录路径。

    返回:
        每个元素为一个字典，包含：
            label: 标签名
            detection_count: rectangle 类型实例数量
            obb_count: rotation 类型实例数量
            polygon_count: polygon 类型实例数量
        结果按标签名字母顺序排序。
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {folder}")

    # 收集目录下所有图片（按去掉后缀的文件名索引）
    image_paths_by_stem: Dict[str, Path] = {}
    for path in root.iterdir():
        if is_image_file(path):
            image_paths_by_stem.setdefault(path.stem, path)

    # 按标签统计三种 shape_type 的实例数量
    label_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "detection_count": 0,
            "obb_count": 0,
            "polygon_count": 0,
        }
    )
    matched_stems: Set[str] = set()

    for path in root.iterdir():
        if not is_annotation_file(path):
            continue

        data = load_annotation(path)
        if data is None:
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        image_stem = resolve_image_stem(path, data)
        if image_stem is None or image_stem not in image_paths_by_stem:
            continue
        if image_stem in matched_stems:
            continue

        matched_stems.add(image_stem)

        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            label = shape.get("label")
            shape_type = shape.get("shape_type")
            if not isinstance(label, str) or not isinstance(shape_type, str):
                continue
            counts = label_counts[label]
            if shape_type == "rectangle":
                counts["detection_count"] += 1
            elif shape_type == "rotation":
                counts["obb_count"] += 1
            elif shape_type == "polygon":
                counts["polygon_count"] += 1

    return [
        {"label": label, **counts}
        for label, counts in sorted(label_counts.items(), key=lambda item: item[0])
    ]
