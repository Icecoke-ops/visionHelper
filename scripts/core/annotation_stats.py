#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注统计核心实现。

遍历目录下的图片与对应的 X-AnyLabeling JSON 标注文件，输出整体统计与
按标签的实例统计；同时提供 CLI 输出 JSON 块的生成 / 解析工具，便于 GUI
通过子进程方式调用。

所有常量（如边界标记）从 :mod:`scripts.config` 读取，所有日志通过
:mod:`scripts.logging_utils` 的 :func:`log` 输出。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from scripts._common import iter_images, iter_matched_pairs
from scripts.core.annotation_type import AnnotationType, AnnotationTypeChecker
from scripts.config import STATS_RESULT_BEGIN_MARKER, STATS_RESULT_END_MARKER
from scripts.logging_utils import log


__all__ = [
    "collect_annotation_stats",
    "collect_annotation_label_stats",
    "parse_machine_block",
    "emit_machine_block",
    "print_stats_human",
    "print_label_stats_human",
]


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
            unannotated_images: 未标注图片数
            detection_images: 包含 rectangle 的图片数
            obb_images: 包含 rotation 的图片数
            polygon_images: 包含 polygon 的图片数
            manual_images: 手动标注图片数
            auto_images: 自动标注图片数
            auto_corrected_images: 自动标注并手动矫正图片数

    异常:
        ValueError: 目录不存在或不是文件夹。
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {folder}")

    # 图片总数（按 stem 去重，与匹配逻辑保持一致）
    total_images = len({p.stem for p in iter_images(root)})

    annotated_images = 0
    detection_images = 0
    obb_images = 0
    polygon_images = 0
    manual_images = 0
    auto_images = 0
    auto_corrected_images = 0

    type_checker = AnnotationTypeChecker()

    for _image_path, ann_path, data in iter_matched_pairs(root, require_shapes=True):
        annotated_images += 1
        shapes = data.get("shapes", [])
        shape_types = _detect_shape_types(shapes)

        if "rectangle" in shape_types:
            detection_images += 1
        if "rotation" in shape_types:
            obb_images += 1
        if "polygon" in shape_types:
            polygon_images += 1

        try:
            json_mtime = ann_path.stat().st_mtime
        except OSError:
            json_mtime = 0.0
        ann_type = type_checker.check(data, json_mtime=json_mtime)
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

    遍历与图片匹配的 X-AnyLabeling JSON 文件，根据 ``label`` 与 ``shape_type``
    聚合统计每个标签的实例数量。

    参数:
        folder: 待统计的目录路径。

    返回:
        每个元素为一个字典，包含 ``label`` / ``detection_count`` /
        ``obb_count`` / ``polygon_count``。结果按标签名升序排序。

    异常:
        ValueError: 目录不存在或不是文件夹。
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {folder}")

    label_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "detection_count": 0,
            "obb_count": 0,
            "polygon_count": 0,
        }
    )

    for _image_path, _ann_path, data in iter_matched_pairs(root, require_shapes=True):
        for shape in data.get("shapes", []):
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


# --------------------------------------------------------------------------- #
# CLI 输出协议
# --------------------------------------------------------------------------- #

def print_stats_human(stats: Dict[str, int]) -> None:
    """以易读格式打印整体统计。"""
    log("===== 整体统计 =====")
    log(f"  图片总数        : {stats['total_images']}")
    log(f"  已标注          : {stats['annotated_images']}")
    log(f"  未标注          : {stats['unannotated_images']}")
    log(f"  目标检测数量    : {stats['detection_images']}")
    log(f"  OBB 数量        : {stats['obb_images']}")
    log(f"  多边形数量      : {stats['polygon_images']}")
    log(f"  手动标注数量    : {stats['manual_images']}")
    log(f"  自动标注数量    : {stats['auto_images']}")
    log(f"  手动矫正数量    : {stats['auto_corrected_images']}")


def print_label_stats_human(label_stats: List[Dict[str, int]]) -> None:
    """以易读格式打印按标签统计。"""
    log("===== 按标签统计 =====")
    if not label_stats:
        log("  （未发现任何标签实例）")
        return
    header = f"  {'标签名':<24}{'检测':>8}{'OBB':>8}{'多边形':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for item in label_stats:
        log(
            f"  {item.get('label', ''):<24}"
            f"{item.get('detection_count', 0):>8}"
            f"{item.get('obb_count', 0):>8}"
            f"{item.get('polygon_count', 0):>8}"
        )


def emit_machine_block(payload: Dict[str, object]) -> None:
    """
    输出供 GUI / 脚本解析的 JSON 块（用边界标记包裹）。

    JSON 内容会被 :data:`scripts.config.STATS_RESULT_BEGIN_MARKER` 与
    :data:`scripts.config.STATS_RESULT_END_MARKER` 包围，方便上层定位。
    """
    log(STATS_RESULT_BEGIN_MARKER)
    log(json.dumps(payload, ensure_ascii=False))
    log(STATS_RESULT_END_MARKER)


def parse_machine_block(output: str) -> Dict[str, object]:
    """
    从 CLI 输出中提取以边界标记包裹的 JSON 块并解析为字典。

    参数:
        output: 子进程输出的全部文本（通常是 ``stdout``）。

    返回:
        包含 ``stats`` 与 ``label_stats`` 字段的字典。

    异常:
        ValueError: 找不到边界标记或 JSON 解析失败时抛出。
    """
    if not isinstance(output, str):
        raise ValueError("output 必须为字符串")

    begin = output.rfind(STATS_RESULT_BEGIN_MARKER)
    if begin < 0:
        raise ValueError(f"未找到结果起始标记 {STATS_RESULT_BEGIN_MARKER}")
    end = output.find(STATS_RESULT_END_MARKER, begin + len(STATS_RESULT_BEGIN_MARKER))
    if end < 0:
        raise ValueError(f"未找到结果结束标记 {STATS_RESULT_END_MARKER}")

    payload_text = output[begin + len(STATS_RESULT_BEGIN_MARKER):end].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"解析结果 JSON 失败: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("结果 JSON 顶层必须是对象")

    payload.setdefault("stats", {})
    payload.setdefault("label_stats", [])
    return payload
