#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``python scripts/vh.py datasets stats`` 命令实现。

遍历目录下的图片与对应的 X-AnyLabeling JSON 标注文件，输出整体统计与
按标签的实例统计；同时提供供 GUI 解析的 JSON 块生成 / 解析工具。

支持两种标注形式：

- 目标检测 / OBB / 分割：标注存放在 ``shapes`` 中（``rectangle`` / ``rotation`` / ``polygon``）。
- 图像分类：标注存放在顶层 ``flags`` 中（值为 True 的 key 即为该图所属类别）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from scripts.common.annotation_type import AnnotationType, AnnotationTypeChecker
from scripts.common.config import (
    STATS_RESULT_BEGIN_MARKER,
    STATS_RESULT_END_MARKER,
)
from scripts.common.logging import log
from scripts.common.utils import iter_images, iter_matched_pairs


__all__ = [
    "collect_all_stats",
    "collect_annotation_stats",
    "collect_annotation_label_stats",
    "emit_machine_block",
    "parse_machine_block",
    "print_stats_human",
    "print_label_stats_human",
    "main",
]


# --------------------------------------------------------------------------- #
# 核心统计
# --------------------------------------------------------------------------- #

def _detect_shape_types(shapes: List[dict]) -> Set[str]:
    """提取 shapes 中出现的 shape_type 集合。"""
    types: Set[str] = set()
    for shape in shapes:
        if isinstance(shape, dict):
            shape_type = shape.get("shape_type")
            if isinstance(shape_type, str):
                types.add(shape_type)
    return types


def _classification_labels(data: dict) -> List[str]:
    """提取分类标注标签：顶层 ``flags`` 中值为 True 的 key。"""
    flags = data.get("flags")
    if not isinstance(flags, dict):
        return []
    return [
        key for key, value in flags.items()
        if isinstance(key, str) and key and isinstance(value, bool) and value
    ]


def _iter_annotated_pairs(root: Path):
    """产出目录下所有"已标注"图片的 ``(annotation_path, data, labels)`` 三元组。

    ``labels`` 为该图的分类标签（顶层 ``flags`` 为 True 的 key），一次遍历
    计算完成，供整体统计与按标签统计共享，避免对同一 JSON 重复解析。
    仅遍历一次文件系统，避免重复 I/O。
    """
    for _image_path, ann_path, data in iter_matched_pairs(root, require_shapes=False):
        labels = _classification_labels(data)
        if data.get("shapes") or labels:
            yield ann_path, data, labels


def _apply_stats_pair(
        stats: Dict[str, int],
        ann_path: Path,
        data: dict,
        type_checker: AnnotationTypeChecker,
        classification_labels: List[str],
) -> None:
    """把一张已标注图片计入整体统计。"""
    stats["annotated_images"] += 1
    shapes = data.get("shapes") or []
    shape_types = _detect_shape_types(shapes)

    if "rectangle" in shape_types:
        stats["detection_images"] += 1
    if "rotation" in shape_types:
        stats["obb_images"] += 1
    if "polygon" in shape_types:
        stats["polygon_images"] += 1
    if classification_labels:
        stats["classification_images"] += 1

    try:
        json_mtime = ann_path.stat().st_mtime
    except OSError:
        json_mtime = 0.0
    ann_type = type_checker.check(data, json_mtime=json_mtime)
    if ann_type == AnnotationType.MANUAL:
        stats["manual_images"] += 1
    elif ann_type == AnnotationType.AUTO:
        stats["auto_images"] += 1
    elif ann_type == AnnotationType.AUTO_CORRECTED:
        stats["auto_corrected_images"] += 1


def _apply_label_pair(
        label_counts: Dict[str, Dict[str, int]],
        data: dict,
        classification_labels: List[str],
) -> None:
    """把一张已标注图片计入按标签统计。"""
    for shape in data.get("shapes") or []:
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

    for label in classification_labels:
        label_counts[label]["classification_count"] += 1


def _new_label_counts() -> Dict[str, int]:
    """创建一份全新的按标签统计计数表。"""
    return {
        "detection_count": 0,
        "obb_count": 0,
        "polygon_count": 0,
        "classification_count": 0,
    }


# 整体统计中由单张图片累计的计数键（其余字段在结果组装时派生）
_STATS_COUNT_KEYS: Tuple[str, ...] = (
    "annotated_images",
    "detection_images",
    "obb_images",
    "polygon_images",
    "classification_images",
    "manual_images",
    "auto_images",
    "auto_corrected_images",
)


def _new_stats() -> Dict[str, int]:
    """创建一份全新的整体统计计数表（全部归零）。"""
    return {key: 0 for key in _STATS_COUNT_KEYS}


def _require_stats_dir(path_str: str) -> Path:
    """校验统计目录存在并返回其 Path 对象。"""
    root = Path(path_str)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {path_str}")
    return root


def _build_stats_result(total_images: int, stats: Dict[str, int]) -> Dict[str, int]:
    """把计数表与图片总数组装为完整的整体统计字典。"""
    return {
        "total_images": total_images,
        "annotated_images": stats["annotated_images"],
        "unannotated_images": total_images - stats["annotated_images"],
        **{key: stats[key] for key in _STATS_COUNT_KEYS[1:]},
    }


def _sort_label_stats(label_counts: Dict[str, Dict[str, int]]) -> List[Dict[str, int]]:
    """把按标签计数表按标签名升序输出为列表。"""
    return [
        {"label": label, **counts}
        for label, counts in sorted(label_counts.items(), key=lambda item: item[0])
    ]


def collect_annotation_stats(folder: str) -> Dict[str, int]:
    """
    统计目录下的图片与标注信息。

    参数:
        folder: 待统计的目录路径。

    返回:
        包含 total_images、annotated_images、unannotated_images、
        detection_images、obb_images、polygon_images、classification_images、
        manual_images、auto_images、auto_corrected_images 的统计字典。

    异常:
        ValueError: 目录不存在或不是文件夹。
    """
    root = _require_stats_dir(folder)

    # 使用 stem 去重：同一张图片可能有 .jpg/.png 等多个扩展名版本，按 stem 计为一张
    total_images = len({p.stem for p in iter_images(root)})

    stats = _new_stats()
    type_checker = AnnotationTypeChecker()

    for ann_path, data, classification_labels in _iter_annotated_pairs(root):
        _apply_stats_pair(stats, ann_path, data, type_checker, classification_labels)

    return _build_stats_result(total_images, stats)


def collect_annotation_label_stats(folder: str) -> List[Dict[str, int]]:
    """
    按标签统计目录下的标注实例数量。

    目标检测 / OBB / 分割标签取自 ``shapes``，分类标签取自顶层 ``flags``
    （值为 True 的 key）。同一标签可同时拥有多种类型计数。

    参数:
        folder: 待统计的目录路径。

    返回:
        每个标签的实例数量列表，元素包含 label、detection_count、
        obb_count、polygon_count、classification_count，按标签名升序排列。

    异常:
        ValueError: 目录不存在或不是文件夹。
    """
    root = _require_stats_dir(folder)

    label_counts: Dict[str, Dict[str, int]] = defaultdict(_new_label_counts)

    for _ann_path, data, classification_labels in _iter_annotated_pairs(root):
        _apply_label_pair(label_counts, data, classification_labels)

    return _sort_label_stats(label_counts)


def collect_all_stats(input_dir: str):
    """
    单次遍历同时返回整体统计与按标签统计。

    相对于分别调用 :func:`collect_annotation_stats` 和
    :func:`collect_annotation_label_stats`，本函数只需遍历一次文件，
    避免重复 I/O。

    参数:
        input_dir: 待统计的目录路径。

    返回:
        ``(stats, label_stats)`` 元组，含义同上述两个函数。
    """
    root = _require_stats_dir(input_dir)

    total_images = len({p.stem for p in iter_images(root)})

    stats = _new_stats()
    type_checker = AnnotationTypeChecker()
    label_counts: Dict[str, Dict[str, int]] = defaultdict(_new_label_counts)

    for ann_path, data, classification_labels in _iter_annotated_pairs(root):
        _apply_stats_pair(stats, ann_path, data, type_checker, classification_labels)
        _apply_label_pair(label_counts, data, classification_labels)

    return _build_stats_result(total_images, stats), _sort_label_stats(label_counts)


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
    log(f"  分类数量        : {stats.get('classification_images', 0)}")
    log(f"  手动标注数量    : {stats['manual_images']}")
    log(f"  自动标注数量    : {stats['auto_images']}")
    log(f"  手动矫正数量    : {stats['auto_corrected_images']}")


def print_label_stats_human(label_stats: List[Dict[str, int]]) -> None:
    """以易读格式打印按标签统计。"""
    log("===== 按标签统计 =====")
    if not label_stats:
        log("  （未发现任何标签实例）")
        return
    header = f"  {'标签名':<24}{'检测':>8}{'OBB':>8}{'多边形':>8}{'分类':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for item in label_stats:
        log(
            f"  {item.get('label', ''):<24}"
            f"{item.get('detection_count', 0):>8}"
            f"{item.get('obb_count', 0):>8}"
            f"{item.get('polygon_count', 0):>8}"
            f"{item.get('classification_count', 0):>8}"
        )


def emit_machine_block(payload: Dict[str, object]) -> None:
    """
    输出供 GUI / 脚本解析的 JSON 块（用边界标记包裹）。
    """
    log(STATS_RESULT_BEGIN_MARKER)
    log(json.dumps(payload, ensure_ascii=False))
    log(STATS_RESULT_END_MARKER)


def parse_machine_block(output: str) -> Dict[str, object]:
    """
    从 CLI 输出中提取以边界标记包裹的 JSON 块并解析为字典。

    返回:
        包含 ``stats`` 与 ``label_stats`` 字段的字典。

    异常:
        ValueError: 找不到边界标记或 JSON 解析失败。
    """
    if not isinstance(output, str):
        raise ValueError("output 必须为字符串")

    begin = output.rfind(STATS_RESULT_BEGIN_MARKER)  # rfind 取最后一个块，避免之前的 log 输出干扰
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


# --------------------------------------------------------------------------- #
# CLI 入口
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py datasets stats",
        description=(
            "统计目录下的图片与 X-AnyLabeling JSON 标注情况，"
            "支持整体统计与按标签统计。"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="待统计的图片目录路径。",
    )
    parser.add_argument(
        "--label-stats",
        action="store_true",
        help="同时输出按标签的实例数量统计。",
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="仅输出供机器解析的 JSON 块（不打印人类可读日志）。",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    folder = Path(args.input)
    if not folder.exists():
        raise ValueError(f"目录不存在：{args.input}")
    if not folder.is_dir():
        raise ValueError(f"路径不是目录：{args.input}")


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回:
        0=成功；2=参数非法；1=运行时错误；130=用户中断。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2

    try:
        if args.label_stats:
            stats, label_stats = collect_all_stats(args.input)
        else:
            stats = collect_annotation_stats(args.input)
            label_stats = []
    except ValueError as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log("[已取消] 用户中断。", stream=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 统计失败: {exc}", stream=sys.stderr)
        return 1

    if not args.json_only:
        print_stats_human(stats)
        if args.label_stats:
            print_label_stats_human(label_stats)

    emit_machine_block({"stats": stats, "label_stats": label_stats})
    return 0


if __name__ == "__main__":
    sys.exit(main())
