#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按标注类型批量清除 X-AnyLabeling JSON 标注文件的核心实现。

只会删除目录顶层与图片相匹配的 JSON 标注文件，不会触碰图片本身、子目录
或其它无关 JSON。判定 "自动 / 矫正" 的时间容差通过 ``tolerance_seconds``
控制，默认值取自 :data:`scripts.config.DEFAULT_TOLERANCE_SECONDS`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from scripts._common import iter_matched_pairs
from scripts.core.annotation_type import AnnotationType, AnnotationTypeChecker
from scripts.config import DEFAULT_TOLERANCE_SECONDS
from scripts.logging_utils import log


__all__ = ["clear_annotations"]


def clear_annotations(
        folder: str,
        include_auto: bool = False,
        include_auto_corrected: bool = False,
        include_manual: bool = False,
        tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> Dict[str, object]:
    """
    清除目录下指定类型的 X-AnyLabeling JSON 标注文件。

    参数:
        folder: 待清理的目录路径。
        include_auto: 是否删除自动标注的 JSON。
        include_auto_corrected: 是否删除自动标注后人工矫正的 JSON。
        include_manual: 是否删除手动标注的 JSON。
        tolerance_seconds: 区分自动 / 矫正 的时间容差（秒）。

    返回:
        ``{"scanned", "deleted", "by_type", "failed"}`` 字典：
            - scanned: 扫描到的有效 JSON 数量
            - deleted: 实际删除的 JSON 数量
            - by_type: ``{"auto", "auto_corrected", "manual"}`` 三类删除数量
            - failed: 删除失败的文件路径列表

    异常:
        ValueError: 目录不存在、不是文件夹，或所有清除开关均为关闭时抛出。
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {folder}")

    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds 必须 >= 0")

    if not any([include_auto, include_auto_corrected, include_manual]):
        raise ValueError("至少需要选择一种待清除的标注类型")

    include_map: Dict[AnnotationType, bool] = {
        AnnotationType.AUTO: include_auto,
        AnnotationType.AUTO_CORRECTED: include_auto_corrected,
        AnnotationType.MANUAL: include_manual,
    }

    type_checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)

    scanned = 0
    deleted = 0
    by_type: Dict[str, int] = {
        AnnotationType.AUTO.value: 0,
        AnnotationType.AUTO_CORRECTED.value: 0,
        AnnotationType.MANUAL.value: 0,
    }
    failed: List[str] = []

    # 仅遍历与图片相匹配的 JSON（不要求 shapes 非空，空标注同样支持清除）
    for _image_path, ann_path, data in iter_matched_pairs(root, require_shapes=False):
        scanned += 1

        try:
            ann_type = type_checker.check(data, json_mtime=ann_path.stat().st_mtime)
        except OSError:
            failed.append(str(ann_path))
            continue

        if not include_map.get(ann_type, False):
            continue

        try:
            ann_path.unlink()
        except OSError as exc:
            log(f"[错误] 删除失败 {ann_path.name}: {exc}")
            failed.append(str(ann_path))
            continue

        deleted += 1
        by_type[ann_type.value] = by_type.get(ann_type.value, 0) + 1

    log(
        "清除标签完成：\n"
        f"  扫描到匹配图片的标注: {scanned}\n"
        f"  实际删除: {deleted}\n"
        f"  按类型: 自动 {by_type[AnnotationType.AUTO.value]}, "
        f"矫正 {by_type[AnnotationType.AUTO_CORRECTED.value]}, "
        f"手动 {by_type[AnnotationType.MANUAL.value]}\n"
        f"  删除失败: {len(failed)}"
    )

    return {
        "scanned": scanned,
        "deleted": deleted,
        "by_type": by_type,
        "failed": failed,
    }
