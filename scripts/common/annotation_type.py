#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注类型判断工具。

根据 X-AnyLabeling JSON 标注文件的内容与文件系统最后修改时间，判断一张图片的
标注属于以下三类之一：

- 手动标注（manual）：不携带 ``auto_annotated_time`` 字段的标注。
- 自动标注（auto）：携带 ``auto_annotated_time`` 字段，且 JSON 文件的
  最后修改时间与该字段值的差距不超过 2 秒。
- 自动标注并手动矫正（auto_corrected）：携带 ``auto_annotated_time`` 字段，
  且 JSON 文件的最后修改时间与该字段值的差距超过 2 秒。

用法示例：
    from scripts.common.annotation_type import AnnotationType, AnnotationTypeChecker

    checker = AnnotationTypeChecker()
    ann_type = checker.check_file("/path/to/image.json")
    print(ann_type.value)  # "manual" / "auto" / "auto_corrected"
"""

import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from scripts.common.logging import log


class AnnotationType(Enum):
    """标注类型枚举。"""

    MANUAL = "manual"
    AUTO = "auto"
    AUTO_CORRECTED = "auto_corrected"


class AnnotationTypeChecker:
    """
    判断 X-AnyLabeling 标注文件属于手动标注、自动标注还是自动标注并手动矫正。

    判断规则：
        1. 手动标注：JSON 中不存在 ``auto_annotated_time`` 字段。
        2. 自动标注：存在 ``auto_annotated_time`` 字段，且 JSON 文件的
           最后修改时间与该字段时间戳相差不超过 2 秒。
        3. 自动标注并手动矫正：存在 ``auto_annotated_time`` 字段，且 JSON
           文件的最后修改时间与该字段时间戳相差超过 2 秒。

    ``tolerance_seconds`` 用于控制“自动标注”与“自动标注并手动矫正”的
    判定阈值，默认 2 秒。
    """

    _TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    def __init__(self, tolerance_seconds: float = 2.0):
        """
        初始化判断器。

        参数:
            tolerance_seconds: 判定为“自动标注”的最大时间差阈值，单位秒，
                必须为非负数。默认 2.0。

        异常:
            ValueError: 阈值小于 0 时抛出。
        """
        if tolerance_seconds < 0:
            raise ValueError("tolerance_seconds 必须大于等于 0")
        self.tolerance_seconds = tolerance_seconds

    @classmethod
    def _parse_time(cls, value: str) -> Optional[datetime]:
        """
        将 ``auto_annotated_time`` 字符串解析为 datetime 对象。

        优先尝试 ISO-8601 格式（``datetime.fromisoformat``，兼容 ``2024-01-02T03:04:05``
        以及带毫秒、时区的写法），失败后回退到固定 ``%Y-%m-%d %H:%M:%S`` 格式。
        """
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            pass
        try:
            return datetime.strptime(value, cls._TIME_FORMAT)
        except (ValueError, TypeError):
            return None

    def check(
            self,
            annotation: dict,
            json_mtime: Optional[Union[datetime, float]] = None,
    ) -> AnnotationType:
        """
        根据标注字典与文件最后修改时间判断标注类型。

        参数:
            annotation: X-AnyLabeling JSON 解析后的字典。
            json_mtime: JSON 文件的最后修改时间。可传入 datetime 对象或
                Unix 时间戳（秒）。如未提供，则仅根据字典中是否存在
                ``auto_annotated_time`` 字段判断，存在则返回
                ``AnnotationType.AUTO``，不存在则返回
                ``AnnotationType.MANUAL``。

        返回:
            对应的 ``AnnotationType`` 枚举值。
        """
        auto_time_value = annotation.get("auto_annotated_time")
        if not auto_time_value:
            return AnnotationType.MANUAL

        auto_time = self._parse_time(auto_time_value)
        if auto_time is None or json_mtime is None:
            log(
                f"[AnnotationTypeChecker] auto_time={auto_time!r}, json_mtime={json_mtime!r} — "
                f"解析失败，保守视为 AUTO",
                stream=sys.stderr,
            )
            return AnnotationType.AUTO

        if isinstance(json_mtime, (int, float)):
            mtime_dt = datetime.fromtimestamp(json_mtime)
        elif isinstance(json_mtime, datetime):
            mtime_dt = json_mtime
        else:
            return AnnotationType.AUTO

        diff_seconds = abs((mtime_dt - auto_time).total_seconds())
        if diff_seconds <= self.tolerance_seconds:
            return AnnotationType.AUTO
        return AnnotationType.AUTO_CORRECTED

    def check_file(self, json_path: Union[str, Path]) -> AnnotationType:
        """
        根据 JSON 文件路径判断标注类型。

        参数:
            json_path: X-AnyLabeling 标注文件路径。

        返回:
            对应的 ``AnnotationType`` 枚举值。

        异常:
            FileNotFoundError: 文件不存在时抛出。
            ValueError: 文件内容不是有效的 JSON 时抛出。
        """
        json_path = Path(json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"标注文件不存在: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)

        mtime = json_path.stat().st_mtime
        return self.check(annotation, mtime)

    def check_image_annotation(
            self,
            image_path: Union[str, Path],
    ) -> AnnotationType:
        """
        根据图片路径查找同目录下的同名 JSON 标注文件并判断标注类型。

        参数:
            image_path: 图片文件路径。

        返回:
            对应的 ``AnnotationType`` 枚举值。如果图片不存在对应 JSON 文件，
            返回 ``AnnotationType.MANUAL``（视为未自动标注）。

        异常:
            FileNotFoundError: 图片文件不存在时抛出。
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        # 形如 ``foo.tar.gz`` 的多后缀图片名（极少出现，但 with_suffix 只替换最后一段，
        # 仍然安全）；此处保留与图片 stem 同名的 ``<stem>.json`` 约定。
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            return AnnotationType.MANUAL

        return self.check_file(json_path)
