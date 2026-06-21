#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 ``AnnotationTypeChecker`` 的核心判定逻辑。"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts.core.annotation_type import AnnotationType, AnnotationTypeChecker


def _set_mtime(path: Path, dt: datetime) -> None:
    """将文件 mtime 设置为指定 datetime（本地时间）。"""
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def test_init_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        AnnotationTypeChecker(tolerance_seconds=-1)


def test_no_auto_time_is_manual(tmp_path: Path, make_image, make_annotation):
    img = make_image(tmp_path / "a.jpg")
    ann = make_annotation(
        tmp_path / "a.json",
        image_path=img,
        shapes=[{"label": "x", "shape_type": "rectangle",
                 "points": [[0, 0], [1, 1]]}],
        auto_annotated_time=None,
    )

    checker = AnnotationTypeChecker()
    assert checker.check_file(ann) == AnnotationType.MANUAL


def test_auto_when_mtime_close_to_auto_time(
    tmp_path: Path, make_image, make_annotation
):
    img = make_image(tmp_path / "a.jpg")
    auto_dt = datetime(2024, 1, 1, 12, 0, 0)
    ann = make_annotation(
        tmp_path / "a.json",
        image_path=img,
        shapes=[{"label": "x", "shape_type": "rectangle",
                 "points": [[0, 0], [1, 1]]}],
        auto_annotated_time=auto_dt.isoformat(),
    )
    # mtime 与 auto 时间相差 1 秒，tolerance=2 → AUTO
    _set_mtime(ann, datetime(2024, 1, 1, 12, 0, 1))

    checker = AnnotationTypeChecker(tolerance_seconds=2.0)
    assert checker.check_file(ann) == AnnotationType.AUTO


def test_auto_corrected_when_mtime_far_from_auto_time(
    tmp_path: Path, make_image, make_annotation
):
    img = make_image(tmp_path / "a.jpg")
    auto_dt = datetime(2024, 1, 1, 12, 0, 0)
    ann = make_annotation(
        tmp_path / "a.json",
        image_path=img,
        shapes=[{"label": "x", "shape_type": "rectangle",
                 "points": [[0, 0], [1, 1]]}],
        auto_annotated_time=auto_dt.isoformat(),
    )
    # mtime 与 auto 时间相差 1 小时，tolerance=2 → AUTO_CORRECTED
    _set_mtime(ann, datetime(2024, 1, 1, 13, 0, 0))

    checker = AnnotationTypeChecker(tolerance_seconds=2.0)
    assert checker.check_file(ann) == AnnotationType.AUTO_CORRECTED


def test_check_image_annotation_without_json(tmp_path: Path, make_image):
    img = make_image(tmp_path / "a.jpg")
    checker = AnnotationTypeChecker()
    # 图片同名 JSON 不存在 → 视为未自动标注（MANUAL）
    assert checker.check_image_annotation(img) == AnnotationType.MANUAL
