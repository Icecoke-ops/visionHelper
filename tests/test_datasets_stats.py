#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证标注统计对分类标注（顶层 flags）的支持。"""

from __future__ import annotations

from pathlib import Path

from scripts.datasets.stats import (
    collect_all_stats,
    collect_annotation_label_stats,
    collect_annotation_stats,
)


def test_classification_only_images_are_annotated(
    tmp_path: Path, make_image, make_annotation
):
    """只有分类标注（顶层 flags）的图片也应计入已标注。"""
    for name, flags in [("a", {"cat": True, "dog": False}), ("b", {"dog": True})]:
        img = make_image(tmp_path / f"{name}.jpg")
        make_annotation(
            tmp_path / f"{name}.json",
            image_path=img,
            shapes=[],
            extra={"flags": flags},
        )
    # 未标注图片
    make_image(tmp_path / "c.jpg")

    stats = collect_annotation_stats(str(tmp_path))
    assert stats["total_images"] == 3
    assert stats["annotated_images"] == 2
    assert stats["unannotated_images"] == 1
    assert stats["classification_images"] == 2
    assert stats["detection_images"] == 0
    assert stats["obb_images"] == 0
    assert stats["polygon_images"] == 0


def test_empty_flags_not_annotated(tmp_path: Path, make_image, make_annotation):
    """flags 全为 False（或空）且无 shapes 时不视为已标注。"""
    img = make_image(tmp_path / "a.jpg")
    make_annotation(
        tmp_path / "a.json",
        image_path=img,
        shapes=[],
        extra={"flags": {"cat": False, "dog": False}},
    )

    stats = collect_annotation_stats(str(tmp_path))
    assert stats["total_images"] == 1
    assert stats["annotated_images"] == 0
    assert stats["classification_images"] == 0


def test_label_stats_include_classification(
    tmp_path: Path, make_image, make_annotation
):
    """按标签统计应包含分类标签数量。"""
    img_a = make_image(tmp_path / "a.jpg")
    make_annotation(
        tmp_path / "a.json",
        image_path=img_a,
        shapes=[],
        extra={"flags": {"cat": True}},
    )
    img_b = make_image(tmp_path / "b.jpg")
    make_annotation(
        tmp_path / "b.json",
        image_path=img_b,
        shapes=[{"label": "cat", "shape_type": "rectangle",
                 "points": [[0, 0], [10, 10]]}],
        extra={"flags": {"cat": True}},
    )

    label_stats = collect_annotation_label_stats(str(tmp_path))
    assert label_stats == [
        {
            "label": "cat",
            "detection_count": 1,
            "obb_count": 0,
            "polygon_count": 0,
            "classification_count": 2,
        }
    ]


def test_collect_all_stats_matches_individual(
    tmp_path: Path, make_image, make_annotation
):
    """单次遍历的合并结果与分函数结果一致。"""
    for name, flags in [("a", {"cat": True}), ("b", {"dog": True})]:
        img = make_image(tmp_path / f"{name}.jpg")
        make_annotation(
            tmp_path / f"{name}.json",
            image_path=img,
            shapes=[],
            extra={"flags": flags},
        )

    stats, label_stats = collect_all_stats(str(tmp_path))
    assert stats == collect_annotation_stats(str(tmp_path))
    assert label_stats == collect_annotation_label_stats(str(tmp_path))
    assert stats["classification_images"] == 2
    assert {item["label"] for item in label_stats} == {"cat", "dog"}