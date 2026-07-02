#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``scripts.datasets.export.export_yolo_dataset`` 的端到端单元测试。

通过在临时目录构造少量图片 + X-AnyLabeling JSON 标注（detect 任务的
rectangle shape），验证：

1. 输出目录结构（``images/train``、``labels/train``、``data.yaml``）；
2. YOLO label 文件内容为 ``class_id cx cy w h``，且全部归一化在 ``[0, 1]``；
3. ``copy_mode='copy'`` 时图片被真正复制（而非硬/软链接）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.common.utils import validate_split_ratios
from scripts.datasets.export import export_yolo_dataset, main as export_main


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_rect_annotation(
        json_path: Path,
        image_path: Path,
        *,
        label: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_w: int = 32,
        img_h: int = 32,
) -> None:
    """构造一个仅含单个 rectangle shape 的 X-AnyLabeling JSON 文件。"""
    data = {
        "version": "2.4.0",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
            }
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_unsupported_shape_annotation(json_path: Path, image_path: Path) -> None:
    """构造一个 detect/obb/segment 均不会导出的 shape。"""
    data = {
        "version": "2.4.0",
        "flags": {},
        "shapes": [
            {
                "label": "cat",
                "points": [[8.0, 8.0], [16.0, 16.0]],
                "shape_type": "circle",
            }
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": 32,
        "imageWidth": 32,
    }
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_export_detect_creates_expected_structure(tmp_path, make_image):
    """detect 任务下，应生成 images/train、labels/train、data.yaml。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    # 两张图，各一个 rectangle 标注，统一进入训练集
    for i in range(2):
        img = make_image(input_dir / f"img_{i}.jpg", size=(32, 32), color=(i * 10, 0, 0))
        _make_rect_annotation(
            input_dir / f"img_{i}.json",
            img,
            label="cat",
            x1=8.0,
            y1=8.0,
            x2=24.0,
            y2=24.0,
        )

    counts = export_yolo_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        task="detect",
        train_ratio=1.0,
        test_ratio=0.0,
        seed=0,
        copy_mode="copy",
    )

    assert counts["train"] == 2
    assert counts["test"] == 0

    assert (output_dir / "images" / "train").is_dir()
    assert (output_dir / "labels" / "train").is_dir()
    assert (output_dir / "data.yaml").is_file()

    # data.yaml 中应当包含 'cat' 类别
    yaml_text = (output_dir / "data.yaml").read_text(encoding="utf-8")
    assert "cat" in yaml_text
    assert "nc: 1" in yaml_text


def test_export_detect_label_values_are_normalized(tmp_path, make_image):
    """YOLO label 行格式：``class_id cx cy w h``，所有坐标值需归一化到 [0, 1]。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    img = make_image(input_dir / "img.jpg", size=(32, 32))
    # 框：(8,8)~(24,24)，中心 (16,16)，宽高 16；归一化后均为 0.5
    _make_rect_annotation(
        input_dir / "img.json",
        img,
        label="cat",
        x1=8.0,
        y1=8.0,
        x2=24.0,
        y2=24.0,
    )

    export_yolo_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        task="detect",
        train_ratio=1.0,
        test_ratio=0.0,
        seed=0,
        copy_mode="copy",
    )

    label_files = list((output_dir / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 1

    lines = label_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parts = lines[0].split()
    assert len(parts) == 5
    class_id = int(parts[0])
    cx, cy, w, h = (float(x) for x in parts[1:])

    assert class_id == 0
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0
    # 在 32x32 图上框为 (8,8)~(24,24)，期望接近 0.5
    assert cx == pytest.approx(0.5, abs=1e-3)
    assert cy == pytest.approx(0.5, abs=1e-3)
    assert w == pytest.approx(0.5, abs=1e-3)
    assert h == pytest.approx(0.5, abs=1e-3)


def test_export_copy_mode_copy_creates_real_file(tmp_path, make_image):
    """copy 模式下输出图片应是独立文件，而非硬/软链接。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_rect_annotation(
        input_dir / "img.json",
        img,
        label="dog",
        x1=4.0,
        y1=4.0,
        x2=28.0,
        y2=28.0,
    )

    export_yolo_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        task="detect",
        train_ratio=1.0,
        test_ratio=0.0,
        seed=0,
        copy_mode="copy",
    )

    out_img = output_dir / "images" / "train" / "img.jpg"
    assert out_img.is_file()
    assert not out_img.is_symlink()
    # copy 模式下 inode 应当与源不同
    assert out_img.stat().st_ino != img.stat().st_ino


def test_export_allows_all_test_split(tmp_path, make_image):
    """应允许 train_ratio=0.0/test_ratio=1.0，方便小数据集调试。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_rect_annotation(
        input_dir / "img.json",
        img,
        label="cat",
        x1=4.0,
        y1=4.0,
        x2=28.0,
        y2=28.0,
    )

    counts = export_yolo_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        task="detect",
        train_ratio=0.0,
        test_ratio=1.0,
        seed=0,
        copy_mode="copy",
    )

    assert counts == {"train": 0, "test": 1}
    assert (output_dir / "images" / "test" / "img.jpg").is_file()


def test_export_invalid_ratios_raise(tmp_path, make_image):
    """train_ratio + test_ratio != 1 时应当抛出 ValueError。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_rect_annotation(
        input_dir / "img.json",
        img,
        label="cat",
        x1=4.0,
        y1=4.0,
        x2=28.0,
        y2=28.0,
    )

    with pytest.raises(ValueError):
        export_yolo_dataset(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "out"),
            task="detect",
            train_ratio=0.5,
            test_ratio=0.2,
        )


def test_export_zero_sum_ratios_raise(tmp_path, make_image):
    """train_ratio/test_ratio 不能同时为 0。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_rect_annotation(
        input_dir / "img.json",
        img,
        label="cat",
        x1=4.0,
        y1=4.0,
        x2=28.0,
        y2=28.0,
    )

    with pytest.raises(ValueError):
        export_yolo_dataset(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "out"),
            task="detect",
            train_ratio=0.0,
            test_ratio=0.0,
        )


def test_export_cli_all_train_split_is_allowed(tmp_path, make_image):
    """CLI 与核心函数一致，允许 1.0/0.0 划分。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_rect_annotation(
        input_dir / "img.json",
        img,
        label="cat",
        x1=4.0,
        y1=4.0,
        x2=28.0,
        y2=28.0,
    )

    code = export_main([
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--task", "detect",
        "--train-ratio", "1.0",
        "--test-ratio", "0.0",
    ])

    assert code == 0
    assert (output_dir / "images" / "train" / "img.jpg").is_file()


def test_validate_split_ratios_rejects_invalid_values():
    """划分比例必须为 [0, 1] 内的有限数字，且总和为 1。"""
    invalid_cases = [
        (-0.1, 1.1),
        (1.1, -0.1),
        (math.nan, 1.0),
        (1.0, math.nan),
        (math.inf, 0.0),
        (0.0, -math.inf),
        (0.5, 0.2),
    ]

    for train_ratio, test_ratio in invalid_cases:
        with pytest.raises(ValueError):
            validate_split_ratios(train_ratio, test_ratio)


def test_validate_split_ratios_allows_single_split_side():
    """允许全训练集或全测试集划分。"""
    validate_split_ratios(1.0, 0.0)
    validate_split_ratios(0.0, 1.0)


def test_export_unknown_task_raises(tmp_path):
    """未支持的 task 应当抛出 ValueError。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(ValueError):
        export_yolo_dataset(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "out"),
            task="pose",  # 不支持
            train_ratio=1.0,
            test_ratio=0.0,
        )


def test_export_skips_empty_labels_by_default(tmp_path, make_image):
    """默认不导出空标签样本；仅显式 export_empty_labels=True 时才保留。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_unsupported_shape_annotation(input_dir / "img.json", img)

    counts = export_yolo_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        task="detect",
        train_ratio=1.0,
        test_ratio=0.0,
    )

    assert counts == {"train": 0, "test": 0}
    assert not (output_dir / "images" / "train" / "img.jpg").exists()
    assert not (output_dir / "labels" / "train" / "img.txt").exists()


def test_export_empty_labels_when_explicitly_enabled(tmp_path, make_image):
    """export_empty_labels=True 时，空标签样本应被导出为空 txt。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    img = make_image(input_dir / "img.jpg", size=(32, 32))
    _make_unsupported_shape_annotation(input_dir / "img.json", img)

    counts = export_yolo_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        task="detect",
        train_ratio=1.0,
        test_ratio=0.0,
        export_empty_labels=True,
    )

    label_path = output_dir / "labels" / "train" / "img.txt"
    assert counts == {"train": 1, "test": 0}
    assert (output_dir / "images" / "train" / "img.jpg").is_file()
    assert label_path.is_file()
    assert label_path.read_text(encoding="utf-8") == ""
