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
from pathlib import Path

import pytest

from scripts.datasets.export import export_yolo_dataset


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
