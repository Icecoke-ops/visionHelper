#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 ``scripts.common.utils`` 中的图片/标注迭代器与发现工具。"""

from __future__ import annotations

from pathlib import Path

from scripts.common.utils import (
    discover_trained_models,
    is_annotation_file,
    is_image_file,
    iter_annotations,
    iter_images,
    iter_matched_pairs,
)


def test_is_image_and_annotation_file(tmp_path: Path, make_image, make_annotation):
    img = make_image(tmp_path / "a.JPG")  # 大写后缀也应识别
    ann = make_annotation(tmp_path / "a.json", image_path=img)
    txt = tmp_path / "note.txt"
    txt.write_text("hi", encoding="utf-8")

    assert is_image_file(img)
    assert is_annotation_file(ann)
    assert not is_image_file(ann)
    assert not is_annotation_file(img)
    assert not is_image_file(txt)


def test_iter_images_sorted_and_filtered(tmp_path: Path, make_image):
    make_image(tmp_path / "b.png")
    make_image(tmp_path / "a.jpg")
    # 非图片文件不应被纳入
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")

    results = [p.name for p in iter_images(tmp_path)]
    assert results == ["a.jpg", "b.png"]


def test_iter_annotations_sorted(tmp_path: Path, make_image, make_annotation):
    img_a = make_image(tmp_path / "a.jpg")
    img_b = make_image(tmp_path / "b.jpg")
    make_annotation(tmp_path / "b.json", image_path=img_b)
    make_annotation(tmp_path / "a.json", image_path=img_a)

    names = [p.name for p in iter_annotations(tmp_path)]
    assert names == ["a.json", "b.json"]


def test_iter_matched_pairs_require_shapes(
    tmp_path: Path, make_image, make_annotation
):
    img_a = make_image(tmp_path / "a.jpg")
    img_b = make_image(tmp_path / "b.jpg")
    # a 有 shapes，b 没有
    make_annotation(
        tmp_path / "a.json",
        image_path=img_a,
        shapes=[
            {
                "label": "cat",
                "shape_type": "rectangle",
                "points": [[0, 0], [10, 10]],
            }
        ],
    )
    make_annotation(tmp_path / "b.json", image_path=img_b, shapes=[])

    # 不要求 shapes：两张都返回
    pairs = list(iter_matched_pairs(tmp_path, require_shapes=False))
    assert [p[0].name for p in pairs] == ["a.jpg", "b.jpg"]

    # 要求 shapes：只剩 a
    pairs = list(iter_matched_pairs(tmp_path, require_shapes=True))
    assert [p[0].name for p in pairs] == ["a.jpg"]
    img, ann, data = pairs[0]
    assert isinstance(data, dict)
    assert data["shapes"][0]["label"] == "cat"


def test_iter_matched_pairs_missing_image(tmp_path: Path, make_annotation):
    """图片缺失时该 (img, ann, data) 不应被产出。"""
    make_annotation(
        tmp_path / "ghost.json",
        image_path=tmp_path / "ghost.jpg",  # 不真正创建文件
        shapes=[{"label": "x", "shape_type": "rectangle",
                 "points": [[0, 0], [1, 1]]}],
    )
    assert list(iter_matched_pairs(tmp_path)) == []


def test_iter_matched_pairs_invalid_json(tmp_path: Path, make_image):
    """损坏的 JSON 标注文件不会让迭代器崩溃。"""
    make_image(tmp_path / "a.jpg")
    (tmp_path / "a.json").write_text("{ not valid json", encoding="utf-8")

    assert list(iter_matched_pairs(tmp_path)) == []


def test_discover_trained_models(tmp_path: Path):
    """构造 ``runs/<name>/weights/*.pt`` 结构后应能被发现。"""
    runs = tmp_path / "runs"
    (runs / "exp1" / "weights").mkdir(parents=True)
    (runs / "exp1" / "weights" / "best.pt").write_bytes(b"\x00")
    (runs / "exp1" / "weights" / "last.pt").write_bytes(b"\x00")

    (runs / "exp2" / "weights").mkdir(parents=True)
    (runs / "exp2" / "weights" / "best.pt").write_bytes(b"\x00")

    # 干扰文件：weights 下的非 pt
    (runs / "exp1" / "weights" / "readme.txt").write_text("x", encoding="utf-8")

    models = discover_trained_models(str(runs))
    names = {display for display, _ in models}
    # 期望显示名包含训练名称-权重名格式
    assert "exp1-best" in names
    assert "exp1-last" in names
    assert "exp2-best" in names

    # 路径都应当存在
    for _, path in models:
        assert Path(path).is_file()


def test_discover_trained_models_missing_dir(tmp_path: Path):
    """目录不存在时应返回空列表，而不是抛异常。"""
    assert discover_trained_models(str(tmp_path / "nope")) == []
