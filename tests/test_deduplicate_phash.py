#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``scripts.images.dedup.deduplicate`` 的 phash 后端测试。

phash 后端仅依赖 Pillow + numpy（DCT 不可用时自动回退到 dHash），
因此可以在没有 torch / transformers 的 CI 环境中跑通。
"""

from __future__ import annotations

from PIL import Image

from scripts.images.dedup import deduplicate


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _make_solid(path, size=(64, 64), color=(255, 0, 0)) -> None:
    """生成一张纯色图片。"""
    Image.new("RGB", size, color).save(path, format="JPEG", quality=95)


def _make_gradient(path, size=(64, 64), shift: int = 0) -> None:
    """生成一张横向渐变图片，可通过 ``shift`` 控制小幅偏移。"""
    w, h = size
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(w):
        v = (x + shift) % 256
        for y in range(h):
            pixels[x, y] = (v, v, v)
    img.save(path, format="JPEG", quality=95)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_phash_marks_identical_images_as_duplicates(tmp_path):
    """两张完全相同的图片应被 phash 后端判为重复。"""
    folder = tmp_path / "imgs"
    folder.mkdir()

    _make_solid(folder / "a.jpg", color=(120, 30, 200))
    _make_solid(folder / "b.jpg", color=(120, 30, 200))  # 与 a 完全相同

    result = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        hash_size=8,
    )

    keep = result["keep"]
    dups = result["duplicates"]

    assert len(keep) == 1
    assert len(dups) == 1
    # keep + duplicates 总数应等于输入图片数
    assert {p.name for p in keep} | {p.name for p in dups} == {"a.jpg", "b.jpg"}


def test_phash_keeps_distinct_images(tmp_path):
    """颜色 / 内容完全不同的图片应当全部保留。

    注：phash 对纯色图缺乏判别力（DCT 低频几乎一致），因此使用渐变 +
    纯色这种结构不同的组合，并采用较高阈值。
    """
    folder = tmp_path / "imgs"
    folder.mkdir()

    _make_gradient(folder / "grad.jpg", shift=0)
    _make_solid(folder / "solid.jpg", color=(10, 200, 50))

    result = deduplicate(
        folder=str(folder),
        threshold=0.99,
        backend="phash",
        hash_size=8,
    )

    # 渐变 vs 纯色，phash 相似度应明显低于 0.99
    assert len(result["keep"]) == 2
    assert len(result["duplicates"]) == 0


def test_phash_move_to_moves_duplicate_files(tmp_path):
    """指定 move_to 后，重复图片应被移动至目标目录。"""
    folder = tmp_path / "imgs"
    folder.mkdir()
    bin_dir = tmp_path / "bin"

    _make_solid(folder / "a.jpg", color=(50, 150, 220))
    _make_solid(folder / "b.jpg", color=(50, 150, 220))

    result = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        move_to=str(bin_dir),
        hash_size=8,
    )

    assert len(result["duplicates"]) == 1
    # 重复图片应当已经从源目录移走
    remaining = sorted(p.name for p in folder.iterdir())
    assert len(remaining) == 1
    # bin 目录中应该出现一张被移走的图片
    assert bin_dir.is_dir()
    moved = list(bin_dir.iterdir())
    assert len(moved) == 1


def test_phash_empty_folder_returns_empty(tmp_path):
    """空目录调用 deduplicate 不应报错，返回空列表。"""
    folder = tmp_path / "empty"
    folder.mkdir()

    result = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
    )

    assert result == {"keep": [], "duplicates": []}
