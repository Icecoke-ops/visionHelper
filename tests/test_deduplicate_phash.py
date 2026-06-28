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


# ---------------------------------------------------------------------------
# 网格分块（grid）模式测试
# ---------------------------------------------------------------------------

def _make_grid_different(tmp_path, prefix, size=(128, 128), diff_tile=None):
    """生成测试用图片，可选择在某个子图块上叠加白色方块造成局部差异。

    Args:
        diff_tile: ``(row, col)`` 坐标，在该位置叠加白色方块；
                   ``None`` 表示不叠加。
    """
    from PIL import ImageDraw

    img = Image.new("RGB", size, (200, 200, 200))  # 浅灰背景
    if diff_tile is not None:
        draw = ImageDraw.Draw(img)
        tile_w = size[0] // 2
        tile_h = size[1] // 2
        left = diff_tile[1] * tile_w
        top = diff_tile[0] * tile_h
        draw.rectangle(
            [left + 4, top + 4, left + tile_w // 2, top + tile_h // 2],
            fill=(255, 255, 255),
        )
    path = tmp_path / f"{prefix}.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


def test_phash_grid_keeps_images_with_local_diff(tmp_path):
    """两张图仅局部（一个子图块）不同时，grid 模式应保留两者。"""
    folder = tmp_path / "imgs"
    folder.mkdir()

    # 图 A：纯灰背景
    _make_grid_different(folder, "a", diff_tile=None)
    # 图 B：右上角子图块有白色方块（小目标缺陷模拟）
    _make_grid_different(folder, "b", diff_tile=(0, 1))

    # 常规模式（无 grid）→ 全局相似度高，可能误判为重复
    result_no_grid = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        hash_size=8,
        grid_size=1,
    )

    # grid 模式（2×2）→ 逐格比较，右上格子相似度低，应保留两张
    result_grid = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        hash_size=8,
        grid_size=2,
    )

    # grid 模式下应有 2 张保留
    assert len(result_grid["keep"]) == 2, (
        f"grid 模式应保留 2 张，实际保留 {len(result_grid['keep'])}"
    )
    assert len(result_grid["duplicates"]) == 0, (
        f"grid 模式不应判为重复，实际重复 {len(result_grid['duplicates'])}"
    )


def test_phash_grid_marks_identical_images_as_duplicates(tmp_path):
    """两张完全相同的图片在 grid 模式下仍应判为重复。"""
    folder = tmp_path / "imgs"
    folder.mkdir()

    _make_grid_different(folder, "a", diff_tile=None)
    _make_grid_different(folder, "b", diff_tile=None)

    result = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        hash_size=8,
        grid_size=2,
    )

    assert len(result["keep"]) == 1
    assert len(result["duplicates"]) == 1
    assert {p.name for p in result["keep"]} | {p.name for p in result["duplicates"]} == {"a.jpg", "b.jpg"}


def test_phash_grid_size_1_equals_original(tmp_path):
    """grid_size=1 时行为应与不指定时一致。"""
    folder = tmp_path / "imgs"
    folder.mkdir()

    from PIL import Image
    Image.new("RGB", (64, 64), (120, 30, 200)).save(folder / "a.jpg", format="JPEG", quality=95)
    Image.new("RGB", (64, 64), (120, 30, 200)).save(folder / "b.jpg", format="JPEG", quality=95)

    ref = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        hash_size=8,
    )
    grid = deduplicate(
        folder=str(folder),
        threshold=0.95,
        backend="phash",
        hash_size=8,
        grid_size=1,
    )

    assert ref == grid
