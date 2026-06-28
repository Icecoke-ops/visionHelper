#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证图片数据增强的随机性与基础效果。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from scripts.images.augment import (
    apply_channel_transform,
    apply_random_cut,
    apply_random_occlusion,
    apply_random_rotation,
    augment_image,
    augment_images,
)


def _gradient_image(width: int = 48, height: int = 32) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        [
            (x * 5) % 256,
            (y * 7) % 256,
            ((x + y) * 3) % 256,
        ],
        axis=2,
    ).astype(np.uint8)


def test_rotation_changes_image_shape_or_pixels():
    img = _gradient_image()

    out = apply_random_rotation(img, degrees=45, p=1.0, rng=random.Random(7))

    assert out.shape[:2] != img.shape[:2] or not np.array_equal(out, img)


def test_cut_changes_image_when_resized():
    img = _gradient_image()

    out = apply_random_cut(img, scale=0.6, ratio=1.5, p=1.0, resize=True, rng=random.Random(3))

    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_occlusion_varies_with_rng_state():
    img = np.full((40, 40, 3), 127, dtype=np.uint8)
    rng = random.Random(11)

    first = apply_random_occlusion(img, count=3, size=0.3, p=1.0, rng=rng)
    second = apply_random_occlusion(img, count=3, size=0.3, p=1.0, rng=rng)

    assert not np.array_equal(first, second)


def test_channel_transform_not_always_green():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 0] = 40
    img[:, :, 1] = 120
    img[:, :, 2] = 220
    rng = random.Random(5)

    outputs = [apply_channel_transform(img, p=1.0, rng=rng) for _ in range(8)]

    assert len({tuple(out[0, 0].tolist()) for out in outputs}) > 1
    assert any(out[0, 0, 1] == 0 or out[0, 0, 1] != out[0, 0].max() for out in outputs)


def test_augment_image_reuses_supplied_rng_for_different_results():
    img = _gradient_image()
    rng = random.Random(13)

    first = augment_image(
        img,
        rotate_enabled=False,
        cut_enabled=True,
        cut_prob=1.0,
        cut_scale=0.5,
        occlusion_enabled=True,
        occlusion_prob=1.0,
        channel_enabled=False,
        rng=rng,
    )
    second = augment_image(
        img,
        rotate_enabled=False,
        cut_enabled=True,
        cut_prob=1.0,
        cut_scale=0.5,
        occlusion_enabled=True,
        occlusion_prob=1.0,
        channel_enabled=False,
        rng=rng,
    )

    assert not np.array_equal(first, second)


def test_augment_images_seed_is_reproducible_but_not_reset_per_image(tmp_path: Path):
    import cv2

    input_dir = tmp_path / "in"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    input_dir.mkdir()
    for idx in range(2):
        cv2.imwrite(str(input_dir / f"{idx}.png"), _gradient_image() + idx)

    saved_a = augment_images(
        str(input_dir),
        str(out_a),
        rotate_enabled=False,
        cut_enabled=False,
        occlusion_enabled=True,
        occlusion_prob=1.0,
        occlusion_count=2,
        occlusion_size=0.25,
        channel_enabled=False,
        seed=21,
        ext="png",
    )
    saved_b = augment_images(
        str(input_dir),
        str(out_b),
        rotate_enabled=False,
        cut_enabled=False,
        occlusion_enabled=True,
        occlusion_prob=1.0,
        occlusion_count=2,
        occlusion_size=0.25,
        channel_enabled=False,
        seed=21,
        ext="png",
    )

    imgs_a = [cv2.imread(path) for path in saved_a]
    imgs_b = [cv2.imread(path) for path in saved_b]

    assert all(np.array_equal(a, b) for a, b in zip(imgs_a, imgs_b))
    assert not np.array_equal(imgs_a[0], imgs_a[1])
