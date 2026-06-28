#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片数据增强模块（``python scripts/vh.py images augment`` 实现）。

支持四种数据增强操作，每种可独立启用/禁用：

    1. 随机切割：随机裁剪一块区域并缩放到原图尺寸。
    2. 随机遮挡：在图片随机位置绘制矩形遮挡块（Cutout）。
    3. 通道变换：随机 BGR 通道打乱、单通道保留、转灰度或通道强度调整。
    4. 随机旋转：以随机角度旋转整张图片，超出部分以黑色填充。

所有操作以固定随机种子确保可复现性。

零副作用约定
------------
``cv2`` / ``numpy`` 等重依赖仅在函数内部延迟 import，模块顶层不触发
重型依赖加载。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

from scripts.common.logging import ProgressLogger, log


def apply_random_rotation(
    img, degrees: float, p: float = 1.0, rng: Optional[random.Random] = None
):
    """随机旋转：以概率 ``p`` 在 ``[-degrees, +degrees]`` 范围内旋转图片。

    Args:
        img: OpenCV 图片数组 (H, W, C) BGR 格式。
        degrees: 最大旋转角度（正值）。
        p: 应用概率。
        rng: 随机数生成器。

    Returns:
        变换后的图片。
    """
    import cv2

    _rng = rng or random
    if _rng.random() >= p:
        return img

    if degrees <= 0:
        return img

    h, w = img.shape[:2]
    angle = _rng.uniform(-degrees, degrees)
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        img, matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def apply_random_cut(
    img,
    scale: float,
    ratio: float,
    p: float = 1.0,
    resize: bool = True,
    rng: Optional[random.Random] = None,
):
    """随机切割：以概率 ``p`` 随机裁剪一块区域。

    裁剪区域的面积占原图比例在 ``[1-scale, 1]`` 之间，宽高比在 ``[1/ratio, ratio]`` 之间。
    当 ``resize=True`` 时，裁剪后的区域会缩放回原始尺寸（类似 RandomResizedCrop）；
    当 ``resize=False`` 时，直接保留裁剪后的原始尺寸。

    Args:
        img: OpenCV 图片数组 (H, W, C)。
        scale: 裁剪面积缩放因子（0~1），值越大裁剪区域越接近原图。
            实际面积比例为 ``[1-scale, 1]``。
        ratio: 宽高比变化范围，裁剪区域的宽高比在 ``[1/ratio, ratio]`` 之间。
        p: 应用概率。
        resize: 是否将裁剪结果缩放回原始尺寸。
        rng: 随机数生成器。

    Returns:
        变换后的图片。
    """
    import cv2
    import numpy as np

    _rng = rng or random
    if _rng.random() >= p:
        return img

    h, w = img.shape[:2]
    if scale <= 0 or h <= 1 or w <= 1:
        return img

    area = h * w
    min_area = max(1, area * (1.0 - scale))

    for _attempt in range(10):
        target_area = _rng.uniform(min_area, area)
        aspect = _rng.uniform(1.0 / ratio, ratio)
        crop_w = min(w, max(1, int(round(np.sqrt(target_area * aspect)))))
        crop_h = min(h, max(1, int(round(np.sqrt(target_area / aspect)))))
        if 0 < crop_w <= w and 0 < crop_h <= h:
            x = _rng.randint(0, w - crop_w)
            y = _rng.randint(0, h - crop_h)
            cropped = img[y : y + crop_h, x : x + crop_w]
            if resize:
                return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            return cropped

    return img


def apply_random_occlusion(
    img, count: int, size: float, p: float = 1.0, rng: Optional[random.Random] = None
):
    """随机遮挡：以概率 ``p`` 在图片上绘制 ``count`` 个矩形遮挡块（Cutout）。

    每个遮挡块的宽高分别占图片宽高的 ``[size*0.5, size]`` 比例。

    Args:
        img: OpenCV 图片数组 (H, W, C)。
        count: 遮挡块数量。
        size: 遮挡块最大尺寸比例（相对于图片宽高）。
        p: 应用概率。
        rng: 随机数生成器。

    Returns:
        变换后的图片。
    """
    import numpy as np

    _rng = rng or random
    if _rng.random() >= p or count <= 0 or size <= 0:
        return img

    h, w = img.shape[:2]
    result = img.copy()

    for _ in range(count):
        block_w = int(_rng.uniform(size * 0.5, size) * w)
        block_h = int(_rng.uniform(size * 0.5, size) * h)
        block_w = max(1, min(block_w, w))
        block_h = max(1, min(block_h, h))
        x = _rng.randint(0, w - block_w)
        y = _rng.randint(0, h - block_h)
        color = (
            [_rng.randint(0, 255) for _channel in range(img.shape[2])]
            if img.ndim == 3
            else _rng.randint(0, 255)
        )
        result[y : y + block_h, x : x + block_w] = color

    return result


def apply_channel_transform(img, p: float = 1.0, rng: Optional[random.Random] = None):
    """通道变换：以概率 ``p`` 随机打乱 RGB 三通道或转为灰度。

    变换方式随机选择：
        - BGR 通道打乱
        - 随机保留单个通道
        - 转为 3 通道灰度图
        - 随机调整各通道强度

    Args:
        img: OpenCV 图片数组 (H, W, C) BGR 格式。
        p: 应用概率。
        rng: 随机数生成器。

    Returns:
        变换后的图片。
    """
    import cv2
    import numpy as np

    _rng = rng or random
    if _rng.random() >= p:
        return img

    mode = _rng.choice(("permute", "single", "gray", "jitter"))

    if mode == "permute":
        perm = list(range(3))
        while perm == [0, 1, 2]:
            _rng.shuffle(perm)
        return img[:, :, perm]

    if mode == "single":
        ch = _rng.randrange(3)
        single = np.zeros_like(img)
        single[:, :, ch] = img[:, :, ch]
        return single

    if mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    factors = np.array(
        [_rng.uniform(0.6, 1.4) for _channel in range(3)], dtype=np.float32
    )
    adjusted = img.astype(np.float32) * factors.reshape(1, 1, 3)
    return np.clip(adjusted, 0, 255).astype(img.dtype)


def augment_image(
    img,
    rotate_enabled: bool = True,
    rotate_degrees: float = 30,
    rotate_prob: float = 0.5,
    cut_enabled: bool = True,
    cut_scale: float = 0.3,
    cut_ratio: float = 1.5,
    cut_prob: float = 0.5,
    cut_resize: bool = True,
    occlusion_enabled: bool = True,
    occlusion_count: int = 3,
    occlusion_size: float = 0.15,
    occlusion_prob: float = 0.5,
    channel_enabled: bool = True,
    channel_prob: float = 0.5,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
):
    """对单张图片按顺序(切割→遮挡→通道→旋转)应用数据增强。

    每个功能可通过 ``*_enabled`` 独立启停。

    Args:
        img: OpenCV 图片数组 (H, W, C) BGR 格式。
        rotate_enabled: 是否启用随机旋转。
        rotate_degrees: 最大旋转角度。
        rotate_prob: 旋转应用概率。
        cut_enabled: 是否启用随机切割。
        cut_scale: 裁剪面积缩放因子。
        cut_ratio: 裁剪宽高比范围。
        cut_prob: 切割应用概率。
        cut_resize: 切割后是否缩放回原始尺寸。
        occlusion_enabled: 是否启用随机遮挡。
        occlusion_count: 遮挡块数量。
        occlusion_size: 遮挡块尺寸比例。
        occlusion_prob: 遮挡应用概率。
        channel_enabled: 是否启用通道变换。
        channel_prob: 通道变换应用概率。
        seed: 随机种子。
        rng: 随机数生成器；优先级高于 seed。

    Returns:
        增强后的图片。
    """
    if rng is None:
        rng = random.Random(seed)

    result = img.copy()

    if cut_enabled:
        result = apply_random_cut(result, cut_scale, cut_ratio, cut_prob, cut_resize, rng)

    if occlusion_enabled:
        result = apply_random_occlusion(
            result, occlusion_count, occlusion_size, occlusion_prob, rng
        )

    if channel_enabled:
        result = apply_channel_transform(result, channel_prob, rng)

    if rotate_enabled:
        result = apply_random_rotation(result, rotate_degrees, rotate_prob, rng)

    return result


# --------------------------------------------------------------------------- #
# 批量处理
# --------------------------------------------------------------------------- #


def augment_images(
    input_dir: str,
    output_dir: str,
    rotate_enabled: bool = True,
    rotate_degrees: float = 30,
    rotate_prob: float = 0.5,
    cut_enabled: bool = True,
    cut_scale: float = 0.3,
    cut_ratio: float = 1.5,
    cut_prob: float = 0.5,
    cut_resize: bool = True,
    occlusion_enabled: bool = True,
    occlusion_count: int = 3,
    occlusion_size: float = 0.15,
    occlusion_prob: float = 0.5,
    channel_enabled: bool = True,
    channel_prob: float = 0.5,
    seed: Optional[int] = None,
    ext: str = "jpg",
    quality: int = 95,
    prefix: str = "aug",
) -> List[str]:
    """批量对 ``input_dir`` 下的图片做数据增强，保存到 ``output_dir``。

    Args:
        input_dir: 输入图片目录。
        output_dir: 输出目录（不存在则自动创建）。
        rotate_enabled: 是否启用随机旋转。
        rotate_degrees: 最大旋转角度。
        rotate_prob: 旋转应用概率。
        cut_enabled: 是否启用随机切割。
        cut_scale: 裁剪面积缩放因子。
        cut_ratio: 裁剪宽高比范围。
        cut_prob: 切割应用概率。
        cut_resize: 切割后是否缩放回原始尺寸。
        occlusion_enabled: 是否启用随机遮挡。
        occlusion_count: 遮挡块数量。
        occlusion_size: 遮挡块尺寸比例。
        occlusion_prob: 遮挡应用概率。
        channel_enabled: 是否启用通道变换。
        channel_prob: 通道变换应用概率。
        seed: 随机种子。
        ext: 输出图片格式（jpg/png/webp）。
        quality: 输出图片质量 1-100。
        prefix: 输出文件名前缀，默认 ``"aug"``，最终文件名为 ``{prefix}_{原文件名}.{ext}``。

    Returns:
        保存的图片路径列表。
    """
    import cv2

    from scripts.common.utils import is_image_file

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ext = ext.lower().lstrip(".") or "jpg"
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    dot_ext = f".{ext}"
    if dot_ext not in supported:
        raise ValueError(f"不支持的输出格式: .{ext}，支持: {sorted(supported)}")

    image_files = sorted(
        [f for f in input_path.iterdir() if is_image_file(f)]
    )
    if not image_files:
        log(f"输入目录中没有图片: {input_dir}")
        return []

    saved: List[str] = []
    rng = random.Random(seed)
    progress = ProgressLogger(total=len(image_files), desc="数据增强")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            log(f"[跳过] 无法读取: {img_path.name}")
            progress.update(1)
            continue

        aug = augment_image(
            img,
            rotate_enabled=rotate_enabled,
            rotate_degrees=rotate_degrees,
            rotate_prob=rotate_prob,
            cut_enabled=cut_enabled,
            cut_scale=cut_scale,
            cut_ratio=cut_ratio,
            cut_prob=cut_prob,
            cut_resize=cut_resize,
            occlusion_enabled=occlusion_enabled,
            occlusion_count=occlusion_count,
            occlusion_size=occlusion_size,
            occlusion_prob=occlusion_prob,
            channel_enabled=channel_enabled,
            channel_prob=channel_prob,
            rng=rng,
        )

        save_name = f"{prefix}_{img_path.stem}.{ext}"
        save_path = output_path / save_name
        params = []
        if ext in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == "png":
            comp = max(0, min(9, 9 - quality // 11))
            params = [cv2.IMWRITE_PNG_COMPRESSION, comp]
        elif ext == "webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]

        ok = cv2.imwrite(str(save_path), aug, params)
        if not ok:
            raise RuntimeError(f"保存图片失败: {save_path}")

        saved.append(str(save_path))
        progress.update(1)

    progress.close()
    log(f"数据增强完成，共处理 {len(saved)} 张图片，保存到: {output_dir}")
    return saved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py images augment",
        description="图片数据增强工具：支持随机旋转、切割、遮挡、通道变换。",
    )
    parser.add_argument("--input", "-i", required=True, help="输入图片目录路径")
    parser.add_argument(
        "--output", "-o", required=True, help="输出图片保存目录（不存在会自动创建）"
    )
    parser.add_argument("--ext", type=str, default="jpg", help="输出图片格式，默认 jpg")
    parser.add_argument(
        "--quality", type=int, default=95, help="输出图片质量 1-100，默认 95"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--prefix", type=str, default="aug", help="输出文件名前缀，默认 aug"
    )

    # 旋转
    parser.add_argument(
        "--no-rotate", action="store_false", dest="rotate_enabled", help="禁用随机旋转"
    )
    parser.add_argument(
        "--rotate-degrees", type=float, default=30, help="最大旋转角度，默认 30"
    )
    parser.add_argument(
        "--rotate-prob", type=float, default=0.5, help="旋转应用概率，默认 0.5"
    )

    # 切割
    parser.add_argument(
        "--no-cut", action="store_false", dest="cut_enabled", help="禁用随机切割"
    )
    parser.add_argument(
        "--cut-scale", type=float, default=0.3, help="裁剪面积缩放因子，默认 0.3"
    )
    parser.add_argument(
        "--cut-ratio", type=float, default=1.5, help="裁剪宽高比范围，默认 1.5"
    )
    parser.add_argument(
        "--cut-prob", type=float, default=0.5, help="切割应用概率，默认 0.5"
    )
    parser.add_argument(
        "--no-cut-resize",
        action="store_false",
        dest="cut_resize",
        help="切割后不缩放回原始尺寸",
    )

    # 遮挡
    parser.add_argument(
        "--no-occlusion", action="store_false", dest="occlusion_enabled", help="禁用随机遮挡"
    )
    parser.add_argument(
        "--occlusion-count", type=int, default=3, help="遮挡块数量，默认 3"
    )
    parser.add_argument(
        "--occlusion-size", type=float, default=0.15, help="遮挡块尺寸比例，默认 0.15"
    )
    parser.add_argument(
        "--occlusion-prob", type=float, default=0.5, help="遮挡应用概率，默认 0.5"
    )

    # 通道变换
    parser.add_argument(
        "--no-channel", action="store_false", dest="channel_enabled", help="禁用通道变换"
    )
    parser.add_argument(
        "--channel-prob", type=float, default=0.5, help="通道变换应用概率，默认 0.5"
    )

    return parser


def _normalize_args(args: argparse.Namespace) -> dict:
    """将 CLI args 标准化为关键字参数字典。"""
    return dict(
        input_dir=args.input,
        output_dir=args.output,
        ext=args.ext,
        quality=args.quality,
        seed=args.seed,
        prefix=args.prefix,
        rotate_enabled=args.rotate_enabled,
        rotate_degrees=args.rotate_degrees,
        rotate_prob=args.rotate_prob,
        cut_enabled=args.cut_enabled,
        cut_scale=args.cut_scale,
        cut_ratio=args.cut_ratio,
        cut_prob=args.cut_prob,
        cut_resize=args.cut_resize,
        occlusion_enabled=args.occlusion_enabled,
        occlusion_count=args.occlusion_count,
        occlusion_size=args.occlusion_size,
        occlusion_prob=args.occlusion_prob,
        channel_enabled=args.channel_enabled,
        channel_prob=args.channel_prob,
    )


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {args.input}")
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是目录: {args.input}")

    if args.quality is not None and not (1 <= args.quality <= 100):
        raise ValueError(f"--quality 必须在 1~100 之间，当前为 {args.quality}")

    if args.rotate_degrees is not None and args.rotate_degrees < 0:
        raise ValueError(f"--rotate-degrees 必须 >= 0，当前为 {args.rotate_degrees}")

    if args.cut_scale is not None and not (0 <= args.cut_scale <= 1):
        raise ValueError(f"--cut-scale 必须在 [0, 1] 之间，当前为 {args.cut_scale}")

    if args.cut_ratio is not None and args.cut_ratio < 1:
        raise ValueError(f"--cut-ratio 必须 >= 1，当前为 {args.cut_ratio}")

    if args.occlusion_count is not None and args.occlusion_count < 0:
        raise ValueError(f"--occlusion-count 必须 >= 0，当前为 {args.occlusion_count}")

    if args.occlusion_size is not None and not (0 <= args.occlusion_size <= 1):
        raise ValueError(
            f"--occlusion-size 必须在 [0, 1] 之间，当前为 {args.occlusion_size}"
        )

    if args.rotate_prob is not None and not (0 <= args.rotate_prob <= 1):
        raise ValueError(f"--rotate-prob 必须在 [0, 1] 之间，当前为 {args.rotate_prob}")

    if args.cut_prob is not None and not (0 <= args.cut_prob <= 1):
        raise ValueError(f"--cut-prob 必须在 [0, 1] 之间，当前为 {args.cut_prob}")

    if args.occlusion_prob is not None and not (0 <= args.occlusion_prob <= 1):
        raise ValueError(
            f"--occlusion-prob 必须在 [0, 1] 之间，当前为 {args.occlusion_prob}"
        )

    if args.channel_prob is not None and not (0 <= args.channel_prob <= 1):
        raise ValueError(
            f"--channel-prob 必须在 [0, 1] 之间，当前为 {args.channel_prob}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    """``python scripts/vh.py images augment`` 命令行入口。

    Returns:
        0 成功；1 运行时错误；2 参数非法。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[参数错误] {exc}", stream=sys.stderr)
        return 2

    kwargs = _normalize_args(args)

    try:
        saved = augment_images(**kwargs)
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except Exception as exc:
        log(f"[错误] 数据增强失败: {exc}", stream=sys.stderr)
        return 1

    log(f"[完成] 共增强 {len(saved)} 张图片")
    return 0


if __name__ == "__main__":
    sys.exit(main())
