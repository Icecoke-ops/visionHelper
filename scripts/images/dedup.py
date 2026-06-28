#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片去重模块（``python scripts/vh.py images dedup`` 实现）。

合并旧 ``scripts.core.deduplicate_images`` 与 ``scripts.deduplicate_images``
CLI 门面，提供 :func:`deduplicate` 核心实现与 ``main`` 命令行入口。

零副作用约定
------------

``torch`` / ``transformers`` 等重依赖仅在 ``backend=vit`` 且调用相关函数时
按需 import，模块顶层仅引入 ``Pillow`` / ``numpy``。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from scripts.common.config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_PHASH_SIZE,
    DEFAULT_VIT_BATCH_SIZE,
    DEFAULT_VIT_MODEL,
    SUPPORTED_DEDUP_BACKENDS as SUPPORTED_BACKENDS,
)
from scripts.common.logging import ProgressLogger, log
from scripts.common.utils import is_image_file

__all__ = [
    "SUPPORTED_BACKENDS",
    "deduplicate",
    "extract_features_phash",
    "extract_features_vit",
    "find_duplicates",
    "list_images",
    "load_model",
    "main",
]


def list_images(folder: str) -> List[Path]:
    """列出文件夹中所有支持的图片文件（不递归）。"""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"路径不存在或不是目录: {folder}")

    images = [p for p in folder_path.iterdir() if is_image_file(p)]
    images.sort(key=lambda x: x.name)
    return images


def _crop_grid(image: Image.Image, grid_size: int):
    """将图片均匀切分为 ``grid_size × grid_size`` 个子图块。

    返回按行主序排列的图块列表。如果图片宽高不能被 ``grid_size``
    整除，最后一行/一列图块会延伸到图片边缘。
    """
    w, h = image.size
    if grid_size > min(w, h):
        raise ValueError(
            f"grid_size 不能大于图片最短边 {min(w, h)}，当前为 {grid_size}"
        )
    tile_w = w // grid_size
    tile_h = h // grid_size
    tiles: list[Image.Image] = []
    for row in range(grid_size):
        top = row * tile_h
        bottom = top + tile_h if row < grid_size - 1 else h
        for col in range(grid_size):
            left = col * tile_w
            right = left + tile_w if col < grid_size - 1 else w
            tiles.append(image.crop((left, top, right, bottom)))
    return tiles


# ---------------------------------------------------------------------- #
# 后端 1：ViT / DINOv2 等 transformers 模型
# ---------------------------------------------------------------------- #


def load_model(model_name: str = DEFAULT_VIT_MODEL):
    """加载 ViT/DINOv2 模型和图像处理器。"""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"使用设备: {device}")
    log(f"正在加载模型: {model_name} ...")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model, device


def extract_features_vit(
        image_paths: List[Path],
        processor,
        model,
        device,
        batch_size: int = DEFAULT_VIT_BATCH_SIZE,
        grid_size: int = DEFAULT_GRID_SIZE,
) -> List[Optional[np.ndarray]]:
    """批量提取图片特征向量（L2 归一化后的 [CLS] / pooled 输出）。

    当 ``grid_size > 1`` 时，每张图片被均匀切分为
    ``grid_size × grid_size`` 个子图块，并分别提取特征，
    返回形状为 ``(grid_size**2, hidden_dim)`` 的二维数组。
    """
    import torch

    features: List[Optional[np.ndarray]] = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    progress = ProgressLogger(
        total=total_batches, desc="提取特征", step_percent=1.0,
    )

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            log(f"[提取特征] 处理批次 {i // batch_size + 1}/{total_batches}（{batch_paths[0].name} ~ {batch_paths[-1].name}）")

            all_tiles: list[Image.Image] = []
            tile_counts: list[int] = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    if grid_size > 1:
                        tiles = _crop_grid(img, grid_size)
                        all_tiles.extend(tiles)
                        tile_counts.append(len(tiles))
                    else:
                        all_tiles.append(img)
                        tile_counts.append(1)
                except Exception as e:
                    log(f"[警告] 无法读取图片 {path}: {e}")
                    tile_counts.append(0)

            if all(c == 0 for c in tile_counts):
                features.extend([None] * len(batch_paths))
                continue

            inputs = processor(images=all_tiles, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            tile_features = last_hidden[:, 0, :].cpu().numpy()

            tile_features = tile_features / (
                    np.linalg.norm(tile_features, axis=1, keepdims=True) + 1e-12
            )

            idx = 0
            for count in tile_counts:
                if count == 0:
                    features.append(None)
                else:
                    if grid_size > 1:
                        features.append(tile_features[idx: idx + count])
                    else:
                        features.append(tile_features[idx])
                    idx += count

            progress.update(1)

    progress.close()
    return features


# ---------------------------------------------------------------------- #
# 后端 2：感知哈希 pHash
# ---------------------------------------------------------------------- #


def _phash_vector(image: Image.Image, hash_size: int = DEFAULT_PHASH_SIZE) -> np.ndarray:
    """计算单张图片的感知哈希向量（DCT 低频系数二值化）。"""
    try:
        from scipy.fftpack import dct  # type: ignore

        img = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        dct_full = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
        block = dct_full[:hash_size, :hash_size]
        median = np.median(block)
        bits = (block > median).astype(np.float32).flatten()
    except Exception:
        img = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        diff = arr[:, 1:] > arr[:, :-1]
        bits = diff.astype(np.float32).flatten()

    vec = bits * 2.0 - 1.0
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def extract_features_phash(
        image_paths: List[Path],
        hash_size: int = DEFAULT_PHASH_SIZE,
        grid_size: int = DEFAULT_GRID_SIZE,
) -> List[Optional[np.ndarray]]:
    """使用感知哈希为每张图片生成归一化向量。

    当 ``grid_size > 1`` 时，每张图片被均匀切分为
    ``grid_size × grid_size`` 个子图块，分别计算哈希向量，
    返回形状为 ``(grid_size**2, hash_size**2)`` 的二维数组。
    """
    features: List[Optional[np.ndarray]] = []
    progress = ProgressLogger(
        total=len(image_paths), desc="计算 pHash", step_percent=1.0,
    )
    for idx, path in enumerate(image_paths):
        log(f"[计算 pHash] {idx + 1}/{len(image_paths)} {path.name}")
        try:
            with Image.open(path) as img:
                if grid_size > 1:
                    tiles = _crop_grid(img, grid_size)
                    tile_vecs = [_phash_vector(t, hash_size=hash_size) for t in tiles]
                    features.append(np.stack(tile_vecs))
                else:
                    features.append(_phash_vector(img, hash_size=hash_size))
        except Exception as e:
            log(f"[警告] 无法读取图片 {path}: {e}")
            features.append(None)
        progress.update(1)
    progress.close()
    return features


# ---------------------------------------------------------------------- #
# 重复识别
# ---------------------------------------------------------------------- #


def find_duplicates(
        features: List[Optional[np.ndarray]],
        threshold: float,
        grid_size: int = DEFAULT_GRID_SIZE,
) -> Tuple[List[int], List[int]]:
    """使用余弦相似度查找重复图片。

    当 ``grid_size == 1`` 时采用原始向量化实现（单向量全图对比）；
    当 ``grid_size > 1`` 时采用逐格对比：两张图片只有 **所有对应子图块**
    的余弦相似度均 ``>= threshold`` 才被判定为重复，否则保留两者。
    这使得图像中任意小区域（如划痕、缺陷）发生变化时不会被误删。
    """
    n = len(features)
    if n == 0:
        return [], []

    valid_indices = [i for i, f in enumerate(features) if f is not None]
    if not valid_indices:
        return [], []

    m = len(valid_indices)
    visited = np.zeros(m, dtype=bool)
    keep: List[int] = []
    duplicate: List[int] = []

    if grid_size <= 1:
        # --- 原始全图模式：向量化矩阵乘法，O(m²) ---
        matrix = np.stack([features[i] for i in valid_indices]).astype(np.float32)
        sim = matrix @ matrix.T
    else:
        # --- 网格分块模式：预计算所有子块的余弦相似度矩阵，取最不相似格 ---
        all_tile_vecs = np.stack(
            [features[i] for i in valid_indices]
        ).astype(np.float32)
        num_tiles = all_tile_vecs.shape[1]

        # (m, m, num_tiles) 各子块的余弦相似度
        tile_sim_3d = np.zeros((m, m, num_tiles), dtype=np.float32)
        for t in range(num_tiles):
            vecs = all_tile_vecs[:, t, :]
            tile_sim_3d[:, :, t] = vecs @ vecs.T

        sim = tile_sim_3d.min(axis=2)

    for local_i in range(m):
        if visited[local_i]:
            continue
        visited[local_i] = True
        keep.append(valid_indices[local_i])

        row = sim[local_i]
        for local_j in range(local_i + 1, m):
            if visited[local_j]:
                continue
            if row[local_j] >= threshold:
                visited[local_j] = True
                duplicate.append(valid_indices[local_j])

    return keep, duplicate


# ---------------------------------------------------------------------- #
# 高层入口
# ---------------------------------------------------------------------- #


def deduplicate(
        folder: str,
        threshold: float = 0.95,
        delete: bool = False,
        move_to: Optional[str] = None,
        model_name: str = DEFAULT_VIT_MODEL,
        batch_size: int = DEFAULT_VIT_BATCH_SIZE,
        backend: str = "vit",
        hash_size: int = DEFAULT_PHASH_SIZE,
        grid_size: int = DEFAULT_GRID_SIZE,
) -> dict:
    """对文件夹中的相似图片进行去重。"""
    if not (0 < threshold <= 1):
        raise ValueError("阈值必须在 (0, 1] 之间")

    if delete and move_to:
        raise ValueError("delete 和 move_to 不能同时为 True/非空")

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"不支持的后端: {backend}，可选值: {SUPPORTED_BACKENDS}"
        )

    if grid_size < 1:
        raise ValueError(f"grid_size 必须 >= 1，当前为 {grid_size}")

    image_paths = list_images(folder)
    if not image_paths:
        log(f"文件夹中没有找到支持的图片: {folder}")
        return {"keep": [], "duplicates": []}

    grid_info = f"，网格 {grid_size}×{grid_size}" if grid_size > 1 else ""
    log(f"共找到 {len(image_paths)} 张图片，使用后端: {backend}{grid_info}")

    if backend == "vit":
        processor, model, device = load_model(model_name)
        features = extract_features_vit(
            image_paths, processor, model, device,
            batch_size=batch_size, grid_size=grid_size,
        )
    else:  # phash
        features = extract_features_phash(
            image_paths, hash_size=hash_size, grid_size=grid_size,
        )

    log("正在比较图片相似度 …")
    keep_indices, duplicate_indices = find_duplicates(features, threshold, grid_size=grid_size)
    log(f"比较完成，{len(keep_indices)} 张保留，{len(duplicate_indices)} 张重复")

    keep_paths = [image_paths[i] for i in keep_indices]
    duplicate_paths = [image_paths[i] for i in duplicate_indices]

    if duplicate_paths:
        log("\n重复图片列表:")
        for path in duplicate_paths:
            log(f"  - {path}")

        if move_to:
            move_dir = Path(move_to)
            move_dir.mkdir(parents=True, exist_ok=True)
            for src in duplicate_paths:
                dst = move_dir / src.name
                counter = 1
                stem = dst.stem
                suffix = dst.suffix
                while dst.exists():
                    dst = move_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.move(str(src), str(dst))
            log(f"\n已将 {len(duplicate_paths)} 张重复图片移动到: {move_dir}")
        elif delete:
            for src in duplicate_paths:
                src.unlink()
            log(f"\n已删除 {len(duplicate_paths)} 张重复图片。")
        else:
            log("\n未指定 delete 或 move_to，仅列出重复图片，未执行任何操作。")

    return {"keep": keep_paths, "duplicates": duplicate_paths}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py images dedup",
        description=(
            "图片去重工具：基于 ViT 特征或感知哈希（pHash）查找相似图片，"
            "可选择仅检测、删除或移动到指定目录。"
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True, help="待处理图片所在的文件夹路径"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="相似度阈值（0~1，越高越严格），默认 0.95。",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="直接删除重复图片（与 --move-to 互斥）。",
    )
    parser.add_argument(
        "--move-to",
        type=str,
        default=None,
        help="将重复图片移动到指定目录（与 --delete 互斥，目标目录不存在会自动创建）。",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vit",
        choices=sorted(SUPPORTED_BACKENDS),
        help="特征后端：vit=高精度但需要 GPU/较慢；phash=快速但仅适合明显重复。默认 vit。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VIT_MODEL,
        help=f"使用的 ViT/DINOv2 模型名称（仅 backend=vit 时生效），默认 {DEFAULT_VIT_MODEL}。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_VIT_BATCH_SIZE,
        help=(
            f"特征提取的批大小（必须为正整数，仅 backend=vit 时生效），"
            f"默认 {DEFAULT_VIT_BATCH_SIZE}。"
        ),
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=DEFAULT_PHASH_SIZE,
        help=(
            f"pHash 哈希尺寸（必须为正整数；向量维度 = hash_size**2；仅 "
            f"backend=phash 时生效），默认 {DEFAULT_PHASH_SIZE}。"
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=(
            f"网格分块大小（1=不分块；N=将每张图均匀切为 N×N 块，逐块比较"
            f"特征，任一子块相似度低于阈值则判为不重复）。默认 {DEFAULT_GRID_SIZE}。"
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    folder = Path(args.input)
    if not folder.exists():
        raise ValueError(f"目录不存在：{args.input}")
    if not folder.is_dir():
        raise ValueError(f"路径不是目录：{args.input}")

    if not (0.0 < args.threshold <= 1.0):
        raise ValueError(
            f"--threshold 必须在 (0, 1] 之间，当前为 {args.threshold}"
        )

    if args.delete and args.move_to:
        raise ValueError("--delete 与 --move-to 不能同时指定，请二选一。")

    if args.move_to:
        move_to = Path(args.move_to)
        if move_to.exists() and not move_to.is_dir():
            raise ValueError(f"--move-to 目标已存在但不是目录：{args.move_to}")

    if args.backend == "vit":
        if args.batch_size is not None and args.batch_size < 1:
            raise ValueError(f"--batch-size 必须为正整数，当前为 {args.batch_size}")
    elif args.backend == "phash":
        if args.hash_size is not None and args.hash_size < 1:
            raise ValueError(f"--hash-size 必须为正整数，当前为 {args.hash_size}")

    if args.grid_size is not None and args.grid_size < 1:
        raise ValueError(f"--grid-size 必须 >= 1，当前为 {args.grid_size}")


def main(argv: Optional[List[str]] = None) -> int:
    """
    ``python scripts/vh.py images dedup`` 命令行入口。

    返回:
        0 成功；1 运行时错误；2 参数非法；130 用户中断。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[参数错误] {exc}", stream=sys.stderr)
        return 2

    try:
        result = deduplicate(
            folder=args.input,
            threshold=args.threshold,
            delete=args.delete,
            move_to=args.move_to,
            model_name=args.model,
            batch_size=args.batch_size,
            backend=args.backend,
            hash_size=args.hash_size,
            grid_size=args.grid_size,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，未对已处理图片造成不可逆变更。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except ImportError as exc:
        log(
            f"[依赖缺失] {exc}\n"
            f"提示：使用 --backend vit 需要安装 torch / transformers；"
            f"或改用 --backend phash 以避免该依赖。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 图片去重失败：{exc}", stream=sys.stderr)
        return 1

    if isinstance(result, dict):
        keep_n = len(result.get("keep", []) or [])
        dup_n = len(result.get("duplicates", []) or [])
        log(f"[完成] 保留 {keep_n} 张，识别重复 {dup_n} 张。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
