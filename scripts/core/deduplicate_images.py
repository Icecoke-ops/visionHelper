#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片去重核心实现，支持多种特征提取后端。

后端（``backend`` 参数）：
    - ``vit``：基于 HuggingFace ``transformers`` 的 ViT / DINOv2 等模型，
      使用 ``[CLS]`` token 的 L2 归一化向量计算余弦相似度（默认）。
    - ``phash``：使用感知哈希（pHash），仅依赖 ``Pillow`` 与 ``numpy``，
      速度快、显存占用低，适合大批量图片的粗筛或无 GPU 环境。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from scripts._common import is_image_file
from scripts.config import (
    DEFAULT_PHASH_SIZE,
    DEFAULT_VIT_BATCH_SIZE,
    DEFAULT_VIT_MODEL,
    SUPPORTED_DEDUP_BACKENDS as SUPPORTED_BACKENDS,
)
from scripts.logging_utils import ProgressLogger, log

__all__ = [
    "SUPPORTED_BACKENDS",
    "deduplicate",
    "extract_features_phash",
    "extract_features_vit",
    "find_duplicates",
    "list_images",
    "load_model",
]


def list_images(folder: str) -> List[Path]:
    """列出文件夹中所有支持的图片文件（不递归）。"""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"路径不存在或不是目录: {folder}")

    images = [p for p in folder_path.iterdir() if is_image_file(p)]
    images.sort(key=lambda x: x.name)
    return images


# ----------------------------------------------------------------------
# 后端 1：ViT / DINOv2 等 transformers 模型
# ----------------------------------------------------------------------

def load_model(model_name: str = DEFAULT_VIT_MODEL):
    """加载 ViT/DINOv2 模型和图像处理器。

    使用 ``transformers.AutoModel`` 以兼容 ViT、DINOv2 等不同架构。
    """
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
) -> List[Optional[np.ndarray]]:
    """批量提取图片特征向量（L2 归一化后的 [CLS] / pooled 输出）。"""
    import torch

    features: List[Optional[np.ndarray]] = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    progress = ProgressLogger(total=total_batches, desc="提取特征")
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            batch_images: List[Optional[Image.Image]] = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    log(f"[警告] 无法读取图片 {path}: {e}")
                    batch_images.append(None)

            valid_images = [img for img in batch_images if img is not None]
            if not valid_images:
                features.extend([None] * len(batch_paths))
                continue

            inputs = processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            batch_features = last_hidden[:, 0, :].cpu().numpy()

            batch_features = batch_features / (
                    np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-12
            )

            feat_iter = iter(batch_features)
            for img in batch_images:
                if img is None:
                    features.append(None)
                else:
                    features.append(next(feat_iter))
            progress.update(1)

    progress.close()
    return features


# ----------------------------------------------------------------------
# 后端 2：感知哈希 pHash
# ----------------------------------------------------------------------

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
) -> List[Optional[np.ndarray]]:
    """使用感知哈希为每张图片生成归一化向量。"""
    features: List[Optional[np.ndarray]] = []
    progress = ProgressLogger(total=len(image_paths), desc="计算 pHash")
    for path in image_paths:
        try:
            with Image.open(path) as img:
                features.append(_phash_vector(img, hash_size=hash_size))
        except Exception as e:
            log(f"[警告] 无法读取图片 {path}: {e}")
            features.append(None)
        progress.update(1)
    progress.close()
    return features


# ----------------------------------------------------------------------
# 重复识别
# ----------------------------------------------------------------------

def find_duplicates(
        features: List[Optional[np.ndarray]],
        threshold: float,
) -> Tuple[List[int], List[int]]:
    """使用余弦相似度查找重复图片（向量化实现）。"""
    n = len(features)
    if n == 0:
        return [], []

    valid_indices = [i for i, f in enumerate(features) if f is not None]
    if not valid_indices:
        return [], []

    matrix = np.stack([features[i] for i in valid_indices]).astype(np.float32)
    sim = matrix @ matrix.T

    m = matrix.shape[0]
    visited = np.zeros(m, dtype=bool)
    keep: List[int] = []
    duplicate: List[int] = []

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


# ----------------------------------------------------------------------
# 高层入口
# ----------------------------------------------------------------------

def deduplicate(
        folder: str,
        threshold: float = 0.95,
        delete: bool = False,
        move_to: Optional[str] = None,
        model_name: str = DEFAULT_VIT_MODEL,
        batch_size: int = DEFAULT_VIT_BATCH_SIZE,
        backend: str = "vit",
        hash_size: int = DEFAULT_PHASH_SIZE,
) -> dict:
    """对文件夹中的相似图片进行去重。"""
    if not (0 <= threshold <= 1):
        raise ValueError("阈值必须在 [0, 1] 之间")

    if delete and move_to:
        raise ValueError("delete 和 move_to 不能同时为 True/非空")

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"不支持的后端: {backend}，可选值: {SUPPORTED_BACKENDS}"
        )

    image_paths = list_images(folder)
    if not image_paths:
        log(f"文件夹中没有找到支持的图片: {folder}")
        return {"keep": [], "duplicates": []}

    log(f"共找到 {len(image_paths)} 张图片，使用后端: {backend}")

    if backend == "vit":
        processor, model, device = load_model(model_name)
        features = extract_features_vit(
            image_paths, processor, model, device, batch_size=batch_size
        )
    else:  # phash
        features = extract_features_phash(image_paths, hash_size=hash_size)

    keep_indices, duplicate_indices = find_duplicates(features, threshold)

    keep_paths = [image_paths[i] for i in keep_indices]
    duplicate_paths = [image_paths[i] for i in duplicate_indices]

    log(f"\n保留图片: {len(keep_paths)} 张")
    log(f"重复图片: {len(duplicate_paths)} 张")

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
