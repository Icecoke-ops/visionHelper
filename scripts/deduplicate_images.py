#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ViT 模型对文件夹中的相似图片进行去重。

该模块暴露以下核心方法：
    - deduplicate(folder, threshold=0.95, delete=False, move_to=None,
                  model_name="google/vit-base-patch16-224", batch_size=8)

用法示例：
    from deduplicate_images import deduplicate
    deduplicate("/path/to/images", threshold=0.95, delete=True)
"""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel

# 支持的图片格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def list_images(folder: str) -> List[Path]:
    """列出文件夹中所有支持的图片文件（不递归）。"""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"路径不存在或不是目录: {folder}")

    images = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    # 按文件名排序，保证结果稳定
    images.sort(key=lambda x: x.name)
    return images


def load_model(model_name: str):
    """加载 ViT 模型和图像处理器。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"正在加载模型: {model_name} ...")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model, device


@torch.no_grad()
def extract_features(
        image_paths: List[Path],
        processor,
        model,
        device,
        batch_size: int = 8,
):
    """批量提取图片特征向量。"""
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
        batch_paths = image_paths[i: i + batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"[警告] 无法读取图片 {path}: {e}")
                batch_images.append(None)

        # 过滤掉读取失败的图片，但保留位置占位
        valid_images = [img for img in batch_images if img is not None]
        if not valid_images:
            features.extend([None] * len(batch_paths))
            continue

        inputs = processor(images=valid_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        # 使用 [CLS] token 的输出作为图片特征
        batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # 归一化特征向量
        batch_features = batch_features / (
                np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-12
        )

        # 将特征放回原来的位置
        feat_iter = iter(batch_features)
        for img in batch_images:
            if img is None:
                features.append(None)
            else:
                features.append(next(feat_iter))

    return features


def find_duplicates(features: List[Optional[np.ndarray]], threshold: float) -> Tuple[List[int], List[int]]:
    """
    使用余弦相似度查找重复图片。
    返回保留图片的索引列表，以及被判定为重复的索引列表。
    """
    n = len(features)
    keep = []
    duplicate = []
    # 记录每个索引是否已经被划分到某个重复组
    visited = [False] * n

    for i in range(n):
        if features[i] is None or visited[i]:
            continue

        # 以第 i 张图片为基准，找到所有相似图片
        group = [i]
        visited[i] = True

        for j in range(i + 1, n):
            if features[j] is None or visited[j]:
                continue
            sim = float(np.dot(features[i], features[j]))
            if sim >= threshold:
                group.append(j)
                visited[j] = True

        # 保留组内第一张图片，其余视为重复
        keep.append(group[0])
        duplicate.extend(group[1:])

    return keep, duplicate


def deduplicate(
        folder: str,
        threshold: float = 0.95,
        delete: bool = False,
        move_to: Optional[str] = None,
        model_name: str = "google/vit-base-patch16-224",
        batch_size: int = 8,
) -> dict:
    """
    对文件夹中的相似图片进行去重。

    参数:
        folder: 待处理图片所在的文件夹路径。
        threshold: 相似度阈值，默认 0.95。余弦相似度大于该值视为重复。
        delete: 是否删除重复图片，默认 False。
        move_to: 将重复图片移动到指定目录。与 delete 互斥。
        model_name: 使用的 ViT 模型名称。
        batch_size: 特征提取的批大小。

    返回:
        包含 keep（保留图片路径列表）和 duplicates（重复图片路径列表）的字典。
    """
    if not (0 <= threshold <= 1):
        raise ValueError("阈值必须在 [0, 1] 之间")

    if delete and move_to:
        raise ValueError("delete 和 move_to 不能同时为 True/非空")

    image_paths = list_images(folder)
    if not image_paths:
        print(f"文件夹中没有找到支持的图片: {folder}")
        return {"keep": [], "duplicates": []}

    print(f"共找到 {len(image_paths)} 张图片")

    processor, model, device = load_model(model_name)
    features = extract_features(image_paths, processor, model, device, batch_size=batch_size)

    keep_indices, duplicate_indices = find_duplicates(features, threshold)

    keep_paths = [image_paths[i] for i in keep_indices]
    duplicate_paths = [image_paths[i] for i in duplicate_indices]

    print(f"\n保留图片: {len(keep_paths)} 张")
    print(f"重复图片: {len(duplicate_paths)} 张")

    if duplicate_paths:
        print("\n重复图片列表:")
        for path in duplicate_paths:
            print(f"  - {path}")

        if move_to:
            move_dir = Path(move_to)
            move_dir.mkdir(parents=True, exist_ok=True)
            for src in duplicate_paths:
                dst = move_dir / src.name
                # 如果目标文件已存在，添加序号后缀
                counter = 1
                stem = dst.stem
                suffix = dst.suffix
                while dst.exists():
                    dst = move_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.move(str(src), str(dst))
            print(f"\n已将 {len(duplicate_paths)} 张重复图片移动到: {move_dir}")
        elif delete:
            for src in duplicate_paths:
                src.unlink()
            print(f"\n已删除 {len(duplicate_paths)} 张重复图片。")
        else:
            print("\n未指定 delete 或 move_to，仅列出重复图片，未执行任何操作。")

    return {"keep": keep_paths, "duplicates": duplicate_paths}


if __name__ == "__main__":
    # 提供简单的命令行入口，便于快速测试
    import argparse

    parser = argparse.ArgumentParser(description="ViT 图片去重工具")
    parser.add_argument("folder", type=str, help="待处理图片所在的文件夹路径")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="相似度阈值，默认 0.95。",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="删除重复图片。",
    )
    parser.add_argument(
        "--move-to",
        type=str,
        default=None,
        help="将重复图片移动到指定目录。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/vit-base-patch16-224",
        help="使用的 ViT 模型名称。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="特征提取的批大小。",
    )
    args = parser.parse_args()

    deduplicate(
        folder=args.folder,
        threshold=args.threshold,
        delete=args.delete,
        move_to=args.move_to,
        model_name=args.model,
        batch_size=args.batch_size,
    )
