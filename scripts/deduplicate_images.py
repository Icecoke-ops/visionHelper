#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片去重工具 CLI 门面（ViT / pHash）。

实际实现位于 :mod:`scripts.core.deduplicate_images`，本模块仅负责：

1. 命令行参数解析；
2. 调用前做 **轻量参数预校验**（阈值范围、互斥选项、批大小、哈希尺寸等）；
3. 统一异常捕获，给出友好的中文提示并返回合适退出码；
4. 对外 re-export 公开符号，保持向后兼容。

注意：``import scripts.deduplicate_images`` 本身不会拉起 ``torch`` /
``transformers``，重依赖仅在调用 :func:`deduplicate` / :func:`load_model`
等核心实现时按需加载。

用法::

    python -m scripts.deduplicate_images <folder> [options]

例（pHash 后端，快速去重）::

    python -m scripts.deduplicate_images ./images --backend phash --threshold 0.92

例（ViT 后端，更高精度但更慢）::

    python -m scripts.deduplicate_images ./images --backend vit \\
        --model google/vit-base-patch16-224 --batch-size 8 --threshold 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.config import (
    DEFAULT_PHASH_SIZE,
    DEFAULT_VIT_BATCH_SIZE,
    DEFAULT_VIT_MODEL,
    SUPPORTED_DEDUP_BACKENDS,
)
from scripts.core.deduplicate_images import (
    SUPPORTED_BACKENDS,
    deduplicate,
    extract_features_phash,
    extract_features_vit,
    find_duplicates,
    list_images,
    load_model,
)
from scripts.logging_utils import log

__all__ = [
    "deduplicate",
    "extract_features_vit",
    "extract_features_phash",
    "find_duplicates",
    "list_images",
    "load_model",
    "SUPPORTED_BACKENDS",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.deduplicate_images",
        description=(
            "图片去重工具：基于 ViT 特征或感知哈希（pHash）查找相似图片，"
            "可选择仅检测、删除或移动到指定目录。"
        ),
    )
    parser.add_argument("folder", type=str, help="待处理图片所在的文件夹路径")
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
        choices=sorted(SUPPORTED_DEDUP_BACKENDS),
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
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    folder = Path(args.folder)
    if not folder.exists():
        raise ValueError(f"目录不存在：{args.folder}")
    if not folder.is_dir():
        raise ValueError(f"路径不是目录：{args.folder}")

    if not (0.0 < args.threshold <= 1.0):
        raise ValueError(
            f"--threshold 必须在 (0, 1] 之间，当前为 {args.threshold}"
        )

    # --delete 与 --move-to 互斥
    if args.delete and args.move_to:
        raise ValueError("--delete 与 --move-to 不能同时指定，请二选一。")

    # --move-to 必须可写
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


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    参数:
        argv: 命令行参数列表（不含程序名），默认从 ``sys.argv[1:]`` 读取。

    返回:
        进程退出码：0 表示成功；2 表示参数非法；1 表示运行时错误；130 表示用户中断。
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
            folder=args.folder,
            threshold=args.threshold,
            delete=args.delete,
            move_to=args.move_to,
            model_name=args.model,
            batch_size=args.batch_size,
            backend=args.backend,
            hash_size=args.hash_size,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，未对已处理图片造成不可逆变更。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except ImportError as exc:
        # 主要针对 backend=vit 时缺失 torch / transformers 的情况
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
