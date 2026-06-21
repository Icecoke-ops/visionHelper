#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 X-AnyLabeling 标注为 YOLO 数据集 CLI 门面。

实际实现位于 :mod:`scripts.core.export_yolo_dataset`，本模块仅负责：

1. 命令行参数解析；
2. 调用前做 **轻量边界校验**（路径存在、比例之和、copy_mode 等）；
3. 统一异常捕获并返回合适退出码；
4. 对外 re-export 公开 API 与常量，保持向后兼容。

数据集仅划分为训练集与测试集，不生成单独验证集（``data.yaml`` 中 ``val``
指向测试集以满足 Ultralytics 训练校验需求）。``classify`` 任务输出
ImageFolder 结构（``images/{train,test}/<class>/<image>``），其它任务输出
标准 ``images/`` + ``labels/`` 结构。

用法::

    python -m scripts.export_yolo_dataset <input_dir> <output_dir> [options]

例（检测任务，软链接落盘）::

    python -m scripts.export_yolo_dataset ./images ./.dataset \\
        --task detect --train-ratio 0.8 --test-ratio 0.2 --copy-mode symlink
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.core.export_yolo_dataset import (
    COPY_MODES,
    DETECT_SHAPE_TYPES,
    OBB_SHAPE_TYPES,
    SEGMENT_SHAPE_TYPES,
    SUPPORTED_TASKS,
    export_yolo_dataset,
)
from scripts.logging_utils import log

__all__ = [
    "export_yolo_dataset",
    "SUPPORTED_TASKS",
    "COPY_MODES",
    "DETECT_SHAPE_TYPES",
    "OBB_SHAPE_TYPES",
    "SEGMENT_SHAPE_TYPES",
    "main",
]

# 比例之和允许的浮点误差范围（避免 0.8 + 0.2 在浮点下不严格等于 1.0）
_RATIO_SUM_TOL = 1e-6


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_yolo_dataset",
        description=(
            "将 X-AnyLabeling JSON 标注的图片目录导出为 YOLO 数据集。"
            "支持 detect / obb / segment / classify 4 种任务。"
        ),
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="包含图片与 X-AnyLabeling JSON 的目录",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="输出的数据集目录（不存在会自动创建）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（0~1），默认 0.8",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集比例（0~1），默认 0.2；train-ratio + test-ratio 须约等于 1。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机划分种子，默认 42",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=sorted(COPY_MODES),
        help=(
            "图片落盘方式：copy=复制（默认）；link=硬链接（同一文件系统）；"
            "symlink=软链接（跨文件系统也可，需目标文件系统支持）。"
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在：{args.input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录：{args.input_dir}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"输出路径已存在但不是目录：{args.output_dir}")

    # 防止用户误把 output 设为 input 本身（会导致原数据被覆盖/链接到自身）
    try:
        if output_dir.exists() and input_dir.resolve() == output_dir.resolve():
            raise ValueError("输入目录与输出目录不能相同，否则会破坏原始数据。")
    except OSError:
        # resolve 失败时（如包含不存在的中间路径）忽略此项校验
        pass

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(
            f"--train-ratio 必须在 (0, 1) 之间，当前为 {args.train_ratio}"
        )
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError(
            f"--test-ratio 必须在 (0, 1) 之间，当前为 {args.test_ratio}"
        )
    if abs(args.train_ratio + args.test_ratio - 1.0) > _RATIO_SUM_TOL:
        raise ValueError(
            f"--train-ratio + --test-ratio 必须约等于 1，当前为 "
            f"{args.train_ratio} + {args.test_ratio} = "
            f"{args.train_ratio + args.test_ratio}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回:
        进程退出码：0=成功；2=参数非法；1=运行时错误；130=用户中断。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[参数错误] {exc}", stream=sys.stderr)
        return 2

    try:
        result = export_yolo_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            task=args.task,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            copy_mode=args.copy_mode,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，输出目录可能处于不完整状态。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except OSError as exc:
        # symlink / hardlink 在跨设备 / 权限不足时常见
        log(
            f"[文件系统错误] {exc}\n"
            f"提示：若使用 --copy-mode link/symlink 失败，可尝试 --copy-mode copy。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 导出 YOLO 数据集失败：{exc}", stream=sys.stderr)
        return 1

    if isinstance(result, dict):
        train_n = result.get("train", 0)
        test_n = result.get("test", 0)
        log(f"[完成] 训练集 {train_n} 张，测试集 {test_n} 张 → {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
