#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动标注工具 CLI 门面。

实际实现位于 :mod:`scripts.core.auto_annotate`，本模块仅负责：

1. 解析命令行参数（``argparse``）；
2. 调用前做 **轻量边界校验**（路径存在性、阈值范围、任务类型、批大小等）；
3. 自动处理 ``include-*`` 开关的默认值（4 个都未指定时回退为"仅未标注"）；
4. 统一捕获常见异常，输出友好中文提示并返回合适退出码；
5. 对外 re-export 公开 API 与常量，保持向后兼容。

注意：``import scripts.auto_annotate`` 本身不会拉起 ``ultralytics`` / ``torch``，
这些重依赖仅在调用 :func:`auto_annotate` 时按需加载。

用法::

    python -m scripts.auto_annotate <work_dir> <model_path> [options]

例（检测任务，仅处理未标注图片，置信度 0.3）::

    python -m scripts.auto_annotate ./images ./runs/train/weights/best.pt \\
        --task detect --threshold 0.3

例（重新刷新所有自动标注 + 自动矫正的图片）::

    python -m scripts.auto_annotate ./images ./best.pt \\
        --task detect --include-auto --include-auto-corrected
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts._common import discover_trained_models
from scripts.config import (
    AUTO_ANNOTATE_DEFAULT_BATCH_SIZE,
    DEFAULT_TOLERANCE_SECONDS,
    SUPPORTED_TASKS,
)
from scripts.core.auto_annotate import auto_annotate
from scripts.logging_utils import log

__all__ = [
    "auto_annotate",
    "discover_trained_models",
    "SUPPORTED_TASKS",
    "AUTO_ANNOTATE_DEFAULT_BATCH_SIZE",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.auto_annotate",
        description=(
            "YOLO 自动标注工具：使用已训练模型为目录中的图片生成 X-AnyLabeling "
            "JSON 标注，支持 detect / obb / segment / classify 4 种任务。"
        ),
    )
    parser.add_argument("work_dir", type=str, help="待标注图片所在的工作目录")
    parser.add_argument("model_path", type=str, help="YOLO 模型权重文件路径（.pt）")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="置信度阈值（0~1），默认 0.25（仅 detect/obb/segment 生效）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="输出 JSON 文件名后缀（追加在 stem 之后），默认空",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，例如 0/cpu/0,1，默认自动选择",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值（0~1），默认 0.45（仅 detect/obb/segment 生效）",
    )
    parser.add_argument(
        "--include-unannotated",
        action="store_true",
        help="处理未标注图片（4 个 include 开关全未指定时此项默认开启）",
    )
    parser.add_argument(
        "--include-auto",
        action="store_true",
        help="处理已被自动标注的图片（会用新模型重新生成标注）",
    )
    parser.add_argument(
        "--include-auto-corrected",
        action="store_true",
        help="处理自动标注后人工矫正过的图片（谨慎使用，会覆盖人工修改）",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="处理纯手动标注的图片（极谨慎使用，会覆盖人工标注）",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help=f"判定自动 / 矫正的时间容差（秒），默认 {DEFAULT_TOLERANCE_SECONDS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=AUTO_ANNOTATE_DEFAULT_BATCH_SIZE,
        help=f"批量推理大小（必须为正整数），默认 {AUTO_ANNOTATE_DEFAULT_BATCH_SIZE}",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        raise ValueError(f"工作目录不存在：{args.work_dir}")
    if not work_dir.is_dir():
        raise ValueError(f"工作目录不是文件夹：{args.work_dir}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise ValueError(f"模型权重文件不存在：{args.model_path}")
    if not model_path.is_file():
        raise ValueError(f"模型权重路径不是文件：{args.model_path}")
    if model_path.suffix.lower() != ".pt":
        # 仅警告，不阻止——某些场景下用户可能确实有非 .pt 但兼容的权重
        log(
            f"[警告] 模型文件后缀为 {model_path.suffix!r}，常规为 .pt，请确认无误。",
            stream=sys.stderr,
        )

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"--threshold 必须在 [0, 1] 之间，当前为 {args.threshold}")
    if not (0.0 <= args.iou <= 1.0):
        raise ValueError(f"--iou 必须在 [0, 1] 之间，当前为 {args.iou}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f"--batch-size 必须为正整数，当前为 {args.batch_size}")
    if args.tolerance_seconds is not None and args.tolerance_seconds < 0:
        raise ValueError(
            f"--tolerance-seconds 必须为非负数，当前为 {args.tolerance_seconds}"
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

    # 若 4 个 include 开关都未指定，向后兼容默认仅处理未标注图片
    include_unannotated = args.include_unannotated
    include_auto = args.include_auto
    include_auto_corrected = args.include_auto_corrected
    include_manual = args.include_manual
    if not any([include_unannotated, include_auto, include_auto_corrected, include_manual]):
        include_unannotated = True
        log(
            "[提示] 未指定任何 --include-* 开关，默认仅处理 *未标注* 图片。",
            stream=sys.stderr,
        )

    try:
        result = auto_annotate(
            work_dir=args.work_dir,
            model_path=args.model_path,
            threshold=args.threshold,
            task=args.task,
            suffix=args.suffix,
            device=args.device,
            iou=args.iou,
            include_unannotated=include_unannotated,
            include_auto=include_auto,
            include_auto_corrected=include_auto_corrected,
            include_manual=include_manual,
            tolerance_seconds=args.tolerance_seconds,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，已写入的 JSON 不会被回滚。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except ImportError as exc:
        log(
            f"[依赖缺失] {exc}\n"
            f"提示：自动标注需要安装 ultralytics（含 torch）。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 自动标注失败：{exc}", stream=sys.stderr)
        return 1

    if isinstance(result, dict):
        total = result.get("total", 0)
        annotated = result.get("annotated", 0)
        skipped = result.get("skipped", 0)
        log(f"[完成] 总计 {total} 张，标注 {annotated} 张，跳过 {skipped} 张。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
