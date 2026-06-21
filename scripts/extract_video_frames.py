#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频抽帧工具 CLI 门面。

实际实现位于 :mod:`scripts.core.extract_video_frames`，本模块仅负责：

1. 解析命令行参数（``argparse``）；
2. 在调用核心实现前做 **轻量的参数边界校验**，尽早给出友好提示；
3. 统一捕获常见异常（``ValueError`` / ``FileNotFoundError`` / ``KeyboardInterrupt``）并
   以非 0 退出码返回，便于 GUI 子进程通过 returncode 与 stderr 显示结果；
4. 对外 re-export 核心函数与常量。

注意：本模块刻意 **不直接 import OpenCV / numpy 等重依赖**——这些会在
``scripts.core.extract_video_frames`` 内部按需加载，从而保证 ``import
scripts.extract_video_frames`` 本身仍是低开销的。

用法::

    python -m scripts.extract_video_frames <input_video> <output_dir> [options]

例::

    python -m scripts.extract_video_frames input.mp4 ./out \\
        --frame-step 5 --ext jpg --quality 95 --prefix frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.config import SUPPORTED_SEEK_MODES
from scripts.core.extract_video_frames import (
    SUPPORTED_EXTENSIONS,
    extract_video_frames,
)
from scripts.logging_utils import log

__all__ = [
    "extract_video_frames",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_SEEK_MODES",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（独立函数便于单元测试）。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.extract_video_frames",
        description="视频抽帧工具：按指定间隔抽取视频画面并保存为图片。",
    )
    parser.add_argument("input_video", type=str, help="输入视频文件路径")
    parser.add_argument("output_dir", type=str, help="输出图片保存目录（不存在会自动创建）")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="抽帧间隔（必须为正整数），默认 1（逐帧抽取）。",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        help=(
            "输出图片格式，默认 jpg。"
            f"支持：{', '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))}"
        ),
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="输出图片质量（取值范围 1-100），默认 100（原始画质）。",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="输出文件名前缀，默认 frame。",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="开始抽取的时间（秒或 HH:MM:SS/MM:SS/SS 格式），默认从视频开头。",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="结束抽取的时间（秒或 HH:MM:SS/MM:SS/SS 格式），默认抽到视频结尾。",
    )
    parser.add_argument(
        "--seek-mode",
        type=str,
        default="decode_all",
        choices=sorted(SUPPORTED_SEEK_MODES),
        help=(
            "跳帧策略：decode_all=顺序解码每一帧（默认，安全且通用）；"
            "seek=主动跳到下一目标帧（大 frame_step 下更快，对部分编码可能略有偏差）。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中已存在的同名文件（默认 False，重名自动追加 _1/_2 后缀）。",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。

    任何不合法情况都以 :class:`ValueError` 抛出，由 :func:`main` 统一处理。
    """
    # 输入视频必须存在且为文件
    input_path = Path(args.input_video)
    if not input_path.exists():
        raise ValueError(f"输入视频不存在：{args.input_video}")
    if not input_path.is_file():
        raise ValueError(f"输入视频不是文件：{args.input_video}")

    # 输出目录不能是已存在的同名文件
    output_path = Path(args.output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"输出路径已存在但不是目录：{args.output_dir}")

    if args.frame_step is not None and args.frame_step < 1:
        raise ValueError(f"--frame-step 必须为正整数，当前为 {args.frame_step}")

    if args.quality is not None and not (1 <= args.quality <= 100):
        raise ValueError(f"--quality 必须在 1~100 之间，当前为 {args.quality}")

    # 扩展名校验（兼容前置点号写法）
    ext_normalized = "." + args.ext.lower().lstrip(".")
    if ext_normalized not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"不支持的图片格式：{args.ext!r}，请使用以下之一：{supported}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    参数:
        argv: 命令行参数列表（不含程序名），默认从 ``sys.argv[1:]`` 读取，
            主要用于单元测试。

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
        saved = extract_video_frames(
            input_video=args.input_video,
            output_dir=args.output_dir,
            frame_step=args.frame_step,
            image_extension=args.ext,
            quality=args.quality,
            prefix=args.prefix,
            start_time=args.start_time,
            end_time=args.end_time,
            seek_mode=args.seek_mode,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，部分帧可能未写入完成。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        # core 内部已做的参数/路径校验在此统一汇报为友好提示
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 视频抽帧失败：{exc}", stream=sys.stderr)
        return 1

    log(f"[完成] 共保存 {len(saved)} 张图片到 {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
