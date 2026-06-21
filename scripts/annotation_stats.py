#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据标注统计 CLI 门面。

实际实现位于 :mod:`scripts.core.annotation_stats`，本模块仅负责：

1. 命令行参数解析；
2. 调用前做 **轻量边界校验**（路径存在性等）；
3. 调用核心实现并以人类可读 + 机器可读两种形式输出；
4. 统一异常捕获，返回合适退出码；
5. 对外 re-export 公开函数与常量，保持向后兼容。

CLI 在 ``stdout`` 中会以
``===VH_STATS_BEGIN===`` / ``===VH_STATS_END===`` 标记包裹一个 JSON 块，
内部包含 ``stats`` 与 ``label_stats`` 两个字段，方便 GUI 等上层进程稳定解析。

用法::

    python -m scripts.annotation_stats /path/to/images
    python -m scripts.annotation_stats /path/to/images --label-stats
    python -m scripts.annotation_stats /path/to/images --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scripts.config import (
    STATS_RESULT_BEGIN_MARKER,
    STATS_RESULT_END_MARKER,
)
from scripts.core.annotation_stats import (
    collect_annotation_label_stats,
    collect_annotation_stats,
    emit_machine_block,
    parse_machine_block,
    print_label_stats_human,
    print_stats_human,
)
from scripts.logging_utils import log

# 向后兼容的别名（旧代码可能从本模块直接 import）
RESULT_BEGIN_MARKER = STATS_RESULT_BEGIN_MARKER
RESULT_END_MARKER = STATS_RESULT_END_MARKER

__all__ = [
    "collect_annotation_stats",
    "collect_annotation_label_stats",
    "parse_machine_block",
    "emit_machine_block",
    "print_stats_human",
    "print_label_stats_human",
    "RESULT_BEGIN_MARKER",
    "RESULT_END_MARKER",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.annotation_stats",
        description=(
            "统计目录下的图片与 X-AnyLabeling JSON 标注情况，"
            "支持整体统计与按标签统计。"
        ),
    )
    parser.add_argument(
        "folder",
        type=str,
        help="待统计的图片目录路径。",
    )
    parser.add_argument(
        "--label-stats",
        action="store_true",
        help="同时输出按标签的实例数量统计。",
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help=(
            "仅输出供机器解析的 JSON 块（不打印人类可读日志）。"
            "JSON 仍以 ===VH_STATS_BEGIN=== / ===VH_STATS_END=== 包裹。"
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
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2

    # 整体统计
    try:
        stats = collect_annotation_stats(args.folder)
    except ValueError as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log("[已取消] 用户中断。", stream=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 整体统计失败: {exc}", stream=sys.stderr)
        return 1

    # 按标签统计（可选）
    label_stats: List[Dict[str, int]] = []
    if args.label_stats:
        try:
            label_stats = collect_annotation_label_stats(args.folder)
        except ValueError as exc:
            log(f"[错误] {exc}", stream=sys.stderr)
            return 2
        except KeyboardInterrupt:
            log("[已取消] 用户中断。", stream=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001
            log(f"[错误] 标签统计失败: {exc}", stream=sys.stderr)
            return 1

    # 人类可读输出（机器模式时省略）
    if not args.json_only:
        print_stats_human(stats)
        if args.label_stats:
            print_label_stats_human(label_stats)

    # 机器可读 JSON 块——始终输出，便于 GUI / 测试稳定解析
    emit_machine_block({"stats": stats, "label_stats": label_stats})
    return 0


if __name__ == "__main__":
    sys.exit(main())
