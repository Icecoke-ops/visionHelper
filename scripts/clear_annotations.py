#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清除标签工具 CLI 门面。

实际实现位于 :mod:`scripts.core.clear_annotations`，本模块仅负责：

1. 命令行参数解析；
2. 调用前做 **轻量边界校验**（目录存在、至少指定一个 ``--include-*`` 开关等）；
3. 统一异常捕获并返回合适退出码；
4. 对外 re-export 公开 API，保持向后兼容。

**安全提示**：本工具会 *直接删除* 目录顶层匹配条件的 JSON 标注文件，请在执行前
确认已备份重要数据。出于谨慎考虑：当未指定任何 ``--include-*`` 开关时，
工具会拒绝执行并打印帮助说明，避免误删。

用法::

    python -m scripts.clear_annotations <folder> [--include-auto] \\
        [--include-auto-corrected] [--include-manual] \\
        [--tolerance-seconds 2.0]

例（仅清除自动标注 JSON）::

    python -m scripts.clear_annotations ./images --include-auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.config import DEFAULT_TOLERANCE_SECONDS
from scripts.core.clear_annotations import clear_annotations
from scripts.logging_utils import log

__all__ = ["clear_annotations", "main"]


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.clear_annotations",
        description=(
            "按标注类型批量清除 X-AnyLabeling JSON 标注文件。"
            "必须显式指定至少一个 --include-* 开关，避免误删。"
        ),
    )
    parser.add_argument("folder", type=str, help="待清理的图片目录")
    parser.add_argument(
        "--include-auto",
        action="store_true",
        help="删除自动标注的 JSON",
    )
    parser.add_argument(
        "--include-auto-corrected",
        action="store_true",
        help="删除自动标注后人工矫正的 JSON",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="删除手动标注的 JSON",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help=f"判定自动 / 矫正的时间容差（秒），默认 {DEFAULT_TOLERANCE_SECONDS}",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    folder = Path(args.folder)
    if not folder.exists():
        raise ValueError(f"目录不存在：{args.folder}")
    if not folder.is_dir():
        raise ValueError(f"路径不是目录：{args.folder}")

    if not any([args.include_auto, args.include_auto_corrected, args.include_manual]):
        raise ValueError(
            "必须至少指定一个 --include-auto / --include-auto-corrected / "
            "--include-manual，以明确删除范围（避免误删）。"
        )

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

    try:
        result = clear_annotations(
            folder=args.folder,
            include_auto=args.include_auto,
            include_auto_corrected=args.include_auto_corrected,
            include_manual=args.include_manual,
            tolerance_seconds=args.tolerance_seconds,
        )
    except KeyboardInterrupt:
        log(
            "[已取消] 用户中断，部分文件可能已被删除（不可回滚）。",
            stream=sys.stderr,
        )
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 清除标注失败：{exc}", stream=sys.stderr)
        return 1

    if isinstance(result, dict):
        scanned = result.get("scanned", 0)
        deleted = result.get("deleted", 0)
        failed = result.get("failed", []) or []
        log(f"[完成] 扫描 {scanned} 个 JSON，删除 {deleted} 个。")
        if failed:
            log(f"[警告] {len(failed)} 个文件删除失败，请检查权限或被占用。",
                stream=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
