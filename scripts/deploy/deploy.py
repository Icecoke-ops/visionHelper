#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python scripts/vh.py deploy export 命令实现（占位）。

当前功能尚未实现，调用时返回退出码 2。
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from scripts.common.logging import log


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py deploy export",
        description="将模型导出为部署格式（预留功能，当前尚未实现）。",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="模型权重文件路径",
    )
    parser.add_argument(
        "-F", "--format",
        type=str,
        required=True,
        help="导出格式，如 onnx",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="输出目录",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    # 预留：各参数已在 _build_parser 中设为 required，类型校验 parser 会自动处理
    pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    返回:
        退出码 2，表示功能尚未实现。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    log(
        f"deploy export 功能尚未实现（model={args.model}, format={args.format}, "
        f"output={args.output})",
        stream=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
