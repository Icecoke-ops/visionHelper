#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python scripts/vh.py predict run 命令实现（占位）。

当前功能尚未实现，调用时返回退出码 3。
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py predict run",
        description="使用训练好的模型对新图片/视频进行预测（待开发）。",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="模型权重文件路径",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入图片或目录路径",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="detect",
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "-T", "--threshold",
        type=float,
        default=0.25,
        help="置信度阈值，默认 0.25",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出目录",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回:
        退出码 3，表示功能尚未实现。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    print(
        f"predict run 功能尚未实现（model={args.model}, input={args.input}, "
        f"task={args.task}, threshold={args.threshold}, output={args.output})",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    sys.exit(main())
