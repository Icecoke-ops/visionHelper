#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python scripts/vh.py 统一命令行入口。

仅做子命令路由，不实现具体业务逻辑。所有子命令的参数解析、参数校验、
help 信息、业务实现均放在各自子包模块中。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    """构造顶层命令解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py",
        description="visionHelper：视觉模型训练全流程工具。",
        add_help=False,
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="帮助信息",
    )
    parser.add_argument(
        "-V", "--version",
        action="store_true",
        help="显示版本号",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用进度输出",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志文件路径",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="子命令")

    # images
    images = subparsers.add_parser("images", help="图片资源管理", add_help=False)
    images_sub = images.add_subparsers(dest="action", help="images 动作")
    images_sub.add_parser("import", help="从视频抽帧导入图片", add_help=False)
    images_sub.add_parser("dedup", help="图片去重", add_help=False)

    # datasets
    datasets = subparsers.add_parser("datasets", help="数据集制作", add_help=False)
    datasets_sub = datasets.add_subparsers(dest="action", help="datasets 动作")
    datasets_sub.add_parser("stats", help="标注统计", add_help=False)
    datasets_sub.add_parser("auto", help="自动标注", add_help=False)
    datasets_sub.add_parser("clear", help="清除标注", add_help=False)
    datasets_sub.add_parser("export", help="导出 YOLO 数据集", add_help=False)

    # train / predict / deploy
    train = subparsers.add_parser("train", help="模型训练", add_help=False)
    train_sub = train.add_subparsers(dest="action", help="train 动作")
    train_sub.add_parser("run", help="启动训练", add_help=False)

    predict = subparsers.add_parser("predict", help="模型预测/推理", add_help=False)
    predict_sub = predict.add_subparsers(dest="action", help="predict 动作")
    predict_sub.add_parser("run", help="启动预测", add_help=False)

    deploy = subparsers.add_parser("deploy", help="模型部署", add_help=False)
    deploy_sub = deploy.add_subparsers(dest="action", help="deploy 动作")
    deploy_sub.add_parser("export", help="导出部署模型", add_help=False)

    return parser


def _print_top_level_help(parser: argparse.ArgumentParser) -> int:
    parser.print_help()
    return 0


def _route(subcommand: Optional[str], action: Optional[str], argv: List[str]) -> int:
    """根据子命令与动作分发到对应入口函数。"""
    if subcommand == "images":
        if action == "import":
            from scripts.images.import_ import main as _main
        elif action == "dedup":
            from scripts.images.dedup import main as _main
        else:
            raise ValueError(f"unknown action: images {action}")
    elif subcommand == "datasets":
        if action == "stats":
            from scripts.datasets.stats import main as _main
        elif action == "auto":
            from scripts.datasets.auto import main as _main
        elif action == "clear":
            from scripts.datasets.clear import main as _main
        elif action == "export":
            from scripts.datasets.export import main as _main
        else:
            raise ValueError(f"unknown action: datasets {action}")
    elif subcommand == "train":
        if action == "run":
            from scripts.train.train import main as _main
        else:
            raise ValueError(f"unknown action: train {action}")
    elif subcommand == "predict":
        if action == "run":
            from scripts.predict.predict import main as _main
        else:
            raise ValueError(f"unknown action: predict {action}")
    elif subcommand == "deploy":
        if action == "export":
            from scripts.deploy.deploy import main as _main
        else:
            raise ValueError(f"unknown action: deploy {action}")
    else:
        raise ValueError(f"unknown subcommand: {subcommand}")
    return _main(argv)


def _apply_global_options(args: argparse.Namespace) -> None:
    """应用全局选项副作用。"""
    if args.no_progress:
        os.environ["VH_NO_PROGRESS"] = "1"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    返回:
        进程退出码：0=成功；2=参数/路由错误；其它由子命令决定。
    """
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.version:
        from scripts.common.config import VERSION
        print(VERSION)
        return 0

    if args.help or args.subcommand is None:
        return _print_top_level_help(parser)

    if args.action is None:
        # 子命令缺少动作，将剩余参数透传过去由其自己输出帮助或报错。
        # 但为了行为一致，这里直接提示并返回 2。
        print(
            f"错误：子命令 {args.subcommand} 缺少动作",
            file=sys.stderr,
        )
        return 2

    _apply_global_options(args)

    try:
        return _route(args.subcommand, args.action, remaining)
    except ValueError as exc:
        print(f"[路由错误] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
