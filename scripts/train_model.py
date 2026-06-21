#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练工具 CLI 门面。

实际实现位于 :mod:`scripts.core.train_model`，本模块仅负责：

1. 命令行参数解析；
2. 调用前做 **轻量边界校验**（数据集 yaml、epochs/imgsz/batch/lr 等）；
3. 统一异常捕获并返回合适退出码；
4. 对外 re-export 公开 API 与常量，保持向后兼容。

注意：``import scripts.train_model`` 本身不会拉起 ``torch`` / ``ultralytics``，
重依赖仅在调用 :func:`train_model` 时按需加载。

用法::

    python -m scripts.train_model <dataset_yaml> [options]

例（检测任务，100 轮）::

    python -m scripts.train_model ./.dataset/data.yaml \\
        --task detect --model yolov8n --epochs 100 --imgsz 640 --batch 16

例（分类任务，``dataset_yaml`` 处传入数据集 ``images/`` 根目录）::

    python -m scripts.train_model ./.dataset/images \\
        --task classify --model yolov8n --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.core.train_model import (
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_TASKS,
    train_model,
)
from scripts.logging_utils import log

__all__ = [
    "train_model",
    "SUPPORTED_TASKS",
    "SUPPORTED_OPTIMIZERS",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.train_model",
        description=(
            "基于 Ultralytics YOLO 训练目标检测 / OBB / 分割 / 分类模型。"
            "脚本会按 --task 自动为基础模型名追加 -obb/-seg/-cls 后缀。"
        ),
    )
    parser.add_argument(
        "dataset_yaml",
        type=str,
        help=(
            "数据集路径：detect/obb/segment 任务传入 data.yaml 文件路径；"
            "classify 任务传入数据集 images/ 根目录。"
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help="基础模型名称，例如 yolov8n/yolov8s/yolov8m/yolo11n 等，默认 yolov8n",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数（必须为正整数），默认 100",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸（必须为正整数，建议为 32 的倍数），默认 640",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批大小（正整数；Ultralytics 中传 -1 表示自动），默认 16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备，例如 0/cpu/0,1,2,3，默认自动选择",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="训练结果保存的父目录（不指定时使用 Ultralytics 默认）",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="训练结果子目录名称",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="早停 patience（多少个 epoch 验证指标不提升就停止），默认 100",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否从最近一次中断的训练继续，默认 False",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=sorted(SUPPORTED_OPTIMIZERS),
        help="优化器，默认 auto（由 Ultralytics 自动选择）",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="初始学习率（必须为正数），默认 0.01",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader 进程数（非负整数），默认 8",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。

    注意：``dataset_yaml`` 在 ``classify`` 任务下表示一个目录，其它任务下表示
    一个 ``.yaml`` 文件，本函数针对两种情况分别校验。
    """
    dataset_path = Path(args.dataset_yaml)
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在：{args.dataset_yaml}")

    if args.task == "classify":
        # classify 期望传入 images/ 根目录
        if not dataset_path.is_dir():
            raise ValueError(
                f"classify 任务需要传入数据集 images/ 根目录（ImageFolder 结构），"
                f"当前路径不是目录：{args.dataset_yaml}"
            )
    else:
        if not dataset_path.is_file():
            raise ValueError(
                f"{args.task} 任务需要传入 data.yaml 文件，当前路径不是文件："
                f"{args.dataset_yaml}"
            )
        if dataset_path.suffix.lower() not in {".yaml", ".yml"}:
            log(
                f"[警告] 数据集文件后缀为 {dataset_path.suffix!r}，常规为 "
                f".yaml/.yml，请确认无误。",
                stream=sys.stderr,
            )

    if args.epochs <= 0:
        raise ValueError(f"--epochs 必须为正整数，当前为 {args.epochs}")
    if args.imgsz <= 0:
        raise ValueError(f"--imgsz 必须为正整数，当前为 {args.imgsz}")
    if args.imgsz % 32 != 0:
        log(
            f"[警告] --imgsz={args.imgsz} 不是 32 的倍数，部分 YOLO 模型可能"
            f"内部自动向下对齐，建议使用 32 的倍数（如 320/416/640/1280）。",
            stream=sys.stderr,
        )
    if args.batch == 0:
        raise ValueError("--batch 不能为 0（正整数表示固定大小，-1 表示自动）")
    if args.lr0 <= 0:
        raise ValueError(f"--lr0 必须为正数，当前为 {args.lr0}")
    if args.workers < 0:
        raise ValueError(f"--workers 必须为非负整数，当前为 {args.workers}")
    if args.patience < 0:
        raise ValueError(f"--patience 必须为非负整数，当前为 {args.patience}")


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
        save_dir = train_model(
            dataset_yaml=args.dataset_yaml,
            task=args.task,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            resume=args.resume,
            optimizer=args.optimizer,
            lr0=args.lr0,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        log(
            "[已取消] 用户中断训练，已保存的 checkpoint 仍可用 --resume 继续。",
            stream=sys.stderr,
        )
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except ImportError as exc:
        log(
            f"[依赖缺失] {exc}\n"
            f"提示：模型训练需要安装 ultralytics（含 torch）。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 训练失败：{exc}", stream=sys.stderr)
        return 1

    if save_dir:
        log(f"[完成] 训练结果保存在：{save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
