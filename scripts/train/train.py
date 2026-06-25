#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python scripts/vh.py train run 命令实现。

基于 Ultralytics YOLO 训练目标检测 / OBB / 分割 / 分类模型。
脚本会按 ``--task`` 自动为基础模型名追加 ``-obb/-seg/-cls`` 后缀。

用法::

    python scripts/vh.py train run -d ./dataset/data.yaml -t detect -m yolov8n -e 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.common.config import (
    IMAGES_FOLDER,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_TASKS,
    TASK_MODEL_SUFFIX,
)
from scripts.common.logging import log

__all__ = [
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_TASKS",
    "main",
    "train_model",
]


def _resolve_model_name(model: str, task: str) -> str:
    """根据任务类型补全模型权重文件名。"""
    model = model.strip()
    if not model:
        raise ValueError("模型名称不能为空")

    if model.endswith(".pt"):
        return model

    suffix = TASK_MODEL_SUFFIX.get(task, "")
    if suffix and suffix not in model:
        model = f"{model}{suffix}"
    return f"{model}.pt"


def train_model(
    dataset_yaml: str,
    task: str = "detect",
    model: str = "yolov8n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    patience: int = 100,
    resume: bool = False,
    optimizer: str = "auto",
    lr0: float = 0.01,
    workers: int = 8,
) -> str:
    """使用 Ultralytics YOLO 训练目标检测 / OBB / 分割 / 分类模型。

    返回训练结果目录路径（best.pt 所在目录），若 ultralytics 没有暴露
    ``trainer.save_dir`` 则返回空字符串。
    """
    yaml_path = Path(dataset_yaml)
    if not yaml_path.is_file():
        raise ValueError(f"数据集配置文件不存在: {dataset_yaml}")

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}"
        )

    if epochs < 1:
        raise ValueError("epochs 必须大于 0")
    if imgsz < 32:
        raise ValueError("imgsz 必须大于等于 32")
    if batch < 1:
        raise ValueError("batch 必须大于 0")
    if patience < 0:
        raise ValueError("patience 必须 >= 0")
    if workers < 0:
        raise ValueError("workers 必须 >= 0")
    if lr0 <= 0:
        raise ValueError("lr0 必须 > 0")
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"不支持的优化器: {optimizer}，仅支持 {sorted(SUPPORTED_OPTIMIZERS)}"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：pip install ultralytics"
        ) from exc

    model_name = _resolve_model_name(model, task)
    log(f"加载模型: {model_name}")
    yolo_model = YOLO(model_name)

    # classify 任务需要的 data 参数为分类数据集根目录（含 train/val 子目录）。
    if task == "classify":
        classify_root = yaml_path.parent / IMAGES_FOLDER
        if not classify_root.is_dir():
            raise ValueError(
                f"分类数据集目录不存在: {classify_root}，"
                f"请确认已使用 task=classify 导出数据集"
            )
        data_arg = str(classify_root.resolve())
    else:
        data_arg = str(yaml_path.resolve())

    train_kwargs = {
        "data": data_arg,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "resume": resume,
        "optimizer": optimizer,
        "lr0": lr0,
        "workers": workers,
    }
    if device is not None:
        train_kwargs["device"] = device
    if project is not None:
        train_kwargs["project"] = project
    if name is not None:
        train_kwargs["name"] = name

    log(f"开始训练 {task} 模型...")
    log(f"  data: {train_kwargs['data']}")
    log(f"  epochs: {epochs}, imgsz: {imgsz}, batch: {batch}")
    log(f"  optimizer: {optimizer}, lr0: {lr0}, patience: {patience}, "
        f"workers: {workers}, resume: {resume}")

    try:
        yolo_model.train(**train_kwargs)
    except Exception as exc:
        raise RuntimeError(f"训练过程出错: {exc}") from exc

    result_dir: Optional[Path] = None
    trainer = getattr(yolo_model, "trainer", None)
    if trainer is not None:
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir is not None:
            candidate = Path(save_dir)
            if candidate.is_dir():
                result_dir = candidate

    if result_dir is not None:
        best_pt = result_dir / "weights" / "best.pt"
        log(f"训练完成，结果目录: {result_dir}")
        if best_pt.is_file():
            log(f"最佳权重: {best_pt}")
        return str(result_dir)

    log("训练完成")
    return ""


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py train run",
        description=(
            "基于 Ultralytics YOLO 训练目标检测 / OBB / 分割 / 分类模型。"
            "脚本会按 --task 自动为基础模型名追加 -obb/-seg/-cls 后缀。"
        ),
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="数据集配置文件路径（YOLO data.yaml）",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolov8n",
        help="基础模型名称，例如 yolov8n/yolov8s/yolov8m/yolo11n 等，默认 yolov8n",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="训练轮数（必须为正整数），默认 100",
    )
    parser.add_argument(
        "-s", "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸（必须为正整数，建议为 32 的倍数），默认 640",
    )
    parser.add_argument(
        "-b", "--batch",
        type=int,
        default=16,
        help="批大小（正整数；Ultralytics 中传 -1 表示自动），默认 16",
    )
    parser.add_argument(
        "-D", "--device",
        type=str,
        default=None,
        help="训练设备，例如 0/cpu/0,1,2,3，默认自动选择",
    )
    parser.add_argument(
        "-P", "--project",
        type=str,
        default=None,
        help="训练结果保存的父目录（不指定时使用 Ultralytics 默认）",
    )
    parser.add_argument(
        "-n", "--name",
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
        "-r", "--resume",
        action="store_true",
        help="是否从最近一次中断的训练继续，默认 False",
    )
    parser.add_argument(
        "-O", "--optimizer",
        type=str,
        default="auto",
        choices=sorted(SUPPORTED_OPTIMIZERS),
        help="优化器，默认 auto（由 Ultralytics 自动选择）",
    )
    parser.add_argument(
        "-l", "--lr0",
        type=float,
        default=0.01,
        help="初始学习率（必须为正数），默认 0.01",
    )
    parser.add_argument(
        "-W", "--workers",
        type=int,
        default=8,
        help="DataLoader 进程数（非负整数），默认 8",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise ValueError(f"数据集路径不存在：{args.data}")
    if not dataset_path.is_file():
        raise ValueError(f"--data 必须指向一个文件：{args.data}")
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
            dataset_yaml=args.data,
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
