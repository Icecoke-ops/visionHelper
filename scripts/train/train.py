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
import json
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
from scripts.common.sahi_config import SAHIConfig

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
    project: str,
    name: str,
    task: str = "detect",
    model: str = "yolov8n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    patience: int = 100,
    resume: bool = False,
    optimizer: str = "auto",
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    label_smoothing: float = 0.0,
    close_mosaic: int = 10,
    amp: bool = True,
    freeze: Optional[int] = None,
    workers: int = 8,
    # 数据增强参数
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    # SAHI 小目标优化参数
    sahi_enabled: bool = False,
    sahi_slice_height: int = 512,
    sahi_slice_width: int = 512,
    sahi_overlap_height_ratio: float = 0.25,
    sahi_overlap_width_ratio: float = 0.25,
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
    if lrf <= 0 or lrf > 1:
        raise ValueError("lrf 必须在 (0, 1] 范围内")
    if momentum < 0 or momentum > 1:
        raise ValueError("momentum 必须在 [0, 1] 范围内")
    if weight_decay < 0:
        raise ValueError("weight_decay 必须 >= 0")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs 必须 >= 0")
    if warmup_momentum < 0 or warmup_momentum > 1:
        raise ValueError("warmup_momentum 必须在 [0, 1] 范围内")
    if warmup_bias_lr < 0:
        raise ValueError("warmup_bias_lr 必须 >= 0")
    if box <= 0:
        raise ValueError("box 必须 > 0")
    if cls <= 0:
        raise ValueError("cls 必须 > 0")
    if dfl <= 0:
        raise ValueError("dfl 必须 > 0")
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError("label_smoothing 必须在 [0, 1) 范围内")
    if close_mosaic < 0:
        raise ValueError("close_mosaic 必须 >= 0")
    if freeze is not None and freeze < 0:
        raise ValueError("freeze 必须 >= 0")
    if hsv_h < 0 or hsv_h > 1:
        raise ValueError("hsv_h 必须在 [0, 1] 范围内")
    if hsv_s < 0 or hsv_s > 1:
        raise ValueError("hsv_s 必须在 [0, 1] 范围内")
    if hsv_v < 0 or hsv_v > 1:
        raise ValueError("hsv_v 必须在 [0, 1] 范围内")
    if degrees < 0:
        raise ValueError("degrees 必须 >= 0")
    if translate < 0 or translate > 1:
        raise ValueError("translate 必须在 [0, 1] 范围内")
    if scale < 0:
        raise ValueError("scale 必须 >= 0")
    if shear < 0:
        raise ValueError("shear 必须 >= 0")
    if perspective < 0 or perspective > 0.001:
        raise ValueError("perspective 必须在 [0, 0.001] 范围内")
    if flipud < 0 or flipud > 1:
        raise ValueError("flipud 必须在 [0, 1] 范围内")
    if fliplr < 0 or fliplr > 1:
        raise ValueError("fliplr 必须在 [0, 1] 范围内")
    if mosaic < 0 or mosaic > 1:
        raise ValueError("mosaic 必须在 [0, 1] 范围内")
    if mixup < 0 or mixup > 1:
        raise ValueError("mixup 必须在 [0, 1] 范围内")
    if copy_paste < 0 or copy_paste > 1:
        raise ValueError("copy_paste 必须在 [0, 1] 范围内")
    # SAHI 参数验证
    if sahi_enabled:
        if sahi_slice_height < 32:
            raise ValueError(f"sahi_slice_height 必须 >= 32，当前值: {sahi_slice_height}")
        if sahi_slice_width < 32:
            raise ValueError(f"sahi_slice_width 必须 >= 32，当前值: {sahi_slice_width}")
        if sahi_overlap_height_ratio < 0 or sahi_overlap_height_ratio > 1:
            raise ValueError(
                f"sahi_overlap_height_ratio 必须在 [0, 1] 范围内，"
                f"当前值: {sahi_overlap_height_ratio}"
            )
        if sahi_overlap_width_ratio < 0 or sahi_overlap_width_ratio > 1:
            raise ValueError(
                f"sahi_overlap_width_ratio 必须在 [0, 1] 范围内，"
                f"当前值: {sahi_overlap_width_ratio}"
            )
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
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
        "box": box,
        "cls": cls,
        "dfl": dfl,
        "label_smoothing": label_smoothing,
        "close_mosaic": close_mosaic,
        "amp": amp,
        "workers": workers,
        # 数据增强参数
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "flipud": flipud,
        "fliplr": fliplr,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
    }
    if device is not None:
        train_kwargs["device"] = device
    train_kwargs["project"] = project
    train_kwargs["name"] = name
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    log(f"开始训练 {task} 模型...")
    log(f"  data: {train_kwargs['data']}")
    log(f"  epochs: {epochs}, imgsz: {imgsz}, batch: {batch}")
    log(f"  optimizer: {optimizer}, lr0: {lr0}, lrf: {lrf}, momentum: {momentum}")
    log(f"  weight_decay: {weight_decay}, warmup_epochs: {warmup_epochs}")
    log(f"  patience: {patience}, workers: {workers}, resume: {resume}")
    log(f"  box: {box}, cls: {cls}, dfl: {dfl}, label_smoothing: {label_smoothing}")
    log(f"  augment: mosaic={mosaic}, mixup={mixup}, flipud={flipud}, fliplr={fliplr}")
    if sahi_enabled:
        log(f"  SAHI 小目标优化: 启用, 切片大小={sahi_slice_height}x{sahi_slice_width}")
        log(f"    重叠比例: height={sahi_overlap_height_ratio}, width={sahi_overlap_width_ratio}")

    sahi_config: Optional[SAHIConfig] = None
    if sahi_enabled:
        sahi_config = SAHIConfig(
            enabled=True,
            slice_height=sahi_slice_height,
            slice_width=sahi_slice_width,
            overlap_height_ratio=sahi_overlap_height_ratio,
            overlap_width_ratio=sahi_overlap_width_ratio,
        )
        sahi_config.validate()

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
        if sahi_config is not None:
            config_path = result_dir / "sahi_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(sahi_config.to_dict(), f, ensure_ascii=False, indent=2)
            log(f"SAHI 配置: {config_path}")
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
        required=True,
        help="训练结果保存的父目录",
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
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
        "--lrf",
        type=float,
        default=0.01,
        help="最终学习率因子（lr0 * lrf），必须在 (0, 1] 范围内，默认 0.01",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD 动量/Adam beta1，必须在 [0, 1] 范围内，默认 0.937",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="L2 正则化系数（权重衰减），必须 >= 0，默认 0.0005",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="预热训练轮数（可以为小数），必须 >= 0，默认 3.0",
    )
    parser.add_argument(
        "--warmup-momentum",
        type=float,
        default=0.8,
        help="预热阶段初始动量，必须在 [0, 1] 范围内，默认 0.8",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=0.1,
        help="预热阶段偏置学习率，必须 >= 0，默认 0.1",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=7.5,
        help="边界框损失权重，必须 > 0，默认 7.5",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=0.5,
        help="分类损失权重，必须 > 0，默认 0.5",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=1.5,
        help="分布焦点损失权重，必须 > 0，默认 1.5",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="标签平滑系数，必须在 [0, 1) 范围内，默认 0.0",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="最后 N 个 epoch 关闭马赛克增强，必须 >= 0，默认 10",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        default=True,
        help="禁用自动混合精度训练（默认启用 AMP）",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="冻结模型前 N 层（用于迁移学习），默认不冻结",
    )
    parser.add_argument(
        "-W", "--workers",
        type=int,
        default=8,
        help="DataLoader 进程数（非负整数），默认 8",
    )
    # 数据增强参数
    aug_group = parser.add_argument_group("数据增强参数")
    aug_group.add_argument(
        "--hsv-h",
        type=float,
        default=0.015,
        help="HSV 色调增强系数，必须在 [0, 1] 范围内，默认 0.015",
    )
    aug_group.add_argument(
        "--hsv-s",
        type=float,
        default=0.7,
        help="HSV 饱和度增强系数，必须在 [0, 1] 范围内，默认 0.7",
    )
    aug_group.add_argument(
        "--hsv-v",
        type=float,
        default=0.4,
        help="HSV 明度增强系数，必须在 [0, 1] 范围内，默认 0.4",
    )
    aug_group.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="图像旋转角度（+/- 度），必须 >= 0，默认 0.0",
    )
    aug_group.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="图像平移系数（+/- 分数），必须在 [0, 1] 范围内，默认 0.1",
    )
    aug_group.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="图像缩放系数（+/- 增益），必须 >= 0，默认 0.5",
    )
    aug_group.add_argument(
        "--shear",
        type=float,
        default=0.0,
        help="图像剪切角度（+/- 度），必须 >= 0，默认 0.0",
    )
    aug_group.add_argument(
        "--perspective",
        type=float,
        default=0.0,
        help="图像透视变换系数，必须在 [0, 0.001] 范围内，默认 0.0",
    )
    aug_group.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="上下翻转概率，必须在 [0, 1] 范围内，默认 0.0",
    )
    aug_group.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="左右翻转概率，必须在 [0, 1] 范围内，默认 0.5",
    )
    aug_group.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="马赛克增强概率，必须在 [0, 1] 范围内，默认 1.0",
    )
    aug_group.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="MixUp 增强概率，必须在 [0, 1] 范围内，默认 0.0",
    )
    aug_group.add_argument(
        "--copy-paste",
        type=float,
        default=0.0,
        help="复制粘贴增强概率（仅分割任务），必须在 [0, 1] 范围内，默认 0.0",
    )
    # SAHI 小目标优化参数
    sahi_group = parser.add_argument_group("SAHI 小目标优化参数")
    sahi_group.add_argument(
        "--sahi-enabled",
        action="store_true",
        default=False,
        help="启用 SAHI 切片推理优化（主要用于推理阶段的小目标检测）",
    )
    sahi_group.add_argument(
        "--sahi-slice-height",
        type=int,
        default=512,
        help="SAHI 切片高度（像素），必须 >= 32，默认 512",
    )
    sahi_group.add_argument(
        "--sahi-slice-width",
        type=int,
        default=512,
        help="SAHI 切片宽度（像素），必须 >= 32，默认 512",
    )
    sahi_group.add_argument(
        "--sahi-overlap-height-ratio",
        type=float,
        default=0.25,
        help="SAHI 垂直重叠比例，必须在 [0, 1] 范围内，默认 0.25",
    )
    sahi_group.add_argument(
        "--sahi-overlap-width-ratio",
        type=float,
        default=0.25,
        help="SAHI 水平重叠比例，必须在 [0, 1] 范围内，默认 0.25",
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
    if args.lrf <= 0 or args.lrf > 1:
        raise ValueError(f"--lrf 必须在 (0, 1] 范围内，当前为 {args.lrf}")
    if args.momentum < 0 or args.momentum > 1:
        raise ValueError(f"--momentum 必须在 [0, 1] 范围内，当前为 {args.momentum}")
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay 必须 >= 0，当前为 {args.weight_decay}")
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup-epochs 必须 >= 0，当前为 {args.warmup_epochs}")
    if args.warmup_momentum < 0 or args.warmup_momentum > 1:
        raise ValueError(f"--warmup-momentum 必须在 [0, 1] 范围内，当前为 {args.warmup_momentum}")
    if args.warmup_bias_lr < 0:
        raise ValueError(f"--warmup-bias-lr 必须 >= 0，当前为 {args.warmup_bias_lr}")
    if args.box <= 0:
        raise ValueError(f"--box 必须 > 0，当前为 {args.box}")
    if args.cls <= 0:
        raise ValueError(f"--cls 必须 > 0，当前为 {args.cls}")
    if args.dfl <= 0:
        raise ValueError(f"--dfl 必须 > 0，当前为 {args.dfl}")
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        raise ValueError(f"--label-smoothing 必须在 [0, 1) 范围内，当前为 {args.label_smoothing}")
    if args.close_mosaic < 0:
        raise ValueError(f"--close-mosaic 必须 >= 0，当前为 {args.close_mosaic}")
    if args.freeze is not None and args.freeze < 0:
        raise ValueError(f"--freeze 必须 >= 0，当前为 {args.freeze}")
    if args.workers < 0:
        raise ValueError(f"--workers 必须为非负整数，当前为 {args.workers}")
    if args.patience < 0:
        raise ValueError(f"--patience 必须为非负整数，当前为 {args.patience}")
    # 数据增强参数验证
    if args.hsv_h < 0 or args.hsv_h > 1:
        raise ValueError(f"--hsv-h 必须在 [0, 1] 范围内，当前为 {args.hsv_h}")
    if args.hsv_s < 0 or args.hsv_s > 1:
        raise ValueError(f"--hsv-s 必须在 [0, 1] 范围内，当前为 {args.hsv_s}")
    if args.hsv_v < 0 or args.hsv_v > 1:
        raise ValueError(f"--hsv-v 必须在 [0, 1] 范围内，当前为 {args.hsv_v}")
    if args.degrees < 0:
        raise ValueError(f"--degrees 必须 >= 0，当前为 {args.degrees}")
    if args.translate < 0 or args.translate > 1:
        raise ValueError(f"--translate 必须在 [0, 1] 范围内，当前为 {args.translate}")
    if args.scale < 0:
        raise ValueError(f"--scale 必须 >= 0，当前为 {args.scale}")
    if args.shear < 0:
        raise ValueError(f"--shear 必须 >= 0，当前为 {args.shear}")
    if args.perspective < 0 or args.perspective > 0.001:
        raise ValueError(f"--perspective 必须在 [0, 0.001] 范围内，当前为 {args.perspective}")
    if args.flipud < 0 or args.flipud > 1:
        raise ValueError(f"--flipud 必须在 [0, 1] 范围内，当前为 {args.flipud}")
    if args.fliplr < 0 or args.fliplr > 1:
        raise ValueError(f"--fliplr 必须在 [0, 1] 范围内，当前为 {args.fliplr}")
    if args.mosaic < 0 or args.mosaic > 1:
        raise ValueError(f"--mosaic 必须在 [0, 1] 范围内，当前为 {args.mosaic}")
    if args.mixup < 0 or args.mixup > 1:
        raise ValueError(f"--mixup 必须在 [0, 1] 范围内，当前为 {args.mixup}")
    if args.copy_paste < 0 or args.copy_paste > 1:
        raise ValueError(f"--copy-paste 必须在 [0, 1] 范围内，当前为 {args.copy_paste}")
    # SAHI 参数验证
    if args.sahi_enabled:
        if args.sahi_slice_height < 32:
            raise ValueError(
                f"--sahi-slice-height 必须 >= 32，当前为 {args.sahi_slice_height}"
            )
        if args.sahi_slice_width < 32:
            raise ValueError(
                f"--sahi-slice-width 必须 >= 32，当前为 {args.sahi_slice_width}"
            )
        if args.sahi_overlap_height_ratio < 0 or args.sahi_overlap_height_ratio > 1:
            raise ValueError(
                f"--sahi-overlap-height-ratio 必须在 [0, 1] 范围内，"
                f"当前为 {args.sahi_overlap_height_ratio}"
            )
        if args.sahi_overlap_width_ratio < 0 or args.sahi_overlap_width_ratio > 1:
            raise ValueError(
                f"--sahi-overlap-width-ratio 必须在 [0, 1] 范围内，"
                f"当前为 {args.sahi_overlap_width_ratio}"
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
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            box=args.box,
            cls=args.cls,
            dfl=args.dfl,
            label_smoothing=args.label_smoothing,
            close_mosaic=args.close_mosaic,
            amp=args.amp,
            freeze=args.freeze,
            workers=args.workers,
            # 数据增强参数
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            # SAHI 小目标优化参数
            sahi_enabled=args.sahi_enabled,
            sahi_slice_height=args.sahi_slice_height,
            sahi_slice_width=args.sahi_slice_width,
            sahi_overlap_height_ratio=args.sahi_overlap_height_ratio,
            sahi_overlap_width_ratio=args.sahi_overlap_width_ratio,
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
