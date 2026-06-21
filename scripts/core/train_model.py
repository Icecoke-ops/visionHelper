#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Ultralytics YOLO 的模型训练核心实现。

支持任务类型：detect / obb / segment / classify。详细规则见外层 CLI 文档。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.config import (
    IMAGES_FOLDER,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_TASKS,
    TASK_MODEL_SUFFIX,
)
from scripts.logging_utils import log

__all__ = [
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_TASKS",
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
