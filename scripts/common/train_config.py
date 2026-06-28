#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置数据类。

将 ``TrainingAPI.train_model`` 的 50+ 参数封装为单一配置对象，
便于维护、测试与向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """YOLO 模型训练配置。"""

    # 必填参数
    dataset_yaml: str
    project: str
    name: str
    task: str = "detect"
    model: str = "yolov8n"

    # 基础训练参数
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: Optional[str] = None
    patience: int = 100
    resume: bool = False

    # 优化器参数
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # 损失权重
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    label_smoothing: float = 0.0
    close_mosaic: int = 10

    # 训练加速
    amp: bool = True
    freeze: Optional[int] = None
    workers: int = 8

    # 数据增强参数
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    # SAHI 小目标优化参数
    sahi_enabled: bool = False
    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    sahi_overlap_height_ratio: float = 0.25
    sahi_overlap_width_ratio: float = 0.25

    # 内部：验证后的标准化值（不直接设置）
    _task_norm: str = field(default="", init=False, repr=False)
    _optimizer_norm: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        """标准化与基础校验。"""
        self._task_norm = (self.task or "detect").lower()
        self._optimizer_norm = self.optimizer or "auto"

    @property
    def task_norm(self) -> str:
        return self._task_norm

    @property
    def optimizer_norm(self) -> str:
        return self._optimizer_norm

    def to_train_kwargs(self) -> dict:
        """转换为 ``scripts.train.train.train_model`` 接受的参数字典。"""
        return {
            "dataset_yaml": self.dataset_yaml,
            "task": self._task_norm,
            "model": self.model,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "project": self.project,
            "name": self.name,
            "patience": self.patience,
            "resume": self.resume,
            "optimizer": self._optimizer_norm,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "box": self.box,
            "cls": self.cls,
            "dfl": self.dfl,
            "label_smoothing": self.label_smoothing,
            "close_mosaic": self.close_mosaic,
            "amp": self.amp,
            "freeze": self.freeze,
            "workers": self.workers,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "sahi_enabled": self.sahi_enabled,
            "sahi_slice_height": self.sahi_slice_height,
            "sahi_slice_width": self.sahi_slice_width,
            "sahi_overlap_height_ratio": self.sahi_overlap_height_ratio,
            "sahi_overlap_width_ratio": self.sahi_overlap_width_ratio,
        }


__all__ = ["TrainConfig"]