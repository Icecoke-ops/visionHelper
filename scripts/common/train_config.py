#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置数据类。

将 ``TrainingAPI.train_model`` 的 50+ 参数封装为单一配置对象，
便于维护、测试与向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from scripts.common.config import SUPPORTED_TASKS, SUPPORTED_OPTIMIZERS
from scripts.common.sahi_config import SAHIConfig


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

    # SAHI 小目标优化配置（组合而非平铺）
    sahi: SAHIConfig = field(default_factory=SAHIConfig)

    def __post_init__(self) -> None:
        """标准化与基础校验。"""
        self.task = (self.task or "detect").lower()
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"不支持的任务类型: {self.task}，支持的: {sorted(SUPPORTED_TASKS)}"
            )
        self.optimizer = self.optimizer or "auto"
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"不支持的优化器: {self.optimizer}，支持的: {sorted(SUPPORTED_OPTIMIZERS)}"
            )

    @property
    def task_norm(self) -> str:
        return self.task

    @property
    def optimizer_norm(self) -> str:
        return self.optimizer

    def to_train_kwargs(self) -> dict:
        """转换为平铺的 SAHI 字段（已弃用，保留向后兼容）。"""
        result = asdict(self)
        result.pop("sahi", None)
        result.update(asdict(self.sahi))
        return result


__all__ = ["TrainConfig"]