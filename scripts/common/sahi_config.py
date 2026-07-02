#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAHI (Slicing Aided Hyper Inference) 配置模块。

提供小目标检测优化的切片推理配置参数。
SAHI 主要用于推理阶段，通过将大图像切成小块进行检测，然后合并结果，
从而提高小目标检测性能。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SAHIConfig:
    """SAHI 切片推理配置。

    属性:
        sahi_enabled: 是否启用 SAHI 切片推理。
        sahi_slice_height: 切片高度（像素）。
        sahi_slice_width: 切片宽度（像素）。
        sahi_overlap_height_ratio: 垂直重叠比例（0.0-1.0）。
        sahi_overlap_width_ratio: 水平重叠比例（0.0-1.0）。
        sahi_postprocess_type: 后处理类型（'NMS' 或 'WBF'）。
        sahi_postprocess_match_threshold: 后处理匹配阈值。
        sahi_postprocess_match_class_agnostic: 是否进行类别无关的匹配。
        sahi_postprocess_individually: 是否单独处理每个切片。
        sahi_postprocess_verbose: 是否输出详细信息。
        sahi_min_image_size: 最小图像尺寸（像素），小于该尺寸的图像不进行切片。
        sahi_auto_slice_resolution: 是否自动计算切片分辨率。
    """

    sahi_enabled: bool = False
    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    sahi_overlap_height_ratio: float = 0.25
    sahi_overlap_width_ratio: float = 0.25
    sahi_postprocess_type: str = "NMS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_postprocess_match_class_agnostic: bool = False
    sahi_postprocess_individually: bool = False
    sahi_postprocess_verbose: bool = False
    sahi_min_image_size: int = 640
    sahi_auto_slice_resolution: bool = True

    def validate(self) -> None:
        """验证配置参数的有效性。"""
        if self.sahi_slice_height < 32:
            raise ValueError(f"sahi_slice_height 必须 >= 32，当前值: {self.sahi_slice_height}")
        if self.sahi_slice_width < 32:
            raise ValueError(f"sahi_slice_width 必须 >= 32，当前值: {self.sahi_slice_width}")
        if not (0.0 <= self.sahi_overlap_height_ratio <= 1.0):
            raise ValueError(
                f"sahi_overlap_height_ratio 必须在 [0, 1] 范围内，当前值: {self.sahi_overlap_height_ratio}"
            )
        if not (0.0 <= self.sahi_overlap_width_ratio <= 1.0):
            raise ValueError(
                f"sahi_overlap_width_ratio 必须在 [0, 1] 范围内，当前值: {self.sahi_overlap_width_ratio}"
            )
        if self.sahi_postprocess_type not in ("NMS", "WBF"):
            raise ValueError(
                f"sahi_postprocess_type 必须为 'NMS' 或 'WBF'，当前值: {self.sahi_postprocess_type}"
            )
        if not (0.0 <= self.sahi_postprocess_match_threshold <= 1.0):
            raise ValueError(
                f"sahi_postprocess_match_threshold 必须在 [0, 1] 范围内，"
                f"当前值: {self.sahi_postprocess_match_threshold}"
            )
        if self.sahi_min_image_size < 32:
            raise ValueError(
                f"sahi_min_image_size 必须 >= 32，当前值: {self.sahi_min_image_size}"
            )

    def to_dict(self) -> dict:
        """将配置转换为字典格式。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SAHIConfig":
        """从字典创建配置实例。"""
        instance = cls(**data)
        instance.validate()
        return instance


def get_default_sahi_config() -> SAHIConfig:
    """获取默认 SAHI 配置。"""
    return SAHIConfig()


def create_sahi_config_for_training(
    imgsz: int = 640,
    enable_sahi: bool = False,
    slice_size: Optional[int] = None,
    overlap_ratio: float = 0.25,
) -> SAHIConfig:
    """为训练创建 SAHI 配置。

    参数:
        imgsz: 训练图像尺寸，用于自动计算切片大小。
        enable_sahi: 是否启用 SAHI。
        slice_size: 切片大小（像素），None 时自动计算。
        overlap_ratio: 重叠比例。

    返回:
        SAHIConfig 实例。
    """
    if slice_size is None:
        slice_size = min(imgsz // 2, 512)
        slice_size = max(slice_size, 128)

    return SAHIConfig(
        sahi_enabled=enable_sahi,
        sahi_slice_height=slice_size,
        sahi_slice_width=slice_size,
        sahi_overlap_height_ratio=overlap_ratio,
        sahi_overlap_width_ratio=overlap_ratio,
    )
