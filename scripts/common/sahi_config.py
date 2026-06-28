#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAHI (Slicing Aided Hyper Inference) 配置模块。

提供小目标检测优化的切片推理配置参数。
SAHI 主要用于推理阶段，通过将大图像切成小块进行检测，然后合并结果，
从而提高小目标检测性能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SAHIConfig:
    """SAHI 切片推理配置。

    属性:
        enabled: 是否启用 SAHI 切片推理。
        slice_height: 切片高度（像素）。
        slice_width: 切片宽度（像素）。
        overlap_height_ratio: 垂直重叠比例（0.0-1.0）。
        overlap_width_ratio: 水平重叠比例（0.0-1.0）。
        postprocess_type: 后处理类型（'NMS' 或 'WBF'）。
        postprocess_match_threshold: 后处理匹配阈值。
        postprocess_match_class_agnostic: 是否进行类别无关的匹配。
        postprocess_individually: 是否单独处理每个切片。
        postprocess_verbose: 是否输出详细信息。
        min_image_size: 最小图像尺寸（像素），小于该尺寸的图像不进行切片。
        auto_slice_resolution: 是否自动计算切片分辨率。
    """

    enabled: bool = False
    slice_height: int = 512
    slice_width: int = 512
    overlap_height_ratio: float = 0.25
    overlap_width_ratio: float = 0.25
    postprocess_type: str = "NMS"
    postprocess_match_threshold: float = 0.5
    postprocess_match_class_agnostic: bool = False
    postprocess_individually: bool = False
    postprocess_verbose: bool = False
    min_image_size: int = 640
    auto_slice_resolution: bool = True

    def validate(self) -> None:
        """验证配置参数的有效性。"""
        if self.slice_height < 32:
            raise ValueError(f"slice_height 必须 >= 32，当前值: {self.slice_height}")
        if self.slice_width < 32:
            raise ValueError(f"slice_width 必须 >= 32，当前值: {self.slice_width}")
        if not (0.0 <= self.overlap_height_ratio <= 1.0):
            raise ValueError(
                f"overlap_height_ratio 必须在 [0, 1] 范围内，当前值: {self.overlap_height_ratio}"
            )
        if not (0.0 <= self.overlap_width_ratio <= 1.0):
            raise ValueError(
                f"overlap_width_ratio 必须在 [0, 1] 范围内，当前值: {self.overlap_width_ratio}"
            )
        if self.postprocess_type not in ("NMS", "WBF"):
            raise ValueError(
                f"postprocess_type 必须为 'NMS' 或 'WBF'，当前值: {self.postprocess_type}"
            )
        if not (0.0 <= self.postprocess_match_threshold <= 1.0):
            raise ValueError(
                f"postprocess_match_threshold 必须在 [0, 1] 范围内，"
                f"当前值: {self.postprocess_match_threshold}"
            )
        if self.min_image_size < 32:
            raise ValueError(
                f"min_image_size 必须 >= 32，当前值: {self.min_image_size}"
            )

    def to_dict(self) -> dict:
        """将配置转换为字典格式。"""
        return {
            "sahi_enabled": self.enabled,
            "sahi_slice_height": self.slice_height,
            "sahi_slice_width": self.slice_width,
            "sahi_overlap_height_ratio": self.overlap_height_ratio,
            "sahi_overlap_width_ratio": self.overlap_width_ratio,
            "sahi_postprocess_type": self.postprocess_type,
            "sahi_postprocess_match_threshold": self.postprocess_match_threshold,
            "sahi_postprocess_match_class_agnostic": self.postprocess_match_class_agnostic,
            "sahi_postprocess_individually": self.postprocess_individually,
            "sahi_postprocess_verbose": self.postprocess_verbose,
            "sahi_min_image_size": self.min_image_size,
            "sahi_auto_slice_resolution": self.auto_slice_resolution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SAHIConfig":
        """从字典创建配置实例。"""
        return cls(
            enabled=data.get("sahi_enabled", False),
            slice_height=data.get("sahi_slice_height", 512),
            slice_width=data.get("sahi_slice_width", 512),
            overlap_height_ratio=data.get("sahi_overlap_height_ratio", 0.25),
            overlap_width_ratio=data.get("sahi_overlap_width_ratio", 0.25),
            postprocess_type=data.get("sahi_postprocess_type", "NMS"),
            postprocess_match_threshold=data.get("sahi_postprocess_match_threshold", 0.5),
            postprocess_match_class_agnostic=data.get(
                "sahi_postprocess_match_class_agnostic", False
            ),
            postprocess_individually=data.get("sahi_postprocess_individually", False),
            postprocess_verbose=data.get("sahi_postprocess_verbose", False),
            min_image_size=data.get("sahi_min_image_size", 640),
            auto_slice_resolution=data.get("sahi_auto_slice_resolution", True),
        )


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
        # 自动计算切片大小：使用训练图像尺寸的 1/2 或 1/4
        slice_size = min(imgsz // 2, 512)
        slice_size = max(slice_size, 128)  # 最小 128 像素

    return SAHIConfig(
        enabled=enable_sahi,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )
