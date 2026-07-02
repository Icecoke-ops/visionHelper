"""图片处理相关 API。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from scripts.api._validators import (
    _require_existing_dir,
    _require_in_range,
    _require_non_empty_str,
    _require_non_negative_int,
)


class ImageAPI:
    """图片处理相关 API。"""

    @staticmethod
    def augment(
            input_dir: str,
            output_dir: str,
            rotate_enabled: bool = True,
            rotate_degrees: float = 30,
            rotate_prob: float = 0.5,
            cut_enabled: bool = True,
            cut_scale: float = 0.3,
            cut_ratio: float = 1.5,
            cut_prob: float = 0.5,
            cut_resize: bool = True,
            occlusion_enabled: bool = True,
            occlusion_count: int = 3,
            occlusion_size: float = 0.15,
            occlusion_prob: float = 0.5,
            channel_enabled: bool = True,
            channel_prob: float = 0.5,
            seed: Optional[int] = None,
            ext: str = "jpg",
            quality: int = 95,
            prefix: str = "aug",
    ) -> List[str]:
        """
        对目录下图片执行数据增强（旋转/切割/遮挡/通道变换）。
        """
        _require_existing_dir(input_dir, "input_dir")
        _require_non_empty_str(output_dir, "output_dir")
        _require_in_range(quality, "quality", 1, 100)
        if rotate_degrees < 0:
            raise ValueError(f"rotate_degrees 必须 >= 0，当前值: {rotate_degrees}")
        _require_in_range(cut_scale, "cut_scale", 0, 1)
        _require_in_range(cut_ratio, "cut_ratio", 1, float("inf"))
        _require_non_negative_int(occlusion_count, "occlusion_count")
        _require_in_range(occlusion_size, "occlusion_size", 0, 1)
        _require_in_range(rotate_prob, "rotate_prob", 0, 1)
        _require_in_range(cut_prob, "cut_prob", 0, 1)
        _require_in_range(occlusion_prob, "occlusion_prob", 0, 1)
        _require_in_range(channel_prob, "channel_prob", 0, 1)

        from scripts.images.augment import augment_images

        return augment_images(
            input_dir=input_dir,
            output_dir=output_dir,
            rotate_enabled=rotate_enabled,
            rotate_degrees=rotate_degrees,
            rotate_prob=rotate_prob,
            cut_enabled=cut_enabled,
            cut_scale=cut_scale,
            cut_ratio=cut_ratio,
            cut_prob=cut_prob,
            cut_resize=cut_resize,
            occlusion_enabled=occlusion_enabled,
            occlusion_count=occlusion_count,
            occlusion_size=occlusion_size,
            occlusion_prob=occlusion_prob,
            channel_enabled=channel_enabled,
            channel_prob=channel_prob,
            seed=seed,
            ext=ext,
            quality=quality,
            prefix=prefix,
        )

    @staticmethod
    def deduplicate(
            folder: str,
            threshold: float = 0.95,
            delete: bool = False,
            move_to: Optional[str] = None,
            model_name: str = "google/vit-base-patch16-224",
            batch_size: int = 8,
            backend: str = "vit",
            hash_size: int = 16,
            grid_size: int = 1,
    ) -> dict:
        """
        对目录下相似图片进行去重。

        基于图片特征向量两两计算余弦相似度，按顺序保留首次出现的图片，
        将后续相似度 ``>= threshold`` 的图片视为重复。
        """
        _require_existing_dir(folder, "folder")
        _require_in_range(threshold, "threshold", 0.0, 1.0, inclusive_lo=False)

        if delete and move_to:
            raise ValueError(
                "delete=True 与 move_to=<path> 互斥，不能同时使用"
            )
        if move_to is not None:
            _require_non_empty_str(move_to, "move_to")

        backend_norm = (backend or "").lower()
        valid_backends = {"vit", "phash"}
        if backend_norm not in valid_backends:
            raise ValueError(
                f"不支持的 backend: {backend!r}，仅支持 {sorted(valid_backends)}"
            )

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"batch_size 必须是 >=1 的整数，当前值: {batch_size}"
            )
        if not isinstance(hash_size, int) or hash_size < 1:
            raise ValueError(
                f"hash_size 必须是 >=1 的整数，当前值: {hash_size}"
            )
        if not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError(
                f"grid_size 必须是 >=1 的整数，当前值: {grid_size}"
            )
        if backend_norm == "vit":
            _require_non_empty_str(model_name, "model_name")

        from scripts.images.dedup import deduplicate

        return deduplicate(
            folder=folder,
            threshold=threshold,
            delete=delete,
            move_to=move_to,
            model_name=model_name,
            batch_size=batch_size,
            backend=backend_norm,
            hash_size=hash_size,
            grid_size=grid_size,
        )
