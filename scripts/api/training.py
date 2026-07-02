"""模型训练相关 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from scripts.api._validators import (
    _require_existing_dir,
    _require_existing_file,
    _require_in_range,
    _require_non_empty_str,
    _require_non_negative_float,
    _require_non_negative_int,
    _require_positive_int,
)
from scripts.common.train_config import TrainConfig


VALID_COPY_MODES = {"copy", "link", "symlink"}


class TrainingAPI:
    """模型训练相关 API。"""

    @staticmethod
    def export_yolo_dataset(
            input_dir: str,
            output_dir: str,
            task: str = "detect",
            train_ratio: float = 0.8,
            test_ratio: float = 0.2,
            seed: int = 42,
            copy_mode: str = "copy",
            export_empty_labels: bool = False,
            export_unlabeled: bool = False,
    ) -> Dict[str, int]:
        """
        将 X-AnyLabeling 标注图片导出为 YOLO 数据集。
        """
        _require_existing_dir(input_dir, "input_dir")
        _require_non_empty_str(output_dir, "output_dir")
        _require_positive_int(seed, "seed")

        task_norm = (task or "").lower()
        valid_tasks = {"detect", "obb", "segment", "classify"}
        if task_norm not in valid_tasks:
            raise ValueError(
                f"不支持的任务类型: {task!r}，仅支持 {sorted(valid_tasks)}"
            )

        copy_mode_norm = (copy_mode or "").lower()
        if copy_mode_norm not in VALID_COPY_MODES:
            raise ValueError(
                f"不支持的 copy_mode: {copy_mode!r}，"
                f"仅支持 {sorted(VALID_COPY_MODES)}"
            )

        from scripts.common.utils import validate_split_ratios
        validate_split_ratios(train_ratio, test_ratio)

        in_resolved = Path(input_dir).expanduser().resolve()
        out_resolved = Path(output_dir).expanduser().resolve()
        if in_resolved == out_resolved:
            raise ValueError("input_dir 与 output_dir 不能为同一路径")

        from scripts.datasets.export import export_yolo_dataset

        return export_yolo_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            task=task_norm,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            seed=seed,
            copy_mode=copy_mode_norm,
            export_empty_labels=export_empty_labels,
            export_unlabeled=export_unlabeled,
        )

    @staticmethod
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
            sahi_enabled: bool = False,
            sahi_slice_height: int = 512,
            sahi_slice_width: int = 512,
            sahi_overlap_height_ratio: float = 0.25,
            sahi_overlap_width_ratio: float = 0.25,
    ) -> str:
        """
        基于 Ultralytics YOLO 训练模型。

        此为向后兼容签名，推荐使用 :meth:`train_model_config` 配合
        :class:`scripts.common.train_config.TrainConfig`。
        """
        _require_existing_file(dataset_yaml, "dataset_yaml")
        _require_non_empty_str(project, "project")
        _require_non_empty_str(name, "name")
        _require_non_empty_str(model, "model")
        _require_positive_int(epochs, "epochs")
        _require_non_negative_int(patience, "patience")
        _require_non_negative_int(workers, "workers")
        config = TrainConfig(
            dataset_yaml=dataset_yaml,
            task=task,
            model=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=patience,
            resume=resume,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            box=box,
            cls=cls,
            dfl=dfl,
            label_smoothing=label_smoothing,
            close_mosaic=close_mosaic,
            amp=amp,
            freeze=freeze,
            workers=workers,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
            sahi_enabled=sahi_enabled,
            sahi_slice_height=sahi_slice_height,
            sahi_slice_width=sahi_slice_width,
            sahi_overlap_height_ratio=sahi_overlap_height_ratio,
            sahi_overlap_width_ratio=sahi_overlap_width_ratio,
        )
        return TrainingAPI.train_model_config(config)

    @staticmethod
    def train_model_config(config: TrainConfig) -> str:
        """
        基于 Ultralytics YOLO 训练模型（配置对象版）。
        """
        _require_existing_file(config.dataset_yaml, "dataset_yaml")
        _require_non_empty_str(config.model, "model")
        _require_non_empty_str(config.project, "project")
        _require_non_empty_str(config.name, "name")

        valid_tasks = {"detect", "obb", "segment", "classify"}
        if config.task_norm not in valid_tasks:
            raise ValueError(
                f"不支持的任务类型: {config.task!r}，仅支持 {sorted(valid_tasks)}"
            )

        _require_positive_int(config.epochs, "epochs")
        if config.imgsz < 32:
            raise ValueError(f"imgsz 必须是 >=32 的整数，当前值: {config.imgsz}")
        if config.batch == 0 or config.batch < -1:
            raise ValueError(f"batch 必须为正整数，或 -1 表示自动 batch，当前值: {config.batch}")
        _require_non_negative_int(config.patience, "patience")
        _require_non_negative_int(config.workers, "workers")
        _require_in_range(config.lr0, "lr0", 0.0, float("inf"), inclusive_lo=False)
        _require_in_range(config.lrf, "lrf", 0.0, 1.0, inclusive_lo=False)
        _require_in_range(config.momentum, "momentum", 0.0, 1.0)
        _require_non_negative_float(config.weight_decay, "weight_decay")
        _require_non_negative_float(config.warmup_epochs, "warmup_epochs")
        _require_in_range(config.warmup_momentum, "warmup_momentum", 0.0, 1.0)
        _require_non_negative_float(config.warmup_bias_lr, "warmup_bias_lr")
        _require_in_range(config.box, "box", 0.0, float("inf"), inclusive_lo=False)
        _require_in_range(config.cls, "cls", 0.0, float("inf"), inclusive_lo=False)
        _require_in_range(config.dfl, "dfl", 0.0, float("inf"), inclusive_lo=False)
        _require_in_range(config.label_smoothing, "label_smoothing", 0.0, 1.0, inclusive_hi=False)
        _require_non_negative_int(config.close_mosaic, "close_mosaic")
        if config.freeze is not None:
            _require_non_negative_int(config.freeze, "freeze")

        _require_in_range(config.hsv_h, "hsv_h", 0.0, 1.0)
        _require_in_range(config.hsv_s, "hsv_s", 0.0, 1.0)
        _require_in_range(config.hsv_v, "hsv_v", 0.0, 1.0)
        _require_non_negative_float(config.degrees, "degrees")
        _require_in_range(config.translate, "translate", 0.0, 1.0)
        _require_non_negative_float(config.scale, "scale")
        _require_non_negative_float(config.shear, "shear")
        _require_in_range(config.perspective, "perspective", 0.0, 0.001)
        _require_in_range(config.flipud, "flipud", 0.0, 1.0)
        _require_in_range(config.fliplr, "fliplr", 0.0, 1.0)
        _require_in_range(config.mosaic, "mosaic", 0.0, 1.0)
        _require_in_range(config.mixup, "mixup", 0.0, 1.0)
        _require_in_range(config.copy_paste, "copy_paste", 0.0, 1.0)

        if config.sahi.sahi_enabled:
            if config.sahi.sahi_slice_height < 32:
                raise ValueError(
                    f"sahi_slice_height 必须是 >=32 的整数，当前值: {config.sahi.sahi_slice_height}"
                )
            if config.sahi.sahi_slice_width < 32:
                raise ValueError(
                    f"sahi_slice_width 必须是 >=32 的整数，当前值: {config.sahi.sahi_slice_width}"
                )
            _require_in_range(
                config.sahi.sahi_overlap_height_ratio, "sahi_overlap_height_ratio", 0.0, 1.0
            )
            _require_in_range(
                config.sahi.sahi_overlap_width_ratio, "sahi_overlap_width_ratio", 0.0, 1.0
            )

        from scripts.common.config import SUPPORTED_OPTIMIZERS
        if config.optimizer_norm not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"不支持的优化器: {config.optimizer!r}，"
                f"仅支持 {sorted(SUPPORTED_OPTIMIZERS)}"
            )

        from scripts.train.train import TrainingConfig, train_model

        train_cfg = TrainingConfig(
            dataset_yaml=config.dataset_yaml,
            project=config.project,
            name=config.name,
            task=config.task,
            model=config.model,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            patience=config.patience,
            resume=config.resume,
            optimizer=config.optimizer,
            lr0=config.lr0,
            lrf=config.lrf,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            warmup_epochs=config.warmup_epochs,
            warmup_momentum=config.warmup_momentum,
            warmup_bias_lr=config.warmup_bias_lr,
            box=config.box,
            cls=config.cls,
            dfl=config.dfl,
            label_smoothing=config.label_smoothing,
            close_mosaic=config.close_mosaic,
            amp=config.amp,
            freeze=config.freeze,
            workers=config.workers,
            hsv_h=config.hsv_h,
            hsv_s=config.hsv_s,
            hsv_v=config.hsv_v,
            degrees=config.degrees,
            translate=config.translate,
            scale=config.scale,
            shear=config.shear,
            perspective=config.perspective,
            flipud=config.flipud,
            fliplr=config.fliplr,
            mosaic=config.mosaic,
            mixup=config.mixup,
            copy_paste=config.copy_paste,
            sahi_enabled=config.sahi.sahi_enabled,
            sahi_slice_height=config.sahi.sahi_slice_height,
            sahi_slice_width=config.sahi.sahi_slice_width,
            sahi_overlap_height_ratio=config.sahi.sahi_overlap_height_ratio,
            sahi_overlap_width_ratio=config.sahi.sahi_overlap_width_ratio,
        )
        return train_model(train_cfg)
