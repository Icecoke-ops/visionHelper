"""模型预测相关 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.api._validators import (
    _require_in_range,
    _require_non_empty_str,
)


class PredictAPI:
    """模型预测相关 API。"""

    @staticmethod
    def predict(
            model_path: str,
            input_path: str,
            output_dir: str,
            threshold: float = 0.25,
            task: str = "detect",
            device: Optional[str] = None,
            iou: float = 0.45,
    ) -> dict:
        """
        使用 YOLO 模型对图片或视频进行预测，将可视化结果保存到输出目录。

        参数:
            model_path: YOLO 模型权重文件路径（.pt），或可触发自动下载的
                裸模型名（如 ``yolov8n``）；裸模型名仅在显式路径（含目录
                分隔符）时校验存在性。
        """
        _require_non_empty_str(model_path, "model_path")
        model_p = Path(model_path).expanduser()
        if len(model_p.parts) > 1:
            # 显式路径（含目录分隔符）必须指向已存在的文件
            if not model_p.exists():
                raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
            if not model_p.is_file():
                raise FileNotFoundError(f"模型权重路径不是文件: {model_path}")
        _require_non_empty_str(input_path, "input_path")
        _require_non_empty_str(output_dir, "output_dir")

        input_p = Path(input_path).expanduser()
        if not input_p.exists():
            raise FileNotFoundError(f"输入路径不存在: {input_path}")

        task_norm = (task or "").lower()
        valid_tasks = {"detect", "obb", "segment", "classify"}
        if task_norm not in valid_tasks:
            raise ValueError(
                f"不支持的任务类型: {task!r}，仅支持 {sorted(valid_tasks)}"
            )

        _require_in_range(threshold, "threshold", 0.0, 1.0, inclusive_lo=False)
        _require_in_range(iou, "iou", 0.0, 1.0, inclusive_lo=False)

        from scripts.predict.predict import predict

        return predict(
            model_path=model_path,
            input_path=input_path,
            output_dir=output_dir,
            threshold=threshold,
            task=task_norm,
            device=device,
            iou=iou,
        )
