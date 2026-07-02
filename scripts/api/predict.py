"""模型预测相关 API。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from scripts.api._validators import (
    _require_existing_file,
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
        """
        _require_existing_file(model_path, "model_path")
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
