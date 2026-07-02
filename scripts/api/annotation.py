"""数据标注相关 API。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.api._validators import (
    _require_existing_dir,
    _require_existing_file,
    _require_in_range,
    _require_non_empty_str,
    _require_non_negative_float,
)


class AnnotationAPI:
    """数据标注相关 API。"""

    @staticmethod
    def annotation_stats(folder: str) -> dict:
        """
        统计目录下的图片标注情况（进程内执行）。

        参数:
            folder: 待统计的图片目录路径。

        返回:
            包含 total_images、annotated_images、unannotated_images、
            detection_images、obb_images、polygon_images、manual_images、
            auto_images、auto_corrected_images 的统计字典。
        """
        _require_existing_dir(folder, "folder")

        from scripts.datasets.stats import collect_annotation_stats

        return collect_annotation_stats(folder)

    @staticmethod
    def annotation_label_stats(folder: str) -> List[Dict[str, int]]:
        """
        按标签统计目录下的标注实例数量（进程内执行）。
        """
        _require_existing_dir(folder, "folder")

        from scripts.datasets.stats import collect_annotation_label_stats

        return collect_annotation_label_stats(folder)

    @staticmethod
    def annotation_stats_cli(
            folder: str,
            include_label_stats: bool = True,
            python_executable: Optional[str] = None,
            timeout: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        通过子进程调用 ``python scripts/vh.py datasets stats`` 完成标注统计。

        与 :meth:`annotation_stats` / :meth:`annotation_label_stats` 在独立
        Python 进程中执行，避免在主进程内 import 重型依赖。
        """
        _require_existing_dir(folder, "folder")
        if timeout is not None and (
                not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise ValueError(f"timeout 必须为正数或 None，当前值: {timeout!r}")

        executable = python_executable or sys.executable
        if not executable:
            raise RuntimeError(
                "未找到可用的 Python 解释器（sys.executable 为空），"
                "请显式传入 python_executable。"
            )

        cwd = str(Path(__file__).resolve().parent.parent)

        cmd: List[str] = [executable, "-m", "scripts.vh", "datasets", "stats", "--input", folder]
        if include_label_stats:
            cmd.append("--label-stats")
        cmd.append("--json")

        completed = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            raise RuntimeError(
                f"annotation_stats 子进程退出码 {completed.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )

        from scripts.datasets.stats import parse_machine_block

        try:
            payload = parse_machine_block(stdout)
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"解析 annotation_stats 子进程输出失败: {exc}\n"
                f"stdout:\n{stdout}"
            ) from exc

        stats = payload.get("stats")
        label_stats = payload.get("label_stats")
        if not isinstance(stats, dict):
            raise RuntimeError(f"子进程输出缺少 stats 字段:\n{stdout}")
        if not isinstance(label_stats, list):
            label_stats = []

        return {
            "stats": stats,
            "label_stats": label_stats,
            "stdout": stdout,
            "stderr": stderr,
        }

    # -------- 自动标注 / 清理 --------

    @staticmethod
    def auto_annotate(
            work_dir: str,
            model_path: str,
            threshold: float = 0.25,
            task: str = "detect",
            suffix: str = "",
            device: Optional[str] = None,
            iou: float = 0.45,
            include_unannotated: bool = True,
            include_auto: bool = False,
            include_auto_corrected: bool = False,
            include_manual: bool = False,
            tolerance_seconds: float = 2.0,
            batch_size: int = 8,
    ) -> dict:
        """
        使用 YOLO 模型对工作目录下指定状态的图片进行自动标注。
        """
        _require_existing_dir(work_dir, "work_dir")
        _require_existing_file(model_path, "model_path")

        task_norm = (task or "").lower()
        valid_tasks = {"detect", "obb", "segment", "classify"}
        if task_norm not in valid_tasks:
            raise ValueError(
                f"不支持的任务类型: {task!r}，仅支持 {sorted(valid_tasks)}"
            )

        _require_in_range(threshold, "threshold", 0.0, 1.0, inclusive_lo=False)
        _require_in_range(iou, "iou", 0.0, 1.0, inclusive_lo=False)
        _require_non_negative_float(tolerance_seconds, "tolerance_seconds")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size 必须是 >=1 的整数，当前值: {batch_size}")

        if not any([
            include_unannotated, include_auto,
            include_auto_corrected, include_manual,
        ]):
            raise ValueError(
                "至少需要选择一种处理范围（include_unannotated / include_auto / "
                "include_auto_corrected / include_manual 至少一个为 True）"
            )

        from scripts.datasets.auto import auto_annotate

        return auto_annotate(
            work_dir=work_dir,
            model_path=model_path,
            threshold=threshold,
            task=task_norm,
            suffix=suffix,
            device=device,
            iou=iou,
            include_unannotated=include_unannotated,
            include_auto=include_auto,
            include_auto_corrected=include_auto_corrected,
            include_manual=include_manual,
            tolerance_seconds=tolerance_seconds,
            batch_size=batch_size,
        )

    @staticmethod
    def clear_annotations(
            folder: str,
            include_auto: bool = False,
            include_auto_corrected: bool = False,
            include_manual: bool = False,
            tolerance_seconds: float = 2.0,
            dry_run: bool = False,
    ) -> Dict[str, object]:
        """
        清除目录下指定类型的 X-AnyLabeling JSON 标注文件。
        """
        _require_existing_dir(folder, "folder")
        _require_non_negative_float(tolerance_seconds, "tolerance_seconds")
        if not any([include_auto, include_auto_corrected, include_manual]):
            raise ValueError(
                "至少需要选择一种待清除的标注类型（include_auto / "
                "include_auto_corrected / include_manual 至少一个为 True）"
            )

        from scripts.datasets.clear import clear_annotations

        return clear_annotations(
            folder=folder,
            include_auto=include_auto,
            include_auto_corrected=include_auto_corrected,
            include_manual=include_manual,
            tolerance_seconds=tolerance_seconds,
            dry_run=dry_run,
        )

    @staticmethod
    def check_annotation_type(
            json_path: str,
            tolerance_seconds: float = 2.0,
    ) -> str:
        """
        判断单个 X-AnyLabeling JSON 标注文件的类型。
        """
        _require_existing_file(json_path, "json_path")
        _require_non_negative_float(tolerance_seconds, "tolerance_seconds")

        from scripts.common.annotation_type import AnnotationTypeChecker

        checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)
        return checker.check_file(json_path).value

    @staticmethod
    def check_image_annotation_type(
            image_path: str,
            tolerance_seconds: float = 2.0,
    ) -> str:
        """
        根据图片路径查找同名 JSON 标注文件并判断标注类型。
        """
        _require_existing_file(image_path, "image_path")
        _require_non_negative_float(tolerance_seconds, "tolerance_seconds")

        from scripts.common.annotation_type import AnnotationTypeChecker

        checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)
        return checker.check_image_annotation(image_path).value

    @staticmethod
    def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
        """
        扫描 runs 目录下的训练模型。
        """
        _require_non_empty_str(runs_dir, "runs_dir")
        if not Path(runs_dir).expanduser().is_dir():
            return []

        from scripts.common.utils import discover_trained_models

        return discover_trained_models(runs_dir)
