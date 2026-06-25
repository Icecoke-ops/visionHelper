#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper 对外统一 API 接口。

该模块以类方法的形式暴露项目内的核心能力，便于上层（GUI、脚本、测试）统一调用：

    - :meth:`VideoAPI.extract_video_frames`             视频抽帧
    - :meth:`ImageAPI.deduplicate`                      图片去重
    - :meth:`AnnotationAPI.auto_annotate`               YOLO 自动标注
    - :meth:`AnnotationAPI.clear_annotations`           按类型清理标注
    - :meth:`AnnotationAPI.annotation_stats`            进程内统计标注
    - :meth:`AnnotationAPI.annotation_label_stats`      进程内按标签统计
    - :meth:`AnnotationAPI.annotation_stats_cli`        子进程统计（GUI 推荐）
    - :meth:`AnnotationAPI.check_annotation_type`       判断单个 JSON 标注类型
    - :meth:`AnnotationAPI.check_image_annotation_type` 根据图片判断标注类型
    - :meth:`AnnotationAPI.discover_trained_models`     扫描 runs 训练产物
    - :meth:`TrainingAPI.export_yolo_dataset`           导出 YOLO 数据集
    - :meth:`TrainingAPI.train_model`                   训练 YOLO 模型

设计要点
========

* **零副作用导入**：所有重型依赖（torch、ultralytics、opencv、transformers 等）
  都通过方法体内的 *懒导入* 在真正调用时才被加载，因此
  ``import scripts.api`` 自身是无副作用的，即使部分依赖缺失，
  也不会影响未使用功能的可用性。
* **轻量参数校验**：每个 API 方法在调用核心实现前会做一次便宜的边界检查
  （路径存在性、阈值范围、互斥选项等），尽早抛出
  ``ValueError`` / ``FileNotFoundError``，避免在重型流程中途失败。
* **签名稳定**：方法签名与默认值保持向后兼容，调用方升级版本无需修改代码。

用法示例
========

::

    from scripts.api import VideoAPI, ImageAPI

    VideoAPI.extract_video_frames(
        input_video="/path/to/video.mp4",
        output_dir="/path/to/output",
        frame_step=5,
    )

    result = ImageAPI.deduplicate(
        folder="/path/to/images",
        threshold=0.95,
        delete=False,
    )
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 内部辅助：轻量参数校验
# ---------------------------------------------------------------------------
def _require_non_empty_str(value: object, name: str) -> str:
    """校验 ``value`` 是非空字符串，否则抛 :class:`ValueError`。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"参数 {name!r} 必须为非空字符串，当前值: {value!r}")
    return value


def _require_existing_dir(value: object, name: str) -> Path:
    """校验 ``value`` 指向已存在的目录，返回 :class:`pathlib.Path`。"""
    _require_non_empty_str(value, name)
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} 路径不存在: {value}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} 不是目录: {value}")
    return path


def _require_existing_file(value: object, name: str) -> Path:
    """校验 ``value`` 指向已存在的常规文件，返回 :class:`pathlib.Path`。"""
    _require_non_empty_str(value, name)
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} 文件不存在: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不是文件: {value}")
    return path


def _require_in_range(
        value: float,
        name: str,
        lo: float,
        hi: float,
        inclusive_lo: bool = True,
        inclusive_hi: bool = True,
) -> float:
    """
    校验数值 ``value`` 位于 ``[lo, hi]``（端点开闭可配置）内，
    否则抛 :class:`ValueError`。
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"参数 {name!r} 必须是数值，当前值: {value!r}")
    lo_ok = value >= lo if inclusive_lo else value > lo
    hi_ok = value <= hi if inclusive_hi else value < hi
    if not (lo_ok and hi_ok):
        lb = "[" if inclusive_lo else "("
        rb = "]" if inclusive_hi else ")"
        raise ValueError(
            f"参数 {name!r} 必须位于 {lb}{lo}, {hi}{rb}，当前值: {value}"
        )
    return float(value)


# ---------------------------------------------------------------------------
# VideoAPI
# ---------------------------------------------------------------------------
class VideoAPI:
    """视频处理相关 API。"""

    @staticmethod
    def extract_video_frames(
            input_video: str,
            output_dir: str,
            frame_step: int = 1,
            image_extension: str = "jpg",
            quality: int = 100,
            prefix: str = "frame",
            start_time: Optional[float] = None,
            end_time: Optional[float] = None,
            seek_mode: str = "decode_all",
            overwrite: bool = False,
    ) -> List[str]:
        """
        将视频抽帧并保存为图片。

        参数:
            input_video: 输入视频文件路径。
            output_dir: 输出图片保存目录。
            frame_step: 抽帧间隔，必须 >= 1，默认 1（逐帧抽取）。
            image_extension: 输出图片格式，默认 jpg。
            quality: 输出图片质量 1-100，默认 100（原始画质）。
            prefix: 输出文件名前缀，默认 "frame"。
            start_time: 开始抽取的时间（秒或时间字符串），默认从视频开头。
            end_time: 结束抽取的时间（秒或时间字符串），默认抽到视频结尾。
            seek_mode: 跳帧策略，``decode_all`` 顺序解码（默认）或 ``seek``
                直接跳到下一目标帧。
            overwrite: 是否覆盖已存在的同名文件，默认 False（重名追加 _1/_2 后缀）。

        返回:
            保存的图片路径列表（按帧顺序）。

        异常:
            ValueError: 参数非法（如 ``frame_step`` < 1、``quality`` 超出 1-100）。
            FileNotFoundError: ``input_video`` 不是文件。
        """
        _require_existing_file(input_video, "input_video")
        _require_non_empty_str(output_dir, "output_dir")
        if not isinstance(frame_step, int) or frame_step < 1:
            raise ValueError(f"frame_step 必须是 >=1 的整数，当前值: {frame_step}")
        _require_in_range(quality, "quality", 1, 100)

        from scripts.images.import_ import extract_video_frames

        return extract_video_frames(
            input_video=input_video,
            output_dir=output_dir,
            frame_step=frame_step,
            image_extension=image_extension,
            quality=quality,
            prefix=prefix,
            start_time=start_time,
            end_time=end_time,
            seek_mode=seek_mode,
            overwrite=overwrite,
        )


# ---------------------------------------------------------------------------
# AnnotationAPI
# ---------------------------------------------------------------------------
class AnnotationAPI:
    """数据标注相关 API。"""

    # -------- 统计 --------

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

        异常:
            FileNotFoundError: ``folder`` 不存在或不是目录。
        """
        _require_existing_dir(folder, "folder")

        from scripts.datasets.stats import collect_annotation_stats

        return collect_annotation_stats(folder)

    @staticmethod
    def annotation_label_stats(folder: str) -> List[Dict[str, int]]:
        """
        按标签统计目录下的标注实例数量（进程内执行）。

        参数:
            folder: 待统计的图片目录路径。

        返回:
            每个标签的实例数量列表，元素包含 label、detection_count、
            obb_count、polygon_count。

        异常:
            FileNotFoundError: ``folder`` 不存在或不是目录。
        """
        _require_existing_dir(folder, "folder")

        from scripts.datasets.stats import collect_annotation_label_stats

        return collect_annotation_label_stats(folder)

    @staticmethod
    def annotation_stats_cli(
            folder: str,
            include_label_stats: bool = True,
            python_executable: Optional[str] = None,
            cwd: Optional[str] = None,
            timeout: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        通过子进程调用 ``python scripts/vh.py datasets stats`` 完成标注统计。

        与 :meth:`annotation_stats` / :meth:`annotation_label_stats` 的区别：

        - 在独立 Python 进程中执行，便于 GUI 通过命令行解耦调用，
          避免在主进程内 import 重型依赖；
        - 直接捕获子进程 ``stdout`` 并解析其中的结果 JSON 块；
        - 同时返回原始日志，便于上层界面展示。

        参数:
            folder: 待统计的图片目录路径。
            include_label_stats: 是否同时计算按标签的实例统计，默认 True。
            python_executable: 用于运行子进程的 Python 解释器路径。
                默认使用当前进程的 ``sys.executable``。
            cwd: 子进程工作目录。默认使用项目根目录，
                以保证 ``scripts/vh.py datasets stats`` 能够定位到包。
            timeout: 超时秒数，``None`` 表示不限制。

        返回:
            字典，包含 ``stats`` / ``label_stats`` / ``stdout`` / ``stderr``。

        异常:
            ValueError: ``folder`` 为空字符串、或 ``timeout`` 非正数。
            FileNotFoundError: ``folder`` 不存在或不是目录。
            RuntimeError: 未找到可用 Python 解释器、子进程退出码非 0，
                或输出中找不到结果 JSON 块时抛出。
            subprocess.TimeoutExpired: 子进程超时未结束。
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

        if cwd is None:
            cwd = str(Path(__file__).resolve().parent.parent)

        cmd: List[str] = [executable, "scripts/vh.py", "datasets", "stats", "--input", folder]
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
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"解析 annotation_stats 子进程输出失败: {exc}\n"
                f"stdout:\n{stdout}"
            ) from exc

        stats = payload.get("stats") if isinstance(payload, dict) else None
        label_stats = payload.get("label_stats") if isinstance(payload, dict) else None
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

        参数:
            work_dir: 待标注图片所在的工作目录。
            model_path: YOLO 模型权重文件路径（.pt）。
            threshold: 置信度阈值，必须位于 (0, 1]，默认 0.25。
            task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``。
            suffix: 输出 JSON 文件名后缀。
            device: 推理设备，例如 ``0``、``cpu``，默认自动选择。
            iou: NMS IoU 阈值，必须位于 (0, 1]，默认 0.45。
            include_unannotated: 是否处理未标注图片，默认 True。
            include_auto: 是否处理已自动标注图片（覆盖 / 合并），默认 False。
            include_auto_corrected: 是否处理自动标注后矫正过的图片，默认 False。
            include_manual: 是否处理手动标注图片，默认 False。
            tolerance_seconds: 自动 / 矫正 时间戳容差（秒），必须 >= 0，默认 2.0。
            batch_size: 推理批大小，必须 >= 1，默认 8。

        返回:
            包含 ``total`` / ``skipped`` / ``annotated`` 等键的字典。

        异常:
            FileNotFoundError: ``work_dir`` 或 ``model_path`` 不存在。
            ValueError: ``task`` 非法或数值参数越界、或所有 include_* 开关均为 False。
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
        if not isinstance(tolerance_seconds, (int, float)) or tolerance_seconds < 0:
            raise ValueError(
                f"tolerance_seconds 必须 >= 0，当前值: {tolerance_seconds}"
            )
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
    ) -> Dict[str, object]:
        """
        清除目录下指定类型的 X-AnyLabeling JSON 标注文件。

        参数:
            folder: 待清理的目录路径。
            include_auto: 是否删除自动标注的 JSON。
            include_auto_corrected: 是否删除自动标注后人工矫正的 JSON。
            include_manual: 是否删除手动标注的 JSON。
            tolerance_seconds: 区分自动 / 矫正 的时间容差（秒），必须 >= 0。

        返回:
            ``{"scanned", "deleted", "by_type", "failed"}`` 字典。

        异常:
            FileNotFoundError: ``folder`` 不存在或不是目录。
            ValueError: 所有 include_* 开关均为 False，或 ``tolerance_seconds`` < 0。
        """
        _require_existing_dir(folder, "folder")
        if not isinstance(tolerance_seconds, (int, float)) or tolerance_seconds < 0:
            raise ValueError(
                f"tolerance_seconds 必须 >= 0，当前值: {tolerance_seconds}"
            )
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
        )

    # -------- 标注类型识别 / 训练产物发现 --------

    @staticmethod
    def check_annotation_type(
            json_path: str,
            tolerance_seconds: float = 2.0,
    ) -> str:
        """
        判断单个 X-AnyLabeling JSON 标注文件的类型。

        参数:
            json_path: 标注文件路径。
            tolerance_seconds: 判定为自动标注的最大时间差阈值（秒），必须 >= 0，默认 2.0。

        返回:
            ``"manual"`` / ``"auto"`` / ``"auto_corrected"`` 之一。

        异常:
            FileNotFoundError: 文件不存在。
            ValueError: ``tolerance_seconds`` < 0 或文件内容不是合法 JSON。
        """
        _require_existing_file(json_path, "json_path")
        if not isinstance(tolerance_seconds, (int, float)) or tolerance_seconds < 0:
            raise ValueError(
                f"tolerance_seconds 必须 >= 0，当前值: {tolerance_seconds}"
            )

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

        参数:
            image_path: 图片文件路径。
            tolerance_seconds: 判定为自动标注的最大时间差阈值（秒），必须 >= 0，默认 2.0。

        返回:
            ``"manual"`` / ``"auto"`` / ``"auto_corrected"`` 之一。
            若图片没有对应 JSON 文件，返回 ``"manual"``。

        异常:
            FileNotFoundError: 图片文件不存在。
            ValueError: ``tolerance_seconds`` < 0。
        """
        _require_existing_file(image_path, "image_path")
        if not isinstance(tolerance_seconds, (int, float)) or tolerance_seconds < 0:
            raise ValueError(
                f"tolerance_seconds 必须 >= 0，当前值: {tolerance_seconds}"
            )

        from scripts.common.annotation_type import AnnotationTypeChecker

        checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)
        return checker.check_image_annotation(image_path).value

    @staticmethod
    def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
        """
        扫描 runs 目录下的训练模型。

        参数:
            runs_dir: 训练结果根目录。

        返回:
            ``(显示名称, 权重路径)`` 列表，显示名称形如 ``训练名称-权重名``。
            目录不存在时返回空列表（与 GUI 调用习惯一致）。
        """
        _require_non_empty_str(runs_dir, "runs_dir")
        if not Path(runs_dir).expanduser().is_dir():
            return []

        from scripts.common.utils import discover_trained_models

        return discover_trained_models(runs_dir)


# ---------------------------------------------------------------------------
# TrainingAPI
# ---------------------------------------------------------------------------
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
    ) -> Dict[str, int]:
        """
        将 X-AnyLabeling 标注图片导出为 YOLO 数据集。

        参数:
            input_dir: 含图片与 X-AnyLabeling JSON 的源目录。
            output_dir: 数据集输出根目录，会自动创建子目录
                ``images/{train,test}`` 与 ``labels/{train,test}``，
                并写入 ``data.yaml``。
            task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``，
                默认 ``detect``。
            train_ratio: 训练集占比，必须位于 (0, 1)，默认 0.8。
            test_ratio: 测试集占比，必须位于 (0, 1)，默认 0.2。
                ``train_ratio + test_ratio`` 必须约等于 1。
            seed: 随机划分使用的随机种子，默认 42。
            copy_mode: 图片落盘策略，``copy`` / ``link`` / ``symlink``，
                默认 ``copy``。``link`` 会优先尝试硬链接，失败时回退为复制；
                ``symlink`` 会创建符号链接（在不支持的文件系统上可能出错）。

        返回:
            ``{"train": int, "test": int}``，各集合实际落盘的样本数量。

        异常:
            FileNotFoundError: ``input_dir`` 不存在或不是目录。
            ValueError: 参数非法（task / copy_mode 不在枚举内、划分比例越界等）。
            RuntimeError: 目录中未找到任何有效标注 / 可导出样本。
        """
        _require_existing_dir(input_dir, "input_dir")
        _require_non_empty_str(output_dir, "output_dir")

        task_norm = (task or "").lower()
        valid_tasks = {"detect", "obb", "segment", "classify"}
        if task_norm not in valid_tasks:
            raise ValueError(
                f"不支持的任务类型: {task!r}，仅支持 {sorted(valid_tasks)}"
            )

        copy_mode_norm = (copy_mode or "").lower()
        valid_copy_modes = {"copy", "link", "symlink"}
        if copy_mode_norm not in valid_copy_modes:
            raise ValueError(
                f"不支持的 copy_mode: {copy_mode!r}，"
                f"仅支持 {sorted(valid_copy_modes)}"
            )

        _require_in_range(
            train_ratio, "train_ratio", 0.0, 1.0,
            inclusive_lo=False, inclusive_hi=False,
        )
        _require_in_range(
            test_ratio, "test_ratio", 0.0, 1.0,
            inclusive_lo=False, inclusive_hi=False,
        )
        if abs((train_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + test_ratio 必须约等于 1，"
                f"当前为 {train_ratio + test_ratio}"
            )

        # 避免源 / 目标为同一路径，防止边读边写导致数据损坏。
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
        )

    @staticmethod
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
        """
        基于 Ultralytics YOLO 训练 detect / obb / segment / classify 模型。

        参数:
            dataset_yaml: 数据集配置：

                * 对 detect / obb / segment：YOLO ``data.yaml`` 文件路径；
                * 对 classify：分类数据集 ``data.yaml`` 文件路径
                  （其同级目录下需存在 ``images/`` 子目录）。
            task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``。
            model: 模型名称或权重路径。传入不含 ``.pt`` 后缀的名称
                （如 ``yolov8n``）时，会根据 ``task`` 自动补全合适的
                后缀（如 ``yolov8n-seg.pt``）。
            epochs: 训练总轮数，必须 >= 1，默认 100。
            imgsz: 训练输入图片尺寸（短边），必须 >= 32，默认 640。
            batch: 训练批大小，必须 >= 1，默认 16。
            device: 训练设备，例如 ``"0"`` / ``"cpu"`` / ``"0,1"``，
                ``None`` 表示由 Ultralytics 自动选择。
            project: 训练输出根目录，``None`` 时使用 Ultralytics 默认值
                （通常为 ``runs/<task>``）。
            name: 训练运行名称，``None`` 时由 Ultralytics 自动编号
                （``train``、``train2`` ...）。
            patience: 早停轮数，0 表示不启用早停，必须 >= 0，默认 100。
            resume: 是否从最近一次中断恢复训练，默认 False。
            optimizer: 优化器名称，可选值见
                :data:`scripts.common.config.SUPPORTED_OPTIMIZERS`，默认 ``auto``。
            lr0: 初始学习率，必须 > 0，默认 0.01。
            workers: DataLoader worker 数量，必须 >= 0，默认 8。

        返回:
            训练结果目录路径（``best.pt`` 所在目录的父目录），
            若 Ultralytics 未暴露 ``trainer.save_dir`` 则返回空字符串。

        异常:
            FileNotFoundError: ``dataset_yaml`` 不是文件。
            ValueError: 参数非法（task / optimizer 不在枚举内、数值越界等）。
            RuntimeError: 未安装 ``ultralytics`` 或训练过程抛出异常时。
        """
        _require_existing_file(dataset_yaml, "dataset_yaml")
        _require_non_empty_str(model, "model")

        task_norm = (task or "").lower()
        valid_tasks = {"detect", "obb", "segment", "classify"}
        if task_norm not in valid_tasks:
            raise ValueError(
                f"不支持的任务类型: {task!r}，仅支持 {sorted(valid_tasks)}"
            )

        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs 必须是 >=1 的整数，当前值: {epochs}")
        if not isinstance(imgsz, int) or imgsz < 32:
            raise ValueError(f"imgsz 必须是 >=32 的整数，当前值: {imgsz}")
        if not isinstance(batch, int) or batch < 1:
            raise ValueError(f"batch 必须是 >=1 的整数，当前值: {batch}")
        if not isinstance(patience, int) or patience < 0:
            raise ValueError(f"patience 必须是 >=0 的整数，当前值: {patience}")
        if not isinstance(workers, int) or workers < 0:
            raise ValueError(f"workers 必须是 >=0 的整数，当前值: {workers}")
        if (
                not isinstance(lr0, (int, float))
                or isinstance(lr0, bool)
                or lr0 <= 0
        ):
            raise ValueError(f"lr0 必须 > 0，当前值: {lr0}")

        # 懒读取支持的优化器集合，避免类定义阶段引入额外耦合。
        from scripts.common.config import SUPPORTED_OPTIMIZERS
        if optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"不支持的优化器: {optimizer!r}，"
                f"仅支持 {sorted(SUPPORTED_OPTIMIZERS)}"
            )

        from scripts.train.train import train_model

        return train_model(
            dataset_yaml=dataset_yaml,
            task=task_norm,
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
            workers=workers,
        )


# ---------------------------------------------------------------------------
# ImageAPI
# ---------------------------------------------------------------------------
class ImageAPI:
    """图片处理相关 API。"""

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
    ) -> dict:
        """
        对目录下相似图片进行去重。

        基于图片特征向量两两计算余弦相似度，按顺序保留首次出现的图片，
        将后续相似度 ``>= threshold`` 的图片视为重复。

        参数:
            folder: 待去重的图片目录（不递归子目录）。
            threshold: 相似度阈值，必须位于 (0, 1]，默认 0.95。
                值越高判定越严格（仅极相似的图片才会被认为是重复）。
            delete: 是否直接删除重复图片，默认 False。与 ``move_to`` 互斥。
            move_to: 将重复图片移动到的目录路径，``None`` 表示不移动。
                设置时会自动创建目录；与 ``delete`` 互斥。
            model_name: 当 ``backend="vit"`` 时使用的 HuggingFace 模型名称，
                默认 ``google/vit-base-patch16-224``。
            batch_size: ViT 推理批大小，必须 >= 1，默认 8。仅 ``vit`` 后端使用。
            backend: 特征提取后端：

                * ``"vit"``（默认）：基于 ViT / DINOv2 等深度特征，
                  精度高但需要安装 ``torch`` / ``transformers``，建议有 GPU；
                * ``"phash"``：感知哈希，仅依赖 ``Pillow`` / ``numpy``
                  （可选 ``scipy``），速度快、显存占用低，适合大批量粗筛
                  或纯 CPU 环境。
            hash_size: ``phash`` 后端的哈希尺寸，必须 >= 1，默认 16
                （对应 256 维向量）。仅 ``phash`` 后端使用。

        返回:
            ``{"keep": List[Path], "duplicates": List[Path]}``，
            分别为保留与重复的图片路径列表。

        异常:
            FileNotFoundError: ``folder`` 不存在或不是目录。
            ValueError: 参数非法（threshold / batch_size / hash_size 越界、
                backend 不在枚举内、``delete`` 与 ``move_to`` 同时设置等）。
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
        )
