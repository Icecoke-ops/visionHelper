#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper 对外统一 API 接口。

该模块将项目中的各个工具函数以类方法的形式暴露，方便上层调用与测试：
    - VideoAPI.extract_video_frames(...)
    - ImageAPI.deduplicate(...)

用法示例：
    from api import VideoAPI, ImageAPI

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

from typing import Dict, List, Optional, Tuple

from scripts.annotation_stats import (
    collect_annotation_label_stats as _collect_annotation_label_stats,
    collect_annotation_stats as _collect_annotation_stats,
)
from scripts.annotation_type import (
    AnnotationTypeChecker as _AnnotationTypeChecker,
)
from scripts.auto_annotate import auto_annotate as _auto_annotate
from scripts.auto_annotate import discover_trained_models as _discover_trained_models
from scripts.deduplicate_images import deduplicate as _deduplicate
from scripts.export_yolo_dataset import export_yolo_dataset as _export_yolo_dataset
from scripts.extract_video_frames import extract_video_frames as _extract_video_frames
from scripts.train_model import train_model as _train_model



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
    ) -> List[str]:
        """
        将视频抽帧并保存为图片。

        参数:
            input_video: 输入视频文件路径。
            output_dir: 输出图片保存目录。
            frame_step: 抽帧间隔，默认 1（逐帧抽取）。
            image_extension: 输出图片格式，默认 jpg。
            quality: 输出图片质量，默认 100（原始画质）。
            prefix: 输出文件名前缀，默认 "frame"。
            start_time: 开始抽取的时间（秒或时间字符串），默认从视频开头。
            end_time: 结束抽取的时间（秒或时间字符串），默认抽到视频结尾。

        返回:
            保存的图片路径列表（按帧顺序）。
        """
        return _extract_video_frames(
            input_video=input_video,
            output_dir=output_dir,
            frame_step=frame_step,
            image_extension=image_extension,
            quality=quality,
            prefix=prefix,
            start_time=start_time,
            end_time=end_time,
        )


class AnnotationAPI:
    """数据标注相关 API。"""

    @staticmethod
    def annotation_stats(folder: str) -> dict:
        """
        统计目录下的图片标注情况。

        参数:
            folder: 待统计的图片目录路径。

        返回:
            包含 total_images、annotated_images、unannotated_images、
            detection_images、obb_images、polygon_images、manual_images、
            auto_images、auto_corrected_images 的统计字典。
        """
        return _collect_annotation_stats(folder)

    @staticmethod
    def annotation_label_stats(folder: str) -> List[Dict[str, int]]:
        """
        按标签统计目录下的标注实例数量。

        参数:
            folder: 待统计的图片目录路径。

        返回:
            每个标签的实例数量列表，元素包含 label、detection_count、
            obb_count、polygon_count。
        """
        return _collect_annotation_label_stats(folder)

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
    ) -> dict:
        """
        使用 YOLO 模型对工作目录下指定状态的图片进行自动标注。

        参数:
            work_dir: 待标注图片所在的工作目录。
            model_path: YOLO 模型权重文件路径（.pt）。
            threshold: 置信度阈值，默认 0.25。
            task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``，
                默认 ``detect``。
            suffix: 输出 JSON 文件名后缀。
            device: 推理设备，例如 ``0``、``cpu``，默认自动选择。
            iou: NMS IoU 阈值，默认 0.45。
            include_unannotated: 是否处理未标注图片，默认 True。
            include_auto: 是否处理已自动标注图片（覆盖 / 合并），默认 False。
            include_auto_corrected: 是否处理自动标注后矫正过的图片，默认 False。
            include_manual: 是否处理手动标注图片，默认 False。
            tolerance_seconds: 自动标注时间戳容差（秒），用于区分自动 / 矫正
                / 手动状态，默认 2.0。

        返回:
            包含 total、skipped、annotated 的字典。
        """
        return _auto_annotate(
            work_dir=work_dir,
            model_path=model_path,
            threshold=threshold,
            task=task,
            suffix=suffix,
            device=device,
            iou=iou,
            include_unannotated=include_unannotated,
            include_auto=include_auto,
            include_auto_corrected=include_auto_corrected,
            include_manual=include_manual,
            tolerance_seconds=tolerance_seconds,
        )

    @staticmethod
    def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
        """
        扫描 runs 目录下的训练模型。

        参数:
            runs_dir: 训练结果根目录。

        返回:
            模型显示名称与模型文件路径列表，显示名称格式为
            ``训练名称-模型权重名称``。
        """
        return _discover_trained_models(runs_dir)

    @staticmethod
    def check_annotation_type(
            json_path: str,
            tolerance_seconds: float = 2.0,
    ) -> str:
        """
        判断单个 LabelMe JSON 标注文件的类型。

        参数:
            json_path: LabelMe 标注文件路径。
            tolerance_seconds: 判定为自动标注的最大时间差阈值，单位秒，
                默认 2.0。

        返回:
            标注类型字符串：``manual``（手动标注）、``auto``（自动标注）、
            ``auto_corrected``（自动标注并手动矫正）。
        """
        checker = _AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)
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
            tolerance_seconds: 判定为自动标注的最大时间差阈值，单位秒，
                默认 2.0。

        返回:
            标注类型字符串：``manual``、``auto`` 或 ``auto_corrected``。
            若图片没有对应 JSON 文件，返回 ``manual``。
        """
        checker = _AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)
        return checker.check_image_annotation(image_path).value


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
    ) -> dict:
        """
        将 LabelMe 标注图片导出为 YOLO 数据集。

        数据集仅划分为训练集与测试集，不生成单独的验证集。
        ``classify`` 任务输出 ImageFolder 结构（``images/{train,test}/<class>/``），
        其它任务输出标准 ``images/`` + ``labels/`` 结构。

        参数:
            input_dir: 包含图片与 LabelMe JSON 标注文件的目录。
            output_dir: 导出的数据集目录。
            task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``。
            train_ratio: 训练集比例。
            test_ratio: 测试集比例。
            seed: 随机划分种子。

        返回:
            包含 train/test 数量的字典。
        """
        return _export_yolo_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            task=task,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            seed=seed,
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
    ) -> str:
        """
        使用 Ultralytics YOLO 训练目标检测/OBB/分割/分类模型。

        参数:
            dataset_yaml: YOLO 数据集 data.yaml 文件路径。
            task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``。
            model: 模型名称，例如 ``yolov8n``。
            epochs: 训练轮数。
            imgsz: 输入图片尺寸。
            batch: 批大小。
            device: 训练设备。
            project: 训练结果保存的父目录。
            name: 训练结果子目录名称。

        返回:
            训练结果目录路径。
        """
        return _train_model(
            dataset_yaml=dataset_yaml,
            task=task,
            model=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
        )


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
    ) -> dict:
        """
        对文件夹中的相似图片进行去重。

        参数:
            folder: 待处理图片所在的文件夹路径。
            threshold: 相似度阈值，默认 0.95。
            delete: 是否删除重复图片，默认 False。
            move_to: 将重复图片移动到指定目录。与 delete 互斥。
            model_name: 使用的 ViT 模型名称。
            batch_size: 特征提取的批大小。

        返回:
            包含 keep（保留图片路径列表）和 duplicates（重复图片路径列表）的字典。
        """
        return _deduplicate(
            folder=folder,
            threshold=threshold,
            delete=delete,
            move_to=move_to,
            model_name=model_name,
            batch_size=batch_size,
        )
