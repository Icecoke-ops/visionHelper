#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Ultralytics YOLO 的模型训练工具。

该模块暴露以下核心方法：
    - train_model(dataset_yaml, task="detect", model="yolov8n",
                  epochs=100, imgsz=640, batch=16, device=None,
                  project=None, name=None)

支持任务类型：
    - ``detect``:   目标检测（使用 ``yolov8n.pt`` 等水平框模型）
    - ``obb``:      旋转框检测（使用 ``yolov8n-obb.pt`` 等 OBB 模型）
    - ``segment``:  实例分割（使用 ``yolov8n-seg.pt`` 等分割模型）
    - ``classify``: 图像分类（使用 ``yolov8n-cls.pt`` 等分类模型，
                    ``data`` 传分类数据集根目录而非 yaml）

用法示例：
    from scripts.train_model import train_model

    train_model(
        dataset_yaml="/path/to/.dataset/data.yaml",
        task="detect",
        model="yolov8n",
        epochs=100,
        imgsz=640,
        batch=16,
    )
"""

import argparse
from pathlib import Path
from typing import Optional, Set

# 支持的任务类型
SUPPORTED_TASKS: Set[str] = {"detect", "obb", "segment", "classify"}

# 任务类型对应的模型名后缀
_TASK_MODEL_SUFFIX = {
    "detect": "",
    "obb": "-obb",
    "segment": "-seg",
    "classify": "-cls",
}


def _resolve_model_name(model: str, task: str) -> str:
    """根据任务类型补全模型权重文件名。"""
    model = model.strip()
    if not model:
        raise ValueError("模型名称不能为空")

    # 如果用户已指定 .pt 文件，直接返回
    if model.endswith(".pt"):
        return model

    suffix = _TASK_MODEL_SUFFIX.get(task, "")
    if suffix and suffix not in model:
        model = f"{model}{suffix}"
    return f"{model}.pt"


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
        dataset_yaml: YOLO 数据集的 data.yaml 文件路径。对于 ``classify`` 任务，
                      训练实际使用的根目录为 ``data.yaml`` 同级的 ``images/`` 目录。
        task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``。
        model: 模型名称，例如 ``yolov8n``、``yolov8s``、``yolov8m`` 等。
        epochs: 训练轮数。
        imgsz: 输入图片尺寸。
        batch: 批大小。
        device: 训练设备，例如 ``0``、``cpu``、``0,1,2,3``。
                默认自动选择（ultralytics 内部逻辑）。
        project: 训练结果保存的父目录。
        name: 训练结果子目录名称。

    返回:
        训练结果目录路径（best.pt 所在目录）。

    异常:
        ValueError: 参数校验失败。
        RuntimeError: 训练过程中发生错误。
    """
    yaml_path = Path(dataset_yaml)
    if not yaml_path.is_file():
        raise ValueError(f"数据集配置文件不存在: {dataset_yaml}")

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}"
        )

    if epochs < 1:
        raise ValueError("epochs 必须大于 0")
    if imgsz < 32:
        raise ValueError("imgsz 必须大于等于 32")
    if batch < 1:
        raise ValueError("batch 必须大于 0")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "未安装 ultralytics，请先执行：pip install ultralytics"
        ) from exc

    model_name = _resolve_model_name(model, task)
    print(f"加载模型: {model_name}")
    yolo_model = YOLO(model_name)

    # classify 任务需要的 data 参数为分类数据集根目录（含 train/val 子目录），
    # 而 detect/obb/segment 接收 data.yaml 路径。
    if task == "classify":
        # 约定导出的分类数据集结构为 <output_dir>/images/{train,test}/<class>/
        # data.yaml 与 images/ 目录平级，因此根目录为 yaml 同级的 images/。
        classify_root = yaml_path.parent / "images"
        if not classify_root.is_dir():
            raise ValueError(
                f"分类数据集目录不存在: {classify_root}，请确认已使用 task=classify 导出数据集"
            )
        data_arg = str(classify_root.resolve())
    else:
        data_arg = str(yaml_path.resolve())

    train_kwargs = {
        "data": data_arg,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
    }
    if device is not None:
        train_kwargs["device"] = device
    if project is not None:
        train_kwargs["project"] = project
    if name is not None:
        train_kwargs["name"] = name

    print(f"开始训练 {task} 模型...")
    print(f"  data: {train_kwargs['data']}")
    print(f"  epochs: {epochs}, imgsz: {imgsz}, batch: {batch}")

    try:
        yolo_model.train(**train_kwargs)
    except Exception as exc:
        raise RuntimeError(f"训练过程出错: {exc}") from exc

    # Ultralytics 训练结果保存在 project/name 或 runs/<task>/name 下
    result_dir = Path(yolo_model.trainer.best.parent) if yolo_model.trainer else None
    if result_dir and result_dir.is_dir():
        print(f"训练完成，最佳权重: {result_dir / 'weights' / 'best.pt'}")
        return str(result_dir)

    print("训练完成")
    return ""


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="模型训练工具")
    parser.add_argument("dataset_yaml", type=str, help="YOLO 数据集 data.yaml 文件路径")
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help="模型名称，例如 yolov8n/yolov8s/yolov8m 等，默认 yolov8n",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数，默认 100",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸，默认 640",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批大小，默认 16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备，例如 0/cpu/0,1,2,3，默认自动选择",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="训练结果保存的父目录",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="训练结果子目录名称",
    )
    args = parser.parse_args()

    train_model(
        dataset_yaml=args.dataset_yaml,
        task=args.task,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
