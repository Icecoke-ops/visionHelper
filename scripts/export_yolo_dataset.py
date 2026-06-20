#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LabelMe 标注图片导出为 YOLO 格式数据集。

该模块暴露以下核心方法：
    - export_yolo_dataset(input_dir, output_dir, task="detect",
                          train_ratio=0.8, test_ratio=0.2, seed=42)

支持任务类型：
    - ``detect``: 目标检测（rectangle / polygon / rotation 均转为水平外接矩形，
      格式为 ``class cx cy w h``）
    - ``obb``:  旋转框（rectangle 转为四个角点，rotation 使用标注的四个角点，
      格式为 ``class x1 y1 x2 y2 x3 y3 x4 y4``）
    - ``segment``: 实例分割（polygon 转为归一化多边形，
      格式为 ``class x1 y1 x2 y2 ... xn yn``）
    - ``classify``: 图像分类（仅支持单标签，从 LabelMe JSON 顶层 ``flags``
      中取第一个值为 ``True`` 的 key 作为类别），输出 ImageFolder 结构：
      ``output_dir/images/{train,test}/<class>/<image>``

数据集仅划分为训练集与测试集，不生成单独的验证集；生成的 ``data.yaml`` 中
``val`` 指向测试集，以满足 Ultralytics YOLO 训练时的校验需求。

用法示例：
    from export_yolo_dataset import export_yolo_dataset

    export_yolo_dataset(
        input_dir="/path/to/annotated_images",
        output_dir="/path/to/.dataset",
        task="detect",
        train_ratio=0.8,
        test_ratio=0.2,
    )
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from scripts._common import (
    is_annotation_file as _is_annotation_file,
    is_image_file as _is_image_file,
    load_annotation as _load_annotation,
    resolve_image_path as _resolve_image_path,
)

# YOLO 数据集中图片子目录名称
_IMAGES_FOLDER = "images"

# 支持的任务类型
SUPPORTED_TASKS: Set[str] = {"detect", "obb", "segment", "classify"}

# LabelMe 中支持的 shape_type
DETECT_SHAPE_TYPES: Set[str] = {"rectangle", "polygon", "rotation"}
OBB_SHAPE_TYPES: Set[str] = {"rectangle", "rotation"}
SEGMENT_SHAPE_TYPES: Set[str] = {"polygon"}


def _points_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将点集转换为水平外接矩形 ``(cx, cy, w, h)``。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy, w, h


def _rectangle_to_obb(points: List[List[float]]) -> List[float]:
    """将 LabelMe rectangle（左上角、右下角）转为顺时针四个角点。"""
    (x1, y1), (x2, y2) = points
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _normalize_detect(
        cx: float, cy: float, w: float, h: float, img_w: float, img_h: float
) -> Tuple[float, float, float, float]:
    """将水平框归一化到 ``[0, 1]``。"""
    return (
        max(0.0, min(1.0, cx / img_w)),
        max(0.0, min(1.0, cy / img_h)),
        max(0.0, min(1.0, w / img_w)),
        max(0.0, min(1.0, h / img_h)),
    )


def _normalize_obb(points: List[float], img_w: float, img_h: float) -> List[float]:
    """将旋转框四个角点归一化到 ``[0, 1]``。"""
    normalized = []
    for i, coord in enumerate(points):
        if i % 2 == 0:
            normalized.append(max(0.0, min(1.0, coord / img_w)))
        else:
            normalized.append(max(0.0, min(1.0, coord / img_h)))
    return normalized


def _shape_to_detect_label(
        shape: dict, class_map: Dict[str, int], img_w: float, img_h: float
) -> Optional[str]:
    """将单个 shape 转换为目标检测 YOLO 行。"""
    shape_type = shape.get("shape_type")
    if shape_type not in DETECT_SHAPE_TYPES:
        return None

    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None

    try:
        points = [[float(p[0]), float(p[1])] for p in points]
    except (TypeError, ValueError, IndexError):
        return None

    label = shape.get("label")
    if not isinstance(label, str) or label not in class_map:
        return None

    cx, cy, w, h = _points_to_bbox(points)
    if w <= 0 or h <= 0:
        return None

    cx, cy, w, h = _normalize_detect(cx, cy, w, h, img_w, img_h)
    if w <= 0 or h <= 0:
        return None

    return f"{class_map[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _shape_to_obb_label(
        shape: dict, class_map: Dict[str, int], img_w: float, img_h: float
) -> Optional[str]:
    """将单个 shape 转换为 OBB YOLO 行。"""
    shape_type = shape.get("shape_type")
    if shape_type not in OBB_SHAPE_TYPES:
        return None

    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None

    try:
        points = [[float(p[0]), float(p[1])] for p in points]
    except (TypeError, ValueError, IndexError):
        return None

    label = shape.get("label")
    if not isinstance(label, str) or label not in class_map:
        return None

    if shape_type == "rectangle":
        obb_points = _rectangle_to_obb(points)
    else:
        if len(points) != 4:
            return None
        obb_points = [coord for pt in points for coord in pt]

    obb_points = _normalize_obb(obb_points, img_w, img_h)
    if not obb_points:
        return None

    return f"{class_map[label]} " + " ".join(f"{v:.6f}" for v in obb_points)


def _shape_to_segment_label(
        shape: dict, class_map: Dict[str, int], img_w: float, img_h: float
) -> Optional[str]:
    """将单个 shape 转换为实例分割 YOLO 行（归一化多边形）。"""
    shape_type = shape.get("shape_type")
    if shape_type not in SEGMENT_SHAPE_TYPES:
        return None

    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 3:
        return None

    try:
        points = [[float(p[0]), float(p[1])] for p in points]
    except (TypeError, ValueError, IndexError):
        return None

    label = shape.get("label")
    if not isinstance(label, str) or label not in class_map:
        return None

    if img_w <= 0 or img_h <= 0:
        return None

    flat: List[float] = []
    for x, y in points:
        nx = max(0.0, min(1.0, x / img_w))
        ny = max(0.0, min(1.0, y / img_h))
        flat.append(nx)
        flat.append(ny)

    return f"{class_map[label]} " + " ".join(f"{v:.6f}" for v in flat)


def _resolve_classification_label(
        annotation: dict, class_map: Dict[str, int]
) -> Optional[str]:
    """从 LabelMe JSON 顶层 ``flags`` 中解析单标签分类的类别名。

    规则：
        - 仅取 ``flags`` 字典中第一个值为 ``True`` 的 key（按插入顺序）；
        - 若没有 True 项或类别不在 ``class_map`` 中，则返回 ``None``。
    """
    flags = annotation.get("flags")
    if not isinstance(flags, dict):
        return None

    for key, value in flags.items():
        if not isinstance(key, str):
            continue
        if not key:
            continue
        if bool(value) and key in class_map:
            return key
    return None


def _collect_labels(input_dir: Path, task: str) -> Set[str]:
    """遍历标注文件收集所有标签名称。

    - 对 ``classify`` 任务：从 JSON 顶层 ``flags`` 的所有 key 中收集；
    - 其它任务：从 ``shapes[*].label`` 中收集。
    """
    labels: Set[str] = set()
    for path in input_dir.iterdir():
        if not _is_annotation_file(path):
            continue
        data = _load_annotation(path)
        if data is None:
            continue

        if task == "classify":
            flags = data.get("flags")
            if isinstance(flags, dict):
                for key in flags.keys():
                    if isinstance(key, str) and key:
                        labels.add(key)
        else:
            for shape in data.get("shapes", []):
                if isinstance(shape, dict):
                    label = shape.get("label")
                    if isinstance(label, str) and label:
                        labels.add(label)
    return labels


def _split_indices(
        total: int, train_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    """按比例随机划分为训练集与测试集索引。"""
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(total * train_ratio)
    return indices[:train_end], indices[train_end:]


def _export_classify(
        input_path: Path,
        output_path: Path,
        annotated_samples: List[Tuple[Path, Path, dict]],
        class_map: Dict[str, int],
        train_ratio: float,
        test_ratio: float,
        seed: int,
) -> Dict[str, int]:
    """分类任务的导出逻辑（ImageFolder 结构）。"""
    # 仅保留能解析出有效类别的样本
    classify_samples: List[Tuple[Path, str]] = []
    for image_path, _ann_path, data in annotated_samples:
        cls_name = _resolve_classification_label(data, class_map)
        if cls_name is None:
            continue
        classify_samples.append((image_path, cls_name))

    if not classify_samples:
        raise RuntimeError(
            "未找到任何带有效分类标签（顶层 flags 中至少一个 True 的 key）的样本"
        )

    classify_samples.sort(key=lambda item: item[0].name)

    # 预创建每个类别的 train/test 子目录
    splits_root = {
        "train": output_path / _IMAGES_FOLDER / "train",
        "test": output_path / _IMAGES_FOLDER / "test",
    }
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
    for split_dir in splits_root.values():
        for cls_name in class_names:
            (split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = _split_indices(
        len(classify_samples), train_ratio, test_ratio, seed
    )
    split_map: Dict[int, str] = {}
    for idx in train_idx:
        split_map[idx] = "train"
    for idx in test_idx:
        split_map[idx] = "test"

    counts = {"train": 0, "test": 0}
    for i, (image_path, cls_name) in enumerate(
            tqdm(classify_samples, desc="导出 YOLO 分类数据集")
    ):
        split = split_map[i]
        dst_image = splits_root[split] / cls_name / image_path.name
        try:
            shutil.copy2(str(image_path), str(dst_image))
        except Exception as exc:
            print(f"[警告] 复制图片失败 {image_path}: {exc}")
            continue
        counts[split] += 1

    # 生成 data.yaml（分类任务）：path 指向 images/，train/val/test 用相对子目录
    yaml_path = output_path / "data.yaml"
    yaml_content = (
        f"path: {(output_path / _IMAGES_FOLDER).resolve()}\n"
        f"train: train\n"
        f"val: test\n"
        f"test: test\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(
        f"YOLO 数据集导出完成（task=classify）：\n"
        f"  训练集: {counts['train']}\n"
        f"  测试集: {counts['test']}\n"
        f"  类别数: {len(class_names)}\n"
        f"  保存路径: {output_path}"
    )
    return counts


def export_yolo_dataset(
        input_dir: str,
        output_dir: str,
        task: str = "detect",
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        seed: int = 42,
) -> Dict[str, int]:
    """
    将工作目录下已标注的图片导出为 YOLO 数据集。

    数据集仅划分为训练集与测试集，不生成单独的验证集；生成的 ``data.yaml`` 中
    ``val`` 指向测试集，以满足 Ultralytics YOLO 训练时的校验需求。

    参数:
        input_dir: 包含图片与 LabelMe JSON 标注文件的目录。
        output_dir: 导出的数据集目录，通常为 ``<work_dir>/.dataset``。
        task: 任务类型，支持 ``detect`` / ``obb`` / ``segment`` / ``classify``。
        train_ratio: 训练集比例。
        test_ratio: 测试集比例。
        seed: 划分随机种子。

    返回:
        包含 train/test 数量的字典。

    异常:
        ValueError: 参数校验失败或目录不存在。
        RuntimeError: 没有可用的标注数据或图片。
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"输入目录不存在: {input_dir}")

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}"
        )

    ratios = [train_ratio, test_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("划分比例必须大于等于 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须等于 1，当前为 {sum(ratios)}")

    labels = _collect_labels(input_path, task)
    if not labels:
        raise RuntimeError(f"目录下未找到任何有效标注: {input_dir}")

    class_map = {label: idx for idx, label in enumerate(sorted(labels))}

    # 收集所有存在对应图片的标注
    annotated_samples: List[Tuple[Path, Path, dict]] = []
    for ann_path in input_path.iterdir():
        if not _is_annotation_file(ann_path):
            continue
        data = _load_annotation(ann_path)
        if data is None:
            continue
        image_path = _resolve_image_path(ann_path, data)
        if image_path is None:
            continue
        annotated_samples.append((image_path, ann_path, data))

    if not annotated_samples:
        raise RuntimeError(f"未找到任何与图片匹配的有效标注: {input_dir}")

    annotated_samples.sort(key=lambda item: item[0].name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 分类任务走独立分支（ImageFolder 结构）
    if task == "classify":
        return _export_classify(
            input_path=input_path,
            output_path=output_path,
            annotated_samples=annotated_samples,
            class_map=class_map,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    # detect / obb / segment 共用 images + labels 结构
    splits = {
        "train": output_path / _IMAGES_FOLDER / "train",
        "test": output_path / _IMAGES_FOLDER / "test",
    }
    label_splits = {
        "train": output_path / "labels" / "train",
        "test": output_path / "labels" / "test",
    }
    for split_dir in list(splits.values()) + list(label_splits.values()):
        split_dir.mkdir(parents=True, exist_ok=True)

    train_idx, test_idx = _split_indices(
        len(annotated_samples), train_ratio, test_ratio, seed
    )

    split_map: Dict[int, str] = {}
    for idx in train_idx:
        split_map[idx] = "train"
    for idx in test_idx:
        split_map[idx] = "test"

    counts = {"train": 0, "test": 0}
    if task == "detect":
        converter = _shape_to_detect_label
    elif task == "obb":
        converter = _shape_to_obb_label
    else:  # segment
        converter = _shape_to_segment_label

    for i, (image_path, ann_path, data) in enumerate(
            tqdm(annotated_samples, desc="导出 YOLO 数据集")
    ):
        split = split_map[i]
        img_w = float(data.get("imageWidth") or 0)
        img_h = float(data.get("imageHeight") or 0)

        # 若 JSON 中未记录宽高，则尝试读取图片
        if not (img_w and img_h):
            try:
                from PIL import Image as PilImage

                with PilImage.open(image_path) as pil_img:
                    img_w, img_h = pil_img.size
            except Exception as exc:
                print(f"[警告] 无法读取图片尺寸 {image_path}: {exc}")
                continue

        label_lines: List[str] = []
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            line = converter(shape, class_map, img_w, img_h)
            if line:
                label_lines.append(line)

        dst_image = splits[split] / image_path.name
        dst_label = label_splits[split] / f"{image_path.stem}.txt"

        shutil.copy2(str(image_path), str(dst_image))
        with dst_label.open("w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

        counts[split] += 1

    # 生成 data.yaml
    yaml_path = output_path / "data.yaml"
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
    yaml_content = (
        f"path: {output_path.resolve()}\n"
        f"train: {_IMAGES_FOLDER}/train\n"
        f"val: {_IMAGES_FOLDER}/test\n"
        f"test: {_IMAGES_FOLDER}/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(
        f"YOLO 数据集导出完成（task={task}）：\n"
        f"  训练集: {counts['train']}\n"
        f"  测试集: {counts['test']}\n"
        f"  类别数: {len(class_names)}\n"
        f"  保存路径: {output_path}"
    )
    return counts


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="导出 LabelMe 标注为 YOLO 数据集")
    parser.add_argument("input_dir", type=str, help="包含图片与 LabelMe JSON 的目录")
    parser.add_argument("output_dir", type=str, help="输出的数据集目录")
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例，默认 0.8",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集比例，默认 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机划分种子，默认 42",
    )
    args = parser.parse_args()

    export_yolo_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task=args.task,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
