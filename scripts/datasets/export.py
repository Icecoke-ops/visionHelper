#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 X-AnyLabeling 标注图片导出为 YOLO 格式数据集。

支持任务类型：detect / obb / segment / classify。本模块同时包含核心实现与
``python scripts/vh.py datasets export`` 命令行入口。

用法::

    python scripts/vh.py datasets export -i ./images -o ./dataset -t detect -R 0.8 --test-ratio 0.2 -S 42 -C link
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image as PilImage

from scripts.common.config import (
    COPY_MODES,
    DETECT_SHAPE_TYPES,
    IMAGES_FOLDER,
    LABELS_FOLDER,
    OBB_SHAPE_TYPES,
    SEGMENT_SHAPE_TYPES,
    SUPPORTED_TASKS,
)
from scripts.common.logging import ProgressLogger, log
from scripts.common.utils import (
    is_annotation_file as _is_annotation_file,
    is_image_file as _is_image_file,
    load_annotation as _load_annotation,
    resolve_image_path as _resolve_image_path,
    validate_split_ratios,
)

__all__ = [
    "export_yolo_dataset",
    "main",
]

def _place_image(src: Path, dst: Path, mode: str) -> None:
    """根据 ``mode`` 将源图片放置到目标位置。"""
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except OSError:
            pass

    if mode == "copy":
        shutil.copy2(str(src), str(dst))
        return

    if mode == "symlink":
        os.symlink(os.path.abspath(str(src)), str(dst))
        return

    try:
        os.link(str(src), str(dst))
    except OSError:
        log("[警告] 无法创建硬链接，已回退到复制")
        shutil.copy2(str(src), str(dst))


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
    """将 X-AnyLabeling rectangle（左上角、右下角）转为顺时针四个角点。"""
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
    """从 X-AnyLabeling JSON 顶层 ``flags`` 中解析单标签分类的类别名。"""
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
    """遍历标注文件收集所有标签名称。"""
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
    rng = random.Random(seed)
    rng.shuffle(indices)

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
        copy_mode: str,
) -> Dict[str, int]:
    """分类任务的导出逻辑（ImageFolder 结构）。"""
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

    used_classes = {cls_name for _img, cls_name in classify_samples}
    splits_root = {
        "train": output_path / IMAGES_FOLDER / "train",
        "test": output_path / IMAGES_FOLDER / "test",
    }
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
    for split_dir in splits_root.values():
        for cls_name in used_classes:
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
    progress = ProgressLogger(
        total=len(classify_samples), desc="导出 YOLO 分类数据集"
    )
    for i, (image_path, cls_name) in enumerate(classify_samples):
        split = split_map[i]
        dst_image = splits_root[split] / cls_name / image_path.name
        try:
            _place_image(image_path, dst_image, copy_mode)
        except Exception as exc:
            log(f"[警告] 落盘图片失败（mode={copy_mode}）{image_path}: {exc}")
            progress.update(1)
            continue
        counts[split] += 1
        progress.update(1)
    progress.close()

    yaml_path = output_path / "data.yaml"
    used_class_names = [name for name in class_names if name in used_classes]
    yaml_content = (
        f"path: {(output_path / IMAGES_FOLDER).resolve()}\n"
        f"train: train\n"
        f"val: test\n"
        f"test: test\n\n"
        f"nc: {len(used_class_names)}\n"
        f"names: {used_class_names}\n"
    )
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)

    log(
        f"YOLO 数据集导出完成（task=classify, mode={copy_mode}）：\n"
        f"  训练集: {counts['train']}\n"
        f"  测试集: {counts['test']}\n"
        f"  类别数: {len(used_class_names)}\n"
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
        copy_mode: str = "copy",
        export_empty_labels: bool = False,
        export_unlabeled: bool = False,
) -> Dict[str, int]:
    """将工作目录下已标注的图片导出为 YOLO 数据集。"""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"输入目录不存在: {input_dir}")  # NOTE: duplicated in _validate_args; kept for API safety

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}"
        )

    copy_mode = copy_mode.lower()
    if copy_mode not in COPY_MODES:
        raise ValueError(
            f"不支持的 copy_mode: {copy_mode}，仅支持 {sorted(COPY_MODES)}"
        )

    validate_split_ratios(train_ratio, test_ratio)

    labels = _collect_labels(input_path, task)
    if not labels:
        if not export_empty_labels and not export_unlabeled:
            raise RuntimeError(f"目录下未找到任何有效标注: {input_dir}")

    class_map = {label: idx for idx, label in enumerate(sorted(labels))}

    annotated_samples: List[Tuple[Path, Optional[Path], dict]] = []
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

    if export_unlabeled:
        covered_stems = {img_path.stem for img_path, _, _ in annotated_samples}
        for f in sorted(input_path.iterdir(), key=lambda p: p.name):
            if _is_image_file(f) and f.stem not in covered_stems:
                annotated_samples.append((f, None, {}))

    if not annotated_samples:
        raise RuntimeError(f"未找到任何与图片匹配的有效标注: {input_dir}")

    annotated_samples.sort(key=lambda item: item[0].name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if task == "classify":
        return _export_classify(
            input_path=input_path,
            output_path=output_path,
            annotated_samples=annotated_samples,
            class_map=class_map,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            seed=seed,
            copy_mode=copy_mode,
        )

    splits = {
        "train": output_path / IMAGES_FOLDER / "train",
        "test": output_path / IMAGES_FOLDER / "test",
    }
    label_splits = {
        "train": output_path / LABELS_FOLDER / "train",
        "test": output_path / LABELS_FOLDER / "test",
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
    skipped_empty = 0
    if task == "detect":
        converter = _shape_to_detect_label
    elif task == "obb":
        converter = _shape_to_obb_label
    else:  # segment
        converter = _shape_to_segment_label

    progress = ProgressLogger(
        total=len(annotated_samples), desc="导出 YOLO 数据集"
    )
    for i, (image_path, ann_path, data) in enumerate(annotated_samples):
        split = split_map[i]
        img_w = float(data.get("imageWidth", 0))
        img_h = float(data.get("imageHeight", 0))

        if not (img_w and img_h):
            try:
                with PilImage.open(image_path) as pil_img:
                    img_w, img_h = pil_img.size
            except (OSError, ValueError) as exc:
                log(f"[警告] 无法读取图片尺寸 {image_path}: {exc}")
                progress.update(1)
                continue

        label_lines: List[str] = []
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            line = converter(shape, class_map, img_w, img_h)
            if line:
                label_lines.append(line)

        if not label_lines and not export_empty_labels:
            skipped_empty += 1
            progress.update(1)
            continue

        dst_image = splits[split] / image_path.name
        dst_label = label_splits[split] / f"{image_path.stem}.txt"

        try:
            _place_image(image_path, dst_image, copy_mode)
        except OSError as exc:
            log(f"[警告] 落盘图片失败（mode={copy_mode}）{image_path}: {exc}")
            progress.update(1)
            continue
        with dst_label.open("w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

        counts[split] += 1
        progress.update(1)
    progress.close()

    yaml_path = output_path / "data.yaml"
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
    yaml_content = (
        f"path: {output_path.resolve()}\n"
        f"train: {IMAGES_FOLDER}/train\n"
        f"val: {IMAGES_FOLDER}/test\n"
        f"test: {IMAGES_FOLDER}/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)

    log(
        f"YOLO 数据集导出完成（task={task}, mode={copy_mode}）：\n"
        f"  训练集: {counts['train']}\n"
        f"  测试集: {counts['test']}\n"
        f"  类别数: {len(class_names)}\n"
        f"  跳过空标签样本: {skipped_empty}\n"
        f"  保存路径: {output_path}"
    )
    return counts


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py datasets export",
        description=(
            "将 X-AnyLabeling JSON 标注的图片目录导出为 YOLO 数据集。"
            "支持 detect / obb / segment / classify 4 种任务。"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="包含图片与 X-AnyLabeling JSON 的目录",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="输出的数据集目录（不存在会自动创建）",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "-R", "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（0~1），默认 0.8",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集比例（0~1），默认 0.2；train-ratio + test-ratio 须约等于 1。",
    )
    parser.add_argument(
        "-S", "--seed",
        type=int,
        default=42,
        help="随机划分种子，默认 42",
    )
    parser.add_argument(
        "-C", "--copy-mode",
        type=str,
        default="copy",
        choices=sorted(COPY_MODES),
        help=(
            "图片落盘方式：copy=复制（默认）；link=硬链接（同一文件系统）；"
            "symlink=软链接（跨文件系统也可，需目标文件系统支持）。"
        ),
    )
    parser.add_argument(
        "--export-empty-labels",
        action="store_true",
        help="导出空标签：即使图片没有标注对象也将其纳入数据集（标签文件为空）",
    )
    parser.add_argument(
        "--export-unlabeled",
        action="store_true",
        help="导出未标注图片：将没有对应 JSON 标注文件的图片也纳入数据集（标签文件为空）",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在：{args.input}")
    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录：{args.input}")

    output_dir = Path(args.output)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"输出路径已存在但不是目录：{args.output}")

    try:
        if output_dir.exists() and input_dir.resolve() == output_dir.resolve():
            raise ValueError("输入目录与输出目录不能相同，否则会破坏原始数据。")
    except OSError:
        log("[警告] 路径解析失败")

    try:
        validate_split_ratios(args.train_ratio, args.test_ratio)
    except ValueError as exc:
        raise ValueError(
            f"--train-ratio / --test-ratio 非法：{exc}"
        ) from exc


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回:
        进程退出码：0=成功；2=参数非法；1=运行时错误；130=用户中断。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[参数错误] {exc}", stream=sys.stderr)
        return 2

    try:
        result = export_yolo_dataset(
            input_dir=args.input,
            output_dir=args.output,
            task=args.task,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            copy_mode=args.copy_mode,
            export_empty_labels=args.export_empty_labels,
            export_unlabeled=args.export_unlabeled,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，输出目录可能处于不完整状态。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except OSError as exc:
        log(
            f"[文件系统错误] {exc}\n"
            f"提示：若使用 --copy-mode link/symlink 失败，可尝试 --copy-mode copy。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 导出 YOLO 数据集失败：{exc}", stream=sys.stderr)
        return 1

    if isinstance(result, dict):
        train_n = result.get("train", 0)
        test_n = result.get("test", 0)
        log(f"[完成] 训练集 {train_n} 张，测试集 {test_n} 张 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
