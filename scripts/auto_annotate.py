#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动标注工具。

使用 Ultralytics YOLO 模型对工作目录下的图片进行推理，将满足条件的结果
保存为 LabelMe JSON 格式：

- ``detect``  → 写入 ``shape_type="rectangle"`` 的 shapes
- ``obb``     → 写入 ``shape_type="rotation"`` 的 shapes（4 个角点）
- ``segment`` → 写入 ``shape_type="polygon"`` 的 shapes
- ``classify``→ 仅刷新 JSON 顶层 ``flags`` 字典（单标签：top1 类别为 ``True``，
                其余为 ``False``），不修改 shapes

自动标注生成的标注文件会额外携带 ``auto_annotated_time`` 字段。
处理范围筛选支持四类（未标注 / 自动标注 / 自动标注后矫正 / 手动标注）。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from scripts._common import (
    discover_trained_models as _discover_trained_models,
    is_image_file,
)
from scripts.annotation_type import AnnotationType, AnnotationTypeChecker

# 支持的任务类型
SUPPORTED_TASKS = {"detect", "obb", "segment", "classify"}

# 已有标注状态枚举（字符串）
_STATUS_UNANNOTATED = "unannotated"
_STATUS_MANUAL = "manual"
_STATUS_AUTO = "auto"
_STATUS_AUTO_CORRECTED = "auto_corrected"


def _find_model_class_names(model) -> List[str]:
    """从 Ultralytics YOLO 模型中提取类别名称列表。"""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


def _box_to_rectangle_points(xyxy: List[float]) -> List[List[float]]:
    """将 ``[x1, y1, x2, y2]`` 转换为 LabelMe rectangle 的两个角点。"""
    x1, y1, x2, y2 = xyxy
    return [[float(x1), float(y1)], [float(x2), float(y2)]]


def _obb_to_points(obb_xyxyxyxy) -> List[List[float]]:
    """将 OBB 四边形角点转换为 LabelMe rotation/polygon 的 points。"""
    points = []
    for pt in obb_xyxyxyxy:
        points.append([float(pt[0]), float(pt[1])])
    return points


def _mask_xy_to_points(mask_xy) -> List[List[float]]:
    """将 Ultralytics mask.xy 单个轮廓数组转换为 LabelMe polygon points。"""
    points: List[List[float]] = []
    for pt in mask_xy:
        points.append([float(pt[0]), float(pt[1])])
    return points


def _build_shape(
        label: str,
        points: List[List[float]],
        shape_type: str,
) -> dict:
    """构造单个 LabelMe shape 字典。"""
    return {
        "label": label,
        "points": points,
        "shape_type": shape_type,
        "group_id": None,
        "description": "",
        "difficult": False,
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def _get_image_size(image_path: Path) -> Tuple[int, int]:
    """读取图片宽高。"""
    with Image.open(image_path) as img:
        return img.size


def _build_annotation(
        image_path: Path,
        shapes: List[dict],
        flags: Optional[Dict[str, bool]] = None,
        existing: Optional[dict] = None,
) -> dict:
    """构造完整 LabelMe JSON 标注字典，并记录自动标注时间。

    若提供 ``existing``，则在其基础上覆盖 ``shapes`` / ``flags`` /
    ``auto_annotated_time``，其余字段保留原值（缺失时再补全）。
    """
    width, height = _get_image_size(image_path)
    auto_annotated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base = dict(existing) if isinstance(existing, dict) else {}
    base["version"] = base.get("version", "5.0.1")
    base["flags"] = flags if flags is not None else base.get("flags", {})
    base["auto_annotated_time"] = auto_annotated_time
    base["shapes"] = shapes
    base["imagePath"] = base.get("imagePath", image_path.name) or image_path.name
    if "imageData" not in base:
        base["imageData"] = None
    base["imageHeight"] = base.get("imageHeight") or height
    base["imageWidth"] = base.get("imageWidth") or width
    if "description" not in base:
        base["description"] = ""
    return base


def _save_annotation(annotation: dict, output_path: Path) -> None:
    """将标注字典安全写入 JSON 文件。"""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


def _load_existing_annotation(json_path: Path) -> Optional[dict]:
    """安全加载已有 JSON，失败返回 None。"""
    if not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None


def _classify_existing_status(
        json_path: Path,
        type_checker: AnnotationTypeChecker,
) -> Tuple[str, Optional[dict]]:
    """根据已有 JSON 文件判断现有标注类型。"""
    data = _load_existing_annotation(json_path)
    if data is None:
        return _STATUS_UNANNOTATED, None

    ann_type = type_checker.check(data, json_mtime=json_path.stat().st_mtime)
    if ann_type == AnnotationType.MANUAL:
        return _STATUS_MANUAL, data
    if ann_type == AnnotationType.AUTO:
        return _STATUS_AUTO, data
    if ann_type == AnnotationType.AUTO_CORRECTED:
        return _STATUS_AUTO_CORRECTED, data
    return _STATUS_UNANNOTATED, data


def _build_classify_flags(
        class_names: List[str], pred_label: Optional[str]
) -> Dict[str, bool]:
    """构造分类任务的顶层 ``flags`` 字典（单标签）。"""
    flags: Dict[str, bool] = {}
    for name in class_names:
        flags[name] = (name == pred_label) if pred_label is not None else False
    return flags


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
) -> Dict[str, object]:
    """
    对工作目录下的图片进行自动标注。

    根据 ``include_*`` 四个开关决定要处理的图片范围：

        - ``include_unannotated``：处理无 JSON 标注的图片
        - ``include_auto``：处理 JSON 标记为自动标注的图片
        - ``include_auto_corrected``：处理 JSON 标记为自动标注后人工矫正的图片
        - ``include_manual``：处理 JSON 标记为手动标注的图片

    根据 ``task`` 不同，写入 / 合并 JSON 的策略：

        - ``detect`` / ``obb`` / ``segment``：覆盖 JSON 的 ``shapes``，保留已有
          ``flags``（不破坏分类信息）。
        - ``classify``：仅刷新顶层 ``flags``（单标签），保留已有 ``shapes``。

    分类任务采用单标签策略：top1 类别置 ``True``，其余 ``False``；JSON 中如有
    多个 ``True`` 也只视为单标签。

    参数:
        work_dir: 待标注图片所在目录。
        model_path: YOLO 模型权重文件路径（``.pt``）。
        threshold: 置信度阈值，默认 0.25。仅 detect/obb/segment 生效。
        task: 任务类型，``detect`` / ``obb`` / ``segment`` / ``classify``。
        suffix: 输出 JSON 文件名后缀，默认空。
        device: 推理设备，例如 ``0``、``cpu``，默认自动选择。
        iou: NMS IoU 阈值，默认 0.45。仅 detect/obb/segment 生效。
        include_unannotated: 是否处理未标注图片，默认 True。
        include_auto: 是否处理自动标注图片，默认 False。
        include_auto_corrected: 是否处理自动标注后矫正图片，默认 False。
        include_manual: 是否处理手动标注图片，默认 False。
        tolerance_seconds: 判定自动 / 矫正的时间容差，默认 2.0 秒。

    返回:
        ``{"total", "skipped", "annotated", "by_type"}`` 字典。

    异常:
        ValueError: 参数校验失败或目录/模型不存在。
        RuntimeError: 推理过程中发生错误。
    """
    work_path = Path(work_dir)
    if not work_path.is_dir():
        raise ValueError(f"工作目录不存在: {work_dir}")

    model_file = Path(model_path)
    if not model_file.is_file():
        raise ValueError(f"模型文件不存在: {model_path}")

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}"
        )

    if not (0 < threshold <= 1):
        raise ValueError("threshold 必须在 (0, 1] 范围内")
    if not (0 < iou <= 1):
        raise ValueError("iou 必须在 (0, 1] 范围内")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds 必须 >= 0")

    if not any([include_unannotated, include_auto, include_auto_corrected, include_manual]):
        raise ValueError("至少需要选择一种处理范围")

    include_map = {
        _STATUS_UNANNOTATED: include_unannotated,
        _STATUS_AUTO: include_auto,
        _STATUS_AUTO_CORRECTED: include_auto_corrected,
        _STATUS_MANUAL: include_manual,
    }

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先执行：pip install ultralytics") from exc

    yolo_model = YOLO(str(model_file))
    class_names = _find_model_class_names(yolo_model)

    type_checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)

    images = [p for p in work_path.iterdir() if is_image_file(p)]
    images.sort(key=lambda p: p.name)

    total = len(images)
    skipped = 0
    annotated = 0
    by_type: Dict[str, int] = {
        _STATUS_UNANNOTATED: 0,
        _STATUS_AUTO: 0,
        _STATUS_AUTO_CORRECTED: 0,
        _STATUS_MANUAL: 0,
    }

    for image_path in tqdm(images, desc="自动标注"):
        json_name = f"{image_path.stem}{suffix}.json"
        json_path = work_path / json_name

        status, existing = _classify_existing_status(json_path, type_checker)
        if not include_map.get(status, False):
            skipped += 1
            continue

        try:
            predict_kwargs = {
                "source": str(image_path),
                "device": device,
                "verbose": False,
            }
            if task != "classify":
                # classify 任务不使用 conf / iou
                predict_kwargs["conf"] = threshold
                predict_kwargs["iou"] = iou
            results = yolo_model.predict(**predict_kwargs)
        except Exception as exc:
            print(f"[错误] 推理失败 {image_path.name}: {exc}")
            skipped += 1
            continue

        if not results:
            skipped += 1
            continue

        result = results[0]
        wrote = False

        if task == "obb":
            shapes: List[dict] = []
            boxes = getattr(result, "obb", None)
            if boxes is not None and len(boxes) > 0:
                for cls_id, _conf, xyxyxyxy in zip(
                        boxes.cls.tolist(),
                        boxes.conf.tolist(),
                        boxes.xyxyxyxy.tolist(),
                ):
                    label = class_names[int(cls_id)] if class_names else str(int(cls_id))
                    points = _obb_to_points(xyxyxyxy)
                    shapes.append(_build_shape(label, points, "rotation"))
            if shapes:
                # 保留已有 flags（分类信息），仅替换 shapes
                preserved_flags = (
                    existing.get("flags", {}) if isinstance(existing, dict) else {}
                )
                annotation = _build_annotation(
                    image_path, shapes, flags=preserved_flags, existing=existing
                )
                _save_annotation(annotation, json_path)
                wrote = True

        elif task == "segment":
            shapes = []
            masks = getattr(result, "masks", None)
            boxes = getattr(result, "boxes", None)
            if (
                    masks is not None
                    and getattr(masks, "xy", None) is not None
                    and len(masks.xy) > 0
            ):
                cls_list = (
                    boxes.cls.tolist()
                    if (boxes is not None and getattr(boxes, "cls", None) is not None)
                    else [0] * len(masks.xy)
                )
                for cls_id, polygon in zip(cls_list, masks.xy):
                    if polygon is None or len(polygon) < 3:
                        continue
                    label = class_names[int(cls_id)] if class_names else str(int(cls_id))
                    points = _mask_xy_to_points(polygon)
                    shapes.append(_build_shape(label, points, "polygon"))
            if shapes:
                preserved_flags = (
                    existing.get("flags", {}) if isinstance(existing, dict) else {}
                )
                annotation = _build_annotation(
                    image_path, shapes, flags=preserved_flags, existing=existing
                )
                _save_annotation(annotation, json_path)
                wrote = True

        elif task == "classify":
            probs = getattr(result, "probs", None)
            pred_label: Optional[str] = None
            if probs is not None and class_names:
                top1 = getattr(probs, "top1", None)
                if top1 is not None:
                    try:
                        idx = int(top1)
                        if 0 <= idx < len(class_names):
                            pred_label = class_names[idx]
                    except (TypeError, ValueError):
                        pred_label = None
            new_flags = _build_classify_flags(class_names, pred_label)
            preserved_shapes = (
                existing.get("shapes", []) if isinstance(existing, dict) else []
            )
            if not isinstance(preserved_shapes, list):
                preserved_shapes = []
            annotation = _build_annotation(
                image_path,
                preserved_shapes,
                flags=new_flags,
                existing=existing,
            )
            _save_annotation(annotation, json_path)
            wrote = True

        else:  # task == "detect"
            shapes = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                for cls_id, _conf, xyxy in zip(
                        boxes.cls.tolist(),
                        boxes.conf.tolist(),
                        boxes.xyxy.tolist(),
                ):
                    label = class_names[int(cls_id)] if class_names else str(int(cls_id))
                    points = _box_to_rectangle_points(xyxy)
                    shapes.append(_build_shape(label, points, "rectangle"))
            if shapes:
                preserved_flags = (
                    existing.get("flags", {}) if isinstance(existing, dict) else {}
                )
                annotation = _build_annotation(
                    image_path, shapes, flags=preserved_flags, existing=existing
                )
                _save_annotation(annotation, json_path)
                wrote = True

        if wrote:
            annotated += 1
            by_type[status] = by_type.get(status, 0) + 1
        else:
            skipped += 1

    print(
        f"自动标注完成（task={task}）：\n"
        f"  图片总数: {total}\n"
        f"  跳过: {skipped}\n"
        f"  实际标注: {annotated}\n"
        f"  按类型: 未标注 {by_type[_STATUS_UNANNOTATED]}, "
        f"自动 {by_type[_STATUS_AUTO]}, "
        f"矫正 {by_type[_STATUS_AUTO_CORRECTED]}, "
        f"手动 {by_type[_STATUS_MANUAL]}"
    )
    return {
        "total": total,
        "skipped": skipped,
        "annotated": annotated,
        "by_type": by_type,
    }


def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
    """
    扫描 runs 目录下的训练模型。

    实现已下沉到 :func:`scripts._common.discover_trained_models`，本函数
    保留为向后兼容的薄包装，便于其它模块继续从 ``scripts.auto_annotate``
    导入。

    参数:
        runs_dir: 训练结果根目录，Ultralytics 默认结构为
            ``runs/<train_name>/weights/<name>.pt``。

    返回:
        模型显示名称与模型文件路径的列表。
    """
    return _discover_trained_models(runs_dir)


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="自动标注工具")
    parser.add_argument("work_dir", type=str, help="待标注图片所在的工作目录")
    parser.add_argument("model_path", type=str, help="YOLO 模型权重文件路径（.pt）")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="置信度阈值，默认 0.25（仅 detect/obb/segment 生效）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="输出 JSON 文件名后缀，默认空",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，例如 0/cpu，默认自动选择",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值，默认 0.45（仅 detect/obb/segment 生效）",
    )
    parser.add_argument(
        "--include-unannotated",
        action="store_true",
        help="处理未标注图片",
    )
    parser.add_argument(
        "--include-auto",
        action="store_true",
        help="处理已被自动标注的图片（重新生成）",
    )
    parser.add_argument(
        "--include-auto-corrected",
        action="store_true",
        help="处理自动标注并手动矫正的图片",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="处理手动标注的图片",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=2.0,
        help="判定自动 / 矫正的时间容差，默认 2.0 秒",
    )
    args = parser.parse_args()

    # 若 4 个 include 开关都未指定，向后兼容默认仅处理未标注图片
    include_unannotated = args.include_unannotated
    include_auto = args.include_auto
    include_auto_corrected = args.include_auto_corrected
    include_manual = args.include_manual
    if not any([include_unannotated, include_auto, include_auto_corrected, include_manual]):
        include_unannotated = True

    auto_annotate(
        work_dir=args.work_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        task=args.task,
        suffix=args.suffix,
        device=args.device,
        iou=args.iou,
        include_unannotated=include_unannotated,
        include_auto=include_auto,
        include_auto_corrected=include_auto_corrected,
        include_manual=include_manual,
        tolerance_seconds=args.tolerance_seconds,
    )


if __name__ == "__main__":
    main()
