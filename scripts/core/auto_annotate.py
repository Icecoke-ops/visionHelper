#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动标注核心实现。

使用 Ultralytics YOLO 模型对工作目录下的图片进行推理，将满足条件的结果
保存为 X-AnyLabeling JSON 格式（兼容 LabelMe）：

- ``detect``  → 写入 ``shape_type="rectangle"`` 的 shapes
- ``obb``     → 写入 ``shape_type="rotation"`` 的 shapes（4 个角点）
- ``segment`` → 写入 ``shape_type="polygon"`` 的 shapes
- ``classify``→ 仅刷新 JSON 顶层 ``flags`` 字典（单标签：top1 类别为 ``True``，
                其余为 ``False``），不修改 shapes

自动标注生成的标注文件会额外携带 ``auto_annotated_time`` 字段；处理范围
筛选支持四类（未标注 / 自动 / 自动矫正 / 手动）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts._common import is_image_file
from scripts.core.annotation_type import AnnotationType, AnnotationTypeChecker
from scripts.config import (
    AUTO_ANNOTATE_DEFAULT_BATCH_SIZE,
    DEFAULT_TOLERANCE_SECONDS,
    STATUS_AUTO,
    STATUS_AUTO_CORRECTED,
    STATUS_MANUAL,
    STATUS_UNANNOTATED,
    SUPPORTED_TASKS,
)
from scripts.logging_utils import ProgressLogger, log

__all__ = ["auto_annotate"]


# --------------------------------------------------------------------------- #
# 推理结果转换辅助
# --------------------------------------------------------------------------- #

def _find_model_class_names(model) -> List[str]:
    """从 Ultralytics YOLO 模型中提取类别名称列表。"""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


def _box_to_rectangle_points(xyxy: List[float]) -> List[List[float]]:
    """将 ``[x1, y1, x2, y2]`` 转换为 X-AnyLabeling rectangle 的两个角点。"""
    x1, y1, x2, y2 = xyxy
    return [[float(x1), float(y1)], [float(x2), float(y2)]]


def _obb_to_points(obb_xyxyxyxy) -> List[List[float]]:
    """将 OBB 四边形角点转换为 rotation 的 points。"""
    return [[float(pt[0]), float(pt[1])] for pt in obb_xyxyxyxy]


def _mask_xy_to_points(mask_xy) -> List[List[float]]:
    """将 Ultralytics mask.xy 单个轮廓转换为 polygon points。"""
    return [[float(pt[0]), float(pt[1])] for pt in mask_xy]


def _build_shape(label: str, points: List[List[float]], shape_type: str) -> dict:
    """构造单个 X-AnyLabeling shape 字典。"""
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


def _result_image_size(result, image_path: Path) -> Tuple[int, int]:
    """从 Ultralytics 推理结果中获取图片宽高，失败则回退到 PIL。"""
    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is not None:
        try:
            h, w = int(orig_shape[0]), int(orig_shape[1])
            if w > 0 and h > 0:
                return w, h
        except (TypeError, ValueError, IndexError):
            pass

    from PIL import Image  # 延迟 import

    with Image.open(image_path) as img:
        return img.size


def _build_annotation(
        image_path: Path,
        shapes: List[dict],
        image_size: Tuple[int, int],
        flags: Optional[Dict[str, bool]] = None,
        existing: Optional[dict] = None,
) -> dict:
    """构造完整 X-AnyLabeling JSON 标注字典，并记录自动标注时间。"""
    width, height = image_size
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
    """安全加载已有 JSON 标注，失败返回 ``None``。"""
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
        return STATUS_UNANNOTATED, None

    ann_type = type_checker.check(data, json_mtime=json_path.stat().st_mtime)
    if ann_type == AnnotationType.MANUAL:
        return STATUS_MANUAL, data
    if ann_type == AnnotationType.AUTO:
        return STATUS_AUTO, data
    if ann_type == AnnotationType.AUTO_CORRECTED:
        return STATUS_AUTO_CORRECTED, data
    return STATUS_UNANNOTATED, data


def _build_classify_flags(
        class_names: List[str], pred_label: Optional[str]
) -> Dict[str, bool]:
    """构造分类任务的顶层 ``flags`` 字典（单标签）。"""
    return {
        name: ((name == pred_label) if pred_label is not None else False)
        for name in class_names
    }


def _result_to_detect_shapes(result, class_names: List[str]) -> List[dict]:
    """将 detect 推理结果转换为 shapes。"""
    shapes: List[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return shapes
    for cls_id, _conf, xyxy in zip(
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
    ):
        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        shapes.append(_build_shape(label, _box_to_rectangle_points(xyxy), "rectangle"))
    return shapes


def _result_to_obb_shapes(result, class_names: List[str]) -> List[dict]:
    """将 obb 推理结果转换为 shapes。"""
    shapes: List[dict] = []
    boxes = getattr(result, "obb", None)
    if boxes is None or len(boxes) == 0:
        return shapes
    for cls_id, _conf, xyxyxyxy in zip(
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            boxes.xyxyxyxy.tolist(),
    ):
        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        shapes.append(_build_shape(label, _obb_to_points(xyxyxyxy), "rotation"))
    return shapes


def _result_to_segment_shapes(result, class_names: List[str]) -> List[dict]:
    """将 segment 推理结果转换为 shapes。"""
    shapes: List[dict] = []
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if (
            masks is None
            or getattr(masks, "xy", None) is None
            or len(masks.xy) == 0
    ):
        return shapes
    cls_list = (
        boxes.cls.tolist()
        if (boxes is not None and getattr(boxes, "cls", None) is not None)
        else [0] * len(masks.xy)
    )
    for cls_id, polygon in zip(cls_list, masks.xy):
        if polygon is None or len(polygon) < 3:
            continue
        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        shapes.append(_build_shape(label, _mask_xy_to_points(polygon), "polygon"))
    return shapes


def _result_to_classify_label(result, class_names: List[str]) -> Optional[str]:
    """从 classify 推理结果中解析 top1 类别名称。"""
    probs = getattr(result, "probs", None)
    if probs is None or not class_names:
        return None
    top1 = getattr(probs, "top1", None)
    if top1 is None:
        return None
    try:
        idx = int(top1)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return None


def _finalize_and_save(
        task: str,
        image_path: Path,
        json_path: Path,
        result,
        class_names: List[str],
        existing: Optional[dict],
) -> bool:
    """
    根据任务类型把推理结果落盘为 X-AnyLabeling JSON。

    - detect / obb / segment：若无任何 shape 则跳过（返回 False）。
    - classify：始终写入（即使无 top1，也会刷新为全 False 的 flags）。

    返回 True 表示已写盘，False 表示已跳过。
    """
    image_size = _result_image_size(result, image_path)

    if task == "classify":
        pred_label = _result_to_classify_label(result, class_names)
        new_flags = _build_classify_flags(class_names, pred_label)
        preserved_shapes = (
            existing.get("shapes", []) if isinstance(existing, dict) else []
        )
        if not isinstance(preserved_shapes, list):
            preserved_shapes = []
        annotation = _build_annotation(
            image_path,
            preserved_shapes,
            image_size,
            flags=new_flags,
            existing=existing,
        )
        _save_annotation(annotation, json_path)
        return True

    if task == "detect":
        shapes = _result_to_detect_shapes(result, class_names)
    elif task == "obb":
        shapes = _result_to_obb_shapes(result, class_names)
    elif task == "segment":
        shapes = _result_to_segment_shapes(result, class_names)
    else:
        return False

    if not shapes:
        return False

    preserved_flags = (
        existing.get("flags", {}) if isinstance(existing, dict) else {}
    )
    annotation = _build_annotation(
        image_path,
        shapes,
        image_size,
        flags=preserved_flags,
        existing=existing,
    )
    _save_annotation(annotation, json_path)
    return True


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

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
        tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
        batch_size: int = AUTO_ANNOTATE_DEFAULT_BATCH_SIZE,
) -> Dict[str, object]:
    """
    对工作目录下的图片进行自动标注（支持批量推理 + 流式结果）。

    参数说明详见模块 docstring。
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
    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")

    if not any([include_unannotated, include_auto, include_auto_corrected, include_manual]):
        raise ValueError("至少需要选择一种处理范围")

    include_map = {
        STATUS_UNANNOTATED: include_unannotated,
        STATUS_AUTO: include_auto,
        STATUS_AUTO_CORRECTED: include_auto_corrected,
        STATUS_MANUAL: include_manual,
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

    # 先按 include 开关过滤出真正需要推理的图片，避免对跳过项也做 IO
    todo: List[Tuple[Path, Path, str, Optional[dict]]] = []
    skipped = 0
    for image_path in images:
        json_name = f"{image_path.stem}{suffix}.json"
        json_path = work_path / json_name
        status, existing = _classify_existing_status(json_path, type_checker)
        if not include_map.get(status, False):
            skipped += 1
            continue
        todo.append((image_path, json_path, status, existing))

    total = len(images)
    annotated = 0
    by_type: Dict[str, int] = {
        STATUS_UNANNOTATED: 0,
        STATUS_AUTO: 0,
        STATUS_AUTO_CORRECTED: 0,
        STATUS_MANUAL: 0,
    }

    if not todo:
        log(
            f"自动标注完成（task={task}）：\n"
            f"  图片总数: {total}\n"
            f"  跳过: {skipped}\n"
            f"  实际标注: 0"
        )
        return {
            "total": total,
            "skipped": skipped,
            "annotated": 0,
            "by_type": by_type,
        }

    # 公共 predict 参数
    predict_kwargs: Dict[str, object] = {
        "device": device,
        "verbose": False,
        "stream": True,
    }
    if task != "classify":
        predict_kwargs["conf"] = threshold
        predict_kwargs["iou"] = iou

    progress = ProgressLogger(total=len(todo), desc="自动标注")

    # 分批批量推理，按图片顺序消费流式结果
    for start in range(0, len(todo), batch_size):
        chunk = todo[start: start + batch_size]
        sources = [str(item[0]) for item in chunk]

        try:
            results_iter = yolo_model.predict(source=sources, **predict_kwargs)
        except Exception as exc:
            log(f"[错误] 推理失败（批 {start // batch_size}）: {exc}")
            for _image_path, _json_path, _status, _existing in chunk:
                skipped += 1
                progress.update(1)
            continue

        # stream=True 时，predict 返回一个生成器，按 sources 顺序产出
        results_list = list(results_iter)
        if len(results_list) != len(chunk):
            log(
                f"[警告] 批量推理返回数量与输入不匹配："
                f"{len(results_list)} vs {len(chunk)}（已尽力对齐前缀）"
            )

        for idx, (image_path, json_path, status, existing) in enumerate(chunk):
            if idx >= len(results_list):
                skipped += 1
                progress.update(1)
                continue
            result = results_list[idx]
            try:
                wrote = _finalize_and_save(
                    task=task,
                    image_path=image_path,
                    json_path=json_path,
                    result=result,
                    class_names=class_names,
                    existing=existing,
                )
            except Exception as exc:
                log(f"[错误] 写入失败 {image_path.name}: {exc}")
                wrote = False

            if wrote:
                annotated += 1
                by_type[status] = by_type.get(status, 0) + 1
            else:
                skipped += 1
            progress.update(1)

    progress.close()

    log(
        f"自动标注完成（task={task}）：\n"
        f"  图片总数: {total}\n"
        f"  跳过: {skipped}\n"
        f"  实际标注: {annotated}\n"
        f"  按类型: 未标注 {by_type[STATUS_UNANNOTATED]}, "
        f"自动 {by_type[STATUS_AUTO]}, "
        f"矫正 {by_type[STATUS_AUTO_CORRECTED]}, "
        f"手动 {by_type[STATUS_MANUAL]}"
    )
    return {
        "total": total,
        "skipped": skipped,
        "annotated": annotated,
        "by_type": by_type,
    }
