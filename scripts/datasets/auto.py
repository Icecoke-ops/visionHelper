#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 自动标注实现。

使用 Ultralytics YOLO 模型对工作目录下的图片进行推理，将满足条件的结果
保存为 X-AnyLabeling JSON 格式（兼容 LabelMe）。本模块同时包含核心实现与
``python scripts/vh.py datasets auto`` 命令行入口；``ultralytics`` 仅在调用
:func:`auto_annotate` 时按需加载。

用法::

    python scripts/vh.py datasets auto -i ./images -m ./best.pt -t detect -T 0.25
    python scripts/vh.py datasets auto -i ./images -m ./best.pt -t detect -a -c
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.common.annotation_type import AnnotationType, AnnotationTypeChecker
from scripts.common.config import (
    AUTO_ANNOTATE_DEFAULT_BATCH_SIZE,
    DEFAULT_TOLERANCE_SECONDS,
    STATUS_AUTO,
    STATUS_AUTO_CORRECTED,
    STATUS_MANUAL,
    STATUS_UNANNOTATED,
    SUPPORTED_TASKS,
)
from scripts.common.logging import ProgressLogger, log
from scripts.common.utils import (
    find_model_class_names,
    is_image_file,
)


def _check_image_readable(image_path: Path) -> bool:
    """使用 Pillow 预检图片是否可被正常读取。

    对截断、损坏或格式错误的图片提前返回 False，避免在批量推理阶段
    导致整个任务中断。本函数仅依赖 Pillow，不触发 ultralytics/torch。
    """
    try:
        from PIL import Image, UnidentifiedImageError

        with Image.open(image_path) as img:
            img.load()
        return True
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        log(
            f"[警告] 图片读取失败，已跳过 {image_path.name}: {exc}",
            stream=sys.stderr,
        )
        return False


__all__ = ["auto_annotate", "main"]


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

    from PIL import Image, UnidentifiedImageError  # 延迟 import

    try:
        with Image.open(image_path) as img:
            return img.size
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(
            f"无法读取图片尺寸 {image_path.name}: {exc}"
        ) from exc


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
    """根据任务类型把推理结果落盘为 X-AnyLabeling JSON。"""
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
    """对工作目录下的图片进行自动标注（支持批量推理 + 流式结果）。"""
    work_path = Path(work_dir)
    if not work_path.is_dir():
        raise ValueError(f"工作目录不存在: {work_dir}")  # NOTE: duplicated in _validate_args; kept for API safety

    model_file = Path(model_path)
    if not model_file.is_file():
        raise ValueError(f"模型文件不存在: {model_path}")  # NOTE: duplicated in _validate_args; kept for API safety

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}"
        )

    if not (0 < threshold <= 1):
        raise ValueError("threshold 必须在 (0, 1] 范围内")  # NOTE: duplicated in _validate_args
    if not (0 < iou <= 1):
        raise ValueError("iou 必须在 (0, 1] 范围内")  # NOTE: duplicated in _validate_args
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds 必须 >= 0")  # NOTE: duplicated in _validate_args
    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")  # NOTE: duplicated in _validate_args

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

    log("[安全提示] 正在加载模型文件，请确保模型来源可信")
    yolo_model = YOLO(str(model_file))
    class_names = find_model_class_names(yolo_model)

    type_checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)

    images = [p for p in work_path.iterdir() if is_image_file(p)]
    images.sort(key=lambda p: p.name)

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

    predict_kwargs: Dict[str, object] = {
        "device": device,
        "verbose": False,
        "stream": True,
    }
    if task != "classify":
        predict_kwargs["conf"] = threshold
        predict_kwargs["iou"] = iou

    progress = ProgressLogger(total=len(todo), desc="自动标注")

    for start in range(0, len(todo), batch_size):
        chunk = todo[start: start + batch_size]

        # 预检本批图片可读性，提前剔除截断/损坏图片，避免 predict 整批失败。
        readable_chunk: List[Tuple[Path, Path, str, Optional[dict]]] = []
        for item in chunk:
            image_path = item[0]
            if _check_image_readable(image_path):
                readable_chunk.append(item)
            else:
                skipped += 1
                progress.update(1)

        if not readable_chunk:
            continue

        sources = [str(item[0]) for item in readable_chunk]

        try:
            results_iter = yolo_model.predict(source=sources, **predict_kwargs)
        except Exception as exc:
            log(f"[错误] 推理失败（批 {start // batch_size}）: {exc}")
            for _image_path, _json_path, _status, _existing in readable_chunk:
                skipped += 1
                progress.update(1)
            continue

        processed = 0
        for result, (image_path, json_path, status, existing) in zip(results_iter, readable_chunk):
            processed += 1
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
                by_type[status] += 1
            else:
                skipped += 1
            progress.update(1)

        # Consume any leftover results (needed for stream=True generator cleanup)
        extra = sum(1 for _ in results_iter)
        total_results = processed + extra

        if total_results != len(readable_chunk):
            log(
                f"[警告] 批量推理返回数量与输入不匹配："
                f"{total_results} vs {len(readable_chunk)}（已尽力对齐前缀）"
            )
            for _ in range(total_results, len(readable_chunk)):
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


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py datasets auto",
        description=(
            "YOLO 自动标注工具：使用已训练模型为目录中的图片生成 X-AnyLabeling "
            "JSON 标注，支持 detect / obb / segment / classify 4 种任务。"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="待标注图片所在的工作目录",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="YOLO 模型权重文件路径（.pt）",
    )
    parser.add_argument(
        "-T", "--threshold",
        type=float,
        default=0.25,
        help="置信度阈值（0~1），默认 0.25（仅 detect/obb/segment 生效）",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="输出 JSON 文件名后缀（追加在 stem 之后），默认空",
    )
    parser.add_argument(
        "-D", "--device",
        type=str,
        default=None,
        help="推理设备，例如 0/cpu/0,1，默认自动选择",
    )
    parser.add_argument(
        "-u", "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值（0~1），默认 0.45（仅 detect/obb/segment 生效）",
    )
    parser.add_argument(
        "--include-unannotated",
        action="store_true",
        help="处理未标注图片（4 个 include 开关全未指定时此项默认开启）",
    )
    parser.add_argument(
        "-a", "--include-auto",
        action="store_true",
        help="处理已被自动标注的图片（会用新模型重新生成标注）",
    )
    parser.add_argument(
        "-c", "--include-auto-corrected",
        action="store_true",
        help="处理自动标注后人工矫正过的图片（谨慎使用，会覆盖人工修改）",
    )
    parser.add_argument(
        "-M", "--include-manual",
        action="store_true",
        help="处理纯手动标注的图片（极谨慎使用，会覆盖人工标注）",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help=f"判定自动 / 矫正的时间容差（秒），默认 {DEFAULT_TOLERANCE_SECONDS}",
    )
    parser.add_argument(
        "-z", "--batch-size",
        type=int,
        default=AUTO_ANNOTATE_DEFAULT_BATCH_SIZE,
        help=f"批量推理大小（必须为正整数），默认 {AUTO_ANNOTATE_DEFAULT_BATCH_SIZE}",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    work_dir = Path(args.input)
    if not work_dir.exists():
        raise ValueError(f"工作目录不存在：{args.input}")
    if not work_dir.is_dir():
        raise ValueError(f"工作目录不是文件夹：{args.input}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise ValueError(f"模型权重文件不存在：{args.model}")
    if not model_path.is_file():
        raise ValueError(f"模型权重路径不是文件：{args.model}")
    if model_path.suffix.lower() != ".pt":
        log(
            f"[警告] 模型文件后缀为 {model_path.suffix!r}，常规为 .pt，请确认无误。",
            stream=sys.stderr,
        )

    if not (0.0 < args.threshold <= 1.0):
        raise ValueError(f"--threshold 必须在 (0, 1] 之间，当前为 {args.threshold}")
    if not (0.0 < args.iou <= 1.0):
        raise ValueError(f"--iou 必须在 (0, 1] 之间，当前为 {args.iou}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f"--batch-size 必须为正整数，当前为 {args.batch_size}")
    if args.tolerance_seconds is not None and args.tolerance_seconds < 0:
        raise ValueError(
            f"--tolerance-seconds 必须为非负数，当前为 {args.tolerance_seconds}"
        )


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

    include_unannotated = args.include_unannotated
    include_auto = args.include_auto
    include_auto_corrected = args.include_auto_corrected
    include_manual = args.include_manual
    if not any([include_unannotated, include_auto, include_auto_corrected, include_manual]):
        include_unannotated = True
        log(
            "[提示] 未指定任何 --include-* 开关，默认仅处理 *未标注* 图片。",
            stream=sys.stderr,
        )

    try:
        result = auto_annotate(
            work_dir=args.input,
            model_path=args.model,
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
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，已写入的 JSON 不会被回滚。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except ImportError as exc:
        log(
            f"[依赖缺失] {exc}\n"
            f"提示：自动标注需要安装 ultralytics（含 torch）。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 自动标注失败：{exc}", stream=sys.stderr)
        return 1

    total = result.get("total", 0)
    annotated = result.get("annotated", 0)
    skipped = result.get("skipped", 0)
    log(f"[完成] 总计 {total} 张，标注 {annotated} 张，跳过 {skipped} 张。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
