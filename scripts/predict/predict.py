#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python scripts/vh.py predict run 命令实现。

使用训练好的 YOLO 模型对图片或视频进行预测，将结果可视化后保存。

支持:
    - 单张图片或图片目录批量预测
    - 视频预测（带进度条）
    - detect / obb / segment / classify 4 种任务
    - 结果保存为标注后的图片/视频

用法::

    python scripts/vh.py predict run -m ./best.pt -i ./images -o ./output -T 0.25
    python scripts/vh.py predict run -m ./best.pt -i ./video.mp4 -o ./output -T 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scripts.common.config import IMAGE_EXTENSIONS, SUPPORTED_TASKS
from scripts.common.logging import ProgressLogger, log
from scripts.common.utils import find_model_class_names, is_image_file

import cv2

__all__ = ["predict", "main"]


def _draw_detect_results(
        frame,
        result,
        class_names: List[str],
        conf_threshold: float,
) -> None:
    """在图片/帧上绘制检测框和标签。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return

    for cls_id, conf, xyxy in zip(
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
    ):
        if conf < conf_threshold:
            continue
        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        x1, y1, x2, y2 = map(int, xyxy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)

        cv2.putText(
            frame, text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
        )


def _draw_obb_results(
        frame,
        result,
        class_names: List[str],
        conf_threshold: float,
) -> None:
    """在图片/帧上绘制旋转框。"""
    obb = getattr(result, "obb", None)
    if obb is None:
        return

    import numpy as np

    for cls_id, conf, points in zip(
            obb.cls.tolist(),
            obb.conf.tolist(),
            obb.xyxyxyxy.tolist(),
    ):
        if conf < conf_threshold:
            continue
        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)

        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame, text, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )


def _draw_segment_results(
        frame,
        result,
        class_names: List[str],
        conf_threshold: float,
) -> None:
    """在图片/帧上绘制分割掩码。"""
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return

    import numpy as np

    cls_list = (
        boxes.cls.tolist()
        if (boxes is not None and getattr(boxes, "cls", None) is not None)
        else [0] * len(masks.xy)
    )
    conf_list = (
        boxes.conf.tolist()
        if (boxes is not None and getattr(boxes, "conf", None) is not None)
        else [1.0] * len(masks.xy)
    )

    overlay = frame.copy()
    for cls_id, conf, polygon in zip(cls_list, conf_list, masks.xy):
        if conf < conf_threshold:
            continue
        if polygon is None or len(polygon) < 3:
            continue

        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        pts = np.array([[int(p[0]), int(p[1])] for p in polygon], dtype=np.int32)

        # 绘制半透明填充
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        # 绘制轮廓
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # 绘制标签
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame, text, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )

    # 混合半透明效果
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


def _draw_classify_results(
        frame,
        result,
        class_names: List[str],
) -> None:
    """在图片/帧上绘制分类结果。"""
    probs = getattr(result, "probs", None)
    if probs is None or not class_names:
        return

    top1 = getattr(probs, "top1", None)
    top1_conf = getattr(probs, "top1conf", None)
    if top1 is None:
        return

    label = class_names[int(top1)]
    conf = float(top1_conf) if top1_conf is not None else 0.0

    # 在图片顶部绘制分类结果
    text = f"{label}: {conf:.2%}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    h, w = frame.shape[:2]
    x = (w - text_w) // 2
    y = text_h + 20

    cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 255, 0), -1)
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA,
    )


_DRAW_STRATEGIES = {
    "detect": _draw_detect_results,
    "obb": _draw_obb_results,
    "segment": _draw_segment_results,
    "classify": _draw_classify_results,
}


def _draw_results(
        frame,
        result,
        task: str,
        class_names: List[str],
        conf_threshold: float,
) -> None:
    """根据任务类型在图片/帧上绘制推理结果。"""
    draw_fn = _DRAW_STRATEGIES.get(task)
    if draw_fn is None:
        return
    if task == "classify":
        draw_fn(frame, result, class_names)
    else:
        draw_fn(frame, result, class_names, conf_threshold)


def predict(
        model_path: str,
        input_path: str,
        output_dir: str,
        threshold: float = 0.25,
        task: str = "detect",
        device: Optional[str] = None,
        iou: float = 0.45,
        batch_size: int = 16,
) -> Dict[str, object]:
    """
    使用 YOLO 模型对图片或视频进行预测，将可视化结果保存到输出目录。

    参数:
        model_path: YOLO 模型权重文件路径（.pt）。
        input_path: 输入图片、图片目录或视频文件路径。
        output_dir: 输出目录，用于保存预测结果。
        threshold: 置信度阈值，必须位于 (0, 1]，默认 0.25。
        task: 任务类型，detect / obb / segment / classify，默认 detect。
        device: 推理设备，例如 0、cpu，默认自动选择。
        iou: NMS IoU 阈值，必须位于 (0, 1]，默认 0.45。
        batch_size: 目录预测时的批次大小，默认 16。

    返回:
        包含 total / success / failed 等键的字典。

    异常:
        ValueError: 参数非法。
    """

    model_file = Path(model_path)
    input_p = Path(input_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task = task.lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的任务类型: {task}，仅支持 {sorted(SUPPORTED_TASKS)}")

    if not (0 < threshold <= 1):
        raise ValueError("threshold 必须在 (0, 1] 范围内")
    if not (0 < iou <= 1):
        raise ValueError("iou 必须在 (0, 1] 范围内")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，请先执行：pip install ultralytics") from exc

    yolo_model = YOLO(str(model_file))
    class_names = find_model_class_names(yolo_model)

    base_kwargs: Dict[str, object] = {
        "device": device,
        "verbose": False,
    }
    if task != "classify":
        base_kwargs["conf"] = threshold
        base_kwargs["iou"] = iou

    if input_p.is_file():
        ext = input_p.suffix.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
        if ext in video_exts:
            stats = _predict_video(
                yolo_model, input_p, output_path, task, class_names,
                dict(base_kwargs),
            )
        elif ext in IMAGE_EXTENSIONS:
            stats = _predict_single_image(
                yolo_model, input_p, output_path, task, class_names,
                dict(base_kwargs),
            )
        else:
            raise ValueError(f"不支持的文件类型: {ext}")
    elif input_p.is_dir():
        stats = _predict_image_dir(
            yolo_model, input_p, output_path, task, class_names,
            dict(base_kwargs), batch_size=batch_size,
        )
    else:
        raise ValueError(f"输入路径不是文件或目录: {input_path}")

    log(
        f"预测完成（task={task}）：\n"
        f"  输入类型: {stats['input_type']}\n"
        f"  总数: {stats['total']}\n"
        f"  成功: {stats['success']}\n"
        f"  失败: {stats['failed']}"
    )
    return stats


def _predict_single_image(
        yolo_model,
        image_path: Path,
        output_dir: Path,
        task: str,
        class_names: List[str],
        predict_kwargs: Dict[str, object],
) -> Dict[str, object]:
    """预测单张图片并保存结果。"""
    stats: Dict[str, object] = {"total": 0, "success": 0, "failed": 0, "input_type": "image"}
    stats["total"] += 1
    conf_threshold = predict_kwargs.get("conf", 0.25)
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            log(f"[警告] 无法读取图片: {image_path.name}")
            stats["failed"] += 1
            return stats

        results = list(yolo_model.predict(source=str(image_path), **predict_kwargs))
        if results:
            _draw_results(frame, results[0], task, class_names, conf_threshold)

        output_file = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
        if not cv2.imwrite(str(output_file), frame):
            log(f"[错误] 无法保存图片: {output_file.name}")
            stats["failed"] += 1
        else:
            stats["success"] += 1
    except Exception as exc:
        log(f"[错误] 预测失败 {image_path.name}: {exc}")
        stats["failed"] += 1
    return stats


def _predict_image_dir(
        yolo_model,
        input_dir: Path,
        output_dir: Path,
        task: str,
        class_names: List[str],
        predict_kwargs: Dict[str, object],
        batch_size: int = 16,
) -> Dict[str, object]:
    """批量预测目录中的图片（分批推理以利用 Ultralytics 内部批处理）。"""
    stats: Dict[str, object] = {"total": 0, "success": 0, "failed": 0, "input_type": "images"}

    images = sorted([p for p in input_dir.iterdir() if is_image_file(p)])
    if not images:
        log("未找到可预测的图片")
        return stats

    stats["total"] += len(images)
    conf_threshold = predict_kwargs.get("conf", 0.25)
    total_batches = (len(images) + batch_size - 1) // batch_size
    progress = ProgressLogger(total=len(images), desc="图片预测")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(images))
        batch_paths = images[start:end]
        sources = [str(p) for p in batch_paths]

        try:
            results = list(yolo_model.predict(source=sources, **predict_kwargs))
        except Exception as exc:
            log(f"[错误] 批次 {batch_idx+1}/{total_batches} 预测失败: {exc}")
            for _ in batch_paths:
                stats["failed"] += 1
                progress.update(1)
            continue

        for image_path, result in zip(batch_paths, results):
            try:
                frame = result.orig_img
                _draw_results(frame, result, task, class_names, conf_threshold)
                output_file = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
                if not cv2.imwrite(str(output_file), frame):
                    log(f"[错误] 无法保存图片: {output_file.name}")
                    stats["failed"] += 1
                else:
                    stats["success"] += 1
            except Exception as exc:
                log(f"[错误] 预测失败 {image_path.name}: {exc}")
                stats["failed"] += 1
            progress.update(1)

    progress.close()
    return stats


def _predict_video(
        yolo_model,
        video_path: Path,
        output_dir: Path,
        task: str,
        class_names: List[str],
        predict_kwargs: Dict[str, object],
) -> Dict[str, object]:
    """预测视频并保存结果，带进度条显示。"""
    import time

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stats: Dict[str, object] = {"total": total_frames, "success": 0, "failed": 0, "input_type": "video"}

    output_video = output_dir / f"{video_path.stem}_pred{video_path.suffix}"
    codecs_to_try = ["mp4v", "avc1", "h264"]
    writer = None
    for codec_str in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if writer.isOpened():
            break

    if writer is None or not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {output_video}")

    log(f"开始预测视频: {video_path.name}")
    log(f"  总帧数: {total_frames}, FPS: {fps:.2f}, 分辨率: {width}x{height}")

    conf_threshold = predict_kwargs.get("conf", 0.25)
    frame_idx = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = list(yolo_model.predict(source=frame, **predict_kwargs))
                if results:
                    _draw_results(frame, results[0], task, class_names, conf_threshold)

                writer.write(frame)
                stats["success"] += 1
            except Exception as exc:
                log(f"[错误] 第 {frame_idx+1} 帧处理失败: {exc}")
                stats["failed"] += 1

            frame_idx += 1

            now = time.time()
            if frame_idx % 10 == 0 or (now - last_print_time) >= 0.5:
                elapsed = now - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                percent = frame_idx * 100.0 / total_frames if total_frames > 0 else 0
                if fps_actual > 0:
                    remaining = (total_frames - frame_idx) / fps_actual
                    remaining_str = f"{int(remaining // 60)}:{int(remaining % 60):02d}"
                else:
                    remaining_str = "计算中..."
                log(
                    f"[视频预测] {frame_idx}/{total_frames} "
                    f"({percent:.1f}%) | {fps_actual:.1f} FPS | 剩余: {remaining_str}"
                )
                last_print_time = now

    except KeyboardInterrupt:
        log("\n[已取消] 用户中断预测")
        stats["failed"] = max(0, total_frames - stats["success"])
        raise
    except Exception as exc:
        log(f"\n[错误] 视频预测失败: {exc}")
        stats["failed"] = total_frames - stats["success"]
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start_time
    log(f"视频预测完成，耗时: {elapsed:.1f}秒，结果保存在: {output_video}")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py predict run",
        description=(
            "使用训练好的 YOLO 模型对图片或视频进行预测，"
            "将可视化结果保存到输出目录。"
        ),
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="YOLO 模型权重文件路径（.pt）",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入图片、图片目录或视频文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="输出目录，用于保存预测结果",
    )
    parser.add_argument(
        "-T", "--threshold",
        type=float,
        default=0.25,
        help="置信度阈值（0~1），默认 0.25",
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="detect",
        choices=sorted(SUPPORTED_TASKS),
        help="任务类型：detect / obb / segment / classify，默认 detect",
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
        help="NMS IoU 阈值（0~1），默认 0.45",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    from pathlib import Path

    model_path = Path(args.model)
    if not model_path.exists():
        raise ValueError(f"模型权重文件不存在：{args.model}")
    if not model_path.is_file():
        raise ValueError(f"模型权重路径不是文件：{args.model}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"输入路径不存在：{args.input}")

    output_path = Path(args.output)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"输出路径不是目录：{args.output}")

    if not (0.0 < args.threshold <= 1.0):
        raise ValueError(f"--threshold 必须在 (0, 1] 之间，当前为 {args.threshold}")
    if not (0.0 < args.iou <= 1.0):
        raise ValueError(f"--iou 必须在 (0, 1] 之间，当前为 {args.iou}")


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
        result = predict(
            model_path=args.model,
            input_path=args.input,
            output_dir=args.output,
            threshold=args.threshold,
            task=args.task,
            device=args.device,
            iou=args.iou,
        )
    except KeyboardInterrupt:
        log("[已取消] 用户中断预测。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except ImportError as exc:
        log(
            f"[依赖缺失] {exc}\n"
            f"提示：模型预测需要安装 ultralytics（含 torch）。",
            stream=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 预测失败：{exc}", stream=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
