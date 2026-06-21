#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将视频按指定间隔抽帧并保存为图片（核心实现）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from scripts.config import (
    SUPPORTED_SEEK_MODES,
    SUPPORTED_VIDEO_FRAME_EXTENSIONS as SUPPORTED_EXTENSIONS,
)
from scripts.logging_utils import ProgressLogger, log

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_SEEK_MODES",
    "extract_video_frames",
]


def _format_index(index: int, width: int = 6) -> str:
    """将索引格式化为固定宽度的字符串。"""
    return str(index).zfill(width)


def _ensure_dir(directory: str) -> Path:
    """确保输出目录存在并返回 Path 对象。"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_extension(ext: str) -> str:
    """校验并规范化图片扩展名。"""
    ext = ext.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的图片格式: {ext}，支持的格式为 {SUPPORTED_EXTENSIONS}"
        )
    return ext


def _time_to_seconds(time_value) -> Optional[float]:
    """将时间参数转换为秒。"""
    if time_value is None:
        return None

    if isinstance(time_value, (int, float)):
        return float(time_value)

    if isinstance(time_value, str):
        parts = time_value.strip().split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"无法解析时间字符串: {time_value}") from exc

        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        raise ValueError(f"时间字符串格式不正确: {time_value}")

    raise TypeError(
        f"start_time/end_time 必须是 int、float 或 str， got {type(time_value)}"
    )


def _get_writer_params(ext: str, quality: int) -> Tuple:
    """根据扩展名返回 cv2.imwrite 所需的参数列表。"""
    if ext in {".jpg", ".jpeg"}:
        return (cv2.IMWRITE_JPEG_QUALITY, quality), False
    if ext == ".png":
        compression = max(0, min(9, 9 - quality // 11))
        return (cv2.IMWRITE_PNG_COMPRESSION, compression), True
    if ext == ".webp":
        return (cv2.IMWRITE_WEBP_QUALITY, quality), False
    return tuple(), False


def _resolve_save_path(
        output_path: Path,
        prefix: str,
        saved_count: int,
        ext: str,
        overwrite: bool,
) -> Path:
    """根据计数生成保存路径；当文件已存在且 ``overwrite=False`` 时，自动追加序号。"""
    base_name = f"{prefix}_{_format_index(saved_count)}"
    candidate = output_path / f"{base_name}{ext}"
    if overwrite or not candidate.exists():
        return candidate

    suffix_idx = 1
    while True:
        candidate = output_path / f"{base_name}_{suffix_idx}{ext}"
        if not candidate.exists():
            return candidate
        suffix_idx += 1


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
    """将视频抽帧并保存为图片。参数详见外层 CLI 文档。"""
    input_path = Path(input_video)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {input_video}")

    if frame_step < 1:
        raise ValueError("frame_step 必须大于等于 1")

    if seek_mode not in SUPPORTED_SEEK_MODES:
        raise ValueError(
            f"不支持的 seek_mode: {seek_mode}，仅支持 {sorted(SUPPORTED_SEEK_MODES)}"
        )

    ext = _validate_extension(image_extension)
    output_path = _ensure_dir(output_dir)

    start_sec = _time_to_seconds(start_time)
    end_sec = _time_to_seconds(end_time)
    if start_sec is not None and end_sec is not None and start_sec >= end_sec:
        raise ValueError("start_time 必须小于 end_time")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"视频帧数无效: {input_video}")

    start_frame = 0
    end_frame = total_frames
    if start_sec is not None:
        start_frame = max(0, int(start_sec * fps))
        if start_frame >= total_frames:
            cap.release()
            raise ValueError(f"start_time {start_sec}s 超出视频时长")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if end_sec is not None:
        end_frame = min(total_frames, int(end_sec * fps))
        if end_frame <= start_frame:
            cap.release()
            raise ValueError(f"end_time {end_sec}s 必须大于 start_time")

    params, _ = _get_writer_params(ext, quality)
    saved_paths: List[str] = []
    frame_index = start_frame
    saved_count = 0

    estimated = max(1, (end_frame - start_frame + frame_step - 1) // frame_step)
    progress = ProgressLogger(total=estimated, desc="抽取视频帧")

    try:
        if seek_mode == "seek" and frame_step > 1:
            target_index = start_frame
            while target_index < end_frame:
                current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current != target_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                ret, frame = cap.read()
                if not ret:
                    break

                save_path = _resolve_save_path(
                    output_path, prefix, saved_count, ext, overwrite
                )
                ok = cv2.imwrite(str(save_path), frame, params)
                if not ok:
                    raise RuntimeError(f"保存图片失败: {save_path}")
                saved_paths.append(str(save_path))
                saved_count += 1
                progress.update(1)
                target_index += frame_step
        else:
            while frame_index < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_index - start_frame) % frame_step == 0:
                    save_path = _resolve_save_path(
                        output_path, prefix, saved_count, ext, overwrite
                    )
                    ok = cv2.imwrite(str(save_path), frame, params)
                    if not ok:
                        raise RuntimeError(f"保存图片失败: {save_path}")
                    saved_paths.append(str(save_path))
                    saved_count += 1
                    progress.update(1)

                frame_index += 1
    finally:
        cap.release()
        progress.close()

    log(f"共抽取 {saved_count} 帧，保存到: {output_path}")
    return saved_paths
