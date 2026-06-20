#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将视频按指定间隔抽帧并保存为图片。

该模块暴露以下核心方法：
    - extract_video_frames(input_video, output_dir, frame_step=1,
                           image_extension="jpg", quality=95,
                           prefix="frame", start_time=None, end_time=None)

用法示例：
    from extract_video_frames import extract_video_frames
    extract_video_frames(
        input_video="/path/to/video.mp4",
        output_dir="/path/to/output",
        frame_step=5,           # 每隔 5 帧抽一张
        image_extension="jpg",
        quality=95,
        prefix="frame",
    )
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from tqdm import tqdm



SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


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
    """
    将时间参数转换为秒。
    支持 int/float（秒数）或 "HH:MM:SS"/"MM:SS"/"SS" 格式字符串。
    """
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

    raise TypeError(f"start_time/end_time 必须是 int、float 或 str， got {type(time_value)}")


def _get_writer_params(ext: str, quality: int) -> Tuple:
    """
    根据扩展名返回 cv2.imwrite 所需的参数列表。

    返回:
        tuple: (params, is_lossless)，其中 params 为 imwrite 参数元组。
    """
    if ext in {".jpg", ".jpeg"}:
        # OpenCV 的 JPEG 质量参数范围是 0-100
        return (cv2.IMWRITE_JPEG_QUALITY, quality), False
    if ext == ".png":
        # PNG 压缩级别 0-9，数值越大压缩率越高；这里使用默认值 3
        compression = max(0, min(9, 9 - quality // 11))
        return (cv2.IMWRITE_PNG_COMPRESSION, compression), True
    if ext == ".webp":
        return (cv2.IMWRITE_WEBP_QUALITY, quality), False
    return tuple(), False


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
        frame_step: 抽帧间隔，默认 1（逐帧抽取）。每间隔 frame_step 帧保存一张。
                    例如 frame_step=5 表示每 5 帧保存一张。
        image_extension: 输出图片格式，支持 jpg/jpeg/png/bmp/webp/tiff/tif，默认 jpg。
        quality: 输出图片质量（仅对 jpg/webp 有效），默认 100（原始画质）。
        prefix: 输出文件名前缀，默认 "frame"。
        start_time: 开始抽取的时间（秒或 "HH:MM:SS"/"MM:SS"/"SS" 格式字符串），默认从视频开头。
        end_time: 结束抽取的时间（秒或时间字符串），默认抽到视频结尾。

    返回:
        保存的图片路径列表（按帧顺序）。

    异常:
        FileNotFoundError: 输入视频不存在。
        ValueError: 参数校验失败。
        RuntimeError: 无法打开视频文件或保存图片失败。
    """
    input_path = Path(input_video)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {input_video}")

    if frame_step < 1:
        raise ValueError("frame_step 必须大于等于 1")

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

    # 计算起始/结束帧索引
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

    # 预估需要处理的帧数，用于进度条
    estimated = max(1, (end_frame - start_frame) // frame_step)
    pbar = tqdm(total=estimated, desc="抽取视频帧")

    while frame_index < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_index - start_frame) % frame_step == 0:
            filename = f"{prefix}_{_format_index(saved_count)}{ext}"
            save_path = output_path / filename
            ok = cv2.imwrite(str(save_path), frame, params)
            if not ok:
                cap.release()
                pbar.close()
                raise RuntimeError(f"保存图片失败: {save_path}")
            saved_paths.append(str(save_path))
            saved_count += 1
            pbar.update(1)

        frame_index += 1

    cap.release()
    pbar.close()

    print(f"共抽取 {saved_count} 帧，保存到: {output_path}")
    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频抽帧工具")
    parser.add_argument("input_video", type=str, help="输入视频文件路径")
    parser.add_argument("output_dir", type=str, help="输出图片保存目录")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="抽帧间隔，默认 1（逐帧抽取）。",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        help="输出图片格式，默认 jpg。",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="输出图片质量，默认 100（原始画质）。",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="输出文件名前缀，默认 frame。",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="开始抽取的时间（秒或 HH:MM:SS/MM:SS/SS 格式）。",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="结束抽取的时间（秒或 HH:MM:SS/MM:SS/SS 格式）。",
    )
    args = parser.parse_args()

    extract_video_frames(
        input_video=args.input_video,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        image_extension=args.ext,
        quality=args.quality,
        prefix=args.prefix,
        start_time=args.start_time,
        end_time=args.end_time,
    )
