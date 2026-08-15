#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频抽帧模块（``python scripts/vh.py images import`` 实现）。

合并旧 ``scripts.core.extract_video_frames`` 与 ``scripts.extract_video_frames``
CLI 门面，提供 :func:`extract_video_frames` 核心实现与 ``main`` 命令行入口。

零副作用约定
------------

``cv2`` 等重依赖仅在函数内部延迟 import，模块顶层不触发重型依赖加载。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.common.config import (
    SUPPORTED_SEEK_MODES,
    SUPPORTED_VIDEO_EXTENSIONS,
    SUPPORTED_VIDEO_FRAME_EXTENSIONS as SUPPORTED_EXTENSIONS,
)
from scripts.common.logging import ProgressLogger, log

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_SEEK_MODES",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "extract_video_frames",
    "extract_video_frames_batch",
    "main",
]


def _format_index(index: int, width: int = 6) -> str:
    """将索引格式化为固定宽度的字符串。"""
    return f"{index:0{width}d}"


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


def _get_writer_params(ext: str, quality: int) -> tuple:
    """根据扩展名返回 cv2.imwrite 所需的参数列表。"""
    import cv2

    if ext in {".jpg", ".jpeg"}:
        return (cv2.IMWRITE_JPEG_QUALITY, quality)
    if ext == ".png":
        compression = max(0, min(9, 9 - quality // 11))
        return (cv2.IMWRITE_PNG_COMPRESSION, compression)
    if ext == ".webp":
        return (cv2.IMWRITE_WEBP_QUALITY, quality)
    return ()


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
    while suffix_idx <= 9999:
        candidate = output_path / f"{base_name}_{suffix_idx}{ext}"
        if not candidate.exists():
            return candidate
        suffix_idx += 1
    raise RuntimeError(
        f"无法生成唯一文件名，已达最大重试次数 9999: {base_name}"
    )


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
    """将视频抽帧并保存为图片。参数详见 CLI 文档。"""
    import cv2

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

    params = _get_writer_params(ext, quality)
    saved_paths: List[str] = []
    frame_index = start_frame
    saved_count = 0

    estimated = max(1, (end_frame - start_frame + frame_step - 1) // frame_step)
    progress = ProgressLogger(total=estimated, desc="抽取视频帧")

    try:
        if seek_mode == "seek" and frame_step > 1:
            target_index = start_frame
            seek_failed = False
            while target_index < end_frame:
                current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current != target_index:
                    # 某些编码器不支持精确 seek，回退到 decode_all
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, target_index):
                        log(f"[警告] seek 模式失败（编码器不支持精确跳帧），回退到 decode_all 模式", stream=sys.stderr)
                        seek_failed = True
                        break
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

            if seek_failed:
                # 重置捕获器并使用 decode_all 模式
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_index = start_frame
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


def _iter_video_files(input_dir: Path) -> List[Path]:
    """递归收集目录下所有受支持的视频文件（按路径排序）。"""
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def _build_video_prefix(prefix: str, video: Path, input_path: Path) -> str:
    """根据视频相对输入目录的路径生成抽帧文件名前缀。

    路径段用 ``_`` 连接（如 ``sub/clip.mp4`` → ``sub_clip``），使不同
    目录下的视频默认拥有独立的输出命名空间。
    """
    rel_key = "_".join(video.relative_to(input_path).with_suffix("").parts)
    return f"{prefix}_{rel_key}"


def _disambiguate_video_prefixes(
        videos: List[Path],
        input_path: Path,
        prefix: str,
) -> Dict[Path, str]:
    """为每个视频生成互不重复的输出文件名前缀。

    当不同视频的相对路径经 ``_`` 拼接后恰好相同（如顶层 ``a_b.mp4``
    与子目录 ``a/b.mp4`` 都会得到 ``a_b``）时，对冲突项追加相对路径的
    短哈希后缀，保证不同视频抽出的帧不会重名。
    """
    keys = [_build_video_prefix(prefix, video, input_path) for video in videos]

    counts: Dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1

    result: Dict[Path, str] = {}
    for video, key in zip(videos, keys):
        if counts[key] == 1:
            result[video] = key
        else:
            rel = str(video.relative_to(input_path)).replace("\\", "/")
            # 仅作文件名消歧，非安全性用途（不需要密码学强度）
            digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
            result[video] = f"{key}_{digest}"
    return result


def extract_video_frames_batch(
        input_dir: str,
        output_dir: str,
        frame_step: int = 1,
        image_extension: str = "jpg",
        quality: int = 100,
        prefix: str = "frame",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        seek_mode: str = "decode_all",
        overwrite: bool = False,
        failures: Optional[List[Tuple[Path, str]]] = None,
) -> List[str]:
    """将目录下所有视频文件逐个抽帧，返回全部保存的图片路径列表。

    每个视频独立计数，输出文件名形如 ``<prefix>_<相对路径>_000001.jpg``，
    其中 ``<相对路径>`` 为该视频相对输入目录的路径（目录分隔符替换为
    ``_``），保证不同视频抽出的帧不会重名。

    参数:
        input_dir: 包含视频文件的输入目录（递归扫描）。
        其余参数含义与 :func:`extract_video_frames` 一致。
        failures: 可选输出列表；部分视频失败时，失败明细 ``(视频路径, 原因)``
            会被追加到此列表（调用方传入空列表即可感知部分失败）。
            默认 ``None`` 时不收集，仅记录日志。

    返回:
        全部视频保存的图片路径列表（按视频顺序、帧顺序）。
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    import cv2

    videos = _iter_video_files(input_path)
    if not videos:
        raise ValueError(f"目录下没有找到支持的视频文件: {input_dir}")

    video_prefixes = _disambiguate_video_prefixes(videos, input_path, prefix)

    saved: List[str] = []
    failures = failures if failures is not None else []
    for video in videos:
        try:
            saved.extend(
                extract_video_frames(
                    input_video=str(video),
                    output_dir=output_dir,
                    frame_step=frame_step,
                    image_extension=image_extension,
                    quality=quality,
                    prefix=video_prefixes[video],
                    start_time=start_time,
                    end_time=end_time,
                    seek_mode=seek_mode,
                    overwrite=overwrite,
                )
            )
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            # 单个视频失败（坏文件、解码不支持等）不中断整批，记录后继续
            failures.append((video, str(exc)))
            log(f"[警告] 视频抽帧失败，跳过 {video}: {exc}", stream=sys.stderr)
        except Exception as exc:
            # 编程错误等意外异常：保留完整 traceback，便于定位问题，
            # 避免内部 bug 被误当作"单视频失败"而静默降级。
            import traceback

            failures.append((video, f"意外错误: {exc}"))
            log(f"[警告] 视频抽帧发生意外错误，跳过 {video}: {exc}", stream=sys.stderr)
            log(traceback.format_exc(), stream=sys.stderr)

    if failures:
        detail = "；".join(f"{video}: {reason}" for video, reason in failures)
        log(
            f"批量抽帧完成：成功 {len(videos) - len(failures)}/{len(videos)} 个视频，"
            f"失败 {len(failures)} 个，共抽取 {len(saved)} 帧，保存到: {output_dir}"
        )
        log(f"失败明细：{detail}", stream=sys.stderr)
        if not saved:
            raise RuntimeError(
                f"批量抽帧失败：{len(failures)} 个视频全部处理失败，详见日志"
            )
    else:
        log(f"共处理 {len(videos)} 个视频，抽取 {len(saved)} 帧，保存到: {output_dir}")
    return saved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py images import",
        description="视频抽帧工具：按指定间隔抽取视频画面并保存为图片。",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="输入视频文件路径，或包含视频文件的目录（目录将递归扫描所有视频）",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="输出图片保存目录（不存在会自动创建）"
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="抽帧间隔（必须为正整数），默认 1（逐帧抽取）。",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="jpg",
        help=(
            "输出图片格式，默认 jpg。"
            f"支持：{', '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))}"
        ),
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="输出图片质量（取值范围 1-100），默认 100（原始画质）。",
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
        help="开始抽取的时间（秒或 HH:MM:SS/MM:SS/SS 格式），默认从视频开头。",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="结束抽取的时间（秒或 HH:MM:SS/MM:SS/SS 格式），默认抽到视频结尾。",
    )
    parser.add_argument(
        "--seek-mode",
        type=str,
        default="decode_all",
        choices=sorted(SUPPORTED_SEEK_MODES),
        help=(
            "跳帧策略：decode_all=顺序解码每一帧（默认，安全且通用）；"
            "seek=主动跳到下一目标帧（大 frame_step 下更快，对部分编码可能略有偏差）。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中已存在的同名文件（默认 False，重名自动追加 _1/_2 后缀）。",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"输入路径不存在：{args.input}")
    if not (input_path.is_file() or input_path.is_dir()):
        raise ValueError(f"输入路径既不是文件也不是目录：{args.input}")

    output_path = Path(args.output)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"输出路径已存在但不是目录：{args.output}")

    if args.frame_step is not None and args.frame_step < 1:
        raise ValueError(f"--frame-step 必须为正整数，当前为 {args.frame_step}")

    if args.quality is not None and not (1 <= args.quality <= 100):
        raise ValueError(f"--quality 必须在 1~100 之间，当前为 {args.quality}")

    ext_normalized = "." + args.ext.lower().lstrip(".")
    if ext_normalized not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"不支持的图片格式：{args.ext!r}，请使用以下之一：{supported}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    """
    ``python scripts/vh.py images import`` 命令行入口。

    返回:
        0 成功；1 运行时错误；2 参数非法；130 用户中断。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[参数错误] {exc}", stream=sys.stderr)
        return 2

    try:
        if Path(args.input).is_dir():
            saved = extract_video_frames_batch(
                input_dir=args.input,
                output_dir=args.output,
                frame_step=args.frame_step,
                image_extension=args.ext,
                quality=args.quality,
                prefix=args.prefix,
                start_time=args.start_time,
                end_time=args.end_time,
                seek_mode=args.seek_mode,
                overwrite=args.overwrite,
            )
        else:
            saved = extract_video_frames(
                input_video=args.input,
                output_dir=args.output,
                frame_step=args.frame_step,
                image_extension=args.ext,
                quality=args.quality,
                prefix=args.prefix,
                start_time=args.start_time,
                end_time=args.end_time,
                seek_mode=args.seek_mode,
                overwrite=args.overwrite,
            )
    except KeyboardInterrupt:
        log("[已取消] 用户中断，部分帧可能未写入完成。", stream=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 视频抽帧失败：{exc}", stream=sys.stderr)
        return 1

    log(f"[完成] 共保存 {len(saved)} 张图片到 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
