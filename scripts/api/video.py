"""视频处理相关 API。"""

from __future__ import annotations

from typing import List, Optional

from scripts.api._validators import (
    _require_existing_dir,
    _require_existing_file,
    _require_in_range,
    _require_non_empty_str,
    _require_positive_int,
)


class VideoAPI:
    """视频处理相关 API。"""

    @staticmethod
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
        """
        将视频抽帧并保存为图片。

        参数:
            input_video: 输入视频文件路径。
            output_dir: 输出图片保存目录。
            frame_step: 抽帧间隔，必须 >= 1，默认 1（逐帧抽取）。
            image_extension: 输出图片格式，默认 jpg。
            quality: 输出图片质量 1-100，默认 100（原始画质）。
            prefix: 输出文件名前缀，默认 "frame"。
            start_time: 开始抽取的时间（秒或时间字符串），默认从视频开头。
            end_time: 结束抽取的时间（秒或时间字符串），默认抽到视频结尾。
            seek_mode: 跳帧策略，``decode_all`` 顺序解码（默认）或 ``seek``
                直接跳到下一目标帧。
            overwrite: 是否覆盖已存在的同名文件，默认 False（重名追加 _1/_2 后缀）。

        返回:
            保存的图片路径列表（按帧顺序）。
        """
        _require_existing_file(input_video, "input_video")
        _require_non_empty_str(output_dir, "output_dir")
        _require_positive_int(frame_step, "frame_step")
        _require_in_range(quality, "quality", 1, 100)
        _require_non_empty_str(image_extension, "image_extension")
        if seek_mode not in {"decode_all", "seek"}:
            raise ValueError(
                f"不支持的 seek_mode: {seek_mode!r}，仅支持 'decode_all' 和 'seek'"
            )
        if not isinstance(overwrite, bool):
            raise ValueError(f"overwrite 必须为 bool 类型，当前值: {overwrite!r}")

        from scripts.images.import_ import extract_video_frames

        return extract_video_frames(
            input_video=input_video,
            output_dir=output_dir,
            frame_step=frame_step,
            image_extension=image_extension,
            quality=quality,
            prefix=prefix,
            start_time=start_time,
            end_time=end_time,
            seek_mode=seek_mode,
            overwrite=overwrite,
        )
