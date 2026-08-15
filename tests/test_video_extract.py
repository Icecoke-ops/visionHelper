#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频抽帧（单文件 + 文件夹批量）测试。"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest


@pytest.fixture()
def make_video(tmp_path) -> callable:
    """生成一张测试视频（10 帧 64x64 纯色变化 mp4）。"""
    cv2 = pytest.importorskip("cv2")

    def _factory(relative: str, frames: int = 10) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (64, 64),
        )
        for i in range(frames):
            import numpy as np
            frame = np.full((64, 64, 3), i * 20 % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _factory


def _extract_single(input_path: Path, output_dir: Path, **kwargs) -> List[str]:
    from scripts.images.import_ import extract_video_frames
    return extract_video_frames(
        input_video=str(input_path),
        output_dir=str(output_dir),
        **kwargs,
    )


def test_extract_video_frames_single(tmp_path, make_video) -> None:
    video = make_video("a.mp4")
    output = tmp_path / "out"

    saved = _extract_single(video, output, frame_step=5, prefix="fr")

    assert len(saved) == 2
    names = sorted(p.name for p in output.iterdir())
    assert names == ["fr_000000.jpg", "fr_000001.jpg"]


def test_extract_video_frames_batch_recursive(tmp_path, make_video) -> None:
    make_video("a.mp4")
    make_video("b.mp4")
    make_video("sub/c.mkv")
    output = tmp_path / "out"

    from scripts.images.import_ import extract_video_frames_batch
    saved = extract_video_frames_batch(
        input_dir=str(tmp_path),
        output_dir=str(output),
        frame_step=3,
        prefix="fr",
    )

    # 每个视频 10 帧按 step=3 应抽 4 帧，共 3 个视频
    assert len(saved) == 12
    names = sorted(p.name for p in output.iterdir())
    assert "fr_a_000000.jpg" in names
    assert "fr_b_000000.jpg" in names
    assert "fr_sub_c_000000.jpg" in names
    assert len(names) == 12
    assert len({p.stem.rsplit("_", 1)[0] for p in output.iterdir()}) == 3


def test_extract_video_frames_batch_empty_dir(tmp_path) -> None:
    from scripts.images.import_ import extract_video_frames_batch

    with pytest.raises(ValueError):
        extract_video_frames_batch(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )


def test_extract_video_frames_batch_prefix_collision_disambiguated(
    tmp_path, make_video
) -> None:
    """rel_key 冲突（顶层 a_b.mp4 与子目录 a/b.mp4 都映射为 a_b）时不重名。"""
    make_video("a_b.mp4")
    make_video("a/b.mp4")
    output = tmp_path / "out"

    from scripts.images.import_ import extract_video_frames_batch
    saved = extract_video_frames_batch(
        input_dir=str(tmp_path),
        output_dir=str(output),
        frame_step=10,
        prefix="fr",
    )

    names = sorted(p.name for p in output.iterdir())
    assert len(saved) == 2
    assert len(names) == len(set(names))
    assert all("a_b" in name for name in names)


def test_extract_video_frames_batch_partial_failure(tmp_path, make_video) -> None:
    """坏视频被跳过，其余视频正常抽帧，不中断整批。"""
    make_video("a.mp4")
    (tmp_path / "broken.mp4").write_bytes(b"not a real video")
    output = tmp_path / "out"

    from scripts.images.import_ import extract_video_frames_batch
    saved = extract_video_frames_batch(
        input_dir=str(tmp_path),
        output_dir=str(output),
        frame_step=5,
        prefix="fr",
    )

    assert len(saved) == 2
    assert "fr_a_000000.jpg" in {p.name for p in output.iterdir()}


def test_extract_video_frames_batch_partial_failure_out_param(
    tmp_path, make_video
) -> None:
    """传入 ``failures`` 列表时可感知部分失败明细。"""
    make_video("a.mp4")
    (tmp_path / "broken.mp4").write_bytes(b"not a real video")
    output = tmp_path / "out"

    from scripts.images.import_ import extract_video_frames_batch
    failures: list = []
    saved = extract_video_frames_batch(
        input_dir=str(tmp_path),
        output_dir=str(output),
        frame_step=5,
        prefix="fr",
        failures=failures,
    )

    assert len(saved) == 2
    assert len(failures) == 1
    assert failures[0][0].name == "broken.mp4"


def test_extract_video_frames_batch_all_fail_raises(tmp_path) -> None:
    """所有视频都失败时抛出 RuntimeError，避免静默"成功"。"""
    (tmp_path / "broken1.mp4").write_bytes(b"not a real video")
    (tmp_path / "broken2.mp4").write_bytes(b"not a real video")
    output = tmp_path / "out"

    from scripts.images.import_ import extract_video_frames_batch
    with pytest.raises(RuntimeError):
        extract_video_frames_batch(
            input_dir=str(tmp_path),
            output_dir=str(output),
        )


def test_extract_video_frames_batch_api(tmp_path, make_video) -> None:
    make_video("a.mp4")
    make_video("b.mp4")
    output = tmp_path / "out"

    from scripts.api import VideoAPI
    saved = VideoAPI.extract_video_frames_batch(
        input_dir=str(tmp_path),
        output_dir=str(output),
        frame_step=5,
    )

    assert len(saved) == 4