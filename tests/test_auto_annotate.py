#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``scripts.datasets.auto.auto_annotate`` 的单元测试。

通过 ``unittest.mock`` 替换 ``ultralytics.YOLO``，验证自动标注在遇见截断/损坏
图片时能够跳过单张图片并继续处理其余图片，而不是整体失败。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest import mock

import pytest

from scripts.common.config import STATUS_UNANNOTATED
from scripts.datasets.auto import auto_annotate


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

class _FakeBoxes:
    """模拟 Ultralytics 的 result.boxes 对象，支持 len() 与 tolist 访问。"""

    def __init__(
        self,
        cls_data: list,
        conf_data: list,
        xyxy_data: list,
    ) -> None:
        self._cls = cls_data
        self._conf = conf_data
        self._xyxy = xyxy_data
        self.cls = SimpleNamespace(tolist=lambda: self._cls)
        self.conf = SimpleNamespace(tolist=lambda: self._conf)
        self.xyxy = SimpleNamespace(tolist=lambda: self._xyxy)

    def __len__(self) -> int:
        return len(self._cls)

    def tolist(self) -> list:
        return [self._cls, self._conf, self._xyxy]


def _make_yolo_result(
    *,
    width: int = 32,
    height: int = 32,
    boxes: list[tuple[int, float, float, float, float, float]] | None = None,
) -> SimpleNamespace:
    """构造一个伪造的 Ultralytics 推理结果对象（detect 任务）。

    Args:
        width: 原始图片宽度。
        height: 原始图片高度。
        boxes: 每个检测框的元组 ``(cls_id, conf, x1, y1, x2, y2)``。
    """
    boxes = boxes or []
    cls = [b[0] for b in boxes]
    conf = [b[1] for b in boxes]
    xyxy = [list(b[2:]) for b in boxes]

    return SimpleNamespace(
        orig_shape=(height, width),
        boxes=_FakeBoxes(cls, conf, xyxy),
    )


@pytest.fixture
def patch_yolo():
    """提供一个 mock ``ultralytics.YOLO`` 的上下文管理器。"""
    return mock.patch("ultralytics.YOLO")


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #

def test_truncated_image_is_skipped_and_normal_image_is_annotated(
    tmp_path: Path,
    make_image: Callable[..., Path],
    patch_yolo,
) -> None:
    """一张截断 JPEG 应被跳过，正常 PNG 继续被标注。"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"")  # mock 后不会真正加载权重

    normal_img = make_image(images_dir / "normal.png", size=(32, 32), color=(255, 0, 0))

    # 构造一张截断的 JPEG：只写入 JPEG 文件头的一部分，导致 Pillow 读取时抛出
    # ``OSError: image file is truncated``。
    truncated_img = images_dir / "truncated.jpg"
    truncated_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    fake_result = _make_yolo_result(
        width=32,
        height=32,
        boxes=[(0, 0.8, 8.0, 8.0, 24.0, 24.0)],
    )

    with patch_yolo as mock_yolo_cls:
        mock_model = mock_yolo_cls.return_value
        mock_model.names = {0: "cat"}
        mock_model.predict.return_value = iter([fake_result])

        result = auto_annotate(
            work_dir=str(images_dir),
            model_path=str(model_path),
            task="detect",
            threshold=0.25,
            include_unannotated=True,
            batch_size=8,
        )

    assert result["total"] == 2
    assert result["annotated"] == 1
    assert result["skipped"] == 1
    assert result["by_type"][STATUS_UNANNOTATED] == 1

    # 正常图片的 JSON 标注应已生成
    json_path = images_dir / "normal.json"
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["imagePath"] == "normal.png"
    assert len(data["shapes"]) == 1
    assert data["shapes"][0]["label"] == "cat"
    assert data["shapes"][0]["shape_type"] == "rectangle"

    # 截断图片不应生成 JSON
    assert not (images_dir / "truncated.json").exists()

    # 模型只应被传入正常图片
    mock_model.predict.assert_called_once()
    call_kwargs = mock_model.predict.call_args
    assert call_kwargs.kwargs.get("stream") is True
    sources = call_kwargs.kwargs.get("source")
    assert sources == [str(normal_img)]


def test_only_truncated_images_result_in_zero_annotated(
    tmp_path: Path,
    patch_yolo,
) -> None:
    """目录中全是截断图片时，任务不抛异常，标注数为 0。"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"")

    for name in ("bad1.jpg", "bad2.jpg"):
        (images_dir / name).write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    with patch_yolo as mock_yolo_cls:
        mock_model = mock_yolo_cls.return_value
        mock_model.predict.return_value = iter([])

        result = auto_annotate(
            work_dir=str(images_dir),
            model_path=str(model_path),
            task="detect",
            include_unannotated=True,
            batch_size=8,
        )

    assert result["total"] == 2
    assert result["annotated"] == 0
    assert result["skipped"] == 2
    # 没有任何可读图片，predict 不应被调用
    mock_model.predict.assert_not_called()


def test_unreadable_image_by_value_error_is_skipped(
    tmp_path: Path,
    make_image: Callable[..., Path],
    patch_yolo,
) -> None:
    """Pillow 读取时抛 ``ValueError`` 的异常图片也应被跳过。"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"")

    normal_img = make_image(images_dir / "normal.png", size=(16, 16))

    # 扩展名使用 .gif，但内容是非法 PPM（maxval=0）。Pillow 按文件内容识别格式，
    # 在 load() 时会抛出 ValueError，验证 _check_image_readable 对 ValueError 分支的处理。
    bad_img = images_dir / "bad.gif"
    bad_img.write_bytes(b"P6\n1 1\n0\n\x00\x00\x00")

    fake_result = _make_yolo_result(
        width=16,
        height=16,
        boxes=[(0, 0.9, 2.0, 2.0, 14.0, 14.0)],
    )

    with patch_yolo as mock_yolo_cls:
        mock_model = mock_yolo_cls.return_value
        mock_model.names = {0: "cat"}
        mock_model.predict.return_value = iter([fake_result])

        result = auto_annotate(
            work_dir=str(images_dir),
            model_path=str(model_path),
            task="detect",
            include_unannotated=True,
            batch_size=8,
        )

    assert result["total"] == 2
    assert result["annotated"] == 1
    assert result["skipped"] == 1
    assert (images_dir / "normal.json").is_file()
    assert not (images_dir / "bad.json").exists()

    mock_model.predict.assert_called_once()
    sources = mock_model.predict.call_args.kwargs.get("source")
    assert sources == [str(normal_img)]
