#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pytest 全局夹具。

负责把仓库根目录注入到 ``sys.path``，以保证 ``import scripts.xxx`` 在
任意目录下执行 ``pytest`` 都能成功；同时提供若干常用的样本数据夹具。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest

# 仓库根目录加入到 sys.path 最前，确保 scripts 包优先被解析
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 子进程中 tqdm 也会因这个变量而被禁用；测试同样关掉以减少噪声
os.environ.setdefault("VH_NO_PROGRESS", "1")


@pytest.fixture()
def make_image() -> Callable[..., Path]:
    """生成一张测试图片（默认 32x32 纯色 PNG）。

    用法::

        path = make_image(tmp_path / "a.jpg", size=(64, 64), color=(255, 0, 0))
    """
    from PIL import Image

    def _factory(
        path: Path,
        size: tuple[int, int] = (32, 32),
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> Path:
        img = Image.new("RGB", size, color=color)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    return _factory


@pytest.fixture()
def make_annotation() -> Callable[..., Path]:
    """生成一份 X-AnyLabeling JSON 标注文件。

    用法::

        ann_path = make_annotation(
            json_path=tmp_path / "a.json",
            image_path=tmp_path / "a.jpg",
            shapes=[{"label": "cat", "shape_type": "rectangle",
                     "points": [[0, 0], [10, 10]]}],
            auto_annotated_time="2024-01-01T00:00:00",
        )
    """

    def _factory(
        json_path: Path,
        image_path: Optional[Path] = None,
        shapes: Optional[list] = None,
        auto_annotated_time: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> Path:
        data: Dict = {
            "version": "2.4.0",
            "flags": {},
            "shapes": list(shapes or []),
            "imagePath": image_path.name if image_path else "",
            "imageData": None,
            "imageHeight": 32,
            "imageWidth": 32,
        }
        if auto_annotated_time is not None:
            data["auto_annotated_time"] = auto_annotated_time
        if extra:
            data.update(extra)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return json_path

    return _factory
