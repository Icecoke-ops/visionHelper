"""``scripts.images`` 包：图片处理（增强 / 去重 / 视频抽帧）。

遵循项目"零副作用导入"约定：导入本包不加载任何子模块，外部依赖
（``numpy`` / ``Pillow`` / ``cv2`` 等）仅在真正访问具体功能时才被引入，
避免仅使用视频抽帧时也被迫安装去重所需的依赖。

对外按需导出：

- :func:`augment_image` → :mod:`scripts.images.augment`
- :func:`deduplicate` → :mod:`scripts.images.dedup`
- :func:`extract_video_frames` / :func:`extract_video_frames_batch` → :mod:`scripts.images.import_`
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "augment_image",
    "deduplicate",
    "extract_video_frames",
    "extract_video_frames_batch",
]

_LAZY_ATTRS: dict[str, str] = {
    "augment_image": "scripts.images.augment",
    "deduplicate": "scripts.images.dedup",
    "extract_video_frames": "scripts.images.import_",
    "extract_video_frames_batch": "scripts.images.import_",
}


def __getattr__(name: str) -> Any:
    """PEP 562 懒加载：首次访问时按需导入对应子模块并缓存到模块命名空间。"""
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value
