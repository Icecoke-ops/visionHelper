#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 配置持久化工具模块。

集中封装基于 :class:`QSettings` 的读写逻辑，避免在多个模块中
（``gui.app`` / ``gui.welcome_page`` 等）散落同样的键名常量与
类型转换代码。

约定：

- 所有 ``QSettings`` 实例使用相同的组织名与应用名
  ``("visionHelper", "MainWindow")``，与历史版本兼容。
- 路径相关的值统一存储为字符串（绝对路径），返回时若为空则统一
  返回空串而非 ``None``，方便调用方直接赋给 ``QLineEdit``。
- 列表类值在某些平台上 QSettings 会反序列化为单个字符串，本模块
  会在读取时统一归一化为 ``list[str]``。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QSettings


# ---------------------------------------------------------------------------
# QSettings 中使用的键名常量
# ---------------------------------------------------------------------------

# 主窗口配置（组织名/应用名）
_ORG = "visionHelper"
_APP = "MainWindow"

# 当前工作目录
_KEY_WORK_DIR = "work_dir"

# 当前 Python 解释器
_KEY_PYTHON_ENV = "python_env"

# 最近使用过的工作目录列表（按最近使用优先）
_KEY_RECENT_WORK_DIRS = "recent_work_dirs"

# 主窗口几何信息（保留接口，目前 MainWindow 暂未持久化）
_KEY_WINDOW_GEOMETRY = "window_geometry"

# 历史目录列表的最大长度，超过的会被截断
MAX_RECENT_WORK_DIRS = 20


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _settings() -> QSettings:
    """获取统一配置存储句柄。"""
    return QSettings(_ORG, _APP)


def _normalize_str_list(raw) -> List[str]:
    """把 QSettings 读出的列表值归一化为 ``list[str]``。"""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if item]


# ---------------------------------------------------------------------------
# 工作目录 / Python 环境
# ---------------------------------------------------------------------------


def load_work_dir() -> str:
    """读取上次保存的工作目录。未配置时返回空串。"""
    return _settings().value(_KEY_WORK_DIR, "", str) or ""


def save_work_dir(path: str) -> None:
    """保存当前工作目录（自动去空白）。"""
    _settings().setValue(_KEY_WORK_DIR, (path or "").strip())


def load_python_env() -> str:
    """读取上次保存的 Python 解释器路径。未配置时返回空串。"""
    return _settings().value(_KEY_PYTHON_ENV, "", str) or ""


def save_python_env(path: str) -> None:
    """保存当前 Python 解释器路径（自动去空白）。"""
    _settings().setValue(_KEY_PYTHON_ENV, (path or "").strip())


# ---------------------------------------------------------------------------
# 最近工作目录列表
# ---------------------------------------------------------------------------


def load_recent_dirs() -> List[str]:
    """读取最近工作目录列表（按最近使用优先排序）。"""
    return _normalize_str_list(_settings().value(_KEY_RECENT_WORK_DIRS, [], list))


def save_recent_dirs(dirs: List[str]) -> None:
    """保存最近工作目录列表。

    自动执行：去空、去重（保持顺序）、长度上限截断。
    """
    cleaned: List[str] = []
    seen = set()
    for d in dirs or []:
        if not d or d in seen:
            continue
        seen.add(d)
        cleaned.append(d)
    _settings().setValue(_KEY_RECENT_WORK_DIRS, cleaned[:MAX_RECENT_WORK_DIRS])


def promote_recent_dir(path: str) -> List[str]:
    """把 ``path`` 移动到最近列表最前面，返回更新后的列表。

    若 ``path`` 已存在则前置，否则插入到首位；超出 :data:`MAX_RECENT_WORK_DIRS`
    的尾部会被截断。
    """
    if not path:
        return load_recent_dirs()
    recent = load_recent_dirs()
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    recent = recent[:MAX_RECENT_WORK_DIRS]
    save_recent_dirs(recent)
    return recent


# ---------------------------------------------------------------------------
# 主窗口几何（预留，便于将来恢复窗口大小/位置）
# ---------------------------------------------------------------------------


def load_window_geometry() -> Optional[bytes]:
    """读取主窗口保存的几何信息，未保存时返回 ``None``。"""
    value = _settings().value(_KEY_WINDOW_GEOMETRY)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def save_window_geometry(data: bytes) -> None:
    """保存主窗口几何信息。"""
    if data is None:
        return
    _settings().setValue(_KEY_WINDOW_GEOMETRY, bytes(data))
