#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端日志模块（``scripts`` 包专用）。

集中实现两类与日志相关的能力：

- :func:`log`：所有脚本/核心实现统一使用的行式日志输出函数（始终 ``flush``），
  替代各模块此前重复定义的 ``_log()`` 工具；
- :class:`ProgressLogger`：用整行 ``print`` 取代 ``tqdm`` 的轻量级进度记录器，
  适配 GUI 子进程日志面板（不会用 ``\\r`` 覆盖整行，因而每条记录都能逐行
  正确显示）。

模块只依赖标准库与 :mod:`scripts.common.config`，零重依赖，可在 GUI 进程中安全 import。
"""

from __future__ import annotations

import os
import sys
import time
from typing import IO, Any, Optional

from scripts.common.config import (
    PROGRESS_DEFAULT_MIN_INTERVAL,
    PROGRESS_DEFAULT_STEP_PERCENT,
    PROGRESS_DISABLE_ENV,
)


def log(*args: Any, stream: Optional[IO[str]] = None, **kwargs: Any) -> None:
    """
    统一的行式日志输出。

    与标准 :func:`print` 一致的位置参数与关键字参数，但额外保证 ``flush=True``
    以便 GUI 子进程能实时读取；并提供 ``stream`` 关键字用于将日志写入
    ``sys.stderr`` 等非默认流。

    参数:
        *args: 与 :func:`print` 相同的位置参数。
        stream: 可选输出流（默认 ``sys.stdout``）。
        **kwargs: 其余传递给 :func:`print` 的关键字参数（``end`` / ``sep`` 等）。
    """
    kwargs.setdefault("flush", True)
    if stream is not None:
        kwargs["file"] = stream
    print(*args, **kwargs)


class ProgressLogger:
    """
    轻量级进度日志记录器（替代 tqdm）。

    与 tqdm 不同，本类通过普通 :func:`print` 输出 **整行** 进度日志（不使用回车
    ``\\r`` 覆盖），确保 GUI 子进程的日志面板能逐行正确显示。

    为避免刷屏，仅在以下情况触发输出：

    - 完成项数达到下一个百分比里程碑（默认每 :data:`scripts.common.config.PROGRESS_DEFAULT_STEP_PERCENT` 个百分点）；
    - 距离上一次输出的时间间隔超过 ``min_interval`` 秒（默认
      :data:`scripts.common.config.PROGRESS_DEFAULT_MIN_INTERVAL`）；
    - 首次（``0/total``）与最终（``total/total``）一定输出。

    支持通过环境变量 :data:`scripts.common.config.PROGRESS_DISABLE_ENV`（``=1``）整体
    关闭进度输出，但首尾两条仍会强制输出，便于知晓任务起止。

    典型用法::

        progress = ProgressLogger(total=len(items), desc="处理")
        for item in items:
            ...
            progress.update(1)
        progress.close()
    """

    def __init__(
            self,
            total: int,
            desc: str = "进度",
            *,
            step_percent: float = PROGRESS_DEFAULT_STEP_PERCENT,
            min_interval: float = PROGRESS_DEFAULT_MIN_INTERVAL,
            stream: Optional[IO[str]] = None,
    ) -> None:
        self.total: int = max(int(total), 0)
        self.desc: str = desc
        self.step_percent: float = max(float(step_percent), 0.1)
        self.min_interval: float = max(float(min_interval), 0.0)
        self._stream: IO[str] = stream if stream is not None else sys.stdout
        self._disabled: bool = (
            os.environ.get(PROGRESS_DISABLE_ENV, "").strip() == "1"
        )
        self._count: int = 0
        self._next_percent: float = 0.0
        self._last_emit_time: float = 0.0
        self._closed: bool = False
        # 起始日志（即使禁用也输出一行，便于知道开始）
        self._emit(force=True)

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #

    def update(self, n: int = 1) -> None:
        """累加 ``n`` 项进度并按节流策略输出日志。"""
        if self._closed or n <= 0:
            return
        if self.total:
            self._count = min(self._count + n, self.total)
        else:
            self._count += n
        self._maybe_emit()

    def set_total(self, total: int) -> None:
        """动态修正 total（在某些场景下任务总数事后才确定）。"""
        self.total = max(int(total), 0)

    def close(self) -> None:
        """输出最终进度并关闭。"""
        if self._closed:
            return
        if self.total > 0:
            self._count = self.total
        self._emit(force=True)
        self._closed = True

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _percent(self) -> float:
        if self.total <= 0:
            return 100.0
        return self._count * 100.0 / self.total

    def _maybe_emit(self) -> None:
        if self._disabled:
            return
        percent = self._percent()
        now = time.monotonic()
        reached_milestone = percent + 1e-9 >= self._next_percent
        time_ok = (now - self._last_emit_time) >= self.min_interval
        if reached_milestone and time_ok:
            self._emit()
        elif self.total > 0 and self._count >= self.total:
            # 兜底：达到总数时输出
            self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        if self._disabled and not force:
            return
        percent = self._percent()
        # 更新下一个百分比里程碑
        if not self._disabled:
            while self._next_percent <= percent + 1e-9:
                self._next_percent += self.step_percent
        if self.total > 0:
            msg = f"[{self.desc}] {self._count}/{self.total} ({percent:.1f}%)"
        else:
            msg = f"[{self.desc}] {self._count}"
        try:
            print(msg, file=self._stream, flush=True)
        except Exception as exc:  # noqa: BLE001 — 任何 IO 失败都不应影响主流程
            sys.stderr.write(f"[ProgressLogger] 输出失败: {exc}\n")
        self._last_emit_time = time.monotonic()


__all__ = ["log", "ProgressLogger"]
