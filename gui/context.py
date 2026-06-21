#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 全局应用上下文。

:class:`AppContext` 集中保存以下跨页面共享的状态：

- ``work_dir``：当前选择的工作目录。
- ``python_env``：用户指定的 Python 解释器路径（可执行文件）。

并对外暴露以下 Qt 信号：

- ``work_dir_changed(str)``：工作目录变化时发射，新值会随信号一起传出。
- ``python_env_changed(str)``：Python 解释器变化时发射。

引入 ``AppContext`` 的目的是替代 :mod:`gui.base_pages` 中 "通过 parent
链向上查找 work_dir / python_env 属性" 的写法。子页面持有 ``ctx``
引用即可：

>>> path = self.ctx.work_dir
>>> self.ctx.work_dir_changed.connect(self._on_work_dir_changed)

这样可以避免父子嵌套关系变化时静默丢失上下文，并显著简化测试。
"""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class AppContext(QObject):
    """visionHelper GUI 的全局上下文对象。

    所有需要跨页面访问的"应用级别"状态都集中在这里。``MainWindow`` 在
    构造时创建并持有一份实例，并将其注入到每个子页面（通过
    :class:`gui.base_pages.BasePage` 的 ``ctx`` 属性）。
    """

    #: 工作目录变化时发射，参数为新的目录路径（可能为空）。
    work_dir_changed = pyqtSignal(str)

    #: Python 解释器路径变化时发射，参数为新的解释器路径（可能为空）。
    python_env_changed = pyqtSignal(str)

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._work_dir: str = ""
        self._python_env: str = ""

    # ------------------------------------------------------------------
    # work_dir
    # ------------------------------------------------------------------

    @property
    def work_dir(self) -> str:
        """当前工作目录（绝对路径），未设置时为空串。"""
        return self._work_dir

    def set_work_dir(self, value: str) -> None:
        """更新工作目录，仅在值发生变化时发射 ``work_dir_changed``。"""
        new_value = (value or "").strip()
        if new_value == self._work_dir:
            return
        self._work_dir = new_value
        self.work_dir_changed.emit(new_value)

    # ------------------------------------------------------------------
    # python_env
    # ------------------------------------------------------------------

    @property
    def python_env(self) -> str:
        """当前 Python 可执行文件路径，未设置时为空串。"""
        return self._python_env

    def set_python_env(self, value: str) -> None:
        """更新 Python 解释器路径，仅在值发生变化时发射信号。"""
        new_value = (value or "").strip()
        if new_value == self._python_env:
            return
        self._python_env = new_value
        self.python_env_changed.emit(new_value)
