#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 页面基类模块。

提供 :class:`BasePage` 与 :class:`BaseTaskPage`，封装统一的页面容器样式、
表单控件以及子进程启动能力。

整体布局约定：

    QScrollArea  ← 提供垂直滚动
        └── QFrame[variant="card"]   ← 白色卡片容器
                └── content_layout    ← 子页面真正放置控件的布局

子页面通过 ``self.content_layout`` 添加自己的控件，无需关心外层卡片
与边距，所有间距、圆角、边框统一由 :mod:`gui.theme` 提供。
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.config import app_root, is_frozen
from gui.run_log_dialog import RunLogDialog
from gui.widgets import FormRow, SecondaryButton



class BasePage(QWidget):
    """所有子页面的基类，提供统一的卡片式滚动容器。"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 外层滚动区，长内容时不会撑出主窗口。
        # 注意：这里使用 QPalette 而非 setStyleSheet 来设置 viewport 背景色，
        # 因为 QScrollArea.viewport().setStyleSheet("background-color: ...") 会
        # 让 Qt 把这条规则当作 ``QWidget { background-color: ... }`` 应用到
        # 整棵子树（包括 QPushButton 的背景），从而覆盖语义化按钮的
        # ``[variant="primary"]`` 等彩色背景，导致按钮变成应用背景色一片。
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        viewport = scroll.viewport()
        viewport.setAutoFillBackground(True)
        palette = viewport.palette()
        palette.setColor(QPalette.Window, QColor(theme.COLOR_BG_APP))
        viewport.setPalette(palette)

        # 内层卡片：白底圆角，统一承载所有页面内容
        self.card = QFrame()
        self.card.setProperty("variant", "card")
        theme.refresh_widget_style(self.card)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(
            theme.CARD_PADDING,
            theme.CARD_PADDING,
            theme.CARD_PADDING,
            theme.CARD_PADDING,
        )
        card_layout.setSpacing(theme.SPACING_MD)

        # 子页面真正放置控件的布局
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(theme.SPACING_MD)
        card_layout.addLayout(self.content_layout)
        card_layout.addStretch(1)

        # 卡片外的留白
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(
            theme.PAGE_MARGIN,
            theme.PAGE_MARGIN,
            theme.PAGE_MARGIN,
            theme.PAGE_MARGIN,
        )
        wrapper_layout.setSpacing(theme.SPACING_MD)
        wrapper_layout.addWidget(self.card)
        wrapper_layout.addStretch(1)

        scroll.setWidget(wrapper)
        outer.addWidget(scroll)

    def _work_dir(self) -> str:
        """从父窗口链中查找并返回当前工作目录。

        约定主窗口（或其上层容器）会维护 ``work_dir`` 属性，子页面
        通过该方法即可读取最新值，而无需直接持有主窗口引用。
        """
        parent = self.parent()
        while parent is not None:
            work_dir = getattr(parent, "work_dir", "")
            if work_dir:
                return work_dir
            parent = parent.parent()
        return ""


class BaseTaskPage(BasePage):
    """任务页面的基类，提供表单控件和子进程启动能力。"""

    PYTHON = sys.executable

    def _add_file_row(self, label: str, is_directory: bool) -> QLineEdit:
        """在内容区追加"标签 + 路径输入框 + 浏览按钮"一行，并返回输入框。"""
        edit = QLineEdit()
        edit.setPlaceholderText("点击右侧按钮选择路径")
        edit.setMinimumWidth(300)

        browse_btn = SecondaryButton("浏览")
        browse_btn.clicked.connect(
            lambda _, e=edit, d=is_directory: self._browse(e, d)
        )

        # 输入框 + 浏览按钮组合放进一个容器，再统一交给 FormRow 处理标签对齐
        combo = QWidget()
        combo_layout = QHBoxLayout(combo)
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.setSpacing(theme.SPACING_SM)
        combo_layout.addWidget(edit, 1)
        combo_layout.addWidget(browse_btn)

        self.content_layout.addWidget(FormRow(label, combo))
        return edit

    def _add_widget_row(self, label: str, widget: QWidget):
        """在内容区追加"固定宽度标签 + 控件"一行。"""
        self.content_layout.addWidget(FormRow(label, widget))

    def _browse(self, edit: QLineEdit, is_directory: bool):
        """弹出文件/目录选择对话框，默认从工作目录开始。"""
        start_dir = self._work_dir() or ""
        if is_directory:
            path = QFileDialog.getExistingDirectory(self, "选择目录", start_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", start_dir)
        if path:
            edit.setText(path)

    def _python_env(self) -> str:
        """从父窗口链中查找并返回当前 Python 可执行文件路径。

        - 开发态（非 frozen）：未设置时回退到 ``sys.executable``，便于直接
          ``python -m gui.app`` 调试。
        - 打包态（frozen）：``sys.executable`` 是 GUI 自身的 bootloader，
          内部并不包含 torch / ultralytics 等重型依赖，绝不能用它来跑
          ``scripts``。此时若用户未选择 Python 环境，本方法返回空串，
          由 :meth:`_start_subprocess` 统一弹窗提示。
        """
        parent = self.parent()
        while parent is not None:
            python_env = getattr(parent, "python_env", "")
            if python_env:
                return python_env
            parent = parent.parent()
        if is_frozen():
            return ""
        return self.PYTHON

    def _start_subprocess(self, arguments: list, title: str):
        """弹出日志窗口并使用当前选定的 Python 环境启动子进程执行任务。

        子进程启动时会：

        1. 将工作目录设置为 :func:`gui.config.app_root`，保证 ``-m scripts.xxx``
           能正确定位到 ``scripts`` 包；
        2. 将 ``app_root`` 注入到子进程的 ``PYTHONPATH``，作为第二重保险，
           即使用户在调用前手动改过 ``cwd`` 也不会影响脚本定位。

        在打包态下，如果用户未配置 Python 环境，会弹窗提示并取消执行。
        """
        python_path = self._python_env()
        if not python_path:
            QMessageBox.warning(
                self,
                "未配置 Python 环境",
                "当前为打包发布版本，必须在导航栏下方手动选择带有所需依赖"
                "（torch / ultralytics 等）的 Python 可执行文件后再运行任务。",
            )
            return

        root = str(app_root())
        dialog = RunLogDialog(
            python_path,
            arguments,
            title=title,
            parent=self,
            working_dir=root,
            extra_pythonpath=[root],
        )
        dialog.exec_()
