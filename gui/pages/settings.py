#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper 设置页面。

集中管理应用级偏好设置：
    - 工作目录：任务运行时的默认路径基准；
    - Python 环境：运行 ``scripts`` 脚本所使用的解释器；
    - 关闭项目：返回欢迎页，用于切换 / 新建工作目录。

所有持久化逻辑复用 :mod:`gui.settings`，视觉风格统一来自
:mod:`gui.theme` 与 :mod:`gui.components.widgets`。
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QWidget,
)

from gui import settings, theme
from gui.context import AppContext
from gui.pages.base import BasePage
from gui.components.widgets import (
    DangerButton,
    FormRow,
    HSeparator,
    MutedLabel,
    SecondaryButton,
    SectionTitle,
)


class SettingsPage(BasePage):
    """visionHelper 设置页面：工作目录 / Python 环境 / 关闭项目。"""

    #: 点击"关闭项目"时发射，由主窗口切换到欢迎页。
    close_project_requested = pyqtSignal()

    def __init__(self, parent: QWidget = None, ctx: Optional[AppContext] = None):
        super().__init__(parent, ctx)
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ===== 工作目录 =====
        self.content_layout.addWidget(SectionTitle("工作目录"))
        self.work_dir_edit = QLineEdit()
        self.work_dir_edit.setPlaceholderText(
            "选择工作环境目录，后续选路径将默认从此处开始"
        )
        self.work_dir_edit.editingFinished.connect(self._save_work_dir)
        if self.ctx is not None:
            self.work_dir_edit.textChanged.connect(self.ctx.set_work_dir)
        self.content_layout.addWidget(
            FormRow(
                "工作目录：",
                self._wrap_with_browse(self.work_dir_edit, self._browse_work_dir),
            )
        )
        self.content_layout.addWidget(
            MutedLabel(
                "任务运行所需的目录，图片目录 / 模型路径等选择将默认从该目录开始。"
            )
        )

        self.content_layout.addWidget(HSeparator())

        # ===== Python 环境 =====
        self.content_layout.addWidget(SectionTitle("Python 环境"))
        self.python_env_edit = QLineEdit()
        self.python_env_edit.setPlaceholderText(
            "选择 Python 可执行文件（例如 /path/to/venv/bin/python），"
            "脚本将通过该环境运行"
        )
        self.python_env_edit.editingFinished.connect(self._save_python_env)
        if self.ctx is not None:
            self.python_env_edit.textChanged.connect(self.ctx.set_python_env)
        self.content_layout.addWidget(
            FormRow(
                "Python 环境：",
                self._wrap_with_browse(self.python_env_edit, self._browse_python_env),
            )
        )
        self.content_layout.addWidget(
            MutedLabel(
                "运行 scripts 脚本所使用的解释器，需包含 torch / ultralytics 等依赖。"
            )
        )

        self.content_layout.addWidget(HSeparator())

        # ===== 项目 =====
        self.content_layout.addWidget(SectionTitle("项目"))
        close_btn = DangerButton("关闭项目")
        close_btn.setMinimumWidth(140)
        close_btn.clicked.connect(self.close_project_requested.emit)
        self.content_layout.addWidget(close_btn, alignment=Qt.AlignLeft)
        self.content_layout.addWidget(
            MutedLabel("关闭当前项目并返回欢迎页，可切换或新建工作目录。")
        )

    def _wrap_with_browse(self, edit: QLineEdit, browse_handler) -> QWidget:
        """把输入框与"浏览"按钮组合为一行。"""
        combo = QWidget()
        layout = QHBoxLayout(combo)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SM)
        layout.addWidget(edit, 1)
        browse_btn = SecondaryButton("浏览")
        browse_btn.clicked.connect(browse_handler)
        layout.addWidget(browse_btn)
        return combo

    # ------------------------------------------------------------------
    # 浏览 / 保存 / 加载
    # ------------------------------------------------------------------

    def _browse_work_dir(self):
        start = self.work_dir_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "选择工作目录", start)
        if path:
            self.work_dir_edit.setText(path)
            self._save_work_dir()
            settings.promote_recent_dir(path)

    def _browse_python_env(self):
        start = self.python_env_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "选择 Python 可执行文件", start)
        if path:
            self.python_env_edit.setText(path)
            self._save_python_env()

    def _save_work_dir(self):
        settings.save_work_dir(self.work_dir_edit.text())

    def _save_python_env(self):
        settings.save_python_env(self.python_env_edit.text())

    def _load_values(self):
        self.work_dir_edit.setText(settings.load_work_dir())
        self.python_env_edit.setText(settings.load_python_env())

    # ------------------------------------------------------------------
    # 外部设置入口（如欢迎页选定目录后由主窗口调用）
    # ------------------------------------------------------------------

    def set_work_dir(self, path: str):
        """设置工作目录并持久化。"""
        self.work_dir_edit.setText(path)
        self._save_work_dir()