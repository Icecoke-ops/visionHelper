#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper GUI 启动引导页。

提供一个简单的入口界面，包含：
    - 历史工作目录列表：每一项可点击进入主界面（即把该目录设置为当前工作目录），
      右侧带一个 "✕" 按钮用于从历史中删除该项；
    - 新增工作目录按钮：弹出目录选择对话框选择新的工作目录，
      选中后会写入历史并直接进入主界面。

历史目录通过 ``QSettings("visionHelper", "MainWindow")`` 中的
``recent_work_dirs`` 键持久化，列表内容按 "最近使用优先" 维护。

页面所有视觉样式（颜色、圆角、字体、间距）统一来自 :mod:`gui.theme` 与
:mod:`gui.components.widgets`，本模块不再写内联样式。
"""

from pathlib import Path
from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.settings import MAX_RECENT_WORK_DIRS
from gui.components.widgets import IconButton, LinkButton, PrimaryButton

__all__ = ["WelcomePage"]


class WelcomePage(QWidget):
    """visionHelper 引导页：历史工作目录列表 + 新增目录按钮。"""

    work_dir_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recent_dirs: List[str] = []
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ===== 中间内容区 =====
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(60, 40, 60, 30)
        content_layout.setSpacing(theme.SPACING_MD)

        # 标题
        title = QLabel("欢迎使用 visionHelper")
        title.setObjectName("welcomeTitle")
        content_layout.addWidget(title)

        subtitle = QLabel("请选择一个工作目录开始：")
        subtitle.setObjectName("welcomeSubtitle")
        content_layout.addWidget(subtitle)

        # 历史目录列表
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.NoSelection)
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.setObjectName("welcomeList")
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(self.list_widget, stretch=1)

        # 空列表占位提示
        self.empty_label = QLabel("暂无历史工作目录，点击下方按钮添加新目录开始使用。")
        self.empty_label.setObjectName("welcomeEmpty")
        self.empty_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.empty_label)

        outer.addWidget(content, 1)

        # ===== 底部按钮区 =====
        footer = QWidget()
        footer.setObjectName("welcomeFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(60, 16, 60, 30)
        footer_layout.addStretch()
        self.add_btn = PrimaryButton("添加新目录")
        self.add_btn.setMinimumWidth(140)
        self.add_btn.clicked.connect(self._on_add_clicked)
        footer_layout.addWidget(self.add_btn)
        outer.addWidget(footer)

    def set_recent_dirs(self, dirs: List[str]):
        """设置并刷新历史目录列表。"""
        seen = set()
        cleaned: List[str] = []
        for d in dirs or []:
            if not d or d in seen:
                continue
            seen.add(d)
            cleaned.append(d)
        self._recent_dirs = cleaned[:MAX_RECENT_WORK_DIRS]
        self._refresh_list()

    def recent_dirs(self) -> List[str]:
        """返回当前历史目录列表的副本。"""
        return list(self._recent_dirs)

    def _refresh_list(self):
        self.list_widget.clear()
        if not self._recent_dirs:
            self.list_widget.setVisible(False)
            self.empty_label.setVisible(True)
            return

        self.list_widget.setVisible(True)
        self.empty_label.setVisible(False)

        for path in self._recent_dirs:
            self._append_row(path)

    def _append_row(self, path: str):
        """向列表中追加一行显示。"""
        item = QListWidgetItem()
        item.setFlags(Qt.NoItemFlags)

        row = QFrame()
        row.setObjectName("welcomeRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # 主按钮：显示路径
        p = Path(path)
        name = p.name or str(p)
        text = f"{name}    {path}" if name != path else path
        open_btn = LinkButton(text)
        open_btn.setToolTip(path)
        open_btn.clicked.connect(lambda _, pth=path: self._on_open_clicked(pth))
        open_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(open_btn, stretch=1)

        # 删除按钮
        remove_btn = IconButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("从历史中移除")
        remove_btn.clicked.connect(lambda _, pth=path: self._on_remove_clicked(pth))
        layout.addWidget(remove_btn)

        item.setSizeHint(row.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row)

    def _on_open_clicked(self, path: str):
        """点击列表项：打开该工作目录。"""
        if not Path(path).is_dir():
            ret = QMessageBox.question(
                self,
                "目录不存在",
                f"目录不存在或已被删除：\n{path}\n\n是否从历史记录中移除？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.Yes:
                self._on_remove_clicked(path)
            return

        self._move_to_front(path)
        self.work_dir_selected.emit(path)

    def _on_remove_clicked(self, path: str):
        """从历史中移除指定目录。"""
        if path in self._recent_dirs:
            self._recent_dirs.remove(path)
            self._refresh_list()

    def _on_add_clicked(self):
        """点击"添加新目录"按钮。"""
        start = self._recent_dirs[0] if self._recent_dirs else ""
        path = QFileDialog.getExistingDirectory(self, "选择工作目录", start)
        if not path:
            return
        self._move_to_front(path)
        self.work_dir_selected.emit(path)

    def _move_to_front(self, path: str):
        """把指定目录移动到列表最前；若不存在则插入。"""
        if path in self._recent_dirs:
            self._recent_dirs.remove(path)
        self._recent_dirs.insert(0, path)
        if len(self._recent_dirs) > MAX_RECENT_WORK_DIRS:
            self._recent_dirs = self._recent_dirs[:MAX_RECENT_WORK_DIRS]
        self._refresh_list()
