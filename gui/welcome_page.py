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
:mod:`gui.widgets`，本模块不再写内联样式。
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
from gui.widgets import IconButton, LinkButton, MutedLabel, PrimaryButton


# 历史目录最多保留多少项
MAX_RECENT = 20


class WelcomePage(QWidget):
    """visionHelper 引导页：历史工作目录列表 + 新增目录按钮。

    通过 :data:`work_dir_selected` 信号将最终选定的工作目录传递给主窗口。
    """

    # 选定一个工作目录后发出的信号，参数为工作目录绝对路径
    work_dir_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._recent_dirs: List[str] = []
        self._init_ui()

    # ------------------------------------------------------------------
    # UI 初始化
    # ------------------------------------------------------------------

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 40, 60, 40)
        outer.setSpacing(theme.SPACING_MD)

        # 标题
        title = QLabel("欢迎使用 visionHelper")
        title_font = theme.app_font()
        title_font.setPointSize(theme.FONT_SIZE_LARGE_TITLE)
        title_font.setBold(True)
        title.setFont(title_font)
        outer.addWidget(title)

        subtitle = MutedLabel("请选择一个工作目录开始：")
        outer.addWidget(subtitle)

        # 历史目录列表（颜色/圆角等由全局 QSS 提供）
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.NoSelection)
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.list_widget, stretch=1)

        # 空列表占位提示
        self.empty_label = MutedLabel("暂无历史工作目录，点击下方按钮添加新目录开始使用。")
        self.empty_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.empty_label)

        # 底部按钮区
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.add_btn = PrimaryButton("➕ 添加新目录")
        self.add_btn.setMinimumWidth(140)
        self.add_btn.clicked.connect(self._on_add_clicked)
        btn_row.addWidget(self.add_btn)
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    # 历史目录管理
    # ------------------------------------------------------------------

    def set_recent_dirs(self, dirs: List[str]):
        """设置并刷新历史目录列表。

        :param dirs: 历史工作目录路径列表（按最近使用优先排序）。
        """
        # 过滤空值并去重，保持原始顺序
        seen = set()
        cleaned: List[str] = []
        for d in dirs or []:
            if not d:
                continue
            if d in seen:
                continue
            seen.add(d)
            cleaned.append(d)
        self._recent_dirs = cleaned[:MAX_RECENT]
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
        # 禁用 item 自身的选中/键盘交互，全部交给行内按钮
        item.setFlags(Qt.NoItemFlags)

        row = QFrame()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(theme.SPACING_SM, theme.SPACING_XS, theme.SPACING_SM, theme.SPACING_XS)
        layout.setSpacing(theme.SPACING_SM)

        # 主按钮：显示路径，点击进入工作目录
        p = Path(path)
        name = p.name or str(p)
        text = f"{name}    {path}" if name != path else path
        open_btn = LinkButton(text)
        open_btn.setToolTip(path)
        open_btn.clicked.connect(lambda _, pth=path: self._on_open_clicked(pth))
        open_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(open_btn, stretch=1)

        # 删除按钮：从历史中移除
        remove_btn = IconButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("从历史中移除")
        remove_btn.clicked.connect(lambda _, pth=path: self._on_remove_clicked(pth))
        layout.addWidget(remove_btn)

        item.setSizeHint(row.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, row)

    # ------------------------------------------------------------------
    # 行内按钮交互
    # ------------------------------------------------------------------

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

        # 打开后将该项前置
        self._move_to_front(path)
        self.work_dir_selected.emit(path)

    def _on_remove_clicked(self, path: str):
        """点击 ✕：从历史中移除指定目录。"""
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
        if len(self._recent_dirs) > MAX_RECENT:
            self._recent_dirs = self._recent_dirs[:MAX_RECENT]
        self._refresh_list()
