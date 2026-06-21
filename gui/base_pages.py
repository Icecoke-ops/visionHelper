#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 页面基类模块。

提供 :class:`BasePage` 与 :class:`BaseTaskPage`，封装统一的页面容器样式、
表单控件以及子进程启动能力。

整体布局约定：

- :class:`BasePage` 内部是一个 :class:`QScrollArea`，承载一个垂直
  排列的 ``wrapper``。
- ``wrapper`` 中可以放任意多张白色卡片（通过 :meth:`BasePage._add_card`
  追加）；首张卡片默认随构造创建。
- 子页面通过 ``self.content_layout`` 向当前卡片添加控件，无需关心
  外层卡片与边距，所有间距 / 圆角 / 边框由 :mod:`gui.theme` 统一控制。

若一个页面需要将不同子任务拆分到多张卡片中，可调用
:meth:`BasePage._add_card` 创建新的卡片块，并将
``self.content_layout`` 切换到新卡片返回的布局上，之后再继续使用
``_add_widget_row`` 等辅助方法即可。
"""


import sys
from pathlib import Path
from typing import Optional

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
from gui._proc import infer_script_name
from gui.config import app_root, is_frozen
from gui.context import AppContext
from gui.run_log_dialog import RunLogDialog
from gui.widgets import FormRow, SecondaryButton


class BasePage(QWidget):
    """所有子页面的基类，提供统一的卡片式滚动容器。

    构造时通过 :class:`gui.context.AppContext` 共享 ``work_dir`` /
    ``python_env``，子页面统一通过 ``self.ctx`` 读取，**不再**通过
    遍历 parent 链查找属性。
    """

    def __init__(self, parent: QWidget = None, ctx: Optional[AppContext] = None):
        super().__init__(parent)
        self.ctx: Optional[AppContext] = ctx
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

        # 卡片外的留白：wrapper 内部以垂直方向排列多张卡片，便于
        # 在需要时通过 :meth:`_add_card` 在末尾继续追加新的卡片块。
        wrapper = QWidget()
        self._wrapper_layout = QVBoxLayout(wrapper)
        self._wrapper_layout.setContentsMargins(
            theme.PAGE_MARGIN,
            theme.PAGE_MARGIN,
            theme.PAGE_MARGIN,
            theme.PAGE_MARGIN,
        )
        self._wrapper_layout.setSpacing(theme.SPACING_MD)
        # 末尾留一个 stretch，使所有卡片靠上排列；新增卡片时插入到
        # stretch 之前。
        self._wrapper_layout.addStretch(1)

        # 第一张默认卡片：``self.card`` 与 ``self.content_layout`` 保持
        # 向后兼容，所有现有子页面无需改动即可继续工作。
        self.card, self.content_layout = self._add_card()

        scroll.setWidget(wrapper)
        outer.addWidget(scroll)

    def _add_card(self) -> tuple:
        """向页面末尾追加一张新的白色卡片，返回 ``(card, content_layout)``。

        子页面可以调用本方法将不同子任务划分到多张卡片中：

        .. code-block:: python

            # 第一张卡片放区域 A 的控件（使用默认的 self.content_layout）
            self._add_widget_row("...", widget_a)

            # 新建第二张卡片，并把 self.content_layout 切到新卡片上
            _, self.content_layout = self._add_card()
            self._add_widget_row("...", widget_b)

        Returns:
            ``(QFrame, QVBoxLayout)``：新卡片本体及其内部承载控件的
            垂直布局。布局已经设置好 ``CARD_PADDING`` 内边距与统一行距。
        """
        card = QFrame()
        card.setProperty("variant", "card")
        theme.refresh_widget_style(card)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(
            theme.CARD_PADDING,
            theme.CARD_PADDING,
            theme.CARD_PADDING,
            theme.CARD_PADDING,
        )
        card_layout.setSpacing(theme.SPACING_MD)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(theme.SPACING_MD)
        card_layout.addLayout(content_layout)
        card_layout.addStretch(1)

        # 插入到末尾 stretch 之前，保持所有卡片靠上排列
        insert_index = max(0, self._wrapper_layout.count() - 1)
        self._wrapper_layout.insertWidget(insert_index, card)
        return card, content_layout

    def _work_dir(self) -> str:
        """读取当前工作目录。

        统一通过 :class:`AppContext` 读取；若 ``ctx`` 未注入，返回空串。
        """
        if self.ctx is not None:
            return self.ctx.work_dir
        return ""


class BaseTaskPage(BasePage):
    """任务页面的基类，提供表单控件和子进程启动能力。"""

    PYTHON = sys.executable

    # ------------------------------------------------------------------
    # 参数校验守卫：统一的"缺失/路径错误"弹窗逻辑
    # ------------------------------------------------------------------

    def _require_work_dir(self) -> Optional[Path]:
        """读取工作目录并校验存在性。

        - 未设置工作目录：弹窗提示并返回 ``None``；
        - 路径不存在或不是目录：弹窗提示并返回 ``None``；
        - 校验通过：返回 :class:`Path` 对象。
        """
        raw = self._work_dir()
        if not raw:
            QMessageBox.warning(self, "参数缺失", "请在导航栏下方设置工作目录")
            return None
        path = Path(raw)
        if not path.is_dir():
            QMessageBox.warning(self, "路径错误", f"工作目录不存在：{raw}")
            return None
        return path

    def _require_existing_dir(self, path: str, hint: str) -> Optional[Path]:
        """校验 ``path`` 为存在的目录，否则弹窗并返回 ``None``。

        :param path: 目录路径。
        :param hint: 校验失败时弹窗中用于标识该路径的中文描述（如"图片目录"）。
        """
        if not path:
            QMessageBox.warning(self, "参数缺失", f"请选择{hint}")
            return None
        p = Path(path)
        if not p.is_dir():
            QMessageBox.warning(self, "路径错误", f"{hint}不存在：{path}")
            return None
        return p

    def _require_existing_file(self, path: str, hint: str) -> Optional[Path]:
        """校验 ``path`` 为存在的文件，否则弹窗并返回 ``None``。"""
        if not path:
            QMessageBox.warning(self, "参数缺失", f"请选择{hint}")
            return None
        p = Path(path)
        if not p.is_file():
            QMessageBox.warning(self, "路径错误", f"{hint}不存在：{path}")
            return None
        return p

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
        """读取当前 Python 可执行文件路径。

        - 统一从注入的 :class:`AppContext` 读取；
        - 开发态（非 frozen）：仍未取到则回退到 ``sys.executable``；
        - 打包态（frozen）：``sys.executable`` 是 GUI 自身的 bootloader，
          内部并不包含 torch / ultralytics 等重型依赖，绝不能用它来跑
          ``scripts``。此时若用户未选择 Python 环境，本方法返回空串，
          由 :meth:`_start_subprocess` 统一弹窗提示。
        """
        if self.ctx is not None and self.ctx.python_env:
            return self.ctx.python_env
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

        # 日志保存目录：优先使用 GUI 当前工作目录下的 ``logs/``，未设置
        # 工作目录时退化为 app_root 下的 ``logs/``，保证总能落盘。
        work_dir = self._work_dir() or root
        log_dir = str(Path(work_dir) / "logs")

        dialog = RunLogDialog(
            python_path,
            arguments,
            title=title,
            parent=self,
            working_dir=root,
            extra_pythonpath=[root],
            log_dir=log_dir,
            log_script_name=infer_script_name(arguments),
        )
        dialog.exec_()
