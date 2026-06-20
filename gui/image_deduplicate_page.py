#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片去重页面模块。

通过子进程调用 ``scripts.deduplicate_images`` 执行图片去重任务。
所有按钮 / 间距 / 颜色统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`。
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QSpinBox, QWidget

from gui import theme
from gui.base_pages import BaseTaskPage
from gui.config import IMAGES_FOLDER, RECYCLE_BIN_FOLDER
from gui.widgets import HintCard, PrimaryButton, SectionTitle


class ImageDeduplicatePage(BaseTaskPage):
    """图片去重页面：通过子进程调用 scripts.deduplicate_images。"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._build_form()

    def _build_form(self):
        self.content_layout.addWidget(SectionTitle("图片去重参数"))

        self.threshold_edit = QLineEdit("0.95")
        self._add_widget_row("相似度阈值：", self.threshold_edit)

        self.model_edit = QLineEdit("google/vit-base-patch16-224")
        self._add_widget_row("模型名称：", self.model_edit)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        self._add_widget_row("批大小：", self.batch_size_spin)

        # 提示卡片：去重结果与回收站位置
        self.content_layout.addWidget(HintCard(
            title="去重提示",
            description=(
                f"待去重的图片来自工作目录下的 {IMAGES_FOLDER} 文件夹；\n"
                f"重复图片将被移动到工作目录下的隐藏文件夹 {RECYCLE_BIN_FOLDER} 中，"
                "可在确认无误后手动清理。"
            ),
        ))

        # 主操作按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.run_btn = PrimaryButton("开始去重")
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._run)
        self.content_layout.addWidget(self.run_btn, alignment=Qt.AlignLeft)

    def _run(self):
        work_dir = self._work_dir()
        if not work_dir:
            QMessageBox.warning(self, "参数缺失", "请在导航栏下方设置工作目录")
            return

        folder = str(Path(work_dir) / IMAGES_FOLDER)
        try:
            threshold = float(self.threshold_edit.text() or "0.95")
        except ValueError:
            QMessageBox.warning(self, "参数错误", "阈值必须是数字")
            return

        recycle_bin = str(Path(work_dir) / RECYCLE_BIN_FOLDER)
        arguments = [
            "-m", "scripts.deduplicate_images",
            folder,
            "--threshold", str(threshold),
            "--model", self.model_edit.text() or "google/vit-base-patch16-224",
            "--batch-size", str(self.batch_size_spin.value()),
            "--move-to", recycle_bin,
        ]
        self._start_subprocess(arguments, title="图片去重日志")
