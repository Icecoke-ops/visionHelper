#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频抽帧页面模块。

通过子进程调用 ``scripts.extract_video_frames`` 执行视频抽帧任务。
所有按钮 / 间距 / 颜色统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`。
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QSpinBox, QWidget

from gui import theme
from gui.base_pages import BaseTaskPage
from gui.config import IMAGES_FOLDER
from gui.widgets import HintCard, PrimaryButton, SectionTitle


class VideoFramePage(BaseTaskPage):
    """视频抽帧页面：通过子进程调用 scripts.extract_video_frames。"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._build_form()

    def _build_form(self):
        self.content_layout.addWidget(SectionTitle("视频抽帧参数"))

        self.video_path_edit = self._add_file_row("视频文件：", is_directory=False)

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 10000)
        self.frame_step_spin.setValue(5)
        self._add_widget_row("抽帧间隔：", self.frame_step_spin)

        self.prefix_edit = QLineEdit("frame")
        self._add_widget_row("文件名前缀：", self.prefix_edit)

        # 提示卡片：抽帧结果输出位置
        self.content_layout.addWidget(HintCard(
            title="抽帧结果",
            description=f"抽出的图片会保存到工作目录下的 {IMAGES_FOLDER} 文件夹中。",
        ))

        # 主操作按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.run_btn = PrimaryButton("开始抽帧")
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._run)
        self.content_layout.addWidget(self.run_btn, alignment=Qt.AlignLeft)

    def _run(self):
        video_path = self.video_path_edit.text().strip()
        work_dir = self._work_dir()
        output_dir = str(Path(work_dir) / IMAGES_FOLDER) if work_dir else ""
        if not video_path:
            QMessageBox.warning(self, "参数缺失", "请选择视频文件")
            return
        if not output_dir:
            QMessageBox.warning(self, "参数缺失", "请在导航栏下方设置工作目录")
            return

        arguments = [
            "-m", "scripts.extract_video_frames",
            video_path,
            output_dir,
            "--frame-step", str(self.frame_step_spin.value()),
            "--ext", "jpg",
            "--prefix", self.prefix_edit.text() or "frame",
        ]
        self._start_subprocess(arguments, title="视频抽帧日志")
