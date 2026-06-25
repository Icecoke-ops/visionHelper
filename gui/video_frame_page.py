#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频抽帧 + 图片去重 页面模块。

该页面整合两个紧密相关的子任务，使用 :meth:`BaseTaskPage._add_card`
将它们分别放在独立的白色卡片中：

    第一张卡片：视频抽帧
        通过子进程调用 ``python scripts/vh.py images import --input <video>
        --output <dir>`` 将视频按帧间隔抽成图片，输出到工作目录下的
        ``IMAGES_FOLDER``。

    第二张卡片：图片去重
        通过子进程调用 ``python scripts/vh.py images dedup --input <folder>``
        对工作目录下的 ``IMAGES_FOLDER`` 做基于 ViT 特征的相似度去重，
        重复图片移动到隐藏回收站文件夹 ``RECYCLE_BIN_FOLDER`` 中。

所有按钮 / 间距 / 颜色统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QSpinBox, QWidget

from gui import theme
from gui._proc import build_script_argv
from gui.base_pages import BaseTaskPage
from gui.config import IMAGES_FOLDER, RECYCLE_BIN_FOLDER
from gui.widgets import PrimaryButton


class VideoFramePage(BaseTaskPage):
    """视频抽帧 + 图片去重页面。

    页面包含两张相互独立的卡片：
        - 顶部卡片：调用 ``python scripts/vh.py images import`` 执行视频抽帧；
        - 底部卡片：调用 ``python scripts/vh.py images dedup`` 对抽帧结果做去重。

    两个任务共享同一份"工作目录"配置：抽帧结果直接落到工作目录下的
    ``IMAGES_FOLDER``，去重任务也从该目录读取图片，形成"抽帧 → 去重"
    一条龙的常用工作流。
    """

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._build_frame_card()
        self._build_dedup_card()

    # ------------------------------------------------------------------
    # 卡片 1：视频抽帧
    # ------------------------------------------------------------------

    def _build_frame_card(self):
        """在默认（第一张）卡片中构建视频抽帧表单。"""
        self.video_path_edit = self._add_file_row("视频文件：", is_directory=False)

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 10000)
        self.frame_step_spin.setValue(5)
        self._add_widget_row("抽帧间隔：", self.frame_step_spin)

        self.prefix_edit = QLineEdit("frame")
        self._add_widget_row("文件名前缀：", self.prefix_edit)

        # 主操作按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.run_btn = PrimaryButton("开始抽帧")
        self.run_btn.setMinimumWidth(120)
        self.run_btn.clicked.connect(self._run_extract)
        self.content_layout.addWidget(self.run_btn, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # 卡片 2：图片去重
    # ------------------------------------------------------------------

    def _build_dedup_card(self):
        """新建第二张卡片，构建图片去重表单。"""
        # 切换 ``self.content_layout`` 到新卡片，复用 BaseTaskPage 的辅助方法
        _, self.content_layout = self._add_card()

        self.threshold_edit = QLineEdit("0.95")
        self._add_widget_row("相似度阈值：", self.threshold_edit)

        self.dedup_model_edit = QLineEdit("google/vit-base-patch16-224")
        self._add_widget_row("模型名称：", self.dedup_model_edit)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        self._add_widget_row("批大小：", self.batch_size_spin)

        # 主操作按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.dedup_btn = PrimaryButton("开始去重")
        self.dedup_btn.setMinimumWidth(120)
        self.dedup_btn.clicked.connect(self._run_dedup)
        self.content_layout.addWidget(self.dedup_btn, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # 子任务执行
    # ------------------------------------------------------------------

    def _run_extract(self):
        """执行视频抽帧任务。"""
        video_path = self._require_existing_file(
            self.video_path_edit.text().strip(), "视频文件"
        )
        if video_path is None:
            return
        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        output_dir = work_dir / IMAGES_FOLDER

        arguments = build_script_argv(
            "images", "import",
            input=video_path,
            output=output_dir,
            frame_step=self.frame_step_spin.value(),
            ext="jpg",
            prefix=self.prefix_edit.text() or "frame",
        )
        self._start_subprocess(arguments, title="视频抽帧日志")

    def _run_dedup(self):
        """执行图片去重任务。"""
        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        try:
            threshold = float(self.threshold_edit.text() or "0.95")
        except ValueError:
            QMessageBox.warning(self, "参数错误", "阈值必须是数字")
            return

        arguments = build_script_argv(
            "images", "dedup",
            input=work_dir / IMAGES_FOLDER,
            threshold=threshold,
            model=self.dedup_model_edit.text() or "google/vit-base-patch16-224",
            batch_size=self.batch_size_spin.value(),
            move_to=work_dir / RECYCLE_BIN_FOLDER,
        )
        self._start_subprocess(arguments, title="图片去重日志")
