#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片处理页面模块：视频抽帧、图片去重、数据增强。

该页面整合三个相互独立的子任务，使用 :meth:`BaseTaskPage._add_card`
将它们分别放在独立的白色卡片中：

    第一张卡片：视频抽帧
        通过子进程调用 ``python scripts/vh.py images import --input <video>
        --output <dir>`` 将视频按帧间隔抽成图片，输出到工作目录下的
        ``IMAGES_FOLDER``。

    第二张卡片：图片去重
        通过子进程调用 ``python scripts/vh.py images dedup --input <folder>``
        对工作目录下的 ``IMAGES_FOLDER`` 做基于 ViT 特征的相似度去重，
        重复图片移动到隐藏回收站文件夹 ``RECYCLE_BIN_FOLDER`` 中。

    第三张卡片：数据增强
        通过子进程调用 ``python scripts/vh.py images augment --input <dir>
        --output <dir>`` 对 ``IMAGES_FOLDER`` 下图片进行随机旋转、切割、
        遮挡、通道变换等增强，结果输出到 ``IMAGES_FOLDER`` 下。

所有按钮 / 间距 / 颜色统一来自 :mod:`gui.theme` 与 :mod:`gui.components.widgets`。
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.utils._proc import build_script_argv
from gui.pages.base import BaseTaskPage
from gui.config import IMAGES_FOLDER, RECYCLE_BIN_FOLDER
from gui.components.widgets import PrimaryButton, SectionTitle


class VideoFramePage(BaseTaskPage):
    """视频抽帧 + 图片去重 + 数据增强页面。"""

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._build_frame_card()
        self._build_dedup_card()
        self._build_augment_card()

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
        _, self.content_layout = self._add_card()

        self.threshold_edit = QLineEdit("0.95")
        self._add_widget_row("相似度阈值：", self.threshold_edit)

        self.dedup_model_edit = QLineEdit("google/vit-base-patch16-224")
        self._add_widget_row("模型名称：", self.dedup_model_edit)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        self._add_widget_row("批大小：", self.batch_size_spin)

        self.small_obj_check = QCheckBox("启用小目标优化")
        self._add_widget_row("", self.small_obj_check)

        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(2, 16)
        self.grid_size_spin.setValue(2)
        self.grid_size_spin.setVisible(False)
        self._add_widget_row("网格大小：", self.grid_size_spin)

        self.small_obj_check.toggled.connect(self._on_small_obj_toggled)

        # 主操作按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.dedup_btn = PrimaryButton("开始去重")
        self.dedup_btn.setMinimumWidth(120)
        self.dedup_btn.clicked.connect(self._run_dedup)
        self.content_layout.addWidget(self.dedup_btn, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # 卡片 3：数据增强
    # ------------------------------------------------------------------

    def _build_augment_card(self):
        """新建第三张卡片，构建数据增强表单。"""
        _, self.content_layout = self._add_card()

        self.content_layout.addWidget(SectionTitle("数据增强设置"))

        # 随机种子（唯一的通用参数）
        self.aug_seed_spin = QSpinBox()
        self.aug_seed_spin.setRange(0, 999999)
        self.aug_seed_spin.setSpecialValueText("随机")
        self.aug_seed_spin.setValue(42)
        self._add_widget_row("随机种子：", self.aug_seed_spin)

        # 4 个增强子功能
        self.content_layout.addSpacing(theme.SPACING_SM)

        self._build_cut_section()
        self._build_occlusion_section()
        self._build_channel_section()
        self._build_rotate_section()

        # 主操作按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.aug_btn = PrimaryButton("开始增强")
        self.aug_btn.setMinimumWidth(120)
        self.aug_btn.clicked.connect(self._run_augment)
        self.content_layout.addWidget(self.aug_btn, alignment=Qt.AlignLeft)

    @staticmethod
    def _build_labeled_spin(label: str, spin: QWidget, label_width: int = 80) -> QWidget:
        """把已有的 spin 控件包成一个带固定宽度标签的紧凑行。"""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(label_width)
        layout.addWidget(lbl)
        layout.addWidget(spin)
        layout.addStretch(1)
        return row

    def _make_aug_section(self, title: str, default_on: bool = True) -> tuple:
        """创建一个不带边框的增强功能区块，返回 ``(body_layout, checkbox)``。

        ``body_layout`` 是用于放置参数控件的内部布局；通过勾选框控制其容器启停。
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACING_SM)

        cb = QCheckBox(title)
        cb.setChecked(default_on)
        cb.setStyleSheet("font-weight: bold;")
        layout.addWidget(cb)

        body = QWidget()
        body.setEnabled(default_on)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(theme.SPACING_SM)
        layout.addWidget(body)

        cb.toggled.connect(body.setEnabled)

        self.content_layout.addWidget(container)
        return body_layout, cb

    def _build_rotate_section(self):
        """构建随机旋转设置区块（双排参数）。"""
        body, self.aug_rotate_cb = self._make_aug_section("随机旋转")

        self.aug_rotate_degrees = QDoubleSpinBox()
        self.aug_rotate_degrees.setRange(0, 360)
        self.aug_rotate_degrees.setValue(30)
        self.aug_rotate_degrees.setSuffix("°")

        self.aug_rotate_prob = QDoubleSpinBox()
        self.aug_rotate_prob.setRange(0, 1)
        self.aug_rotate_prob.setSingleStep(0.05)
        self.aug_rotate_prob.setValue(0.5)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(theme.SPACING_LG)
        row_layout.addWidget(self._build_labeled_spin("最大角度：", self.aug_rotate_degrees))
        row_layout.addWidget(self._build_labeled_spin("应用概率：", self.aug_rotate_prob))
        row_layout.addStretch(1)
        body.addWidget(row)

    def _build_cut_section(self):
        """构建随机切割设置区块（双排参数）。"""
        body, self.aug_cut_cb = self._make_aug_section("随机切割")

        self.aug_cut_scale = QDoubleSpinBox()
        self.aug_cut_scale.setRange(0, 1)
        self.aug_cut_scale.setSingleStep(0.05)
        self.aug_cut_scale.setValue(0.3)

        self.aug_cut_ratio = QDoubleSpinBox()
        self.aug_cut_ratio.setRange(1, 10)
        self.aug_cut_ratio.setSingleStep(0.1)
        self.aug_cut_ratio.setValue(1.5)

        self.aug_cut_prob = QDoubleSpinBox()
        self.aug_cut_prob.setRange(0, 1)
        self.aug_cut_prob.setSingleStep(0.05)
        self.aug_cut_prob.setValue(0.5)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(theme.SPACING_LG)
        row_layout.addWidget(self._build_labeled_spin("裁剪因子：", self.aug_cut_scale))
        row_layout.addWidget(self._build_labeled_spin("宽高比范围：", self.aug_cut_ratio))
        row_layout.addWidget(self._build_labeled_spin("应用概率：", self.aug_cut_prob))
        row_layout.addStretch(1)
        body.addWidget(row)

        self.aug_cut_resize_cb = QCheckBox("切割后缩放回原始尺寸")
        self.aug_cut_resize_cb.setChecked(True)
        body.addWidget(self.aug_cut_resize_cb)

    def _build_occlusion_section(self):
        """构建随机遮挡设置区块（双排参数）。"""
        body, self.aug_occlusion_cb = self._make_aug_section("随机遮挡")

        self.aug_occlusion_count = QSpinBox()
        self.aug_occlusion_count.setRange(1, 20)
        self.aug_occlusion_count.setValue(3)

        self.aug_occlusion_size = QDoubleSpinBox()
        self.aug_occlusion_size.setRange(0.01, 0.5)
        self.aug_occlusion_size.setSingleStep(0.01)
        self.aug_occlusion_size.setValue(0.15)
        self.aug_occlusion_size.setSuffix(" × 图像尺寸")

        self.aug_occlusion_prob = QDoubleSpinBox()
        self.aug_occlusion_prob.setRange(0, 1)
        self.aug_occlusion_prob.setSingleStep(0.05)
        self.aug_occlusion_prob.setValue(0.5)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(theme.SPACING_LG)
        row_layout.addWidget(self._build_labeled_spin("遮挡块数：", self.aug_occlusion_count))
        row_layout.addWidget(self._build_labeled_spin("遮挡块大小：", self.aug_occlusion_size))
        row_layout.addWidget(self._build_labeled_spin("应用概率：", self.aug_occlusion_prob))
        row_layout.addStretch(1)
        body.addWidget(row)

    def _build_channel_section(self):
        """构建通道变换设置区块。"""
        body, self.aug_channel_cb = self._make_aug_section("通道变换")

        self.aug_channel_prob = QDoubleSpinBox()
        self.aug_channel_prob.setRange(0, 1)
        self.aug_channel_prob.setSingleStep(0.05)
        self.aug_channel_prob.setValue(0.5)
        body.addWidget(self._build_labeled_spin("应用概率：", self.aug_channel_prob))

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

    def _on_small_obj_toggled(self, checked: bool) -> None:
        """小目标优化开关：显示/隐藏网格大小设置。"""
        self.grid_size_spin.setVisible(checked)

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

        kwargs = dict(
            input=work_dir / IMAGES_FOLDER,
            threshold=threshold,
            model=self.dedup_model_edit.text() or "google/vit-base-patch16-224",
            batch_size=self.batch_size_spin.value(),
            move_to=work_dir / RECYCLE_BIN_FOLDER,
        )
        if self.small_obj_check.isChecked():
            kwargs["grid_size"] = self.grid_size_spin.value()

        arguments = build_script_argv("images", "dedup", **kwargs)
        self._start_subprocess(arguments, title="图片去重日志")

    def _run_augment(self):
        """执行数据增强任务。"""
        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        input_dir = Path(work_dir) / IMAGES_FOLDER
        output_dir = Path(work_dir) / IMAGES_FOLDER

        if not input_dir.is_dir():
            QMessageBox.warning(self, "目录错误", f"输入目录不存在：{input_dir}")
            return

        kwargs = dict(
            input=input_dir,
            output=output_dir,
            ext="png",
            quality=95,
            prefix="aug",
            seed=None if self.aug_seed_spin.value() == 0 else self.aug_seed_spin.value(),
            rotate_degrees=self.aug_rotate_degrees.value(),
            rotate_prob=self.aug_rotate_prob.value(),
            cut_scale=self.aug_cut_scale.value(),
            cut_ratio=self.aug_cut_ratio.value(),
            cut_prob=self.aug_cut_prob.value(),
            occlusion_count=self.aug_occlusion_count.value(),
            occlusion_size=self.aug_occlusion_size.value(),
            occlusion_prob=self.aug_occlusion_prob.value(),
            channel_prob=self.aug_channel_prob.value(),
        )

        # 对每个禁用的功能追加 --no-* 标记
        if not self.aug_rotate_cb.isChecked():
            kwargs["no_rotate"] = True
        if not self.aug_cut_cb.isChecked():
            kwargs["no_cut"] = True
        if not self.aug_cut_resize_cb.isChecked():
            kwargs["no_cut_resize"] = True
        if not self.aug_occlusion_cb.isChecked():
            kwargs["no_occlusion"] = True
        if not self.aug_channel_cb.isChecked():
            kwargs["no_channel"] = True

        arguments = build_script_argv("images", "augment", **kwargs)
        self._start_subprocess(arguments, title="数据增强日志")
