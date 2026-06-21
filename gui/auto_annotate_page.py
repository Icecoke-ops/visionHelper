#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动标注页面模块。

在"数据标注"功能下提供自动标注能力：
    - 自动扫描工作目录下 ``runs``（TRAIN_FOLDER）子目录中的训练模型
    - 通过下拉框选择模型（显示名称格式：训练名称-模型权重名称）
    - 设置置信度阈值、IoU 阈值、任务类型等参数
    - 调用 ``scripts.auto_annotate`` 对未标注图片生成 X-AnyLabeling JSON 标注（兼容 LabelMe）
    - 自动标注结果在 JSON 中写入 ``auto_annotated_time`` 字段

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`，本模块不再
书写内联样式。
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from gui import theme
from gui._proc import build_script_argv
from gui.base_pages import BaseTaskPage
from gui.config import IMAGES_FOLDER, TRAIN_FOLDER
from gui.widgets import (
    FormRow,
    HSeparator,
    LabeledDoubleSpinBox,
    SecondaryButton,
    SuccessButton,
)
# 注意：不要在模块顶层 import scripts.api，避免在 PyInstaller 打包态下
# GUI 进程因找不到 scripts 包（或顺带触发 torch / ultralytics 等重依赖
# 的 import）而启动失败。需要时改为函数内部局部导入。


class AutoAnnotatePage(BaseTaskPage):
    """自动标注页面：选择模型并对未标注图片进行自动标注。"""

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._build_form()

    def _build_form(self):
        # ===== 模型选择 =====
        model_widget = QWidget()
        model_row = QHBoxLayout(model_widget)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(theme.SPACING_SM)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        self.model_combo.setEditable(False)
        model_row.addWidget(self.model_combo, 1)

        self.refresh_model_btn = SecondaryButton("刷新模型")
        self.refresh_model_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self.refresh_model_btn)

        self.content_layout.addWidget(FormRow("选择模型：", model_widget))

        # 显示当前选中的模型文件路径
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("模型文件路径将在此处显示")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._add_widget_row("模型路径：", self.model_path_edit)

        # ===== 任务参数 =====
        self.task_combo = QComboBox()
        self.task_combo.addItem("目标检测（detect）", "detect")
        self.task_combo.addItem("旋转框（obb）", "obb")
        self.task_combo.addItem("实例分割（segment）", "segment")
        self.task_combo.addItem("图像分类（classify）", "classify")
        self._add_widget_row("任务类型：", self.task_combo)

        # ===== 处理范围（4 个复选框） =====
        scope_widget = QWidget()
        scope_layout = QHBoxLayout(scope_widget)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setSpacing(theme.SPACING_MD)

        self.include_unannotated_cb = QCheckBox("未标注")
        self.include_unannotated_cb.setChecked(True)
        self.include_auto_cb = QCheckBox("自动标注")
        self.include_auto_corrected_cb = QCheckBox("自动标注后矫正")
        self.include_manual_cb = QCheckBox("手动标注")

        scope_layout.addWidget(self.include_unannotated_cb)
        scope_layout.addWidget(self.include_auto_cb)
        scope_layout.addWidget(self.include_auto_corrected_cb)
        scope_layout.addWidget(self.include_manual_cb)
        scope_layout.addStretch()
        self._add_widget_row("处理范围：", scope_widget)

        hparams_widget = QWidget()
        hparams_layout = QHBoxLayout(hparams_widget)
        hparams_layout.setContentsMargins(0, 0, 0, 0)
        hparams_layout.setSpacing(theme.SPACING_MD)

        self.threshold_spin = LabeledDoubleSpinBox("置信度", 0.01, 1.0, 0.25)
        self.iou_spin = LabeledDoubleSpinBox("IoU", 0.01, 1.0, 0.45)
        hparams_layout.addWidget(self.threshold_spin)
        hparams_layout.addWidget(self.iou_spin)
        hparams_layout.addStretch()
        self._add_widget_row("推理参数：", hparams_widget)

        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("可留空，例如 _auto")
        self._add_widget_row("文件后缀：", self.suffix_edit)

        self.content_layout.addWidget(HSeparator())

        # ===== 操作按钮 =====
        self.annotate_btn = SuccessButton("开始自动标注")
        self.annotate_btn.setMinimumWidth(140)
        self.annotate_btn.clicked.connect(self._run_auto_annotate)
        self.content_layout.addWidget(self.annotate_btn, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # 模型扫描
    # ------------------------------------------------------------------

    def _runs_dir(self) -> str:
        """根据当前工作目录返回训练结果目录路径。"""
        work_dir = self._work_dir()
        if not work_dir:
            return ""
        return str(Path(work_dir) / TRAIN_FOLDER)

    def _refresh_models(self):
        """扫描训练结果目录并填充模型下拉框。"""
        self.model_combo.clear()

        runs_dir = self._runs_dir()
        if not runs_dir:
            QMessageBox.warning(self, "参数缺失", "请先设置工作目录")
            return

        # 仅在需要时局部 import 轻量子模块，避免在 GUI 进程启动时拖入
        # scripts.api 顶部的 torch / ultralytics 等重依赖。
        from scripts._common import discover_trained_models

        models = discover_trained_models(runs_dir)
        if not models:
            QMessageBox.information(
                self,
                "未找到模型",
                f"在 {runs_dir} 下未找到训练模型，请确认已完成训练。",
            )
            return

        for display_name, model_path in models:
            self.model_combo.addItem(display_name, model_path)

        self._on_model_changed()

    def _on_model_changed(self):
        """模型选择变化时更新模型路径显示。"""
        model_path = self.model_combo.currentData()
        self.model_path_edit.setText(model_path or "")

    # ------------------------------------------------------------------
    # 自动标注执行
    # ------------------------------------------------------------------

    def _run_auto_annotate(self):
        """收集参数并启动自动标注子进程。"""
        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        image_dir = self._require_existing_dir(
            str(work_dir / IMAGES_FOLDER), "图片目录"
        )
        if image_dir is None:
            return

        model_path = self._require_existing_file(
            self.model_path_edit.text().strip(), "模型文件"
        )
        if model_path is None:
            return

        include_unannotated = self.include_unannotated_cb.isChecked()
        include_auto = self.include_auto_cb.isChecked()
        include_auto_corrected = self.include_auto_corrected_cb.isChecked()
        include_manual = self.include_manual_cb.isChecked()
        if not any([
            include_unannotated,
            include_auto,
            include_auto_corrected,
            include_manual,
        ]):
            QMessageBox.warning(
                self, "参数缺失", "请至少选择一种处理范围"
            )
            return

        arguments = build_script_argv(
            "scripts.auto_annotate",
            image_dir,
            model_path,
            threshold=f"{self.threshold_spin.value():.4f}",
            task=self.task_combo.currentData(),
            iou=f"{self.iou_spin.value():.4f}",
            suffix=self.suffix_edit.text().strip() or None,
            include_unannotated=include_unannotated,
            include_auto=include_auto,
            include_auto_corrected=include_auto_corrected,
            include_manual=include_manual,
        )
        self._start_subprocess(arguments, title="自动标注")

    def on_page_shown(self):
        """页面首次显示时根据工作目录刷新模型列表。"""
        work_dir = self._work_dir()
        if not work_dir:
            return
        self._refresh_models()
