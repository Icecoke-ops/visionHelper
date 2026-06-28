#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测页面模块。

提供模型预测功能：
    - 自动扫描工作目录下 ``runs``（TRAIN_FOLDER）子目录中的训练模型
    - 通过下拉框选择模型（显示名称格式：训练名称-模型权重名称）
    - 支持单张图片、图片目录或视频文件作为输入
    - 设置置信度阈值、任务类型等参数
    - 调用 ``scripts.predict`` 进行预测，结果保存到工作空间 predict 目录

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.components.widgets`，本模块不再
书写内联样式。
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QWidget,
)

from gui import theme
from gui.config import PREDICT_FOLDER
from gui.pages.base import BaseTaskPage
from gui.components.widgets import (
    FormRow,
    HSeparator,
    LabeledDoubleSpinBox,
    SecondaryButton,
    SuccessButton,
)


class PredictPage(BaseTaskPage):
    """预测页面：选择模型并对图片/视频进行预测。"""

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._build_form()

    def _build_form(self):
        # ===== 模型选择 =====
        self._model_widget, self.model_combo, self.refresh_model_btn = self._build_model_selector_widget()
        self.refresh_model_btn.clicked.connect(self._refresh_models)
        self.content_layout.addWidget(FormRow("选择模型：", self._model_widget))

        # ===== 任务参数 =====
        self.task_combo = self._build_task_combo()
        self._add_widget_row("任务类型：", self.task_combo)

        # ===== 输入路径 =====
        self.input_edit = self._add_file_row("输入路径：", is_directory=False)

        # ===== 推理参数 =====
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

        self.content_layout.addWidget(HSeparator())

        # ===== 操作按钮 =====
        self.predict_btn = SuccessButton("开始预测")
        self.predict_btn.setMinimumWidth(140)
        self.predict_btn.clicked.connect(self._run_predict)
        self.content_layout.addWidget(self.predict_btn, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # 模型扫描
    # ------------------------------------------------------------------

    def _refresh_models(self):
        """扫描训练结果目录并填充模型下拉框。"""
        self.model_combo.clear()

        runs_dir = self._runs_dir()
        if not runs_dir:
            QMessageBox.warning(self, "参数缺失", "请先设置工作目录")
            return

        # 使用基类方法发现已训练模型
        self._discover_and_populate_models(self.model_combo)

        if self.model_combo.count() == 0:
            QMessageBox.information(
                self,
                "未找到模型",
                f"在 {runs_dir} 下未找到训练模型，请确认已完成训练。",
            )
            return

    # ------------------------------------------------------------------
    # 预测执行
    # ------------------------------------------------------------------

    def _run_predict(self):
        """收集参数并启动预测子进程。"""
        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        # 校验模型
        model_path = self.model_combo.currentData()
        if not model_path:
            QMessageBox.warning(self, "参数缺失", "请选择模型")
            return
        model_file = self._require_existing_file(model_path, "模型文件")
        if model_file is None:
            return

        # 校验输入路径
        input_path = self.input_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "参数缺失", "请选择输入图片、图片目录或视频文件")
            return
        input_p = Path(input_path)
        if not input_p.exists():
            QMessageBox.warning(self, "路径错误", f"输入路径不存在：{input_path}")
            return

        # 输出目录自动设置为工作目录下的 predict 目录
        output_path = str(work_dir / PREDICT_FOLDER)

        arguments = [
            str(self._scripts_dir() / "vh.py"),
            "predict", "run",
            "--model", str(model_path),
            "--input", input_path,
            "--output", output_path,
            "--threshold", f"{self.threshold_spin.value():.4f}",
            "--task", self.task_combo.currentData(),
            "--iou", f"{self.iou_spin.value():.4f}",
        ]
        self._start_subprocess(arguments, title="模型预测")

    def _scripts_dir(self):
        """获取 scripts 目录路径。"""
        from gui.config import scripts_dir
        return scripts_dir()

    def on_page_shown(self):
        """页面首次显示时刷新模型列表。"""
        work_dir = self._work_dir()
        if not work_dir:
            return
        self._refresh_models()
