#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练页面模块。

整合 YOLO 数据集导出与模型训练两个子任务：
    - 导出数据集：调用 ``scripts.export_yolo_dataset`` 将工作目录下的
      X-AnyLabeling 标注导出为 YOLO 格式，并自动划分 train/test。
    - 训练模型：调用 ``scripts.train_model`` 对导出的数据集进行训练。

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.components.widgets`，本模块不再
书写内联样式。
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from gui import theme
from gui.utils._proc import build_script_argv
from gui.pages.base import BaseTaskPage
from gui.config import DATASET_FOLDER, IMAGES_FOLDER, TRAIN_FOLDER
from gui.components.widgets import (
    FormRow,
    LabeledSpinBox,
    LabeledDoubleSpinBox,
    PrimaryButton,
    SecondaryButton,
    SuccessButton,
)

# 注意：不要在模块顶层 import scripts.api，避免在 PyInstaller 打包态下
# GUI 进程因找不到 scripts 包（或顺带触发 torch / ultralytics 等重依赖
# 的 import）而启动失败。需要时改为函数内部局部导入轻量子模块。

# 常用 YOLO 基模型尺寸，可根据需要扩展
BASE_MODEL_FAMILIES = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
]

def _build_base_models(task: str) -> list:
    """根据任务类型构造基模型名列表（自动追加 ``-obb``/``-seg``/``-cls`` 后缀）。"""
    from scripts.common.config import TASK_MODEL_SUFFIX
    suffix = TASK_MODEL_SUFFIX.get(task, "")
    return [f"{name}{suffix}" for name in BASE_MODEL_FAMILIES]


class ModelTrainingPage(BaseTaskPage):
    """模型训练页面：导出 YOLO 数据集并启动训练。"""

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._build_form()

    def _build_form(self):
        # ===== 数据集导出区域 =====
        self.task_combo = self._build_task_combo()
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        self._add_widget_row("任务类型：", self.task_combo)

        split_widget = QWidget()
        split_layout = QHBoxLayout(split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(theme.SPACING_MD)
        self.train_ratio_spin = LabeledSpinBox("训练", 0, 100, 80)
        self.test_ratio_spin = LabeledSpinBox("测试", 0, 100, 20)
        split_layout.addWidget(self.train_ratio_spin)
        split_layout.addWidget(self.test_ratio_spin)
        split_layout.addStretch()
        self._add_widget_row("划分比例：", split_widget)

        self.export_empty_check = QCheckBox("导出空标签（允许无标注对象的图片）")
        self._add_widget_row("", self.export_empty_check)

        self.export_unlabeled_check = QCheckBox("导出未标注图片（没有 JSON 标注文件的图片）")
        self._add_widget_row("", self.export_unlabeled_check)

        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_edit.setVisible(False)

        self.export_btn = PrimaryButton("导出数据集")
        self.export_btn.setMinimumWidth(140)
        self.export_btn.clicked.connect(self._export_dataset)
        self.content_layout.addWidget(self.export_btn, alignment=Qt.AlignLeft)

        # ===== 模型训练区域 =====
        _, self.content_layout = self._add_card()

        self.name_edit = QLineEdit()
        self._add_widget_row("训练名称：", self.name_edit)

        self._model_widget, self.model_combo, self.refresh_model_btn = self._build_model_selector_widget()
        self.refresh_model_btn.clicked.connect(self._refresh_models)
        self._add_widget_row("选择模型：", self._model_widget)

        self.model_edit = QLineEdit("yolov8n")
        self.model_edit.setPlaceholderText("可填写模型名称（如 yolov8n）或权重文件路径")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._add_widget_row("模型路径：", self.model_edit)

        self._populate_base_models()

        # ===== 统一参数网格布局 =====
        params_widget = QWidget()
        params_grid = QGridLayout(params_widget)
        params_grid.setContentsMargins(0, 0, 0, 0)
        params_grid.setSpacing(8)
        params_grid.setColumnStretch(3, 1)  # 第4列弹性填充

        # 统一尺寸常量
        LABEL_W = 65
        SPIN_W = 130

        # 第0行：基础训练参数
        self.epochs_spin = LabeledSpinBox("轮数", 1, 10000, 100, label_width=LABEL_W, spin_width=SPIN_W)
        self.imgsz_spin = LabeledSpinBox("尺寸", 32, 2048, 640, label_width=LABEL_W, spin_width=SPIN_W)
        self.batch_spin = LabeledSpinBox("批次", -1, 256, 16, label_width=LABEL_W, spin_width=SPIN_W)
        self.batch_spin.spin.setSpecialValueText("自动")
        params_grid.addWidget(self.epochs_spin, 0, 0)
        params_grid.addWidget(self.imgsz_spin, 0, 1)
        params_grid.addWidget(self.batch_spin, 0, 2)

        # 第1行：学习率参数
        self.lr0_spin = LabeledDoubleSpinBox("初始学习率", 0.0001, 1.0, 0.01, decimals=4, step=0.001, label_width=LABEL_W, spin_width=SPIN_W)
        self.lrf_spin = LabeledDoubleSpinBox("最终学习率", 0.0001, 1.0, 0.01, decimals=4, step=0.001, label_width=LABEL_W, spin_width=SPIN_W)
        self.momentum_spin = LabeledDoubleSpinBox("动量", 0.0, 1.0, 0.937, decimals=3, step=0.01, label_width=LABEL_W, spin_width=SPIN_W)
        params_grid.addWidget(self.lr0_spin, 1, 0)
        params_grid.addWidget(self.lrf_spin, 1, 1)
        params_grid.addWidget(self.momentum_spin, 1, 2)

        # 第2行：正则化参数
        self.weight_decay_spin = LabeledDoubleSpinBox("权重衰减", 0.0, 0.1, 0.0005, decimals=4, step=0.0001, label_width=LABEL_W, spin_width=SPIN_W)
        self.label_smoothing_spin = LabeledDoubleSpinBox("标签平滑", 0.0, 0.5, 0.0, decimals=2, step=0.01, label_width=LABEL_W, spin_width=SPIN_W)
        self.warmup_epochs_spin = LabeledDoubleSpinBox("预热轮数", 0.0, 10.0, 3.0, decimals=1, step=0.5, label_width=LABEL_W, spin_width=SPIN_W)
        params_grid.addWidget(self.weight_decay_spin, 2, 0)
        params_grid.addWidget(self.label_smoothing_spin, 2, 1)
        params_grid.addWidget(self.warmup_epochs_spin, 2, 2)

        # 第3行：损失权重
        self.box_spin = LabeledDoubleSpinBox("边界框损失", 0.1, 20.0, 7.5, decimals=1, step=0.5, label_width=LABEL_W, spin_width=SPIN_W)
        self.cls_spin = LabeledDoubleSpinBox("分类损失", 0.1, 5.0, 0.5, decimals=1, step=0.1, label_width=LABEL_W, spin_width=SPIN_W)
        self.dfl_spin = LabeledDoubleSpinBox("分布焦点损失", 0.1, 5.0, 1.5, decimals=1, step=0.1, label_width=LABEL_W, spin_width=SPIN_W)
        params_grid.addWidget(self.box_spin, 3, 0)
        params_grid.addWidget(self.cls_spin, 3, 1)
        params_grid.addWidget(self.dfl_spin, 3, 2)

        # 第4行：SAHI 小目标优化参数
        self.sahi_enabled_check = QCheckBox("启用SAHI小目标优化")
        self.sahi_slice_height_spin = LabeledSpinBox("切片高度", 32, 2048, 512, label_width=LABEL_W, spin_width=SPIN_W)
        self.sahi_slice_width_spin = LabeledSpinBox("切片宽度", 32, 2048, 512, label_width=LABEL_W, spin_width=SPIN_W)
        self.sahi_overlap_ratio_spin = LabeledDoubleSpinBox("重叠比例", 0.0, 1.0, 0.25, decimals=2, step=0.05, label_width=LABEL_W, spin_width=SPIN_W)
        params_grid.addWidget(self.sahi_enabled_check, 4, 0)
        params_grid.addWidget(self.sahi_slice_height_spin, 4, 1)
        params_grid.addWidget(self.sahi_slice_width_spin, 4, 2)
        params_grid.addWidget(self.sahi_overlap_ratio_spin, 4, 3)

        # 默认隐藏 SAHI 参数，勾选后显示
        self.sahi_slice_height_spin.setVisible(False)
        self.sahi_slice_width_spin.setVisible(False)
        self.sahi_overlap_ratio_spin.setVisible(False)
        self.sahi_enabled_check.toggled.connect(self._on_sahi_toggled)

        self._add_widget_row("训练参数：", params_widget)

        sahi_hint = QLabel("说明：当前 SAHI 选项仅在训练结果目录保存 sahi_config.json，暂未接入切片推理流程。")
        sahi_hint.setWordWrap(True)
        sahi_hint.setProperty("hint", True)
        self.content_layout.addWidget(sahi_hint)

        self.train_btn = SuccessButton("开始训练")
        self.train_btn.setMinimumWidth(140)
        self.train_btn.clicked.connect(self._train_model)
        self.content_layout.addWidget(self.train_btn, alignment=Qt.AlignLeft)

    def _populate_base_models(self) -> None:
        """根据当前任务类型，使用对应后缀的基模型填充模型下拉框。"""
        task = self.task_combo.currentData() or "detect"
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for name in _build_base_models(task):
            # display_name 与 data 均为模型名，方便直接传给训练命令
            self.model_combo.addItem(f"基模型：{name}", name)
        self.model_combo.blockSignals(False)
        # 默认选中第一个，并同步到路径输入框
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)
            self._on_model_changed()

    def _on_task_changed(self) -> None:
        """切换任务类型时刷新模型下拉框（含基模型与已训练模型）。"""
        self._refresh_models()

    def _on_sahi_toggled(self, checked: bool) -> None:
        """切换 SAHI 启用状态时显示/隐藏相关参数。"""
        self.sahi_slice_height_spin.setVisible(checked)
        self.sahi_slice_width_spin.setVisible(checked)
        self.sahi_overlap_ratio_spin.setVisible(checked)

    def _refresh_models(self) -> None:
        """重新填充模型下拉框：基模型 + 工作目录下已训练的模型。"""
        self._populate_base_models()

        # 使用基类方法发现已训练模型
        self._discover_and_populate_models(
            self.model_combo,
            include_base_models=True,
            task=self.task_combo.currentData() or "detect",
        )

    def _on_model_changed(self) -> None:
        """下拉框选择变化时同步到模型路径输入框。"""
        data = self.model_combo.currentData()
        if data:
            self.model_edit.setText(str(data))

    def _get_split_ratios(self) -> Optional[tuple]:
        """获取并校验训练/测试划分比例。"""
        train = self.train_ratio_spin.value()
        test = self.test_ratio_spin.value()
        total = train + test
        if total == 0:
            QMessageBox.warning(self, "参数错误", "划分比例不能全为 0")
            return None
        return train / total, test / total

    def _export_dataset(self) -> None:
        """导出 YOLO 数据集。"""
        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        input_dir = self._require_existing_dir(
            str(work_dir / IMAGES_FOLDER), "标注目录"
        )
        if input_dir is None:
            return

        ratios = self._get_split_ratios()
        if ratios is None:
            return
        train_ratio, test_ratio = ratios

        output_dir = work_dir / DATASET_FOLDER
        self.dataset_dir_edit.setText(str(output_dir))

        arguments = build_script_argv(
            "datasets", "export",
            input=input_dir,
            output=output_dir,
            task=self.task_combo.currentData(),
            train_ratio=f"{train_ratio:.4f}",
            test_ratio=f"{test_ratio:.4f}",
            export_empty_labels=self.export_empty_check.isChecked(),
            export_unlabeled=self.export_unlabeled_check.isChecked(),
        )
        self._start_subprocess(arguments, title="导出 YOLO 数据集")

    def on_page_shown(self) -> None:
        """页面首次显示时填充工作目录下的默认值，并刷新可用模型列表。"""
        work_dir = self._work_dir()
        if not work_dir:
            return
        if not self.dataset_dir_edit.text().strip():
            self.dataset_dir_edit.setText(str(Path(work_dir) / DATASET_FOLDER))
        # 刷新模型列表（基模型 + 已训练模型），但保留用户当前已填写的模型路径
        current_model = self.model_edit.text().strip()
        self._refresh_models()
        if current_model:
            self.model_edit.setText(current_model)

    def _train_model(self) -> None:
        """启动模型训练。"""
        dataset_dir = self.dataset_dir_edit.text().strip()
        if not dataset_dir:
            QMessageBox.warning(self, "参数缺失", "请选择数据集目录")
            return

        work_dir = self._require_work_dir()
        if work_dir is None:
            return

        yaml_path = Path(dataset_dir) / "data.yaml"
        project = str(work_dir / TRAIN_FOLDER)
        name = self.name_edit.text().strip() or "train"

        arguments = build_script_argv(
            "train", "run",
            data=yaml_path,
            task=self.task_combo.currentData(),
            model=self.model_edit.text().strip() or "yolov8n",
            epochs=self.epochs_spin.value(),
            imgsz=self.imgsz_spin.value(),
            batch=self.batch_spin.value(),
            project=project,
            name=name,
            # 学习率参数
            lr0=self.lr0_spin.value(),
            lrf=self.lrf_spin.value(),
            momentum=self.momentum_spin.value(),
            # 正则化参数
            weight_decay=self.weight_decay_spin.value(),
            label_smoothing=self.label_smoothing_spin.value(),
            warmup_epochs=self.warmup_epochs_spin.value(),
            # 损失权重
            box=self.box_spin.value(),
            cls=self.cls_spin.value(),
            dfl=self.dfl_spin.value(),
            # SAHI 参数
            sahi_enabled=self.sahi_enabled_check.isChecked(),
            sahi_slice_height=self.sahi_slice_height_spin.value(),
            sahi_slice_width=self.sahi_slice_width_spin.value(),
            # SAHI 使用同一重叠比例同时应用于高度和宽度
            sahi_overlap_height_ratio=self.sahi_overlap_ratio_spin.value(),
            sahi_overlap_width_ratio=self.sahi_overlap_ratio_spin.value(),
        )
        self._start_subprocess(arguments, title="训练模型")
