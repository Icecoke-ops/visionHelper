#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练页面模块。

整合 YOLO 数据集导出与模型训练两个子任务：
    - 导出数据集：调用 ``scripts.export_yolo_dataset`` 将工作目录下的
      X-AnyLabeling 标注导出为 YOLO 格式，并自动划分 train/test。
    - 训练模型：调用 ``scripts.train_model`` 对导出的数据集进行训练。

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`，本模块不再
书写内联样式。
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from gui import theme
from gui._proc import build_script_argv
from gui.base_pages import BaseTaskPage
from gui.config import DATASET_FOLDER, IMAGES_FOLDER, TRAIN_FOLDER
from gui.widgets import (
    FormRow,
    LabeledSpinBox,
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

# 各任务对应的模型名后缀（与 scripts.train_model._TASK_MODEL_SUFFIX 保持一致）
_TASK_MODEL_SUFFIX = {
    "detect": "",
    "obb": "-obb",
    "segment": "-seg",
    "classify": "-cls",
}


def _build_base_models(task: str) -> list:
    """根据任务类型构造基模型名列表（自动追加 ``-obb``/``-seg``/``-cls`` 后缀）。"""
    suffix = _TASK_MODEL_SUFFIX.get(task, "")
    return [f"{name}{suffix}" for name in BASE_MODEL_FAMILIES]


class ModelTrainingPage(BaseTaskPage):
    """模型训练页面：导出 YOLO 数据集并启动训练。"""

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._build_form()

    def _build_form(self):
        # ===== 数据集导出区域 =====
        self.task_combo = QComboBox()
        self.task_combo.addItem("目标检测（detect）", "detect")
        self.task_combo.addItem("旋转框（obb）", "obb")
        self.task_combo.addItem("实例分割（segment）", "segment")
        self.task_combo.addItem("图像分类（classify）", "classify")
        # 任务变化时刷新基模型下拉框（自动追加 -obb/-seg/-cls 后缀）
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

        self.export_btn = PrimaryButton("导出数据集")
        self.export_btn.setMinimumWidth(140)
        self.export_btn.clicked.connect(self._export_dataset)
        self.content_layout.addWidget(self.export_btn, alignment=Qt.AlignLeft)

        # ===== 模型训练区域 =====
        # 新建第二张卡片，将"模型训练"与上方"数据集导出"在视觉上分离
        _, self.content_layout = self._add_card()

        self.dataset_dir_edit = self._add_file_row("数据集目录：", is_directory=True)


        # ===== 模型选择 =====
        # 与"自动标注"页面保持一致的交互：下拉框 + 路径输入框 + 刷新按钮。
        # 下拉框中包含常用基模型，以及工作目录下已训练好的模型；
        # 用户也可以直接在路径输入框中填写：
        #   - 模型名称（如 ``yolov8n``，会自动下载）
        #   - 本地权重路径（如 ``/path/to/best.pt``）
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

        # 模型路径/名称输入框（可手动编辑）
        self.model_edit = QLineEdit("yolov8n")
        self.model_edit.setPlaceholderText("可填写模型名称（如 yolov8n）或权重文件路径")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._add_widget_row("模型路径：", self.model_edit)

        # 初始化下拉框，仅填充基模型；训练模型在工作目录可用时动态刷新
        self._populate_base_models()

        self.name_edit = QLineEdit()
        self._add_widget_row("训练名称：", self.name_edit)

        hparams_widget = QWidget()
        hparams_layout = QHBoxLayout(hparams_widget)
        hparams_layout.setContentsMargins(0, 0, 0, 0)
        hparams_layout.setSpacing(theme.SPACING_MD)

        self.epochs_spin = LabeledSpinBox("轮数", 1, 10000, 100)
        self.imgsz_spin = LabeledSpinBox("尺寸", 32, 2048, 640)
        self.batch_spin = LabeledSpinBox("批次", 1, 256, 16)
        hparams_layout.addWidget(self.epochs_spin)
        hparams_layout.addWidget(self.imgsz_spin)
        hparams_layout.addWidget(self.batch_spin)
        hparams_layout.addStretch()
        self._add_widget_row("训练参数：", hparams_widget)

        self.train_btn = SuccessButton("开始训练")
        self.train_btn.setMinimumWidth(140)
        self.train_btn.clicked.connect(self._train_model)
        self.content_layout.addWidget(self.train_btn, alignment=Qt.AlignLeft)

    def _populate_base_models(self):
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

    def _on_task_changed(self):
        """切换任务类型时刷新模型下拉框（含基模型与已训练模型）。"""
        self._refresh_models()

    def _runs_dir(self) -> str:
        """根据当前工作目录返回训练结果目录路径。"""
        work_dir = self._work_dir()
        if not work_dir:
            return ""
        return str(Path(work_dir) / TRAIN_FOLDER)

    def _refresh_models(self):
        """重新填充模型下拉框：基模型 + 工作目录下已训练的模型。"""
        self._populate_base_models()

        runs_dir = self._runs_dir()
        if not runs_dir or not Path(runs_dir).is_dir():
            return

        try:
            from scripts.common.utils import discover_trained_models

            trained_models = discover_trained_models(runs_dir)
        except Exception:
            trained_models = []

        if not trained_models:
            return

        # 在基模型与训练模型之间插入分隔项
        self.model_combo.insertSeparator(self.model_combo.count())
        for display_name, model_path in trained_models:
            self.model_combo.addItem(f"已训练：{display_name}", model_path)

    def _on_model_changed(self):
        """下拉框选择变化时同步到模型路径输入框。"""
        data = self.model_combo.currentData()
        if data:
            self.model_edit.setText(str(data))

    def _get_split_ratios(self) -> tuple:
        """获取并校验训练/测试划分比例。"""
        train = self.train_ratio_spin.value()
        test = self.test_ratio_spin.value()
        total = train + test
        if total == 0:
            QMessageBox.warning(self, "参数错误", "划分比例不能全为 0")
            return None
        return train / total, test / total

    def _export_dataset(self):
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
        )
        self._start_subprocess(arguments, title="导出 YOLO 数据集")

    def on_page_shown(self):
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

    def _train_model(self):
        dataset_dir = self.dataset_dir_edit.text().strip()
        if not dataset_dir:
            QMessageBox.warning(self, "参数缺失", "请选择数据集目录")
            return

        yaml_path = Path(dataset_dir) / "data.yaml"
        work_dir = self._work_dir()
        project = str(Path(work_dir) / TRAIN_FOLDER) if work_dir else None

        arguments = build_script_argv(
            "train", "run",
            data=yaml_path,
            task=self.task_combo.currentData(),
            model=self.model_edit.text().strip() or "yolov8n",
            epochs=self.epochs_spin.value(),
            imgsz=self.imgsz_spin.value(),
            batch=self.batch_spin.value(),
            project=project,
            name=self.name_edit.text().strip() or None,
        )
        self._start_subprocess(arguments, title="训练模型")
