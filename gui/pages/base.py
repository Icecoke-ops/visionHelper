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
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
from gui.utils._proc import infer_script_name, build_script_argv
from gui.config import TRAIN_FOLDER, IMAGES_FOLDER, app_root, is_frozen
from gui.context import AppContext
from gui.components.run_log import RunLogDialog
from gui.components.widgets import FormRow, SecondaryButton, HSeparator, LabeledDoubleSpinBox, SuccessButton


# 任务类型选项常量，用于多处复用的 QComboBox 初始化
TASK_OPTIONS: List[Tuple[str, str]] = [
    ("目标检测（detect）", "detect"),
    ("旋转框（obb）", "obb"),
    ("实例分割（segment）", "segment"),
    ("图像分类（classify）", "classify"),
]


class AutoAnnotateMixin:
    """自动标注表单与执行逻辑的共享混入类。

    供需要自动标注功能的页面（如 DataAnnotationPage）继承使用。
    要求子类具有 ``ctx`` 属性（AppContext），并已调用 ``BaseTaskPage.__init__``。
    """

    def _build_auto_annotate_form(self):
        """构建自动标注表单：模型选择、任务类型、处理范围、推理参数、文件后缀。"""
        # ===== 模型选择 =====
        self._model_widget, self.model_combo, self.refresh_model_btn = self._build_model_selector_widget()
        self.refresh_model_btn.clicked.connect(self._refresh_models)
        self.content_layout.addWidget(FormRow("选择模型：", self._model_widget))

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("模型文件路径将在此处显示")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._add_widget_row("模型路径：", self.model_path_edit)

        # ===== 任务参数 =====
        self.task_combo = self._build_task_combo()
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

    def _refresh_models(self):
        """扫描训练结果目录并填充模型下拉框。"""
        self.model_combo.clear()

        runs_dir = self._runs_dir()
        if not runs_dir:
            QMessageBox.warning(self, "参数缺失", "请先设置工作目录")
            return

        self._discover_and_populate_models(self.model_combo)

        if self.model_combo.count() == 0:
            QMessageBox.information(
                self,
                "未找到模型",
                f"在 {runs_dir} 下未找到训练模型，请确认已完成训练。",
            )
            return

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
            "datasets", "auto",
            input=image_dir,
            model=model_path,
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

    def on_page_shown(self) -> None:
        """页面首次显示时的钩子方法，子类可重写以填充默认值。

        默认为空操作，子类无需强制实现。
        """
        pass

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
        - 若未设置，统一返回空串，由调用方（如 :meth:`_start_subprocess`）
          弹窗提示用户配置 Python 环境。

        注意：打包态下 ``sys.executable`` 是 GUI 自身的 bootloader，
        内部并不包含 torch / ultralytics 等重型依赖，绝不能用它来跑
        ``scripts``。开发态下 ``sys.executable`` 虽然可用，但为了保持
        行为一致性，也要求用户显式配置。
        """
        if self.ctx is not None and self.ctx.python_env:
            return self.ctx.python_env
        return ""

    def _start_subprocess(self, arguments: list, title: str):
        """弹出日志窗口并使用当前选定的 Python 环境启动子进程执行任务。

        子进程启动时会：

        1. 将工作目录设置为 :func:`gui.config.app_root`，保证 ``scripts/vh.py``
           能正确定位到 ``scripts`` 包；
        2. 将 ``app_root`` 注入到子进程的 ``PYTHONPATH``，作为第二重保险，
           即使用户在调用前手动改过 ``cwd`` 也不会影响脚本定位。

        在打包态下，如果用户未配置 Python 环境，会弹窗提示并取消执行。
        开发态下若未配置，回退到当前解释器 ``sys.executable``。
        """
        python_path = self._python_env()
        if not python_path:
            from gui.config import is_frozen
            if is_frozen():
                QMessageBox.warning(
                    self,
                    "未配置 Python 环境",
                    "当前为打包发布版本，必须在导航栏下方手动选择带有所需依赖"
                    "（torch / ultralytics 等）的 Python 可执行文件后再运行任务。",
                )
                return
            # 开发态：回退到当前解释器
            python_path = sys.executable

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

    def _warn_no_python_env(self) -> None:
        """显示"未配置 Python 环境"的警告弹窗。"""
        from gui.config import is_frozen
        if is_frozen():
            QMessageBox.warning(
                self,
                "未配置 Python 环境",
                "当前为打包发布版本，必须在导航栏下方手动选择带有所需依赖"
                "（torch / ultralytics 等）的 Python 可执行文件后再运行任务。",
            )
        else:
            QMessageBox.warning(
                self,
                "未配置 Python 环境",
                "开发态下建议在导航栏下方配置 Python 环境，\n"
                "当前将使用默认解释器运行，可能缺少依赖。",
            )

    def _runs_dir(self) -> str:
        """根据当前工作目录返回训练结果目录路径。

        统一使用 :data:`gui.config.TRAIN_FOLDER` 常量，避免子类重复实现。
        """
        work_dir = self._work_dir()
        if not work_dir:
            return ""
        return str(Path(work_dir) / TRAIN_FOLDER)

    def _build_task_combo(self) -> QComboBox:
        """创建任务类型下拉框，预填充 detect / obb / segment / classify 四个选项。

        Returns:
            填充好选项的 :class:`QComboBox`，调用方可直接添加到布局中。
        """
        combo = QComboBox()
        for display, data in TASK_OPTIONS:
            combo.addItem(display, data)
        return combo

    def _build_model_selector_widget(self) -> Tuple[QWidget, QComboBox, SecondaryButton]:
        """构建"模型选择下拉框 + 刷新按钮"组合控件。

        返回 ``(container, model_combo, refresh_btn)`` 三元组，调用方
        **必须**将 ``container`` 存储为实例属性（如 ``self._model_widget``），
        否则容器被 GC 回收后子控件将失效。

        注意：Qt 的父子关系会自动管理控件生命周期，因此只要
        ``container`` 被正确添加到布局中，它就不会被 GC 回收。
        本方法返回 container 仅是为了让调用方能够显式控制其生命周期。

        Returns:
            ``(QWidget, QComboBox, SecondaryButton)``：容器、模型下拉框、刷新按钮。
        """
        model_widget = QWidget()
        model_row = QHBoxLayout(model_widget)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(theme.SPACING_SM)

        model_combo = QComboBox()
        model_combo.setMinimumWidth(300)
        model_combo.setEditable(False)
        model_row.addWidget(model_combo, 1)

        refresh_btn = SecondaryButton("刷新模型")
        model_row.addWidget(refresh_btn)

        return model_widget, model_combo, refresh_btn

    def _discover_and_populate_models(
        self,
        model_combo: QComboBox,
        include_base_models: bool = False,
        task: str = "detect",
    ) -> None:
        """扫描训练结果目录并填充模型下拉框。

        通过 :func:`scripts.common.utils.discover_trained_models` 获取
        已训练模型列表（只依赖标准库，无重型依赖）。

        Args:
            model_combo: 要填充的模型下拉框。
            include_base_models: 是否同时填充基模型列表。
            task: 当前任务类型，用于生成带后缀的基模型名。
        """
        runs_dir = self._runs_dir()
        if not runs_dir:
            return

        try:
            from scripts.common.utils import discover_trained_models
        except ImportError:
            return

        trained_models = discover_trained_models(runs_dir)
        if not trained_models:
            return

        if include_base_models and model_combo.count() > 0:
            model_combo.insertSeparator(model_combo.count())

        for display_name, model_path in trained_models:
            model_combo.addItem(f"已训练：{display_name}", model_path)
