#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据标注页面模块。

提供目录级标注信息展示功能，展示：
    - 图片总数
    - 已标注图片数量
    - 包含目标检测框的图片数量
    - 包含 OBB 的图片数量
    - 包含多边形的图片数量
    - 手动标注、自动标注、自动标注并手动矫正的图片数量
    - 各标签下目标检测、OBB、多边形实例数量

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.components.widgets`，本模块不再
书写内联样式。
"""

import sys
from pathlib import Path
from typing import List, Dict

from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from gui import theme
from gui.pages.base import BaseTaskPage, AutoAnnotateMixin
from gui.config import IMAGES_FOLDER, app_root
from gui.components.widgets import (
    DangerButton,
    FormRow,
    HintCard,
    HSeparator,
    LabeledDoubleSpinBox,
    PrimaryButton,
    StatItem,
    SuccessButton,
)
from gui.utils._proc import build_script_argv

# X-AnyLabeling 下载地址
XANYLABELING_RELEASES = "https://github.com/CVHub520/X-AnyLabeling/releases"

# 统计项 key -> 显示标题（按语义分组，渲染为多列网格）
_STAT_GROUPS: List[List[tuple]] = [
    [
        ("total_images", "图片总数"),
        ("annotated_images", "已标注数量"),
        ("unannotated_images", "未标注数量"),
    ],
    [
        ("detection_images", "目标检测数量"),
        ("obb_images", "OBB 数量"),
        ("polygon_images", "多边形数量"),
    ],
    [
        ("manual_images", "手动标注数量"),
        ("auto_images", "自动标注数量"),
        ("auto_corrected_images", "手动矫正数量"),
    ],
]


# CLI 输出标记（与 scripts/common/config.py 同步，避免在 GUI 进程导入 scripts）
_STATS_BEGIN = "===VH_STATS_BEGIN==="
_STATS_END = "===VH_STATS_END==="


class StatsWorkerThread(QThread):
    """后台统计工作线程（通过子进程调用 CLI，不直接 import scripts）。"""

    stats_finished = pyqtSignal(dict, list)
    stats_error = pyqtSignal(str)

    def __init__(self, folder: str, python_exe: str, cwd: str, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.python_exe = python_exe
        self.cwd = cwd

    def run(self):
        try:
            import json
            import subprocess

            cmd = [
                self.python_exe,
                "-m", "scripts.vh", "datasets", "stats",
                "--input", self.folder,
                "--label-stats",
                "--json",
            ]
            completed = subprocess.run(
                cmd, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
            if completed.returncode != 0:
                self.stats_error.emit(
                    f"子进程退出码 {completed.returncode}: {completed.stderr}"
                )
                return

            stdout = completed.stdout or ""
            begin = stdout.rfind(_STATS_BEGIN)
            if begin < 0:
                self.stats_error.emit("未找到统计结果标记")
                return
            end = stdout.find(_STATS_END, begin + len(_STATS_BEGIN))
            if end < 0:
                self.stats_error.emit("未找到统计结果结束标记")
                return

            payload = json.loads(stdout[begin + len(_STATS_BEGIN):end].strip())
            stats = payload.get("stats", {})
            label_stats = payload.get("label_stats", [])
            self.stats_finished.emit(stats, label_stats)

        except Exception as exc:
            self.stats_error.emit(str(exc))


class DataAnnotationPage(BaseTaskPage, AutoAnnotateMixin):
    """数据标注页面：统计目录下的图片与标注信息。"""

    def __init__(self, parent=None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._init_controls()

    def _init_controls(self):
        """初始化页面控件。"""
        # 隐藏的 dir_edit 用于缓存图片目录路径（不显示在界面上）
        self._image_dir: str = ""

        # 基类默认卡片不适用于本页面（需要蓝色 HintCard 在最前面），移除之
        # 使用 deleteLater() 延迟删除，避免立即释放导致的潜在问题
        self._wrapper_layout.removeWidget(self.card)
        self.card.hide()  # 先隐藏，避免在删除前继续显示
        self.card.deleteLater()  # 延迟到事件循环结束时删除
        self.card = None

        # ===== 卡片 1：蓝色提示（无外层白色卡片） =====
        hint_card = self._build_annotation_tool_hint()
        insert_index = max(0, self._wrapper_layout.count() - 1)
        self._wrapper_layout.insertWidget(insert_index, hint_card)

        # ===== 卡片 2：标注统计信息 =====
        _, self.content_layout = self._add_card()
        self._build_stats_card()

        # ===== 卡片 3：自动标注 =====
        _, self.content_layout = self._add_card()
        self._build_auto_annotate_form()

        # ===== 卡片 4：标签清除功能 =====
        _, self.content_layout = self._add_card()
        self._build_clear_card()

    def _build_stats_card(self):
        """构建"标注信息"卡片内容。"""
        # 统计结果区（多列网格布局）
        self.result_labels = {}

        stats_container = QWidget()
        grid = QGridLayout(stats_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(theme.SPACING_LG)
        grid.setVerticalSpacing(theme.SPACING_SM)

        columns = max((len(group) for group in _STAT_GROUPS), default=3)
        for row, group in enumerate(_STAT_GROUPS):
            for col, (key, title) in enumerate(group):
                stat = StatItem(title=title, value="0", kind="success", title_width=96)
                self.result_labels[key] = stat
                grid.addWidget(stat, row, col)
            for col in range(len(group), columns):
                grid.setColumnStretch(col, 1)
        for col in range(columns):
            grid.setColumnStretch(col, 1)

        self.content_layout.addWidget(stats_container)
        self.content_layout.addSpacing(theme.SPACING_SM)

        # 标签统计表格
        self.label_table = QTableWidget()
        self.label_table.setColumnCount(4)
        self.label_table.setHorizontalHeaderLabels(
            ["标签名", "目标检测数据量", "OBB 数量", "多边形数量"]
        )
        for col in range(self.label_table.columnCount()):
            header_item = self.label_table.horizontalHeaderItem(col)
            if header_item is not None:
                header_item.setTextAlignment(Qt.AlignCenter)
        header = self.label_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.label_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.label_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.label_table.setMinimumHeight(160)
        self.content_layout.addWidget(self.label_table)

        # 统计按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.stats_btn = PrimaryButton("开始统计")
        self.stats_btn.setMinimumWidth(120)
        self.stats_btn.clicked.connect(self._run_stats)
        self.content_layout.addWidget(self.stats_btn, alignment=Qt.AlignLeft)
        
        # 初始化后台线程引用
        self._stats_thread = None

    def _build_clear_card(self):
        """构建"标签清除"卡片内容。"""
        clear_scope_widget = QWidget()
        clear_scope_layout = QHBoxLayout(clear_scope_widget)
        clear_scope_layout.setContentsMargins(0, 0, 0, 0)
        clear_scope_layout.setSpacing(theme.SPACING_MD)

        self.clear_auto_cb = QCheckBox("自动标注")
        self.clear_auto_corrected_cb = QCheckBox("自动标注且修正")
        self.clear_dry_run_cb = QCheckBox("预演模式（不删除）")
        clear_scope_layout.addWidget(self.clear_auto_cb)
        clear_scope_layout.addWidget(self.clear_auto_corrected_cb)
        clear_scope_layout.addWidget(self.clear_dry_run_cb)

        clear_scope_layout.addSpacing(theme.SPACING_MD)
        self.clear_btn = DangerButton("清除标签")
        self.clear_btn.setMinimumWidth(120)
        self.clear_btn.clicked.connect(self._run_clear_annotations)
        clear_scope_layout.addWidget(self.clear_btn)
        clear_scope_layout.addStretch()

        self.content_layout.addWidget(FormRow("清除范围：", clear_scope_widget))

    def _build_annotation_tool_hint(self) -> HintCard:
        """构建顶部"标注工具推荐"卡片。"""
        download_btn = PrimaryButton("点击下载 X-AnyLabeling")
        download_btn.setMinimumWidth(180)
        download_btn.clicked.connect(self._open_xanylabeling_download)

        return HintCard(
            title="推荐使用 X-AnyLabeling 进行数据标注",
            description="本工具基于 X-AnyLabeling JSON 格式进行标注信息统计与自动标注",
            action=download_btn,
        )

    def _open_xanylabeling_download(self):
        """打开 X-AnyLabeling 的 GitHub Releases 下载页。"""
        QDesktopServices.openUrl(QUrl(XANYLABELING_RELEASES))

    def on_page_shown(self):
        """页面首次显示时，自动填充工作目录下的 images 目录并刷新模型列表。"""
        work_dir = self._work_dir()
        if work_dir:
            self._image_dir = str(Path(work_dir) / IMAGES_FOLDER)
            self._refresh_models()

    def _resolve_image_dir(self, hint: str) -> str:
        """解析当前应使用的图片目录路径。"""
        folder = self._image_dir
        if not folder:
            work_dir = self._work_dir()
            folder = str(Path(work_dir) / IMAGES_FOLDER) if work_dir else ""
        path = self._require_existing_dir(folder, hint)
        return str(path) if path else ""

    def _run_stats(self):
        """通过子进程执行标注统计（非阻塞线程）。"""
        folder = self._resolve_image_dir("要统计的图片目录")
        if not folder:
            return

        python = self._python_env() or sys.executable
        if not python:
            self._warn_no_python_env()
            return

        self.stats_btn.setEnabled(False)

        self._stats_thread = StatsWorkerThread(folder, python, str(app_root()), self)
        self._stats_thread.stats_finished.connect(self._on_stats_finished)
        self._stats_thread.stats_error.connect(self._on_stats_error)
        self._stats_thread.start()

    def _on_stats_finished(self, stats: Dict[str, int], label_stats: List[Dict[str, int]]):
        """统计完成的回调函数。"""
        for key, stat_item in self.result_labels.items():
            value = stats.get(key, 0)
            stat_item.set_value(str(value))
        
        self._update_label_table(label_stats)
        self.stats_btn.setEnabled(True)

    def _on_stats_error(self, error_msg: str):
        """统计出错的回调函数。"""
        self.stats_btn.setEnabled(True)
        QMessageBox.critical(self, "统计失败", f"标注信息获取失败：{error_msg}")

    def _update_label_table(self, label_stats: List[Dict[str, int]]):
        """更新标签统计表格。"""
        self.label_table.setRowCount(len(label_stats))
        
        for row, item in enumerate(label_stats):
            # 标签名
            label_item = QTableWidgetItem(item.get("label", ""))
            label_item.setTextAlignment(Qt.AlignCenter)
            self.label_table.setItem(row, 0, label_item)
            
            # 目标检测数量
            detection_item = QTableWidgetItem(str(item.get("detection_count", 0)))
            detection_item.setTextAlignment(Qt.AlignCenter)
            self.label_table.setItem(row, 1, detection_item)
            
            # OBB数量
            obb_item = QTableWidgetItem(str(item.get("obb_count", 0)))
            obb_item.setTextAlignment(Qt.AlignCenter)
            self.label_table.setItem(row, 2, obb_item)
            
            # 多边形数量
            polygon_item = QTableWidgetItem(str(item.get("polygon_count", 0)))
            polygon_item.setTextAlignment(Qt.AlignCenter)
            self.label_table.setItem(row, 3, polygon_item)

    def _run_clear_annotations(self):
        """按勾选的标注类型清除目录下的 X-AnyLabeling JSON 标注文件。"""
        folder = self._resolve_image_dir("要清理的图片目录")
        if not folder:
            return

        include_auto = self.clear_auto_cb.isChecked()
        include_auto_corrected = self.clear_auto_corrected_cb.isChecked()
        dry_run = self.clear_dry_run_cb.isChecked()
        if not (include_auto or include_auto_corrected):
            QMessageBox.warning(self, "参数缺失", "请至少勾选一种待清除的标注类型")
            return

        scope_desc_parts: List[str] = []
        if include_auto:
            scope_desc_parts.append("自动标注")
        if include_auto_corrected:
            scope_desc_parts.append("自动标注且修正")
        scope_desc = "、".join(scope_desc_parts)

        if dry_run:
            confirm_text = (
                f"将预演扫描目录\n  {folder}\n"
                f"中类型为【{scope_desc}】的 X-AnyLabeling JSON 标注文件，"
                "不会实际删除。是否继续？"
            )
        else:
            confirm_text = (
                f"将从目录\n  {folder}\n"
                f"中删除类型为【{scope_desc}】的 X-AnyLabeling JSON 标注文件，"
                "该操作不可恢复，是否继续？"
            )

        confirm = QMessageBox.question(
            self,
            "确认清除标签" if not dry_run else "确认预演清除",
            confirm_text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        arguments = build_script_argv(
            "datasets", "clear",
            input=folder,
            include_auto=include_auto,
            include_auto_corrected=include_auto_corrected,
            include_manual=False,
            dry_run=dry_run,
        )
        self._start_subprocess(arguments, title="清除标签")
