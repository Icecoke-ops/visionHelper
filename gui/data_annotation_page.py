#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据标注页面模块。

提供目录级标注统计功能，展示：
    - 图片总数
    - 已标注图片数量
    - 包含目标检测框的图片数量
    - 包含 OBB 的图片数量
    - 包含多边形的图片数量
    - 手动标注、自动标注、自动标注并手动矫正的图片数量
    - 各标签下目标检测、OBB、多边形实例数量

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`，本模块不再
书写内联样式。
"""

from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from gui import theme
from gui.base_pages import BasePage
from gui.config import IMAGES_FOLDER
from gui.widgets import HintCard, PrimaryButton, SectionTitle, StatItem
# 注意：不要在模块顶层 import scripts.api，避免在 PyInstaller 打包态下
# GUI 进程因找不到 scripts 包（或顺带触发 torch / ultralytics 等重依赖
# 的 import）而启动失败。需要时改为函数内部局部导入轻量子模块。


# X-AnyLabeling 项目地址 / 下载地址
XANYLABELING_HOMEPAGE = "https://github.com/CVHub520/X-AnyLabeling"
XANYLABELING_RELEASES = "https://github.com/CVHub520/X-AnyLabeling/releases"


# 统计项 key -> 显示标题
_STAT_FIELDS = [
    ("total_images", "图片总数"),
    ("annotated_images", "已标注数量"),
    ("unannotated_images", "未标注数量"),
    ("detection_images", "目标检测数量"),
    ("obb_images", "OBB 数量"),
    ("polygon_images", "多边形数量"),
    ("manual_images", "手动标注数量"),
    ("auto_images", "自动标注数量"),
    ("auto_corrected_images", "自动标注并手动矫正数量"),
]


class DataAnnotationPage(BasePage):
    """数据标注页面：统计目录下的图片与标注信息。"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._init_controls()

    def _init_controls(self):
        """初始化页面控件。"""
        # 顶部：标注工具说明 + 下载按钮（HintCard）
        self.content_layout.addWidget(self._build_annotation_tool_hint())

        # 目录选择行（不显示，仅保留工作目录缓存）
        self.dir_edit = QLineEdit()
        self.dir_edit.setVisible(False)
        self.content_layout.addWidget(self.dir_edit)

        # 统计结果区
        self.content_layout.addWidget(SectionTitle("整体统计"))
        self.result_labels: dict = {}
        for key, title in _STAT_FIELDS:
            stat = StatItem(title=title, value="0", kind="success")
            self.result_labels[key] = stat
            self.content_layout.addWidget(stat)

        self.content_layout.addSpacing(theme.SPACING_SM)

        # 标签统计表格
        self.content_layout.addWidget(SectionTitle("标签统计"))
        self.label_table = QTableWidget()
        self.label_table.setColumnCount(4)
        self.label_table.setHorizontalHeaderLabels(
            ["标签名", "目标检测数据量", "OBB 数量", "多边形数量"]
        )
        self.label_table.horizontalHeader().setStretchLastSection(True)
        self.label_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.label_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.label_table.setMinimumHeight(160)
        # 表格样式（背景、表头、单元格 padding 等）由全局 QSS 提供
        self.content_layout.addWidget(self.label_table)

        # 统计按钮
        self.content_layout.addSpacing(theme.SPACING_SM)
        self.stats_btn = PrimaryButton("开始统计")
        self.stats_btn.setMinimumWidth(120)
        self.stats_btn.clicked.connect(self._run_stats)
        self.content_layout.addWidget(self.stats_btn, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # 顶部说明区
    # ------------------------------------------------------------------

    def _build_annotation_tool_hint(self) -> HintCard:
        """构建顶部"标注工具推荐"卡片：说明 + 下载按钮。

        使用统一的 :class:`HintCard` 替代原先的内联样式 QFrame。
        """
        download_btn = PrimaryButton("点击下载 X-AnyLabeling")
        download_btn.setMinimumWidth(180)
        download_btn.clicked.connect(self._open_xanylabeling_download)

        return HintCard(
            title="📝 推荐使用 X-AnyLabeling 进行数据标注",
            description=(
                "本工具基于 LabelMe JSON 格式进行统计与自动标注。"
                "请使用 X-AnyLabeling（兼容 LabelMe 格式，支持多种 AI 辅助标注）"
                "完成图片的人工标注，再回到本页查看统计结果。\n"
                f"项目主页：{XANYLABELING_HOMEPAGE}"
            ),
            action=download_btn,
        )

    def _open_xanylabeling_download(self):
        """打开 X-AnyLabeling 的 GitHub Releases 下载页。"""
        QDesktopServices.openUrl(QUrl(XANYLABELING_RELEASES))

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def on_page_shown(self):
        """页面首次显示时，自动填充工作目录下的 images 目录。"""
        work_dir = self._work_dir()
        if work_dir:
            self.dir_edit.setText(str(Path(work_dir) / IMAGES_FOLDER))

    # ------------------------------------------------------------------
    # 行为
    # ------------------------------------------------------------------

    def _run_stats(self):
        """执行统计并刷新界面结果。"""
        folder = self.dir_edit.text().strip()
        if not folder:
            work_dir = self._work_dir()
            folder = str(Path(work_dir) / IMAGES_FOLDER) if work_dir else ""
        if not folder:
            QMessageBox.warning(self, "参数缺失", "请选择要统计的图片目录")
            return

        if not Path(folder).is_dir():
            QMessageBox.warning(self, "路径错误", f"目录不存在：{folder}")
            return

        try:
            from scripts.annotation_stats import collect_annotation_stats

            stats = collect_annotation_stats(folder)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "统计失败", f"统计过程中发生错误：{exc}")
            return

        self._update_result_labels(stats)

        try:
            from scripts.annotation_stats import collect_annotation_label_stats

            label_stats = collect_annotation_label_stats(folder)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "统计失败", f"标签统计过程中发生错误：{exc}")
            return

        self._update_label_table(label_stats)

    def _update_result_labels(self, stats: dict):
        """根据统计结果更新界面显示。"""
        for key, _ in _STAT_FIELDS:
            value = stats.get(key, 0)
            self.result_labels[key].set_value(str(value))

    def _update_label_table(self, label_stats: list):
        """根据按标签统计结果刷新表格。"""
        self.label_table.setRowCount(len(label_stats))
        for row, item in enumerate(label_stats):
            label_name = item.get("label", "")
            detection_count = item.get("detection_count", 0)
            obb_count = item.get("obb_count", 0)
            polygon_count = item.get("polygon_count", 0)

            self.label_table.setItem(row, 0, QTableWidgetItem(str(label_name)))
            self.label_table.setItem(row, 1, QTableWidgetItem(str(detection_count)))
            self.label_table.setItem(row, 2, QTableWidgetItem(str(obb_count)))
            self.label_table.setItem(row, 3, QTableWidgetItem(str(polygon_count)))
