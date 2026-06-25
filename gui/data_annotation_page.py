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

import os
import subprocess
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
from gui.base_pages import BaseTaskPage
from gui.config import IMAGES_FOLDER, app_root
from gui.widgets import (
    DangerButton,
    FormRow,
    HintCard,
    PrimaryButton,
    StatItem,
)
# 注意：不要在模块顶层 import scripts.api，避免在 PyInstaller 打包态下
# GUI 进程因找不到 scripts 包（或顺带触发 torch / ultralytics 等重依赖
# 的 import）而启动失败。需要时改为函数内部局部导入轻量子模块。


# X-AnyLabeling 项目地址 / 下载地址
XANYLABELING_HOMEPAGE = "https://github.com/CVHub520/X-AnyLabeling"
XANYLABELING_RELEASES = "https://github.com/CVHub520/X-AnyLabeling/releases"


def _build_script_env(root: str) -> dict:
    """构造调用 ``python scripts/vh.py`` 同步子进程时的环境变量。

    将 ``root`` 注入到 ``PYTHONPATH`` 头部，确保 ``scripts/vh.py`` 在打包/任意
    启动目录下都能定位到 ``scripts`` 包；同时设置 ``PYTHONUNBUFFERED=1``
    让子进程输出可以即时被捕获。
    """
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing_pp if existing_pp else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


# 统计项 key -> 显示标题
# 这里按"语义分组"组织：每个子列表会渲染为一行（多列网格），
# 这样可以让"整体统计"区域不再是又长又窄的一列，看起来更紧凑。
_STAT_GROUPS = [
    # 第 1 行：总览
    [
        ("total_images", "图片总数"),
        ("annotated_images", "已标注数量"),
        ("unannotated_images", "未标注数量"),
    ],
    # 第 2 行：按标注形状
    [
        ("detection_images", "目标检测数量"),
        ("obb_images", "OBB 数量"),
        ("polygon_images", "多边形数量"),
    ],
    # 第 3 行：按标注方式
    [
        ("manual_images", "手动标注数量"),
        ("auto_images", "自动标注数量"),
        ("auto_corrected_images", "手动矫正数量"),
    ],
]

# 扁平化的字段列表，用于把统计结果按字段名回填到 UI 控件。
_STAT_FIELDS = [item for group in _STAT_GROUPS for item in group]


class DataAnnotationPage(BaseTaskPage):
    """数据标注页面：统计目录下的图片与标注信息。

    继承 :class:`BaseTaskPage` 以复用 ``_python_env()`` 等子进程相关辅助
    方法（``_run_stats`` 通过子进程调用 ``python scripts/vh.py datasets stats``）。
    """

    def __init__(self, parent: QWidget = None, ctx=None):
        super().__init__(parent, ctx=ctx)
        self._init_controls()

    def _init_controls(self):
        """初始化页面控件。

        页面被拆分为三张白色卡片：

            1. **顶部提示卡片**：蓝色 :class:`HintCard`，介绍 X-AnyLabeling
               推荐工具，并提供下载按钮。
            2. **统计信息卡片**：整体统计网格 + 按标签明细表格 +
               "开始统计"按钮。
            3. **标签清除卡片**：勾选清除范围 + "清除标签"危险按钮。
        """
        # 目录选择行（不显示，仅保留工作目录缓存）
        self.dir_edit = QLineEdit()
        self.dir_edit.setVisible(False)

        # 基类已经默认创建了一张空白卡片用于兼容旧页面，但本页面
        # 第一块要展示的是蓝色 HintCard，并希望它能与其它白色卡片
        # 同宽地占满整个 wrapper。这里先移除默认卡片，再将 HintCard
        # 直接放入 wrapper 布局，避免出现"白色卡片包蓝色卡片"的双层框。
        self._wrapper_layout.removeWidget(self.card)
        self.card.deleteLater()
        self.card = None

        # ===== 卡片 1：蓝色提示（无外层白色卡片） =====
        hint_card = self._build_annotation_tool_hint()
        insert_index = max(0, self._wrapper_layout.count() - 1)
        self._wrapper_layout.insertWidget(insert_index, hint_card)
        # 隐藏的 dir_edit 也直接挂在 wrapper 上（不可见，不影响排版）
        insert_index = max(0, self._wrapper_layout.count() - 1)
        self._wrapper_layout.insertWidget(insert_index, self.dir_edit)

        # ===== 卡片 2：标注统计信息 =====
        _, self.content_layout = self._add_card()
        self._build_stats_card()

        # ===== 卡片 3：标签清除功能 =====
        _, self.content_layout = self._add_card()
        self._build_clear_card()

    # ------------------------------------------------------------------
    # 卡片构建
    # ------------------------------------------------------------------

    def _build_stats_card(self):
        """在当前 ``self.content_layout`` 上构建"标注统计"卡片内容。"""
        # 统计结果区（多列网格布局，避免单列过长）
        self.result_labels: dict = {}

        stats_container = QWidget()
        grid = QGridLayout(stats_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(theme.SPACING_LG if hasattr(theme, "SPACING_LG") else 24)
        grid.setVerticalSpacing(theme.SPACING_SM)

        # 自适应列数：默认每行 3 个统计项
        columns = max((len(group) for group in _STAT_GROUPS), default=3)
        for row, group in enumerate(_STAT_GROUPS):
            for col, (key, title) in enumerate(group):
                # 网格内的 StatItem 使用更紧凑的标题宽度
                stat = StatItem(
                    title=title,
                    value="0",
                    kind="success",
                    title_width=96,
                )
                self.result_labels[key] = stat
                grid.addWidget(stat, row, col)
            # 当前行不足列数时，让最后一个单元格拉伸吸收空隙
            for col in range(len(group), columns):
                grid.setColumnStretch(col, 1)
        # 让每列等宽分布
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
        # 表头文字居中显示
        for col in range(self.label_table.columnCount()):
            header_item = self.label_table.horizontalHeaderItem(col)
            if header_item is not None:
                header_item.setTextAlignment(Qt.AlignCenter)
        # 让所有列默认等宽：关闭最后一列拉伸，使用 Stretch 模式平均分配宽度
        header = self.label_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Stretch)
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

    def _build_clear_card(self):
        """在当前 ``self.content_layout`` 上构建"标签清除"卡片内容。"""
        # 两个复选框：自动标注、自动标注且修正
        clear_scope_widget = QWidget()
        clear_scope_layout = QHBoxLayout(clear_scope_widget)
        clear_scope_layout.setContentsMargins(0, 0, 0, 0)
        clear_scope_layout.setSpacing(theme.SPACING_MD)

        self.clear_auto_cb = QCheckBox("自动标注")
        self.clear_auto_corrected_cb = QCheckBox("自动标注且修正")
        clear_scope_layout.addWidget(self.clear_auto_cb)
        clear_scope_layout.addWidget(self.clear_auto_corrected_cb)

        # 清除按钮（红色危险按钮以提示破坏性操作）
        # 与复选框处于同一行，紧跟在"自动标注且修正"右侧
        clear_scope_layout.addSpacing(theme.SPACING_MD)
        self.clear_btn = DangerButton("清除标签")
        self.clear_btn.setMinimumWidth(120)
        self.clear_btn.clicked.connect(self._run_clear_annotations)
        clear_scope_layout.addWidget(self.clear_btn)
        clear_scope_layout.addStretch()

        self.content_layout.addWidget(FormRow("清除范围：", clear_scope_widget))


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
                "本工具基于 X-AnyLabeling JSON 格式进行标注统计与自动标注"
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

    def _resolve_image_dir(self, hint: str) -> str:
        """解析当前应使用的图片目录路径（含工作目录兜底）。

        如果 ``dir_edit`` 为空，会尝试用工作目录下的 ``IMAGES_FOLDER``
        作为兜底；最终路径必须是已存在的目录，否则弹窗并返回空串。
        """
        folder = self.dir_edit.text().strip()
        if not folder:
            work_dir = self._work_dir()
            folder = str(Path(work_dir) / IMAGES_FOLDER) if work_dir else ""
        path = self._require_existing_dir(folder, hint)
        return str(path) if path else ""

    def _run_stats(self):
        """执行统计并刷新界面结果。

        通过 ``subprocess`` 调用 ``python scripts/vh.py datasets stats --input <folder>``
        命令行，从其标准输出中提取以 ``===VH_STATS_BEGIN===`` /
        ``===VH_STATS_END===`` 包裹的 JSON 块并解析为统计结果，再回填到
        界面控件中。
        """
        folder = self._resolve_image_dir("要统计的图片目录")
        if not folder:
            return

        # 组装子进程命令：python scripts/vh.py datasets stats --input <folder> --label-stats
        python_exe = self._python_env()
        if not python_exe:
            # 打包态下若用户未配置 Python 环境，``_python_env()`` 会返回空串，
            # 这里给出与其他任务页面一致的友好提示，避免直接闪退。
            QMessageBox.warning(
                self,
                "未配置 Python 环境",
                "当前为打包发布版本，必须在导航栏下方手动选择带有所需依赖"
                "的 Python 可执行文件后再运行标注统计任务。",
            )
            return
        root = str(app_root())
        cmd = [
            python_exe,
            "scripts/vh.py",
            "datasets",
            "stats",
            "--input",
            folder,
            "--label-stats",
        ]

        # 注入 PYTHONPATH，确保 ``scripts/vh.py`` 在打包/任意启动目录下都能定位到
        # ``scripts`` 包。
        env = _build_script_env(root)

        # 同步调用：标注统计是轻量操作，几百毫秒级别，无需异步日志窗口。
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except FileNotFoundError as exc:
                QMessageBox.critical(
                    self,
                    "统计失败",
                    f"无法启动统计子进程：{exc}",
                )
                return
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(
                    self,
                    "统计失败",
                    f"统计子进程执行异常：{exc}",
                )
                return
        finally:
            QApplication.restoreOverrideCursor()

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            detail = stderr.strip() or stdout.strip() or "（无输出）"
            QMessageBox.critical(
                self,
                "统计失败",
                f"子进程返回非零状态码 {completed.returncode}：\n{detail}",
            )
            return

        # 解析机器块
        try:
            from scripts.datasets.stats import parse_machine_block

            payload = parse_machine_block(stdout)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "统计失败",
                f"解析统计结果失败：{exc}\n\n原始输出：\n{stdout[-2000:]}",
            )
            return

        stats = payload.get("stats") or {}
        label_stats = payload.get("label_stats") or []
        if not isinstance(stats, dict):
            stats = {}
        if not isinstance(label_stats, list):
            label_stats = []

        self._update_result_labels(stats)
        self._update_label_table(label_stats)

    def _run_clear_annotations(self):
        """按勾选的标注类型清除目录下的 X-AnyLabeling JSON 标注文件。"""
        folder = self._resolve_image_dir("要清理的图片目录")
        if not folder:
            return

        include_auto = self.clear_auto_cb.isChecked()
        include_auto_corrected = self.clear_auto_corrected_cb.isChecked()
        if not (include_auto or include_auto_corrected):
            QMessageBox.warning(
                self, "参数缺失", "请至少勾选一种待清除的标注类型"
            )
            return

        # 选定的类型描述，用于二次确认
        scope_desc_parts = []
        if include_auto:
            scope_desc_parts.append("自动标注")
        if include_auto_corrected:
            scope_desc_parts.append("自动标注且修正")
        scope_desc = "、".join(scope_desc_parts)

        confirm = QMessageBox.question(
            self,
            "确认清除标签",
            (
                f"将从目录\n  {folder}\n"
                f"中删除类型为【{scope_desc}】的 X-AnyLabeling JSON 标注文件，"
                "该操作不可恢复，是否继续？"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            from scripts.datasets.clear import clear_annotations

            result = clear_annotations(
                folder=folder,
                include_auto=include_auto,
                include_auto_corrected=include_auto_corrected,
                include_manual=False,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "清除失败", f"清除标签过程中发生错误：{exc}")
            return

        by_type = result.get("by_type", {}) or {}
        deleted = result.get("deleted", 0)
        failed = result.get("failed", []) or []

        msg_lines = [
            f"扫描到匹配图片的标注：{result.get('scanned', 0)}",
            f"实际删除：{deleted}",
            f"  · 自动标注：{by_type.get('auto', 0)}",
            f"  · 自动标注且修正：{by_type.get('auto_corrected', 0)}",
        ]
        if failed:
            msg_lines.append(f"删除失败：{len(failed)} 项（详见控制台日志）")

        QMessageBox.information(self, "清除完成", "\n".join(msg_lines))

        # 清除完成后自动刷新统计，方便用户查看新的标注情况
        self._run_stats()

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

            for col, value in enumerate(
                (label_name, detection_count, obb_count, polygon_count)
            ):
                cell = QTableWidgetItem(str(value))
                cell.setTextAlignment(Qt.AlignCenter)
                self.label_table.setItem(row, col, cell)
