#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper GUI 主程序。

启动后首先展示引导页（:class:`gui.welcome_page.WelcomePage`），用于
选择/管理历史工作目录；选定目录后再切换到原有的主界面（菜单栏 +
各功能子页面）。

启动方式：
    /home/zh/.anaconda3/envs/vision/bin/python -m gui.app
"""

import os
import platform
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QT_VERSION_STR, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui import settings, theme
from gui.auto_annotate_page import AutoAnnotatePage
from gui.context import AppContext
from gui.data_annotation_page import DataAnnotationPage
from gui.model_training_page import ModelTrainingPage
from gui.video_frame_page import VideoFramePage
from gui.welcome_page import WelcomePage
from gui.widgets import (
    HSeparator,
    PrimaryButton,
    SecondaryButton,
    SectionTitle,
)


# visionHelper 版本号
APP_VERSION = "1.0.1"

# visionHelper 项目号（项目编号 / 项目编码）
APP_PROJECT_CODE = "VH-2026-001"

# 作者信息
APP_AUTHOR = "IceCoke"

# 项目仓库地址
PROJECT_HOMEPAGE = "https://github.com/"


class AboutDialog(QDialog):
    """visionHelper 关于对话框。

    展示应用名称、版本、简介、功能列表、运行环境（Python / PyQt 版本、
    操作系统、Python 解释器路径等），并提供"访问主页 / 复制环境信息"
    等便捷操作。所有视觉风格依赖 :mod:`gui.theme`，本类不再写内联样式。
    """

    def __init__(self, parent: QWidget = None, python_env: str = ""):
        super().__init__(parent)
        self.setWindowTitle("关于 visionHelper")
        self.setMinimumSize(560, 540)
        self._python_env = python_env or sys.executable

        self._init_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 顶部 Logo + 标题区
        outer.addWidget(self._build_header())
        outer.addWidget(HSeparator())

        # 中部滚动区域：简介、功能、环境
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 16, 24, 16)
        content_layout.setSpacing(10)

        content_layout.addWidget(SectionTitle("项目简介"))
        content_layout.addWidget(self._build_body(
            "visionHelper 是一个轻量级的计算机视觉辅助工具集，"
            "围绕 YOLO 数据生产与训练流程，把视频抽帧、图片去重、"
            "数据标注统计、自动标注、YOLO 数据集导出与模型训练"
            "等常用步骤整合到同一个图形界面中，帮助你更高效地"
            "完成从原始素材到可用模型的全流程。"
        ))

        content_layout.addWidget(SectionTitle("核心功能"))
        content_layout.addWidget(self._build_body(
            "• 视频抽帧：按帧间隔从视频中抽取图片，支持多种格式与质量参数。\n"
            "• 图片去重：基于 ViT 特征对图片进行相似度去重。\n"
            "• 标注统计：统计 X-AnyLabeling JSON 标注中的图片数 / 实例数。\n"
            "• 自动标注：使用训练好的 YOLO 模型自动生成 X-AnyLabeling 标注。\n"
            "• 数据集导出：将已标注图片导出为标准 YOLO 数据集。\n"
            "• 模型训练：基于 Ultralytics YOLO 进行训练，"
            "支持 detect / obb / segment / classify 四种任务。"
        ))

        content_layout.addWidget(SectionTitle("项目信息"))
        content_layout.addWidget(self._build_body(
            f"项目号：{APP_PROJECT_CODE}\n"
            f"版本号：{APP_VERSION}\n"
            f"作者　：{APP_AUTHOR}\n"
            f"感谢使用 visionHelper，欢迎反馈问题与建议。"
        ))

        content_layout.addWidget(SectionTitle("运行环境"))
        content_layout.addWidget(self._build_env_info())

        content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        outer.addWidget(HSeparator())
        outer.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        icon_label = QLabel()
        icon_label.setFixedSize(64, 64)
        icon_label.setAlignment(Qt.AlignCenter)
        # 优先使用应用窗口图标，没有则用文字占位
        pixmap: QPixmap = QPixmap()
        if self.parent() is not None:
            icon = self.parent().windowIcon()
            if not icon.isNull():
                pixmap = icon.pixmap(64, 64)
        if pixmap.isNull():
            icon_label.setText("🛠️")
            icon_label.setStyleSheet(
                f"font-size: 36pt;"
                f"background-color: {theme.COLOR_PRIMARY};"
                f"color: {theme.COLOR_TEXT_INVERSE};"
                f"border-radius: {theme.RADIUS_LG}px;"
            )
        else:
            icon_label.setPixmap(pixmap)
        layout.addWidget(icon_label)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        title = QLabel("visionHelper")
        title_font = theme.app_font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        text_box.addWidget(title)

        version = QLabel(f"版本 {APP_VERSION}    ·    作者 {APP_AUTHOR}")
        version.setProperty("variant", "muted")
        theme.refresh_widget_style(version)
        text_box.addWidget(version)

        project_code = QLabel(f"项目号 {APP_PROJECT_CODE}")
        project_code.setProperty("variant", "muted")
        theme.refresh_widget_style(project_code)
        text_box.addWidget(project_code)

        tagline = QLabel("轻量级视觉辅助工具集")
        tagline.setProperty("variant", "muted")
        theme.refresh_widget_style(tagline)
        text_box.addWidget(tagline)
        text_box.addStretch(1)

        layout.addLayout(text_box, 1)
        return header

    def _build_body(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    def _build_env_info(self) -> QLabel:
        info_lines = [
            f"项目号  ：{APP_PROJECT_CODE}",
            f"版本    ：{APP_VERSION}",
            f"Python  ：{sys.version.split()[0]}",
            f"PyQt5   ：{QT_VERSION_STR}",
            f"操作系统：{platform.system()} {platform.release()}",
            f"架构    ：{platform.machine()}",
            f"解释器  ：{self._python_env}",
            f"工作目录：{os.getcwd()}",
        ]
        label = QLabel("\n".join(info_lines))
        # 使用等宽字体显示，但颜色仍走全局主题
        label.setStyleSheet(f'font-family: {theme.FONT_FAMILY_MONO}; font-size: 9pt;')
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(8)

        copyright_lbl = QLabel(f"© 2026 visionHelper · {APP_AUTHOR}")
        copyright_lbl.setProperty("variant", "muted")
        theme.refresh_widget_style(copyright_lbl)
        layout.addWidget(copyright_lbl)

        layout.addStretch(1)

        copy_btn = SecondaryButton("复制环境信息")
        copy_btn.clicked.connect(self._copy_env_info)
        layout.addWidget(copy_btn)

        homepage_btn = SecondaryButton("访问主页")
        homepage_btn.clicked.connect(self._open_homepage)
        layout.addWidget(homepage_btn)

        close_btn = PrimaryButton("关闭")
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        return footer

    # ------------------------------------------------------------------
    # 行为
    # ------------------------------------------------------------------

    def _copy_env_info(self):
        """把环境信息复制到剪贴板，便于反馈问题时附带。"""
        info = (
            f"visionHelper {APP_VERSION} ({APP_PROJECT_CODE})\n"
            f"Python : {sys.version.split()[0]}\n"
            f"PyQt5  : {QT_VERSION_STR}\n"
            f"OS     : {platform.system()} {platform.release()} ({platform.machine()})\n"
            f"Python : {self._python_env}\n"
            f"CWD    : {os.getcwd()}\n"
        )
        QApplication.clipboard().setText(info)
        QMessageBox.information(self, "已复制", "环境信息已复制到剪贴板。")

    def _open_homepage(self):
        """打开项目主页。"""
        QDesktopServices.openUrl(QUrl(PROJECT_HOMEPAGE))


class _WorkDirNotifier(QStackedWidget):
    """
    包装 QStackedWidget，在页面切换时通知子页面工作目录已就绪。

    数据标注页面需要在初始化后获取主窗口的工作目录；
    通过 page_shown 信号/回调，让页面在第一次显示时填充默认值。
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._shown_pages: set = set()

    def setCurrentIndex(self, index: int):
        super().setCurrentIndex(index)
        widget = self.widget(index)
        if widget is not None and widget not in self._shown_pages:
            self._shown_pages.add(widget)
            if hasattr(widget, "on_page_shown"):
                widget.on_page_shown()


def _setup_qt_platform_plugin():
    """设置 PyQt5 平台插件路径。

    打包态下 PyInstaller 自带平台插件，无需额外处理；开发态下尝试
    根据 ``PyQt5`` 安装位置推断 ``platforms`` 目录，避免在某些 Linux
    发行版上出现 ``Could not load the Qt platform plugin "xcb"`` 错误。
    """
    if getattr(sys, "frozen", False):
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "")
        return
    try:
        import PyQt5
        plugin_dir = (
            Path(PyQt5.__file__).resolve().parent
            / "Qt5"
            / "plugins"
            / "platforms"
        )
        if plugin_dir.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_dir)
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "")
    except Exception:
        # 推断失败时保持现状，让 PyQt5 自行处理
        pass


class MainWindow(QMainWindow):
    """visionHelper 主窗口。

    结构：

        QStackedWidget
          ├── WelcomePage     ：启动后首先展示的引导页
          └── 主工作界面       ：菜单栏 + 顶部工作目录条 + 页面堆栈

    主窗口对外暴露 ``work_dir`` / ``python_env`` 属性，供子页面通过
    :meth:`gui.base_pages.BasePage._work_dir` 与 ``_python_env`` 读取。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("visionHelper")
        self.setMinimumSize(800, 540)

        # 全局上下文：work_dir / python_env 等跨页面状态
        self.ctx = AppContext(self)

        # 根容器：欢迎页 / 主工作界面 二选一
        self._root_stack = QStackedWidget()
        self.setCentralWidget(self._root_stack)

        self._init_menu()

        self._welcome_page = self._build_welcome_page()
        self._main_widget = self._build_main_widget()
        self._root_stack.addWidget(self._welcome_page)
        self._root_stack.addWidget(self._main_widget)

        self._load_settings()
        self._show_welcome()

    # ------------------------------------------------------------------
    # 子页面访问的属性
    # ------------------------------------------------------------------

    @property
    def work_dir(self) -> str:
        """当前工作目录（绝对路径），供子页面通过父链查询。"""
        return self.work_dir_edit.text().strip() if hasattr(self, "work_dir_edit") else ""

    @property
    def python_env(self) -> str:
        """当前 Python 解释器路径，供子页面通过父链查询。"""
        return self.python_env_edit.text().strip() if hasattr(self, "python_env_edit") else ""

    # ------------------------------------------------------------------
    # 菜单
    # ------------------------------------------------------------------

    def _init_menu(self):
        menubar = self.menuBar()
        # 直接以 action 形式列出，简化菜单结构。
        # 注："视频抽帧"页面内部已合并"图片去重"卡片，故不再单列菜单项。
        menubar.addAction("视频抽帧", lambda: self._switch_page(0))
        menubar.addAction("标注统计", lambda: self._switch_page(1))
        menubar.addAction("模型训练", lambda: self._switch_page(2))
        menubar.addAction("自动标注", lambda: self._switch_page(3))
        menubar.addAction("关闭项目", self._show_welcome)
        menubar.addAction("关于", self._show_about)


    # ------------------------------------------------------------------
    # 欢迎页 / 主界面切换
    # ------------------------------------------------------------------

    def _build_welcome_page(self) -> WelcomePage:
        page = WelcomePage()
        page.work_dir_selected.connect(self._on_work_dir_selected)
        return page

    def _show_welcome(self):
        self._welcome_page.set_recent_dirs(settings.load_recent_dirs())
        self.menuBar().setVisible(False)
        self._root_stack.setCurrentWidget(self._welcome_page)

    def _show_main(self):
        self.menuBar().setVisible(True)
        self._root_stack.setCurrentWidget(self._main_widget)

    def _on_work_dir_selected(self, path: str):
        if not path:
            return
        self.work_dir_edit.setText(path)
        self._save_work_dir()
        # 把欢迎页的最新顺序写回设置
        settings.save_recent_dirs(self._welcome_page.recent_dirs())
        self._switch_page(0)
        self._show_main()

    # ------------------------------------------------------------------
    # 主工作界面构建
    # ------------------------------------------------------------------

    def _build_main_widget(self) -> QWidget:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._init_top_bars(layout)

        self.stacked = _WorkDirNotifier()
        # 把全局 AppContext 注入到每个子页面：页面优先通过 self.ctx 读取
        # work_dir / python_env，避免遍历 parent 链。各 Page 构造器需要
        # 兼容 ``ctx`` 关键字参数（已在 BasePage 中提供默认值）。
        # "图片去重"已作为第二张卡片合并进 VideoFramePage，不再单独注册页面。
        # 这里保持 dict 顺序与菜单索引一致：
        #   index 0 → 视频抽帧 + 图片去重
        #   index 1 → 标注统计
        #   index 2 → 模型训练
        #   index 3 → 自动标注
        self.pages = {
            "video": VideoFramePage(ctx=self.ctx),
            "annotation": DataAnnotationPage(ctx=self.ctx),
            "training": ModelTrainingPage(ctx=self.ctx),
            "auto_annotate": AutoAnnotatePage(ctx=self.ctx),
        }

        for page in self.pages.values():
            self.stacked.addWidget(page)
        layout.addWidget(self.stacked, 1)
        return central

    def _init_top_bars(self, parent_layout: QVBoxLayout):
        """构建顶部"工作目录 + Python 环境"信息条。

        使用 QFrame[variant="topbar"] 应用统一主题样式（浅色底 + 下边框），
        而不再写内联样式。
        """
        bar = QFrame()
        bar.setProperty("variant", "topbar")
        theme.refresh_widget_style(bar)

        v = QVBoxLayout(bar)
        v.setContentsMargins(theme.SPACING_MD, theme.SPACING_SM, theme.SPACING_MD, theme.SPACING_SM)
        v.setSpacing(theme.SPACING_SM)

        bold = theme.app_font()
        bold.setBold(True)
        label_width = 88

        # 工作目录行
        wd_row = QHBoxLayout()
        wd_row.setSpacing(theme.SPACING_SM)
        wd_lbl = QLabel("工作目录：")
        wd_lbl.setFixedWidth(label_width)
        wd_lbl.setFont(bold)
        wd_row.addWidget(wd_lbl)
        self.work_dir_edit = QLineEdit()
        self.work_dir_edit.setPlaceholderText("选择工作环境目录，后续选路径将默认从此处开始")
        self.work_dir_edit.editingFinished.connect(self._save_work_dir)
        self.work_dir_edit.textChanged.connect(self.ctx.set_work_dir)
        wd_row.addWidget(self.work_dir_edit, 1)
        wd_btn = SecondaryButton("浏览")
        wd_btn.clicked.connect(self._browse_work_dir)
        wd_row.addWidget(wd_btn)
        v.addLayout(wd_row)

        # Python 环境行
        py_row = QHBoxLayout()
        py_row.setSpacing(theme.SPACING_SM)
        py_lbl = QLabel("Python 环境：")
        py_lbl.setFixedWidth(label_width)
        py_lbl.setFont(bold)
        py_row.addWidget(py_lbl)
        self.python_env_edit = QLineEdit()
        self.python_env_edit.setPlaceholderText(
            "选择 Python 可执行文件（推荐 /home/zh/.anaconda3/envs/vision/bin/python），"
            "脚本将通过该环境运行"
        )
        self.python_env_edit.editingFinished.connect(self._save_python_env)
        self.python_env_edit.textChanged.connect(self.ctx.set_python_env)
        py_row.addWidget(self.python_env_edit, 1)
        py_btn = SecondaryButton("浏览")
        py_btn.clicked.connect(self._browse_python_env)
        py_row.addWidget(py_btn)
        v.addLayout(py_row)

        parent_layout.addWidget(bar)

    # ------------------------------------------------------------------
    # 顶部按钮交互
    # ------------------------------------------------------------------

    def _browse_work_dir(self):
        start = self.work_dir_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "选择工作目录", start)
        if path:
            self.work_dir_edit.setText(path)
            self._save_work_dir()
            # 同步到历史目录列表（去重 + 截断由 settings 模块统一处理）
            settings.promote_recent_dir(path)

    def _browse_python_env(self):
        start = self.python_env_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "选择 Python 可执行文件", start)
        if path:
            self.python_env_edit.setText(path)
            self._save_python_env()

    # ------------------------------------------------------------------
    # 页面切换
    # ------------------------------------------------------------------

    def _switch_page(self, index: int):
        if hasattr(self, "stacked"):
            self.stacked.setCurrentIndex(index)
        self._show_main()

    # ------------------------------------------------------------------
    # 关于对话框
    # ------------------------------------------------------------------

    def _show_about(self):
        dlg = AboutDialog(self, python_env=self.python_env)
        dlg.exec_()

    # ------------------------------------------------------------------
    # 设置持久化（统一通过 gui.settings 封装的 QSettings 句柄读写）
    # ------------------------------------------------------------------

    def _load_settings(self):
        self.work_dir_edit.setText(settings.load_work_dir())
        self.python_env_edit.setText(settings.load_python_env())

    def _save_work_dir(self):
        settings.save_work_dir(self.work_dir_edit.text())

    def _save_python_env(self):
        settings.save_python_env(self.python_env_edit.text())


def main():
    """visionHelper GUI 入口函数。"""
    _setup_qt_platform_plugin()

    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply_theme(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
