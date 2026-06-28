#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper GUI 主程序。

启动后首先展示引导页（:class:`gui.pages.welcome.WelcomePage`），用于
选择/管理历史工作目录；选定目录后再切换到原有的主界面（菜单栏 +
各功能子页面）。

启动方式：
    python -m gui.app
"""

import os
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui import settings, theme
from gui.pages.data_annotation import DataAnnotationPage
from gui.pages.model_training import ModelTrainingPage
from gui.pages.predict import PredictPage
from gui.pages.video_frame import VideoFramePage
from gui.pages.welcome import WelcomePage
from gui.pages.about import AboutPage
from gui.context import AppContext
from gui.components.widgets import SecondaryButton


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
            # 使用基类定义的 on_page_shown() 方法，不再依赖 hasattr 鸭子类型
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
    except (ImportError, OSError, AttributeError) as exc:
        # 推断失败时保持现状，让 PyQt5 自行处理
        import logging
        logging.debug(f"Failed to detect PyQt5 plugin path: {exc}")
        pass


class MainWindow(QMainWindow):
    """visionHelper 主窗口。

    结构：

        QStackedWidget
          ├── WelcomePage     ：启动后首先展示的引导页
          └── 主工作界面       ：菜单栏 + 顶部工作目录条 + 页面堆栈
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
    # 菜单
    # ------------------------------------------------------------------

    def _init_menu(self):
        menubar = self.menuBar()
        menubar.addAction("图片处理", lambda: self._switch_page_by_name("video"))
        menubar.addAction("标注信息", lambda: self._switch_page_by_name("annotation"))
        menubar.addAction("模型训练", lambda: self._switch_page_by_name("training"))
        menubar.addAction("模型预测", lambda: self._switch_page_by_name("predict"))
        menubar.addAction("关闭项目", self._show_welcome)
        menubar.addAction("关于", lambda: self._switch_page_by_name("about"))


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
        self.pages = {
            "video": VideoFramePage(ctx=self.ctx),
            "annotation": DataAnnotationPage(ctx=self.ctx),
            "training": ModelTrainingPage(ctx=self.ctx),
            "predict": PredictPage(ctx=self.ctx),
            "about": AboutPage(),
        }

        for page in self.pages.values():
            self.stacked.addWidget(page)
        layout.addWidget(self.stacked, 1)
        return central

    def _init_top_bars(self, parent_layout: QVBoxLayout):
        """构建顶部"工作目录 + Python 环境"信息条。"""
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
            "选择 Python 可执行文件（例如 /path/to/venv/bin/python），"
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

    def _switch_page_by_name(self, name: str):
        """根据页面名称切换到对应页面。"""
        if hasattr(self, "pages") and name in self.pages:
            index = list(self.pages.keys()).index(name)
            self._switch_page(index)

    # ------------------------------------------------------------------
    # 设置持久化
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
