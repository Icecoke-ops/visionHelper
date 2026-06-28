#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visionHelper 关于页面。

展示应用名称、版本、简介、功能列表、运行环境（Python / PyQt 版本、
操作系统、Python 解释器路径等），并提供"访问主页 / 复制环境信息"
等便捷操作。所有视觉风格依赖 :mod:`gui.theme`，本类不再写内联样式。
"""

import os
import platform
import sys

from PyQt5.QtCore import Qt, QT_VERSION_STR, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.config import APP_VERSION, APP_PROJECT_CODE, APP_AUTHOR, PROJECT_HOMEPAGE
from gui.components.widgets import HSeparator, PrimaryButton, SecondaryButton, SectionTitle


class AboutPage(QWidget):
    """visionHelper 关于页面：展示项目信息与运行环境。"""

    def __init__(self, parent: QWidget = None, python_env: str = ""):
        super().__init__(parent)
        self._python_env = python_env or sys.executable
        self._init_ui()

    def on_page_shown(self) -> None:
        pass

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
            "围绕 YOLO 数据生产与训练流程，把图片处理、图片去重、"
             "数据标注信息、自动标注、YOLO 数据集导出与模型训练"
            "等常用步骤整合到同一个图形界面中，帮助你更高效地"
            "完成从原始素材到可用模型的全流程。"
        ))

        content_layout.addWidget(SectionTitle("核心功能"))
        content_layout.addWidget(self._build_body(
            "• 图片处理：按帧间隔从视频中抽取图片，支持多种格式与质量参数。\n"
            "• 图片去重：基于 ViT 特征对图片进行相似度去重。\n"
            "• 标注信息：统计 X-AnyLabeling JSON 标注中的图片数 / 实例数。\n"
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
            icon_label.setText("V")  # 使用文字代替emoji，提高兼容性
            icon_label.setStyleSheet(
                f"font-size: 36pt;"
                f"font-weight: bold;"
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
