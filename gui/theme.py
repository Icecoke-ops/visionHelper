#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 主题模块 —— 集中管理 visionHelper 桌面端的视觉风格。

定义全局统一的「设计令牌」（颜色 / 圆角 / 间距 / 控件高度等）以及一份覆盖所有
常用 Qt 控件的全局样式表 :data:`GLOBAL_STYLESHEET`。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication



# ---------------------------------------------------------------------------
# 颜色令牌（语义化）
# ---------------------------------------------------------------------------

# 主操作（蓝）
COLOR_PRIMARY = "#2563eb"
COLOR_PRIMARY_HOVER = "#1d4ed8"
COLOR_PRIMARY_PRESSED = "#1e40af"
COLOR_PRIMARY_DISABLED = "#93c5fd"

# 成功 / 执行（绿）
COLOR_SUCCESS = "#16a34a"
COLOR_SUCCESS_HOVER = "#15803d"
COLOR_SUCCESS_PRESSED = "#166534"
COLOR_SUCCESS_DISABLED = "#86efac"

# 警告（橙）
COLOR_WARNING = "#d97706"
COLOR_WARNING_HOVER = "#b45309"

# 错误 / 危险（红）
COLOR_DANGER = "#dc2626"
COLOR_DANGER_HOVER = "#b91c1c"
COLOR_DANGER_PRESSED = "#991b1b"

# 信息 / 提示（青）
COLOR_INFO = "#0ea5e9"
COLOR_INFO_HOVER = "#0284c7"

# 文本
COLOR_TEXT_PRIMARY = "#1f2937"
COLOR_TEXT_SECONDARY = "#4b5563"
COLOR_TEXT_MUTED = "#9ca3af"
COLOR_TEXT_INVERSE = "#ffffff"
COLOR_TEXT_LINK = COLOR_PRIMARY

# 背景 / 边框
COLOR_BG_APP = "#f5f7fa"
COLOR_BG_CARD = "#ffffff"
COLOR_BG_HOVER = "#f3f4f6"
COLOR_BG_HINT = "#eff6ff"
COLOR_BG_HINT_BORDER = "#bfdbfe"
COLOR_BG_DARK = "#1f2937"
COLOR_BG_DARK_TEXT = "#e5e7eb"

COLOR_BORDER = "#e5e7eb"
COLOR_BORDER_STRONG = "#d1d5db"
COLOR_BORDER_FOCUS = "#93c5fd"

COLOR_SELECTION_BG = "#dbeafe"


# ---------------------------------------------------------------------------
# 圆角 / 间距 / 尺寸
# ---------------------------------------------------------------------------

RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 8

SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 12
SPACING_LG = 20
SPACING_XL = 28

INPUT_HEIGHT = 32
BUTTON_HEIGHT = 32
SMALL_BUTTON_HEIGHT = 28

FORM_LABEL_WIDTH = 96

PAGE_MARGIN = 24
CARD_PADDING = 20


# ---------------------------------------------------------------------------
# 字体
# ---------------------------------------------------------------------------

FONT_FAMILY = "Microsoft YaHei"
FONT_FAMILY_MONO = "Consolas, Menlo, DejaVu Sans Mono, monospace"

FONT_SIZE_BODY = 10
FONT_SIZE_SMALL = 9
FONT_SIZE_TITLE = 12
FONT_SIZE_LARGE_TITLE = 18


def app_font() -> QFont:
    """返回应用统一字体（标准正文字号）。"""
    return QFont(FONT_FAMILY, FONT_SIZE_BODY)


# ---------------------------------------------------------------------------
# 箭头图标（用于 QSpinBox / QComboBox 等控件的 up/down 箭头）
#
# Qt 5 的 QSS 在 sub-control 中绘制 CSS 三角形（border-trick）并不稳定，
# 因此这里把箭头以 SVG 的形式写入临时文件，再通过 ``image: url(...)`` 引用。
# 使用内容哈希作为文件名，避免重复写入；启动时清理旧缓存文件。
# ---------------------------------------------------------------------------

_ARROW_SVG_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
    '<polygon points="{points}" fill="{color}"/>'
    "</svg>"
)

# 箭头三角形的顶点坐标（10x10 画布）
_ARROW_POINTS_DOWN = "1,3 9,3 5,7"
_ARROW_POINTS_UP = "1,7 9,7 5,3"


def _arrow_cache_dir() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "visionhelper_theme"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir



def _write_arrow_svg(name: str, points: str, color: str) -> str:
    """写入 SVG 箭头到临时文件并返回 Qt QSS 可用的 URL 路径（正斜杠）。

    文件名基于内容哈希，相同内容复用同一文件，避免临时目录无限增长。
    """
    import hashlib
    content = f"{name}:{points}:{color}"
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    path = _arrow_cache_dir() / f"{name}_{content_hash}.svg"
    svg = _ARROW_SVG_TEMPLATE.format(points=points, color=color)
    # 仅当内容变化时才写盘
    try:
        if not path.exists() or path.read_text(encoding="utf-8") != svg:
            path.write_text(svg, encoding="utf-8")
    except OSError:
        path.write_text(svg, encoding="utf-8")
    return path.as_posix()


def _cleanup_old_arrow_cache() -> None:
    """清理旧的箭头缓存文件（保留最近 20 个），防止临时目录无限增长。"""
    try:
        cache_dir = _arrow_cache_dir()
        files = sorted(cache_dir.glob("*.svg"), key=lambda p: p.stat().st_mtime)
        for f in files[:-20]:
            try:
                f.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _build_arrow_assets() -> dict:
    """生成默认/禁用 状态下的上下箭头 SVG，返回路径字典。"""
    return {
        "arrow_down": _write_arrow_svg("arrow_down", _ARROW_POINTS_DOWN, COLOR_TEXT_SECONDARY),
        "arrow_up": _write_arrow_svg("arrow_up", _ARROW_POINTS_UP, COLOR_TEXT_SECONDARY),
        "arrow_down_disabled": _write_arrow_svg(
            "arrow_down_disabled", _ARROW_POINTS_DOWN, COLOR_TEXT_MUTED
        ),
        "arrow_up_disabled": _write_arrow_svg(
            "arrow_up_disabled", _ARROW_POINTS_UP, COLOR_TEXT_MUTED
        ),
    }


# 延迟初始化缓存：避免在模块导入时触发文件 I/O
_ARROW_CACHE: dict = {}


# ---------------------------------------------------------------------------
# 全局样式表（按片段拼接，避免单一过长字符串）
# ---------------------------------------------------------------------------

_QSS_BASE = """
QWidget {
    color: %(text)s;
    font-family: "%(font)s";
    font-size: %(body_pt)dpt;
}
QMainWindow, QDialog {
    background-color: %(bg_app)s;
}
QToolTip {
    background-color: %(bg_dark)s;
    color: %(inverse)s;
    border: 1px solid %(bg_dark)s;
    padding: 4px 8px;
    border-radius: %(r_sm)dpx;
}
"""

_QSS_MENU = """
QMenuBar {
    background-color: %(card)s;
    border-bottom: 1px solid %(border)s;
    padding: 4px 8px;
}
QMenuBar::item {
    padding: 8px 16px;
    background-color: transparent;
    color: %(text_sec)s;
    border-radius: %(r_sm)dpx;
    margin: 2px 1px;
}
QMenuBar::item:selected { background-color: %(hover)s; color: %(primary)s; }
QMenuBar::item:pressed  { background-color: %(sel)s;  color: %(primary_pressed)s; }
QMenu {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    padding: 6px 0;
    border-radius: %(r_md)dpx;
}
QMenu::item {
    padding: 8px 24px 8px 16px;
    color: %(text)s;
}
QMenu::item:selected { background-color: %(hover)s; color: %(primary)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 8px; }
"""

_QSS_INPUT = """
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: %(card)s;
    border: 1px solid %(border_strong)s;
    border-radius: %(r_md)dpx;
    padding: 6px 10px;
    min-height: 22px;
    selection-background-color: %(sel)s;
    selection-color: %(text)s;
    color: %(text)s;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid %(primary)s;
    padding: 5px 9px;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background-color: %(hover)s; color: %(muted)s;
}
QLineEdit[readOnly="true"] { background-color: %(hover)s; color: %(text_sec)s; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: url(%(arrow_down)s);
    width: 10px;
    height: 10px;
    margin-right: 8px;
}
QComboBox::down-arrow:on {
    image: url(%(arrow_up)s);
}
QComboBox QAbstractItemView {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: %(r_md)dpx;
    selection-background-color: %(sel)s;
    selection-color: %(text)s;
    outline: 0;
    padding: 4px;
}
QSpinBox, QDoubleSpinBox {
    padding-right: 26px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid %(border)s;
    border-bottom: 1px solid %(border)s;
    border-top-right-radius: %(r_md)dpx;
    background-color: %(card)s;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid %(border)s;
    border-bottom-right-radius: %(r_md)dpx;
    background-color: %(card)s;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: %(hover)s;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: %(sel)s;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url(%(arrow_up)s);
    width: 10px;
    height: 10px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url(%(arrow_down)s);
    width: 10px;
    height: 10px;
}
QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off {
    image: url(%(arrow_up_disabled)s);
}
QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off {
    image: url(%(arrow_down_disabled)s);
}
"""



_QSS_BUTTON_BASE = """
QPushButton {
    background-color: %(card)s;
    color: %(text)s;
    border: 1px solid %(border_strong)s;
    border-radius: %(r_md)dpx;
    padding: 6px 16px;
    min-height: 24px;
    font-weight: 500;
}
QPushButton:hover    { background-color: %(hover)s; border-color: %(primary)s; color: %(primary)s; }
QPushButton:pressed  { background-color: %(sel)s;  border-color: %(primary_pressed)s; color: %(primary_pressed)s; }
QPushButton:disabled { background-color: %(hover)s; color: %(muted)s; border-color: %(border)s; }
"""

_QSS_BUTTON_PRIMARY = """
QPushButton[variant="primary"] {
    background-color: %(primary)s;
    color: %(inverse)s;
    border: 1px solid %(primary)s;
    border-radius: %(r_md)dpx;
    font-weight: bold;
    padding: 6px 20px;
}
QPushButton[variant="primary"]:hover    { background-color: %(primary_hover)s;   border-color: %(primary_hover)s;   color: %(inverse)s; }
QPushButton[variant="primary"]:pressed  { background-color: %(primary_pressed)s; border-color: %(primary_pressed)s; color: %(inverse)s; }
QPushButton[variant="primary"]:disabled { background-color: %(primary_disabled)s; border-color: %(primary_disabled)s; color: %(inverse)s; }
"""

_QSS_BUTTON_SUCCESS = """
QPushButton[variant="success"] {
    background-color: %(success)s;
    color: %(inverse)s;
    border: 1px solid %(success)s;
    border-radius: %(r_md)dpx;
    font-weight: bold;
    padding: 6px 20px;
}
QPushButton[variant="success"]:hover    { background-color: %(success_hover)s;   border-color: %(success_hover)s;   color: %(inverse)s; }
QPushButton[variant="success"]:pressed  { background-color: %(success_pressed)s; border-color: %(success_pressed)s; color: %(inverse)s; }
QPushButton[variant="success"]:disabled { background-color: %(success_disabled)s; border-color: %(success_disabled)s; color: %(inverse)s; }
"""

_QSS_BUTTON_DANGER = """
QPushButton[variant="danger"] {
    background-color: %(card)s;
    color: %(danger)s;
    border: 1px solid %(border_strong)s;
}
QPushButton[variant="danger"]:hover    { background-color: %(danger)s;         color: %(inverse)s; border-color: %(danger)s; }
QPushButton[variant="danger"]:pressed  { background-color: %(danger_pressed)s; color: %(inverse)s; border-color: %(danger_pressed)s; }
QPushButton[variant="link"] {
    background-color: transparent;
    border: none;
    color: %(text)s;
    text-align: left;
    padding: 4px 6px;
}
QPushButton[variant="link"]:hover   { color: %(primary)s; background-color: transparent; border: none; }
QPushButton[variant="link"]:pressed { color: %(primary_pressed)s; background-color: transparent; border: none; }
QPushButton[variant="icon"] {
    background-color: transparent;
    border: none;
    color: %(muted)s;
    padding: 0 6px;
    min-height: 0;
}
QPushButton[variant="icon"]:hover   { color: %(danger)s; background-color: transparent; border: none; }
QPushButton[variant="icon"]:pressed { color: %(danger_pressed)s; background-color: transparent; border: none; }
"""

_QSS_CHECKBOX_RADIO = """
QCheckBox, QRadioButton { color: %(text)s; spacing: 6px; padding: 2px 0; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
    border: 1px solid %(border_strong)s;
    background-color: %(card)s;
    border-radius: 3px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    border: 1px solid %(primary)s;
    background-color: %(primary)s;
    border-radius: 3px;
}
QRadioButton::indicator { border-radius: 8px; }
QRadioButton::indicator:checked { border-radius: 8px; }
"""

_QSS_TABLE = """
QTableWidget, QTableView {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: %(r_md)dpx;
    gridline-color: %(border)s;
    selection-background-color: %(sel)s;
    selection-color: %(text)s;
}
QTableWidget::item, QTableView::item { padding: 6px 8px; }
QTableWidget::item:selected, QTableView::item:selected {
    background-color: %(sel)s; color: %(text)s;
}
QHeaderView::section {
    background-color: %(hover)s;
    color: %(text_sec)s;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid %(border)s;
    border-bottom: 1px solid %(border)s;
    font-weight: bold;
}
QHeaderView::section:last { border-right: none; }
"""

_QSS_LIST_TREE = """
QListWidget, QListView, QTreeWidget, QTreeView {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: %(r_md)dpx;
    padding: 4px;
    outline: 0;
}
QListWidget::item, QListView::item, QTreeWidget::item, QTreeView::item {
    padding: 6px 8px;
    border-radius: %(r_sm)dpx;
}
QListWidget::item:hover, QListView::item:hover,
QTreeWidget::item:hover, QTreeView::item:hover {
    background-color: %(hover)s;
}
QListWidget::item:selected, QListView::item:selected,
QTreeWidget::item:selected, QTreeView::item:selected {
    background-color: %(sel)s;
    color: %(text)s;
}
"""

_QSS_TEXTEDIT = """
QTextEdit, QPlainTextEdit {
    background-color: %(card)s;
    border: 1px solid %(border_strong)s;
    border-radius: %(r_sm)dpx;
    padding: 6px 8px;
    selection-background-color: %(sel)s;
    selection-color: %(text)s;
}
QTextEdit:focus, QPlainTextEdit:focus { border: 1px solid %(primary)s; }
QTextEdit[variant="log"], QPlainTextEdit[variant="log"] {
    background-color: %(bg_dark)s;
    color: %(bg_dark_text)s;
    border: 1px solid %(bg_dark)s;
    border-radius: %(r_md)dpx;
    padding: 10px;
    font-family: %(mono)s;
}
"""

_QSS_PROGRESS_TAB_SCROLL = """
QProgressBar {
    background-color: %(hover)s;
    border: none;
    border-radius: %(r_sm)dpx;
    text-align: center;
    color: %(text_sec)s;
    min-height: 8px;
    max-height: 8px;
}
QProgressBar::chunk {
    background-color: %(primary)s;
    border-radius: %(r_sm)dpx;
}
QTabWidget::pane { border: 1px solid %(border)s; border-radius: %(r_md)dpx; top: -1px; }
QTabBar::tab {
    background-color: %(hover)s;
    color: %(text_sec)s;
    padding: 6px 14px;
    border-top-left-radius: %(r_sm)dpx;
    border-top-right-radius: %(r_sm)dpx;
    margin-right: 2px;
}
QTabBar::tab:selected { background-color: %(card)s; color: %(primary)s; font-weight: bold; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: %(border_strong)s; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: %(border_strong)s; border-radius: 4px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background: %(muted)s; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""

_QSS_CARDS = """
QFrame[variant="card"] {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: %(r_lg)dpx;
}
QFrame[variant="hint"] {
    background-color: %(bg_hint)s;
    border: 1px solid %(bg_hint_border)s;
    border-radius: %(r_md)dpx;
    padding: 4px;
}
QFrame[variant="separator"] {
    background-color: %(border)s;
    max-height: 1px;
    min-height: 1px;
    border: none;
}
QFrame[variant="topbar"] {
    background-color: %(card)s;
    border: none;
    border-bottom: 1px solid %(border)s;
    padding: 2px 0;
}
QLabel[variant="title"] {
    color: %(text)s;
    font-size: %(title_pt)dpt;
    font-weight: bold;
}
QLabel[variant="section"] {
    color: %(primary)s;
    font-size: %(title_pt)dpt;
    font-weight: bold;
    padding: 4px 0 2px 0;
}
QLabel[variant="muted"] { color: %(muted)s; font-size: %(small_pt)dpt; }
QLabel[variant="hint-title"] { color: %(primary_pressed)s; font-weight: bold; font-size: 11pt; }
QLabel[variant="hint-text"] { color: %(text_sec)s; font-size: %(small_pt)dpt; }
QLabel[variant="stat-title"] { color: %(text_sec)s; font-weight: bold; }
QLabel[variant="stat-value"] { color: %(primary)s; font-weight: bold; font-size: %(title_pt)dpt; }
QLabel[variant="stat-value-success"] { color: %(success)s; font-weight: bold; font-size: %(title_pt)dpt; }
QLabel[variant="status-running"] { color: %(info)s; font-weight: bold; }
QLabel[variant="status-success"] { color: %(success)s; font-weight: bold; }
QLabel[variant="status-error"]   { color: %(danger)s;  font-weight: bold; }

/* ===== Welcome Page ===== */
QLabel#welcomeTitle {
    color: %(text)s;
    font-size: 22pt;
    font-weight: bold;
    padding: 0 0 4px 0;
}
QLabel#welcomeSubtitle {
    color: %(text_sec)s;
    font-size: 11pt;
    padding: 0 0 8px 0;
}
QListWidget#welcomeList {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: %(r_lg)dpx;
    padding: 6px;
    outline: 0;
}
QListWidget#welcomeList::item {
    padding: 0px;
    border: none;
}
QFrame#welcomeRow {
    background-color: transparent;
    border-radius: %(r_md)dpx;
}
QFrame#welcomeRow:hover {
    background-color: %(hover)s;
}
QLabel#welcomeEmpty {
    color: %(muted)s;
    font-size: %(body_pt)dpt;
    padding: 20px 0;
}
QWidget#welcomeFooter {
    background-color: transparent;
}

/* ===== Topbar ===== */
QFrame[variant="topbar"] > QLabel {
    font-weight: bold;
}
"""


def _qss_format_args() -> dict:
    """构造 QSS 格式化参数字典，延迟初始化箭头 SVG 资源。"""
    args = {
        "text": COLOR_TEXT_PRIMARY,

        "text_sec": COLOR_TEXT_SECONDARY,
        "muted": COLOR_TEXT_MUTED,
        "inverse": COLOR_TEXT_INVERSE,
        "font": FONT_FAMILY,
        "mono": FONT_FAMILY_MONO,
        "body_pt": FONT_SIZE_BODY,
        "small_pt": FONT_SIZE_SMALL,
        "title_pt": FONT_SIZE_TITLE,
        "bg_app": COLOR_BG_APP,
        "card": COLOR_BG_CARD,
        "hover": COLOR_BG_HOVER,
        "bg_hint": COLOR_BG_HINT,
        "bg_hint_border": COLOR_BG_HINT_BORDER,
        "bg_dark": COLOR_BG_DARK,
        "bg_dark_text": COLOR_BG_DARK_TEXT,
        "border": COLOR_BORDER,
        "border_strong": COLOR_BORDER_STRONG,
        "sel": COLOR_SELECTION_BG,
        "primary": COLOR_PRIMARY,
        "primary_hover": COLOR_PRIMARY_HOVER,
        "primary_pressed": COLOR_PRIMARY_PRESSED,
        "primary_disabled": COLOR_PRIMARY_DISABLED,
        "success": COLOR_SUCCESS,
        "success_hover": COLOR_SUCCESS_HOVER,
        "success_pressed": COLOR_SUCCESS_PRESSED,
        "success_disabled": COLOR_SUCCESS_DISABLED,
        "danger": COLOR_DANGER,
        "danger_pressed": COLOR_DANGER_PRESSED,
        "info": COLOR_INFO,
        "r_sm": RADIUS_SM,
        "r_md": RADIUS_MD,
        "r_lg": RADIUS_LG,
    }
    # 延迟初始化箭头 SVG，避免在模块导入时触发文件 I/O
    if not _ARROW_CACHE:
        _cleanup_old_arrow_cache()
        _ARROW_CACHE.update(_build_arrow_assets())
    args.update(_ARROW_CACHE)
    return args


def build_global_stylesheet() -> str:
    """构造覆盖所有常用 Qt 控件的全局样式表。"""
    parts = [
        _QSS_BASE,
        _QSS_MENU,
        _QSS_INPUT,
        _QSS_BUTTON_BASE,
        _QSS_BUTTON_PRIMARY,
        _QSS_BUTTON_SUCCESS,
        _QSS_BUTTON_DANGER,
        _QSS_CHECKBOX_RADIO,
        _QSS_TABLE,
        _QSS_LIST_TREE,
        _QSS_TEXTEDIT,
        _QSS_PROGRESS_TAB_SCROLL,
        _QSS_CARDS,
    ]
    args = _qss_format_args()
    return "\n".join(part % args for part in parts)


# 延迟初始化：仅在 apply_theme() 调用时构建样式表，避免导入时副作用
GLOBAL_STYLESHEET: str = ""


def apply_theme(app: QApplication) -> None:
    """给整个 ``QApplication`` 注入统一的字体与全局样式。

    应在 ``QApplication`` 实例化之后、主窗口显示之前调用一次即可。
    """
    global GLOBAL_STYLESHEET
    if not GLOBAL_STYLESHEET:
        GLOBAL_STYLESHEET = build_global_stylesheet()
    app.setFont(app_font())
    app.setStyleSheet(GLOBAL_STYLESHEET)


def refresh_widget_style(widget) -> None:
    """重新应用动态属性（如 ``variant``）后刷新控件样式。

    Qt 在通过 ``setProperty('variant', ...)`` 改动属性后并不会自动重绘，
    此辅助函数封装了 ``unpolish/polish`` 调用。
    """
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
