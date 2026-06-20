#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 公共控件模块。

集中存放在多个页面中复用的小型自定义控件，避免重复定义。

包含：
    - :class:`LabeledSpinBox`: 带文字标签的整型数值输入框
    - :class:`LabeledDoubleSpinBox`: 带文字标签的浮点数值输入框
    - 一组语义化按钮：:class:`PrimaryButton` / :class:`SuccessButton` /
      :class:`SecondaryButton` / :class:`DangerButton` / :class:`LinkButton` /
      :class:`IconButton`
    - 文案与排版组件：:class:`SectionTitle` / :class:`MutedLabel` /
      :class:`HSeparator`
    - 容器组件：:class:`HintCard`（蓝色提示卡片） / :class:`FormRow`
      （统一表单行） / :class:`StatItem`（标签 + 数值统计行）
"""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui import theme


# ---------------------------------------------------------------------------
# 数值输入控件
# ---------------------------------------------------------------------------


class LabeledSpinBox(QWidget):
    """带文字标签的整型数值输入框。

    Args:
        label: 显示在数字框左侧的标签文本。
        min_value: 最小值。
        max_value: 最大值。
        default: 默认值。
        parent: Qt 父控件。
    """

    def __init__(
        self,
        label: str,
        min_value: int,
        max_value: int,
        default: int,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QLabel(label)
        self.label.setFixedWidth(50)
        layout.addWidget(self.label)

        self.spin = QSpinBox()
        self.spin.setRange(min_value, max_value)
        self.spin.setValue(default)
        layout.addWidget(self.spin)

    def value(self) -> int:
        """返回当前数值。"""
        return self.spin.value()


class LabeledDoubleSpinBox(QWidget):
    """带文字标签的浮点数值输入框。

    Args:
        label: 显示在数字框左侧的标签文本。
        min_value: 最小值。
        max_value: 最大值。
        default: 默认值。
        decimals: 小数位数，默认为 2。
        step: 单步增减量，默认为 0.01。
        parent: Qt 父控件。
    """

    def __init__(
        self,
        label: str,
        min_value: float,
        max_value: float,
        default: float,
        decimals: int = 2,
        step: float = 0.01,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QLabel(label)
        self.label.setFixedWidth(50)
        layout.addWidget(self.label)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_value, max_value)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setValue(default)
        layout.addWidget(self.spin)

    def value(self) -> float:
        """返回当前数值。"""
        return self.spin.value()


# ---------------------------------------------------------------------------
# 语义化按钮
# ---------------------------------------------------------------------------


def _apply_variant(widget: QWidget, variant: str) -> None:
    """给控件设置 ``variant`` 动态属性，并刷新样式。"""
    widget.setProperty("variant", variant)
    theme.refresh_widget_style(widget)


class _BaseButton(QPushButton):
    """所有自定义按钮的统一基类，仅做尺寸/光标等通用设置。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(theme.BUTTON_HEIGHT)


class PrimaryButton(_BaseButton):
    """主操作按钮（蓝底白字），用于页面中最关键的操作（如"开始抽帧"）。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        _apply_variant(self, "primary")


class SuccessButton(_BaseButton):
    """成功/执行类按钮（绿底白字），用于"开始训练 / 自动标注"等耗时执行。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        _apply_variant(self, "success")


class SecondaryButton(_BaseButton):
    """次要按钮（白底灰字），保持默认 QPushButton 外观即可。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)


class DangerButton(_BaseButton):
    """危险/破坏性操作按钮（白底红字，悬停变红），用于"停止 / 删除"等。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        _apply_variant(self, "danger")


class LinkButton(_BaseButton):
    """无边框链接式按钮，用于列表项 / 信息卡内的次要触发点。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        _apply_variant(self, "link")
        self.setMinimumHeight(0)


class IconButton(_BaseButton):
    """无边框图标按钮（如列表行尾的"✕"），悬停变红。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        _apply_variant(self, "icon")
        self.setMinimumHeight(0)


# ---------------------------------------------------------------------------
# 文案与排版控件
# ---------------------------------------------------------------------------


class SectionTitle(QLabel):
    """区块小标题（粗体），用于页面内多步骤之间的分段。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        _apply_variant(self, "section")


class MutedLabel(QLabel):
    """次要灰色文本（小号），用于辅助说明 / 占位提示等。"""

    def __init__(self, text: str = "", parent: QWidget = None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        _apply_variant(self, "muted")


class HSeparator(QFrame):
    """水平分隔线（1px 浅灰），统一替代页面内零散的 ``QFrame.HLine``。"""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        _apply_variant(self, "separator")
        self.setFixedHeight(1)


# ---------------------------------------------------------------------------
# 容器组件
# ---------------------------------------------------------------------------


class HintCard(QFrame):
    """蓝色提示卡片：左侧文字（标题 + 描述 + 可选额外控件），右侧可选按钮。

    Args:
        title: 卡片粗体标题（可包含 emoji）。
        description: 描述性正文，自动换行。
        action: 可选右侧动作按钮（建议传 :class:`PrimaryButton`）。
        extra_widget: 可选追加在描述下方的控件（如带链接的 ``QLabel``）。
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        action: Optional[QPushButton] = None,
        extra_widget: Optional[QWidget] = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        _apply_variant(self, "hint")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)

        title_label = QLabel(title)
        _apply_variant(title_label, "hint-title")
        text_box.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            _apply_variant(desc_label, "hint-text")
            text_box.addWidget(desc_label)

        if extra_widget is not None:
            text_box.addWidget(extra_widget)

        layout.addLayout(text_box, 1)

        if action is not None:
            action.setMinimumHeight(theme.BUTTON_HEIGHT)
            layout.addWidget(action, alignment=Qt.AlignVCenter)

        self.title_label = title_label


class FormRow(QWidget):
    """统一的表单行：左侧固定宽度的标签 + 右侧填充控件。

    用于替代各页面中手写的 ``QHBoxLayout + QLabel + widget`` 组合，
    保证标签宽度、间距、对齐方式在整个应用中保持一致。
    """

    def __init__(
        self,
        label: str,
        widget: QWidget,
        label_width: int = theme.FORM_LABEL_WIDTH,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignLeft)

        self.label = QLabel(label)
        self.label.setFixedWidth(label_width)
        layout.addWidget(self.label, alignment=Qt.AlignVCenter)

        self.widget = widget
        widget.setSizePolicy(
            QSizePolicy.Expanding,
            widget.sizePolicy().verticalPolicy(),
        )
        layout.addWidget(widget, 1)


class StatItem(QWidget):
    """统计项：一行展示"标题 + 数值"。

    Args:
        title: 左侧标题（如 "图片总数："）。
        value: 右侧数值字符串。
        kind: 数值的颜色风格，可选 ``"primary"``（蓝） / ``"success"``（绿） /
            ``"muted"``（灰），默认为 ``"success"``。
        title_width: 标题列的固定宽度，便于多个 StatItem 上下对齐。
    """

    _KIND_VARIANT = {
        "primary": "stat-value",
        "success": "stat-value-success",
    }

    def __init__(
        self,
        title: str,
        value: str = "0",
        kind: str = "success",
        title_width: int = 160,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignLeft)

        self.title_label = QLabel(title)
        _apply_variant(self.title_label, "stat-title")
        self.title_label.setFixedWidth(title_width)
        layout.addWidget(self.title_label)

        self.value_label = QLabel(str(value))
        variant = self._KIND_VARIANT.get(kind, "stat-value-success")
        if kind == "muted":
            _apply_variant(self.value_label, "muted")
        else:
            _apply_variant(self.value_label, variant)
        layout.addWidget(self.value_label)
        layout.addStretch(1)

    def set_value(self, value) -> None:
        """更新右侧数值文本。"""
        self.value_label.setText(str(value))

    def set_title(self, title: str) -> None:
        """更新左侧标题文本。"""
        self.title_label.setText(title)
