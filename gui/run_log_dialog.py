#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务日志弹窗模块。

提供 :class:`RunLogDialog`，通过 QProcess 运行子任务并实时展示输出。

为了支持"GUI 独立打包、scripts 保持源码"的部署模式，本对话框允许：

- 通过 ``working_dir`` 参数设置子进程工作目录；
- 通过 ``extra_pythonpath`` 参数把额外目录注入到子进程的
  ``PYTHONPATH`` 环境变量中，从而保证 ``python -m scripts.xxx`` 能在
  任意启动位置（如双击 exe）下找到 ``scripts`` 包。

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`，本模块不再
书写内联样式。
"""

import os
from typing import Iterable, Optional

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.widgets import DangerButton, SecondaryButton


class RunLogDialog(QDialog):
    """任务日志弹窗：通过 QProcess 运行子任务并实时展示输出。"""

    def __init__(
        self,
        program: str,
        arguments: list,
        title: str = "任务日志",
        parent: QWidget = None,
        working_dir: Optional[str] = None,
        extra_pythonpath: Optional[Iterable[str]] = None,
    ):
        super().__init__(parent)
        self.program = program
        self.arguments = arguments
        self.working_dir = working_dir
        self.extra_pythonpath = [p for p in (extra_pythonpath or []) if p]
        self._setup_ui(title)
        self._start_process()

    def _setup_ui(self, title: str):
        self.setWindowTitle(title)
        self.setMinimumSize(760, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACING_LG,
            theme.SPACING_LG,
            theme.SPACING_LG,
            theme.SPACING_LG,
        )
        layout.setSpacing(theme.SPACING_MD)

        # 状态标签：根据运行状态切换 variant（running/success/error）
        self.status_label = QLabel("正在启动...")
        self._set_status_variant("running")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # 日志文本框：使用 log variant 呈现深色等宽风格
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setProperty("variant", "log")
        theme.refresh_widget_style(self.log_edit)
        layout.addWidget(self.log_edit, 1)

        # 操作按钮：停止（红） + 关闭（次要）
        button_bar = QHBoxLayout()
        button_bar.setSpacing(theme.SPACING_SM)
        button_bar.addStretch(1)

        self.stop_btn = DangerButton("停止")
        self.stop_btn.clicked.connect(self._stop_process)
        button_bar.addWidget(self.stop_btn)

        self.close_btn = SecondaryButton("关闭")
        self.close_btn.clicked.connect(self.reject)
        self.close_btn.setEnabled(False)
        button_bar.addWidget(self.close_btn)

        layout.addLayout(button_bar)

    def _set_status_variant(self, variant: str) -> None:
        """切换状态标签的语义化样式：``running`` / ``success`` / ``error``。"""
        self.status_label.setProperty("variant", f"status-{variant}")
        theme.refresh_widget_style(self.status_label)

    def _build_process_environment(self) -> QProcessEnvironment:
        """构造子进程环境变量，将 ``extra_pythonpath`` 追加到 ``PYTHONPATH``。"""
        env = QProcessEnvironment.systemEnvironment()
        if self.extra_pythonpath:
            existing = env.value("PYTHONPATH", "")
            parts = list(self.extra_pythonpath)
            if existing:
                parts.append(existing)
            env.insert("PYTHONPATH", os.pathsep.join(parts))
        # 让子进程的 stdout/stderr 不缓冲，避免日志大段延迟
        env.insert("PYTHONUNBUFFERED", "1")
        return env

    def _start_process(self):
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._append_output)
        self.process.errorOccurred.connect(self._on_error)
        self.process.finished.connect(self._on_finished)

        self.process.setProcessEnvironment(self._build_process_environment())
        if self.working_dir:
            self.process.setWorkingDirectory(self.working_dir)

        cmd = " ".join([self.program, *self.arguments])
        if self.working_dir:
            self._append_log(f"[cwd] {self.working_dir}\n")
        if self.extra_pythonpath:
            self._append_log(
                f"[PYTHONPATH+] {os.pathsep.join(self.extra_pythonpath)}\n"
            )
        self._append_log(f"$ {cmd}\n")
        self.process.start(self.program, self.arguments)

        # 进程启动后切换到"运行中"状态
        self.status_label.setText("任务运行中...")
        self._set_status_variant("running")

    def _append_output(self):
        data = self.process.readAllStandardOutput().data()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        self._append_log(text)

    def _append_log(self, text: str):
        self.log_edit.moveCursor(QTextCursor.End)
        self.log_edit.insertPlainText(text)
        self.log_edit.moveCursor(QTextCursor.End)

    def _stop_process(self):
        if self.process.state() != QProcess.NotRunning:
            self._append_log("\n[用户停止任务]\n")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
        self.stop_btn.setEnabled(False)

    def _on_error(self, error: QProcess.ProcessError):
        self._append_log(f"\n[进程错误] {error}\n")
        self.status_label.setText("运行出错")
        self._set_status_variant("error")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

    def _on_finished(self, exit_code: int):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

        if exit_code == 0:
            self.status_label.setText("任务已完成")
            self._set_status_variant("success")
            self._append_log("\n[任务完成]\n")
            QMessageBox.information(self, "完成", "任务执行成功")
        else:
            self.status_label.setText(f"任务失败（退出码：{exit_code}）")
            self._set_status_variant("error")
            self._append_log(f"\n[任务失败，退出码：{exit_code}]\n")
            QMessageBox.critical(self, "失败", f"任务执行失败，退出码：{exit_code}")
