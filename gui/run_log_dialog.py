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
- 通过 ``log_dir`` + ``log_script_name`` 参数把整次运行的日志另存为
  ``<log_dir>/YYYYMMDD_HHMMSS_<script>.log`` 文件，方便事后追溯。

视觉风格统一来自 :mod:`gui.theme` 与 :mod:`gui.widgets`，本模块不再
书写内联样式。
"""

import datetime as _dt
import os
import re
from pathlib import Path
from typing import IO, Iterable, Optional

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


# 用于剥离控制台中的 ANSI 颜色 / 控制序列，避免日志文件里出现乱码 ``\x1b[31m`` 等
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _sanitize_for_log(text: str) -> str:
    """清理 ANSI 转义序列与回车符，得到适合写入日志文件的纯文本。"""
    cleaned = _ANSI_ESCAPE_RE.sub("", text)
    # ``\r`` 通常用于同行刷新，写入文件时统一替换为换行更可读
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


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
        log_dir: Optional[str] = None,
        log_script_name: Optional[str] = None,
    ):
        super().__init__(parent)
        self.program = program
        self.arguments = arguments
        self.working_dir = working_dir
        self.extra_pythonpath = [p for p in (extra_pythonpath or []) if p]

        # 日志文件相关：若 ``log_dir`` 为空则不落盘，仅在界面上显示。
        self._log_dir: Optional[Path] = Path(log_dir) if log_dir else None
        self._log_script_name: str = (log_script_name or "task").strip() or "task"
        self._log_file: Optional[IO[str]] = None
        self._log_file_path: Optional[Path] = None
        self._open_log_file()

        self._setup_ui(title)
        self._start_process()

    # ------------------------------------------------------------------
    # 日志文件
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_filename(name: str) -> str:
        """把脚本名清理成安全的文件名片段。"""
        cleaned = re.sub(r"[^0-9A-Za-z._\-]+", "_", name).strip("._-")
        return cleaned or "task"

    def _open_log_file(self) -> None:
        """根据 ``log_dir`` 准备日志文件句柄，失败时静默回退到不落盘。"""
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{self._safe_filename(self._log_script_name)}.log"
            self._log_file_path = self._log_dir / filename
            self._log_file = self._log_file_path.open(
                "w", encoding="utf-8", buffering=1  # 行缓冲，便于实时查看
            )
        except OSError as exc:
            # 写入磁盘失败时不应阻断任务运行，仅在控制台和窗口上提示一次
            self._log_file = None
            self._log_file_path = None
            print(f"[RunLogDialog] 创建日志文件失败：{exc}")

    def _write_log_file(self, text: str) -> None:
        """将一段文本追加到日志文件，并清理 ANSI 序列。"""
        if self._log_file is None:
            return
        try:
            self._log_file.write(_sanitize_for_log(text))
        except (OSError, ValueError):
            # 文件可能已被外部关闭/磁盘满，安全降级
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

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
        # 让子进程的 stdout/stderr 不缓冲，避免日志大段延迟。
        # 进度由 scripts._common.ProgressLogger 输出整行日志（不使用 \r 覆盖），
        # 因此 GUI 日志面板能够正常显示进度。
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
        if self._log_file_path is not None:
            self._append_log(f"[log] {self._log_file_path}\n")
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
        # 同步写入日志文件（若已配置）。失败时静默降级，不影响界面。
        self._write_log_file(text)

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
        self._close_log_file()

    def _on_finished(self, exit_code: int):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)

        if exit_code == 0:
            self.status_label.setText("任务已完成")
            self._set_status_variant("success")
            self._append_log("\n[任务完成]\n")
            if self._log_file_path is not None:
                self._append_log(f"[日志已保存] {self._log_file_path}\n")
            QMessageBox.information(self, "完成", "任务执行成功")
        else:
            self.status_label.setText(f"任务失败（退出码：{exit_code}）")
            self._set_status_variant("error")
            self._append_log(f"\n[任务失败，退出码：{exit_code}]\n")
            if self._log_file_path is not None:
                self._append_log(f"[日志已保存] {self._log_file_path}\n")
            QMessageBox.critical(self, "失败", f"任务执行失败，退出码：{exit_code}")
        # 关闭日志文件，确保磁盘写入完整
        self._close_log_file()

    def closeEvent(self, event):
        """兜底关闭日志文件，避免对话框被强制关闭时残留打开的句柄。"""
        try:
            self._close_log_file()
        finally:
            super().closeEvent(event)
