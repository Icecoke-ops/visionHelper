#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 子进程参数构造工具。

``visionHelper`` 的各个子页面都通过 ``python -m scripts.xxx --arg value``
的方式启动子进程，参数构造逻辑此前散落在每个页面中：拼接字符串、判断
空值、按需追加 ``--flag``。本模块提供两组工具集中处理：

- :func:`build_script_argv`：把 Python 风格 kwargs 转成统一的 CLI argv，
  自动跳过 ``None``/空串、把 bool 转成 ``--flag`` 形式、把 list 展开。
- :func:`infer_script_name`：从 argv 中反推脚本名，用于日志文件命名。

使用示例::

    argv = build_script_argv(
        "scripts.extract_video_frames",
        input=video_path,
        output=output_dir,
        step=5,
        overwrite=True,
        end_time=None,                # 自动跳过
    )
    # → ["-m", "scripts.extract_video_frames",
    #    "--input", "...", "--output", "...", "--step", "5", "--overwrite"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List


def _to_cli_value(value: Any) -> str:
    """将任意 Python 值转换为 CLI 参数字符串。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        # bool 由调用方判断是否追加 flag，这里不应进入
        return "true" if value else "false"
    return str(value)


def _key_to_flag(key: str) -> str:
    """``snake_case`` → ``--kebab-case``。"""
    return "--" + key.replace("_", "-")


def build_script_argv(module: str, *positional: Any, **options: Any) -> List[str]:
    """构造 ``python -m <module>`` 风格的参数列表。

    参数:
        module: 要运行的 Python 模块名（例如 ``scripts.train_model``）。
        *positional: 位置参数，按顺序追加到 ``-m <module>`` 之后。
            ``None`` 与空串会被忽略。
        **options: 关键字参数。

            - ``bool`` 类型：``True`` 则追加 ``--flag``；``False`` 则不追加。
            - ``None`` / 空串：忽略。
            - ``list`` / ``tuple``：按 ``--key v1 v2 ...`` 展开。
            - 其它类型：作为 ``--key value``。

    返回:
        可直接传给 ``QProcess.start(python_path, argv)`` 的字符串列表，
        以 ``-m`` 开头。
    """
    argv: List[str] = ["-m", module]

    for value in positional:
        if value is None or value == "":
            continue
        argv.append(_to_cli_value(value))

    for key, value in options.items():
        if value is None:
            continue
        flag = _key_to_flag(key)

        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue

        if isinstance(value, (list, tuple)):
            items = [v for v in value if v is not None and v != ""]
            if not items:
                continue
            argv.append(flag)
            argv.extend(_to_cli_value(v) for v in items)
            continue

        if isinstance(value, str) and not value:
            continue

        argv.extend([flag, _to_cli_value(value)])

    return argv


def infer_script_name(arguments: Iterable[Any]) -> str:
    """从 QProcess 的 ``arguments`` 中推断脚本名（用于日志文件命名）。

    支持以下两种最常见的形式：

    - ``["-m", "scripts.xxx", ...]``：取 ``xxx``
    - ``["/path/to/scripts/xxx.py", ...]``：取 ``xxx``

    若都识别不出，回退到 ``"task"``。
    """
    args = list(arguments or [])
    # ``-m scripts.xxx`` 形式
    for i, token in enumerate(args):
        if token == "-m" and i + 1 < len(args):
            module = args[i + 1]
            return module.rsplit(".", 1)[-1] or "task"
    # 文件路径形式：取第一个以 .py 结尾的参数
    for token in args:
        if isinstance(token, str) and token.endswith(".py"):
            return Path(token).stem or "task"
    return "task"


def join_for_display(argv: Iterable[str]) -> str:
    """把 argv 拼成可读的命令行字符串，方便日志展示。"""
    return " ".join(argv)
