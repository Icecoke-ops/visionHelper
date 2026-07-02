#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 子进程参数构造工具。

``visionHelper`` 的各个子页面统一通过 ``python scripts/vh.py <subcommand> <action>
--arg value`` 的方式启动子进程。参数构造逻辑此前散落在每个页面中：拼接
字符串、判断空值、按需追加 ``--flag``。本模块提供两组工具集中处理：

- :func:`build_script_argv`：把 Python 风格 kwargs 转成统一的 CLI argv，
  自动跳过 ``None``/空串、把 bool 转成 ``--flag`` 形式。
- :func:`infer_script_name`：从 argv 中反推脚本名，用于日志文件命名。

使用示例::

    argv = build_script_argv(
        "images", "import",
        input=video_path,
        output=output_dir,
        frame_step=5,
        overwrite=True,
        end_time=None,                # 自动跳过
    )
    # → ["images", "import",
    #    "--input", "...", "--output", "...", "--frame-step", "5", "--overwrite"]
"""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any, Iterable, List


def _to_cli_value(value: Any) -> str:
    """将任意 Python 值转换为 CLI 参数字符串。"""
    if isinstance(value, bool):
        msg = f"bool 应由 build_script_argv 在调用 _to_cli_value 前处理，收到 {value!r}"
        raise TypeError(msg)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _key_to_flag(key: str) -> str:
    """``snake_case`` → ``--kebab-case``。"""
    return "--" + key.replace("_", "-")


def build_script_argv(subcommand: str, action: str, **options: Any) -> List[str]:
    """构造 ``python -m scripts.vh <subcommand> <action> --arg value`` 风格的参数列表。

    使用 ``-m`` 模式而非直接传入脚本路径，保证 ``scripts`` 包始终能被正确解析为
    顶层模块（不会因 ``sys.path[0]`` 指向 ``scripts/`` 内部而找不到父包）。

    参数:
        subcommand: 一级子命令（例如 ``images``、``datasets``）。
        action: 二级动作（例如 ``import``、``export``）。
        **options: 关键字参数，全部转为 ``--<kebab-case> value`` 形式。

            - ``bool`` 类型：``True`` 则追加 ``--flag``；``False`` 则不追加。
            - ``None`` / 空串：忽略。
            - 其它类型：作为 ``--key value``。

    返回:
        可直接传给 ``QProcess.start(python_path, argv)`` 的字符串列表，
        形如 ``["-m", "scripts.vh", "images", "import", "--input", "...", ...]``。
        调用方需把 ``python_path`` 指向 Python 解释器。
    """
    argv: List[str] = ["-m", "scripts.vh", subcommand, action]

    for key, value in options.items():
        if value is None:
            continue
        flag = _key_to_flag(key)

        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue

        if isinstance(value, str) and not value:
            continue

        argv.extend([flag, _to_cli_value(value)])

    return argv


def infer_script_name(arguments: Iterable[Any]) -> str:
    """从 QProcess 的 ``arguments`` 中推断脚本名（用于日志文件命名）。

    支持以下形式：

    - ``["images", "import", ...]``：返回 ``images_import``
    - ``["datasets", "export", ...]``：返回 ``datasets_export``
    - 旧 ``["-m", "scripts.xxx", ...]``：取 ``xxx``（兼容旧日志）
    - ``["/path/to/scripts/xxx.py", ...]``：取 ``xxx``

    若都识别不出，回退到 ``"task"``。
    """
    args = list(arguments or [])

    # ``-m scripts.vh <subcommand> <action>`` 形式
    if (
        len(args) >= 4
        and args[0] == "-m"
        and isinstance(args[1], str)
        and args[1].endswith(".vh")
        and isinstance(args[2], str)
        and isinstance(args[3], str)
    ):
        return f"{args[2]}_{args[3]}"

    # 旧 ``scripts/vh.py <subcommand> <action>`` 形式
    if (
        len(args) >= 3
        and isinstance(args[0], str)
        and isinstance(args[1], str)
        and isinstance(args[2], str)
        and args[0].endswith(".py")
        and not args[1].startswith("-")
    ):
        return f"{args[1]}_{args[2]}"

    # 旧 ``<subcommand> <action>`` 形式（无脚本路径前缀）
    if (
        len(args) >= 2
        and isinstance(args[0], str)
        and isinstance(args[1], str)
        and not args[0].startswith("-")
        and not args[1].startswith("-")
        and not args[0].endswith(".py")
    ):
        return f"{args[0]}_{args[1]}"

    # 旧 ``-m scripts.xxx`` 形式（兼容）
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
    """把 argv 拼成可读且可复制的命令行字符串，方便日志展示。"""
    return shlex.join(str(item) for item in argv)
