"""内部辅助：轻量参数校验。"""

from __future__ import annotations

from pathlib import Path


def _is_number(value: object) -> bool:
    """判断是否为数值类型（排除 bool）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_non_empty_str(value: object, name: str) -> str:
    """校验 ``value`` 是非空字符串，否则抛 :class:`ValueError`。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"参数 {name!r} 必须为非空字符串，当前值: {value!r}")
    return value


def _require_existing_dir(value: object, name: str) -> Path:
    """校验 ``value`` 指向已存在的目录，返回 :class:`pathlib.Path`。"""
    _require_non_empty_str(value, name)
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} 路径不存在: {value}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} 不是目录: {value}")
    return path


def _require_existing_file(value: object, name: str) -> Path:
    """校验 ``value`` 指向已存在的常规文件，返回 :class:`pathlib.Path`。"""
    _require_non_empty_str(value, name)
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} 文件不存在: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不是文件: {value}")
    return path


def _require_in_range(
        value: float,
        name: str,
        lo: float,
        hi: float,
        inclusive_lo: bool = True,
        inclusive_hi: bool = True,
) -> float:
    """
    校验数值 ``value`` 位于 ``[lo, hi]``（端点开闭可配置）内，
    否则抛 :class:`ValueError`。
    """
    if not _is_number(value):
        raise ValueError(f"参数 {name!r} 必须是数值，当前值: {value!r}")
    lo_ok = value >= lo if inclusive_lo else value > lo
    hi_ok = value <= hi if inclusive_hi else value < hi
    if not (lo_ok and hi_ok):
        lb = "[" if inclusive_lo else "("
        rb = "]" if inclusive_hi else ")"
        raise ValueError(
            f"参数 {name!r} 必须位于 {lb}{lo}, {hi}{rb}，当前值: {value}"
        )
    return float(value)


def _require_positive_int(value: object, name: str) -> int:
    """校验 ``value`` 为正整数。"""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"参数 {name!r} 必须是 >=1 的整数，当前值: {value!r}")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    """校验 ``value`` 为非负整数。"""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"参数 {name!r} 必须是 >=0 的整数，当前值: {value!r}")
    return value


def _require_non_negative_float(value: object, name: str) -> float:
    """校验 ``value`` 为非负浮点数。"""
    if not _is_number(value) or value < 0:
        raise ValueError(f"参数 {name!r} 必须是 >=0 的数值，当前值: {value!r}")
    return float(value)
