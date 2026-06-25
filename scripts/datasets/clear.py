#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按标注类型批量清除 X-AnyLabeling JSON 标注文件。

作为 ``scripts/datasets`` 子包的一部分，本模块同时包含核心实现与
``python scripts/vh.py datasets clear`` 命令行入口。``ultralytics`` 等重依赖
不会被模块顶层导入。

用法::

    python scripts/vh.py datasets clear -i ./images -a
    python scripts/vh.py datasets clear -i ./images -a -c -M --tolerance-seconds 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scripts.common.annotation_type import AnnotationType, AnnotationTypeChecker
from scripts.common.config import DEFAULT_TOLERANCE_SECONDS
from scripts.common.logging import log
from scripts.common.utils import iter_matched_pairs

__all__ = ["clear_annotations", "main"]


def clear_annotations(
        folder: str,
        include_auto: bool = False,
        include_auto_corrected: bool = False,
        include_manual: bool = False,
        tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> Dict[str, object]:
    """
    清除目录下指定类型的 X-AnyLabeling JSON 标注文件。

    参数:
        folder: 待清理的目录路径。
        include_auto: 是否删除自动标注的 JSON。
        include_auto_corrected: 是否删除自动标注后人工矫正的 JSON。
        include_manual: 是否删除手动标注的 JSON。
        tolerance_seconds: 区分自动 / 矫正 的时间容差（秒）。

    返回:
        ``{"scanned", "deleted", "by_type", "failed"}`` 字典。

    异常:
        ValueError: 目录不存在、不是文件夹，或所有清除开关均为关闭时抛出。
    """
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"目录不存在或不是文件夹: {folder}")

    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds 必须 >= 0")

    if not any([include_auto, include_auto_corrected, include_manual]):
        raise ValueError("至少需要选择一种待清除的标注类型")

    include_map: Dict[AnnotationType, bool] = {
        AnnotationType.AUTO: include_auto,
        AnnotationType.AUTO_CORRECTED: include_auto_corrected,
        AnnotationType.MANUAL: include_manual,
    }

    type_checker = AnnotationTypeChecker(tolerance_seconds=tolerance_seconds)

    scanned = 0
    deleted = 0
    by_type: Dict[str, int] = {
        AnnotationType.AUTO.value: 0,
        AnnotationType.AUTO_CORRECTED.value: 0,
        AnnotationType.MANUAL.value: 0,
    }
    failed: List[str] = []

    for _image_path, ann_path, data in iter_matched_pairs(root, require_shapes=False):
        scanned += 1

        try:
            ann_type = type_checker.check(data, json_mtime=ann_path.stat().st_mtime)
        except OSError:
            failed.append(str(ann_path))
            continue

        if not include_map.get(ann_type, False):
            continue

        try:
            ann_path.unlink()
        except OSError as exc:
            log(f"[错误] 删除失败 {ann_path.name}: {exc}", stream=sys.stderr)
            failed.append(str(ann_path))
            continue

        deleted += 1
        by_type[ann_type.value] = by_type.get(ann_type.value, 0) + 1

    log(
        "清除标签完成：\n"
        f"  扫描到匹配图片的标注: {scanned}\n"
        f"  实际删除: {deleted}\n"
        f"  按类型: 自动 {by_type[AnnotationType.AUTO.value]}, "
        f"矫正 {by_type[AnnotationType.AUTO_CORRECTED.value]}, "
        f"手动 {by_type[AnnotationType.MANUAL.value]}\n"
        f"  删除失败: {len(failed)}"
    )

    return {
        "scanned": scanned,
        "deleted": deleted,
        "by_type": by_type,
        "failed": failed,
    }


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python scripts/vh.py datasets clear",
        description=(
            "按标注类型批量清除 X-AnyLabeling JSON 标注文件。"
            "必须显式指定至少一个 --include-* 开关，避免误删。"
        ),
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="待清理的图片目录",
    )
    parser.add_argument(
        "-a", "--include-auto",
        action="store_true",
        help="删除自动标注的 JSON",
    )
    parser.add_argument(
        "-c", "--include-auto-corrected",
        action="store_true",
        help="删除自动标注后人工矫正的 JSON",
    )
    parser.add_argument(
        "-M", "--include-manual",
        action="store_true",
        help="删除手动标注的 JSON",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help=f"判定自动 / 矫正的时间容差（秒），默认 {DEFAULT_TOLERANCE_SECONDS}",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """对命令行参数做友好的预校验。"""
    folder = Path(args.input)
    if not folder.exists():
        raise ValueError(f"目录不存在：{args.input}")
    if not folder.is_dir():
        raise ValueError(f"路径不是目录：{args.input}")

    if not any([args.include_auto, args.include_auto_corrected, args.include_manual]):
        raise ValueError(
            "必须至少指定一个 --include-auto / --include-auto-corrected / "
            "--include-manual，以明确删除范围（避免误删）。"
        )

    if args.tolerance_seconds is not None and args.tolerance_seconds < 0:
        raise ValueError(
            f"--tolerance-seconds 必须为非负数，当前为 {args.tolerance_seconds}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。

    返回:
        进程退出码：0=成功；2=参数非法；1=运行时错误；130=用户中断。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except ValueError as exc:
        log(f"[参数错误] {exc}", stream=sys.stderr)
        return 2

    try:
        result = clear_annotations(
            folder=args.input,
            include_auto=args.include_auto,
            include_auto_corrected=args.include_auto_corrected,
            include_manual=args.include_manual,
            tolerance_seconds=args.tolerance_seconds,
        )
    except KeyboardInterrupt:
        log(
            "[已取消] 用户中断，部分文件可能已被删除（不可回滚）。",
            stream=sys.stderr,
        )
        return 130
    except (ValueError, FileNotFoundError) as exc:
        log(f"[错误] {exc}", stream=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        log(f"[错误] 清除标注失败：{exc}", stream=sys.stderr)
        return 1

    if isinstance(result, dict):
        scanned = result.get("scanned", 0)
        deleted = result.get("deleted", 0)
        failed = result.get("failed", []) or []
        log(f"[完成] 扫描 {scanned} 个 JSON，删除 {deleted} 个。")
        if failed:
            log(f"[警告] {len(failed)} 个文件删除失败，请检查权限或被占用。",
                stream=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
