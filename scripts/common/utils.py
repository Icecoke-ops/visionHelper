#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``scripts`` 包内部共享的 **轻量** IO 工具集。

集中管理跨模块复用的辅助逻辑：图片 / 标注文件判定、X-AnyLabeling JSON
读取、图片解析、目录遍历与已训练模型扫描等。所有逻辑只依赖标准库
（不引入 ``torch`` / ``ultralytics`` / ``cv2`` / ``transformers`` 等重依赖），
确保打包后的 GUI 进程也能在不触发 ``scripts.api`` 的情况下直接 import。

- 常量 :data:`IMAGE_EXTENSIONS` 与 :class:`ProgressLogger` 在此处仅作 **向后兼容
  的 re-export**：实现已分别迁移到 :mod:`scripts.common.config` 与 :mod:`scripts.common.logging`。
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Tuple

from scripts.common.config import (
    IMAGE_EXTENSIONS,  # re-export，保持外部 import 兼容
    MODELS_FOLDER,
    TASK_MODEL_SUFFIX,
)
from scripts.common.logging import ProgressLogger  # re-export，保持外部 import 兼容


__all__ = [
    "IMAGE_EXTENSIONS",
    "ProgressLogger",
    "app_root",
    "discover_trained_models",
    "ensure_models_dir",
    "find_model_class_names",
    "is_annotation_file",
    "is_image_file",
    "iter_annotations",
    "iter_images",
    "iter_matched_pairs",
    "load_annotation",
    "models_dir",
    "resolve_image_path",
    "resolve_image_stem",
    "resolve_model_path",
    "validate_split_ratios",
]


def validate_split_ratios(train_ratio: float, test_ratio: float) -> None:
    """校验训练/测试集划分比例之和约等于 1，允许 1/0 或 0/1。

    参数:
        train_ratio: 训练集占比，必须为 ``[0, 1]`` 内的有限数字。
        test_ratio: 测试集占比，必须为 ``[0, 1]`` 内的有限数字。

    异常:
        ValueError: 任一比例不是有限数字、不在 ``[0, 1]`` 内，或总和偏离 1 超过 1e-6。
    """
    ratios = [("train_ratio", train_ratio), ("test_ratio", test_ratio)]
    for name, ratio in ratios:
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            raise ValueError(f"{name} 必须为有限数字，当前为 {ratio!r}")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{name} 必须在 [0, 1] 范围内，当前为 {ratio}")

    ratio_sum = train_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须等于 1，当前为 {ratio_sum}")


def app_root() -> Path:
    """返回应用根目录，即 ``scripts/`` 包所在的父目录。

    本实现是应用根目录规则的单一事实来源；:func:`gui.config.app_root` 会
    优先复用本函数，仅在其不可导入时回退到本地等价实现（GUI 层不反向
    import 本包之外的任何依赖）：

    - 优先使用环境变量 ``VISIONHELPER_APP_ROOT`` 强制覆盖；
    - 打包态（``sys.frozen``）返回可执行文件所在目录；
    - 开发态返回仓库根目录，由本文件位置 ``scripts/common/`` 推断
      （``parents[2]``）。
    """
    override = os.environ.get("VISIONHELPER_APP_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    """返回应用根目录下存放预训练模型（自动下载）的目录路径。

    纯路径计算，不产生目录创建等副作用；需要落盘 / 下载前请调用
    :func:`ensure_models_dir`。与 YOLO 权重、HuggingFace 模型共用
    ``MODELS_FOLDER``，避免模型散落到程序根目录或 HuggingFace 缓存。
    """
    return app_root() / MODELS_FOLDER


def ensure_models_dir() -> Path:
    """确保预训练模型目录存在，不存在则创建；失败时给出友好提示。

    :func:`resolve_model_path` 等纯解析函数不建目录，仅在即将写入模型
    （Ultralytics 自动下载 / HuggingFace 快照下载）前调用本函数，避免在
    只读部署目录下无谓创建或抛出难懂的原始 ``OSError``。

    异常:
        RuntimeError: 目录创建失败（如应用根目录不可写）。
    """
    models_path = models_dir()
    try:
        models_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"无法创建模型目录 {models_path}，"
            f"请检查应用根目录 {app_root()} 是否可写"
        ) from exc
    return models_path


def _complete_model_name(model: str, task: Optional[str] = None) -> str:
    """把裸模型名补全为带任务后缀与 ``.pt`` 的权重文件名。

    与训练侧的任务后缀约定保持一致（单点实现，供 :func:`resolve_model_path`
    复用）：

    - 去掉已有的 ``.pt`` 后按 ``TASK_MODEL_SUFFIX`` 补全任务后缀
      （如 ``yolov8n`` + classify → ``yolov8n-cls``）；
    - 仅当任务后缀缺失时追加，避免 ``yolov8n-cls`` + classify 被重复拼接；
    - 统一补上 ``.pt`` 扩展名。

    参数:
        model: 模型名（可含 ``.pt`` 或不含）。
        task: 任务类型（detect / obb / segment / classify）；为 ``None`` 或
            未知任务时不做任务后缀补全。

    返回:
        补全后的权重文件名（如 ``yolov8n-cls.pt``）。
    """
    stem = model
    if stem.lower().endswith(".pt"):
        stem = stem[:-3]
    if task:
        task_suffix = TASK_MODEL_SUFFIX.get(task, "")
        if task_suffix and not stem.endswith(task_suffix):
            stem = f"{stem}{task_suffix}"
    return f"{stem}.pt"


def resolve_model_path(model: str, task: Optional[str] = None) -> Path:
    """把用户输入的模型名称解析为最终的权重文件路径（纯计算，无副作用）。

    规则（按优先级）：

    - 空输入抛 :class:`ValueError`；
    - 输入是 **已存在的文件** 时原样返回：尊重用户显式指定的位置，这也保证
      ``--model best.pt``（工作目录下的已训练权重）这类常见用法不受影响；
    - 含路径分隔符（绝对 / 相对路径）时原样返回，不做迁移；
    - 否则视为 **裸模型名**（如 ``yolov8n`` / ``yolov8n.pt``），经
      :func:`_complete_model_name` 补全任务后缀与 ``.pt`` 后，统一映射到
      应用根目录下的 :data:`MODELS_FOLDER` 目录，确保 Ultralytics 自动下载
      的预训练权重不会散落到程序根目录。

    参数:
        model: 模型权重路径或裸模型名。
        task: 任务类型（detect / obb / segment / classify），仅对裸模型名
            生效，用于补全正确的预训练变体（如 ``yolov8n-cls.pt``）。

    返回:
        最终权重路径。
    """
    name = (model or "").strip()
    if not name:
        raise ValueError("模型名称不能为空")
    path = Path(name).expanduser()
    if len(path.parts) > 1:
        # 显式路径（绝对 / 相对）原样返回，交由调用方判断存在性
        return path
    if path.suffix.lower() == ".pt" and path.is_file():
        # 裸名但工作目录下存在已训练权重（如 ``--model best.pt``）
        return path
    return models_dir() / _complete_model_name(name, task)


class _YOLOModel(Protocol):
    """Ultralytics YOLO 模型的轻量 Protocol，避免导入 torch/ultralytics。"""
    names: dict[int, str] | list[str] | tuple[str, ...]


def find_model_class_names(model: _YOLOModel) -> list:
    """从 Ultralytics YOLO 模型中提取类别名称列表。"""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, (list, tuple)):
        return list(names)
    return []


def is_image_file(path: Path) -> bool:
    """判断文件是否为支持的图片文件。"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_annotation_file(path: Path) -> bool:
    """判断文件是否为 X-AnyLabeling JSON 标注文件。"""
    return path.is_file() and path.suffix.lower() == ".json"


def load_annotation(annotation_path: Path) -> Optional[dict]:
    """安全加载 X-AnyLabeling JSON 标注文件，失败时返回 ``None``。"""
    try:
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def resolve_image_stem(annotation_path: Path, data: dict) -> Optional[str]:
    """
    根据标注文件内容解析其对应的图片去后缀文件名（stem）。

    优先使用 JSON 中的 ``imagePath`` 字段，回退为标注文件自身的 stem。
    """
    image_path = data.get("imagePath")
    if isinstance(image_path, str) and image_path:
        return Path(image_path).stem
    return annotation_path.stem


def resolve_image_path(annotation_path: Path, data: dict) -> Optional[Path]:
    """
    根据标注文件解析对应的图片绝对路径。

    优先使用 ``imagePath`` 字段；若不存在或对应文件不存在，则退化为
    在标注文件所在目录下按支持的扩展名查找同 stem 的图片文件。
    """
    root = annotation_path.parent
    image_path_value = data.get("imagePath")
    if isinstance(image_path_value, str) and image_path_value:
        candidate = root / image_path_value
        if candidate.is_file() and is_image_file(candidate):
            return candidate

    for candidate in root.glob(f"{annotation_path.stem}.*"):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.is_file():
            return candidate
    return None


def iter_images(folder: Path) -> Iterator[Path]:
    """按文件名排序遍历目录顶层的所有图片文件。"""
    if not folder.is_dir():
        return iter(())
    images = [p for p in folder.iterdir() if is_image_file(p)]
    images.sort(key=lambda p: p.name)
    return iter(images)


def iter_annotations(folder: Path) -> Iterator[Path]:
    """按文件名排序遍历目录顶层的所有 JSON 标注文件。"""
    if not folder.is_dir():
        return iter(())
    anns = [p for p in folder.iterdir() if is_annotation_file(p)]
    anns.sort(key=lambda p: p.name)
    return iter(anns)


def iter_matched_pairs(
        folder: Path,
        require_shapes: bool = False,
        sort_results: bool = True,
) -> Iterator[Tuple[Path, Path, dict]]:
    """
    遍历目录下与图片相匹配的 ``(image_path, annotation_path, data)`` 三元组。

    匹配规则：

    1. 标注文件 JSON 能被成功解析为 dict；
    2. 通过 :func:`resolve_image_path` 解析到的图片实际存在；
    3. 同一张图片（按 stem）只产出一次（标注文件按名称排序后取首个）。

    参数:
        folder: 待扫描的目录。
        require_shapes: 是否要求 ``shapes`` 非空（默认 ``False``）。
        sort_results: 是否按文件名排序后产出（默认 ``True``）。
            设为 ``False`` 可节省内存（不必一次性收集所有匹配对），
            但产出顺序为文件系统遍历的原始顺序。

    产出顺序按图片文件名升序（``sort_results=True`` 时）。
    """
    if not folder.is_dir():
        return

    annotations = sorted(
        (p for p in folder.iterdir() if is_annotation_file(p)),
        key=lambda p: p.name,
    )

    seen_stems: set = set()

    if not sort_results:
        for ann_path in annotations:
            data = load_annotation(ann_path)
            if data is None:
                continue
            if require_shapes and not data.get("shapes"):
                continue
            image_path = resolve_image_path(ann_path, data)
            if image_path is None:
                continue
            if image_path.stem in seen_stems:
                continue
            seen_stems.add(image_path.stem)
            yield (image_path, ann_path, data)
        return

    pairs: List[Tuple[Path, Path, dict]] = []
    for ann_path in annotations:
        data = load_annotation(ann_path)
        if data is None:
            continue
        if require_shapes and not data.get("shapes"):
            continue
        image_path = resolve_image_path(ann_path, data)
        if image_path is None:
            continue
        if image_path.stem in seen_stems:
            continue
        seen_stems.add(image_path.stem)
        pairs.append((image_path, ann_path, data))

    pairs.sort(key=lambda item: item[0].name)
    for item in pairs:
        yield item


def discover_trained_models(runs_dir: str) -> List[Tuple[str, str]]:
    """
    扫描 ``runs`` 目录下的训练模型。

    Ultralytics 默认结构为 ``runs/<train_name>/weights/<name>.pt``，本函数
    枚举该结构并返回 ``(显示名称, 权重绝对路径)`` 列表，显示名称形如
    ``训练名称-模型权重名称``（例如 ``first-best``）。

    本函数只依赖标准库，便于在轻量 GUI 进程中直接调用，避免引入
    ``ultralytics`` / ``torch`` / ``PIL`` 等重型依赖。

    参数:
        runs_dir: 训练结果根目录。

    返回:
        模型显示名称与模型文件路径的列表，按显示名升序排序。
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return []

    models: List[Tuple[str, str]] = []
    for train_dir in runs_path.iterdir():
        if not train_dir.is_dir():
            continue
        weights_dir = train_dir / "weights"
        if not weights_dir.is_dir():
            continue
        for weight_file in weights_dir.iterdir():
            if weight_file.is_file() and weight_file.suffix.lower() == ".pt":
                display_name = f"{train_dir.name}-{weight_file.stem}"
                models.append((display_name, str(weight_file)))

    models.sort(key=lambda item: item[0])
    return models
