#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 目录与文件夹名称统一配置。

集中管理项目中使用的子目录名称，避免在多处硬编码。
所有需要拼接具体工作目录路径的地方，应导入本模块中的常量后
通过 ``Path(work_dir) / FOLDER_NAME`` 的方式使用。

本模块同时提供与"打包独立 GUI、脚本保持源码"部署模式相关的
运行期工具函数：

- :func:`is_frozen`：判断当前是否处于 PyInstaller 等冻结打包态。
- :func:`app_root`：返回 ``scripts/`` 包所在的父目录，子进程启动时
  会被设置为工作目录或注入到 ``PYTHONPATH``，以保证 ``-m scripts.xxx``
  能正确定位到脚本包。
"""

import os
import sys
from pathlib import Path

# 图片文件夹名称：视频抽帧、图片去重等默认输出的图片目录
IMAGES_FOLDER = "images"

# 数据集文件夹名称：YOLO 数据集导出目录
DATASET_FOLDER = "dataset"

# 训练结果文件夹名称：模型训练结果保存目录
TRAIN_FOLDER = "runs"

# 回收站文件夹名称：图片去重时移动重复图片的目标目录
RECYCLE_BIN_FOLDER = "recycle_bin"


def is_frozen() -> bool:
    """判断当前进程是否处于 PyInstaller / cx_Freeze 等冻结打包态。"""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """返回应用根目录，即 ``scripts/`` 包所在的父目录。

    - 开发态：返回仓库根目录 ``<repo>/``，由 ``gui/config.py`` 文件位置
      推断（``parents[1]``）。
    - 打包态（``sys.frozen``）：返回可执行文件所在目录。打包发布物形如::

          dist/visionHelper/
          ├── visionHelper(.exe)        ← sys.executable
          └── scripts/                  ← 与 exe 同级的脚本源码目录

      因此使用 ``Path(sys.executable).parent`` 即可得到 ``scripts/`` 的
      父目录。

    - 允许通过环境变量 ``VISIONHELPER_APP_ROOT`` 强制覆盖，便于自定义部署
      布局或调试。
    """
    override = os.environ.get("VISIONHELPER_APP_ROOT", "").strip()
    if override:
        return Path(override).resolve()

    if is_frozen():
        return Path(sys.executable).resolve().parent

    # gui/config.py -> gui/ -> 仓库根
    return Path(__file__).resolve().parents[1]


def scripts_dir() -> Path:
    """返回 ``scripts/`` 目录的绝对路径。"""
    return app_root() / "scripts"

