# -*- coding: utf-8 -*-
"""
scripts 包：visionHelper 后端核心能力。

本包遵循零副作用导入原则：直接 ``import scripts`` 不会拉起 torch、
ultralytics、cv2、transformers 等重依赖。各子包的具体能力按需导入。
"""

__all__ = []
