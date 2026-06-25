#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train 包：模型训练相关 CLI 与实现。

本包遵循零副作用导入原则：导入 ``scripts.train`` 不会拉起
torch / ultralytics 等重依赖，相关依赖仅在调用具体训练函数时按需加载。
"""

__all__ = []
