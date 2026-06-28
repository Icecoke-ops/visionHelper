#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/predict 包：模型预测/推理相关 CLI 与实现。

本包遵循零副作用导入原则，所有重型依赖在方法体内懒导入。

功能：
    - 使用训练好的 YOLO 模型对图片或视频进行预测
    - 支持 detect / obb / segment / classify 4 种任务
    - 视频预测带进度条显示
    - 结果保存为可视化后的图片/视频
"""

__all__ = []
