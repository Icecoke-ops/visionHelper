#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型训练参数解析与验证测试。"""

from __future__ import annotations

import pytest

from scripts.common.train_config import TrainConfig
from scripts.api import TrainingAPI
from scripts.train.train import _resolve_model_name


def test_resolve_model_name_appends_task_suffix_only_when_missing():
    assert _resolve_model_name("yolov8n", "segment") == "yolov8n-seg.pt"
    assert _resolve_model_name("yolov8n-seg", "segment") == "yolov8n-seg.pt"


def test_resolve_model_name_uses_precise_suffix_check():
    """包含 -seg 子串但不以后缀结尾的名称仍应补全 segment 后缀。"""
    assert _resolve_model_name("my-segment-model", "segment") == "my-segment-model-seg.pt"


def test_train_model_config_allows_auto_batch_minus_one(tmp_path, monkeypatch):
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/test\n", encoding="utf-8")

    captured = {}

    def fake_train_model(cfg):
        captured["batch"] = cfg.batch
        return "runs/train"

    monkeypatch.setattr("scripts.train.train.train_model", fake_train_model)

    result = TrainingAPI.train_model_config(
        TrainConfig(
            dataset_yaml=str(dataset_yaml),
            project=str(tmp_path / "runs"),
            name="train",
            batch=-1,
        )
    )

    assert result == "runs/train"
    assert captured["batch"] == -1


@pytest.mark.parametrize("batch", [0, -2])
def test_train_model_config_rejects_invalid_batch_values(tmp_path, batch):
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="batch"):
        TrainingAPI.train_model_config(
            TrainConfig(
                dataset_yaml=str(dataset_yaml),
                project=str(tmp_path / "runs"),
                name="train",
                batch=batch,
            )
        )
