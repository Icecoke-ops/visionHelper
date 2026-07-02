#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 子进程参数构造工具测试。"""

from __future__ import annotations

from pathlib import Path

from gui.utils._proc import build_script_argv, infer_script_name, join_for_display


def test_build_script_argv_converts_options_to_cli_args():
    argv = build_script_argv(
        "datasets",
        "export",
        input="images",
        output=Path("dataset"),
        train_ratio=0.8,
        export_empty_labels=True,
        export_unlabeled=False,
        suffix="",
        seed=None,
    )

    assert argv == [
        "-m",
        "scripts.vh",
        "datasets",
        "export",
        "--input",
        "images",
        "--output",
        "dataset",
        "--train-ratio",
        "0.8",
        "--export-empty-labels",
    ]


def test_build_script_argv_for_predict_run():
    argv = build_script_argv(
        "predict",
        "run",
        model="best.pt",
        input="image.jpg",
        output="predict",
        threshold="0.2500",
        task="detect",
        iou="0.4500",
    )

    assert argv[:4] == ["-m", "scripts.vh", "predict", "run"]
    assert "--model" in argv
    assert "best.pt" in argv
    assert infer_script_name(argv) == "predict_run"


def test_infer_script_name_supports_legacy_script_path():
    assert infer_script_name(["scripts/vh.py", "images", "dedup"]) == "images_dedup"


def test_join_for_display_quotes_paths_with_spaces():
    command = join_for_display(["-m", "scripts.vh", "predict", "run", "--input", "my images/a.jpg"])

    assert "'my images/a.jpg'" in command or '"my images/a.jpg"' in command
