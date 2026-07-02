#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清除标注功能测试。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts.datasets.clear import clear_annotations, main as clear_main


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def _make_auto_annotation(make_annotation, json_path: Path, image_path: Path, *, corrected: bool) -> Path:
    auto_time = datetime(2026, 1, 1, 12, 0, 0)
    ann = make_annotation(
        json_path=json_path,
        image_path=image_path,
        shapes=[{"label": "cat", "shape_type": "rectangle", "points": [[1, 1], [10, 10]]}],
        auto_annotated_time=auto_time.isoformat(),
    )
    _set_mtime(ann, auto_time + (timedelta(seconds=10) if corrected else timedelta(seconds=0)))
    return ann


def test_clear_annotations_dry_run_does_not_delete(tmp_path, make_image, make_annotation):
    img = make_image(tmp_path / "a.jpg")
    ann = _make_auto_annotation(make_annotation, tmp_path / "a.json", img, corrected=False)

    result = clear_annotations(str(tmp_path), include_auto=True, dry_run=True)

    assert result["scanned"] == 1
    assert result["deleted"] == 0
    assert result["would_delete"] == 1
    assert result["dry_run"] is True
    assert ann.exists()


def test_clear_annotations_deletes_auto_only(tmp_path, make_image, make_annotation):
    auto_img = make_image(tmp_path / "auto.jpg")
    manual_img = make_image(tmp_path / "manual.jpg")
    auto_ann = _make_auto_annotation(make_annotation, tmp_path / "auto.json", auto_img, corrected=False)
    manual_ann = make_annotation(
        json_path=tmp_path / "manual.json",
        image_path=manual_img,
        shapes=[{"label": "dog", "shape_type": "rectangle", "points": [[1, 1], [10, 10]]}],
    )

    result = clear_annotations(str(tmp_path), include_auto=True)

    assert result["deleted"] == 1
    assert result["would_delete"] == 1
    assert not auto_ann.exists()
    assert manual_ann.exists()


def test_clear_annotations_deletes_auto_corrected(tmp_path, make_image, make_annotation):
    img = make_image(tmp_path / "a.jpg")
    ann = _make_auto_annotation(make_annotation, tmp_path / "a.json", img, corrected=True)

    result = clear_annotations(str(tmp_path), include_auto_corrected=True, tolerance_seconds=2.0)

    assert result["deleted"] == 1
    assert not ann.exists()


def test_clear_annotations_requires_scope(tmp_path):
    with pytest.raises(ValueError):
        clear_annotations(str(tmp_path))


def test_clear_cli_dry_run(tmp_path, make_image, make_annotation):
    img = make_image(tmp_path / "a.jpg")
    ann = _make_auto_annotation(make_annotation, tmp_path / "a.json", img, corrected=False)

    code = clear_main([
        "--input", str(tmp_path),
        "--include-auto",
        "--dry-run",
    ])

    assert code == 0
    assert ann.exists()
