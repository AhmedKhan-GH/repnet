"""Tests for the training entrypoint's parameterization.

`train.main` must accept a split count, an output directory, and a write
toggle so a notebook can drive the *release* training code with a small,
configurable number of splits. These tests stub out the dataset and the
per-split training so they run fast and require no PHI data.
"""
import json
import os

import numpy as np

from src import train


def _canned_metrics():
    """A per-split metrics dict with every key main() reads downstream."""
    m = {"auroc": 0.7, "auprc": 0.3, "seconds": 0.0, "best_val_auroc": 0.7}
    for thr in ("youden", "sens80"):
        for metric in ("sens", "spec", "prec", "acc", "f1", "npv", "threshold"):
            m[f"{thr}_{metric}"] = 0.5
    return m


def _patch_training(monkeypatch):
    """Replace dataset loading + per-split training with cheap fakes."""
    X = np.zeros((8, 12, 10), dtype=np.float32)
    y = np.array([0, 1] * 4)
    groups = np.arange(8)
    monkeypatch.setattr(train, "load_dataset", lambda: (X, y, groups))
    monkeypatch.setattr(
        train, "train_val_test_split",
        lambda i, y, groups: (np.array([0, 1]), np.array([2]), np.array([3])))
    import torch
    monkeypatch.setattr(
        train, "train_one_split",
        lambda *a, **k: (_canned_metrics(), {"w": torch.zeros(1)}))


def test_main_runs_requested_number_of_splits(tmp_path, monkeypatch):
    _patch_training(monkeypatch)
    summary, rows = train.main(n_splits=2, out_dir=str(tmp_path), write=True)
    assert summary["n_splits"] == 2
    assert len(rows) == 2
    assert [r["split"] for r in rows] == [0, 1]


def test_main_writes_release_artifacts(tmp_path, monkeypatch):
    _patch_training(monkeypatch)
    train.main(n_splits=2, out_dir=str(tmp_path), write=True)
    for fname in ("summary.json", "per_split.json", "best_model.pt", "median_model.pt"):
        assert os.path.isfile(tmp_path / fname), f"missing {fname}"
    with open(tmp_path / "summary.json") as f:
        assert json.load(f)["n_splits"] == 2


def test_main_write_false_produces_no_files(tmp_path, monkeypatch):
    _patch_training(monkeypatch)
    train.main(n_splits=1, out_dir=str(tmp_path), write=False)
    assert os.listdir(tmp_path) == []
