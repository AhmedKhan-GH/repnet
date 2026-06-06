"""Reproduction test: the shipped checkpoints must reproduce the recorded metrics.

This is the core reproducibility guarantee. It re-evaluates the bundled
`best_model.pt` and `median_model.pt` against the exact patient-grouped test
splits and asserts the numbers match `per_split.json` / `summary.json` to 1e-4.

Skipped automatically if the (PHI) dataset is not present locally.

Run:  python -m pytest tests/ -v        (from the package root)
"""
import json
import os

import numpy as np
import pytest

from src.data import DEFAULT_DATA_DIR, load_dataset
from src.evaluate import evaluate_checkpoint_on_split

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PKG_ROOT, "results", "multisplit_dbb6f49")
TOL = 1e-4

_METRIC_KEYS = [
    "auroc", "auprc",
    "youden_sens", "youden_spec", "youden_prec", "youden_npv", "youden_f1",
    "sens80_sens", "sens80_spec", "sens80_prec", "sens80_npv", "sens80_f1",
]

data_missing = not os.path.isdir(os.path.join(DEFAULT_DATA_DIR, "ekg_data"))
skip_no_data = pytest.mark.skipif(
    data_missing, reason=f"dataset not found at {DEFAULT_DATA_DIR} (PHI, not shipped)")


def _load_per_split():
    with open(os.path.join(RESULTS_DIR, "per_split.json")) as f:
        return json.load(f)


def _load_summary():
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        return json.load(f)


def test_data_dir_resolved_at_call_time(tmp_path, monkeypatch):
    """The loader must honour REPNET_DATA_DIR set AFTER import.

    Regression: the default data dir was bound at import time, so setting
    REPNET_DATA_DIR in a notebook (after `import src.data`) was silently
    ignored and the loader looked in the stale package-relative path.
    """
    import src.data as data

    bogus = tmp_path / "my_data_dir"
    bogus.mkdir()
    monkeypatch.setenv("REPNET_DATA_DIR", str(bogus))

    with pytest.raises(FileNotFoundError) as excinfo:
        data.load_ecg_data()

    # The error must reference the env-var dir, not the import-time default.
    assert str(bogus) in str(excinfo.value)


def test_missing_data_dir_raises_helpful_error(tmp_path, monkeypatch):
    """A missing/empty data dir must fail fast with actionable guidance.

    Pointing REPNET_DATA_DIR at a directory without metadata.csv / ekg_data
    should raise immediately with a message that names the directory, the
    REPNET_DATA_DIR env var, and DATA.md — not a deep pandas/os traceback.
    """
    import src.data as data

    empty = tmp_path / "no_data_here"
    empty.mkdir()
    monkeypatch.setenv("REPNET_DATA_DIR", str(empty))

    with pytest.raises(FileNotFoundError) as excinfo:
        data.load_ecg_data()

    msg = str(excinfo.value)
    assert str(empty) in msg          # names the offending directory
    assert "REPNET_DATA_DIR" in msg   # tells the user the knob to set
    assert "DATA.md" in msg           # points at the format spec


def test_aggregate_self_consistency():
    """summary.json mean/std must equal the mean/std of per_split.json."""
    per_split = _load_per_split()
    summary = _load_summary()
    aurocs = [s["auroc"] for s in per_split]
    auprcs = [s["auprc"] for s in per_split]
    assert len(per_split) == summary["n_splits"] == 30
    assert abs(np.mean(aurocs) - summary["auroc_mean"]) < 1e-6
    assert abs(np.std(aurocs) - summary["auroc_std"]) < 1e-6
    assert abs(np.mean(auprcs) - summary["auprc_mean"]) < 1e-6
    assert abs(np.std(auprcs) - summary["auprc_std"]) < 1e-6


@skip_no_data
@pytest.mark.parametrize("checkpoint", ["best_model.pt", "median_model.pt"])
def test_checkpoint_reproduces_recorded_metrics(checkpoint):
    """Re-evaluating a shipped checkpoint reproduces its recorded split metrics."""
    per_split = _load_per_split()
    aurocs = [s["auroc"] for s in per_split]
    if checkpoint == "best_model.pt":
        split_i = int(np.argmax(aurocs))
    else:  # median model: the split at the median of the sorted AUROCs
        split_i = int(np.argsort(aurocs)[len(aurocs) // 2])

    X, y, groups = load_dataset()
    repro = evaluate_checkpoint_on_split(
        split_i, os.path.join(RESULTS_DIR, checkpoint), X, y, groups)
    recorded = per_split[split_i]

    for key in _METRIC_KEYS:
        assert abs(repro[key] - recorded[key]) < TOL, (
            f"{checkpoint} split {split_i} {key}: "
            f"reproduced {repro[key]:.6f} != recorded {recorded[key]:.6f}")
