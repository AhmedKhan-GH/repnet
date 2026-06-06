"""Tests for the training-time benchmark.

The benchmark must run on synthetic data (no PHI dataset required) and report
positive timing for whatever device is selected. CPU is always available, so
the synthetic CPU benchmark is the portable smoke test.
"""
from src.benchmark import resolve_device, run_benchmark


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_valid():
    dev = resolve_device(None)
    assert dev.type in {"cpu", "cuda", "mps"}


def test_synthetic_benchmark_cpu():
    r = run_benchmark(device="cpu", epochs=1, n_train=64, n_eval=32,
                      batch_size=16, augment=False, synthetic=True)
    assert r["device"] == "cpu"
    assert r["params"] == 29490
    assert r["seconds_per_epoch"] > 0
    assert r["train_samples_per_sec"] > 0
    assert r["eval_samples_per_sec"] > 0
    # projections are derived from the measured per-epoch time
    assert r["est_seconds_per_split"] > 0
    assert r["est_seconds_full_run"] > r["est_seconds_per_split"]
