from __future__ import annotations

import pytest

import omega_native_runtime_r2_canaries as canaries
import run_omega_native_runtime_r2_benchmark as benchmark


def test_synthetic_ratio_and_classification_canaries() -> None:
    result = canaries.run_canaries()
    assert result["status"] == "PASS"
    assert result["checks"]["ratio_10_8"] == 0.8
    assert result["checks"]["ratio_10_11"] == 1.1
    assert result["checks"]["joint_strong"] == "STRONG"
    assert result["checks"]["joint_pass"] == "PASS"
    assert result["checks"]["joint_neutral"] == "NEUTRAL"
    assert result["checks"]["joint_regression"] == "REGRESSION"
    assert result["checks"]["strong_k_blocker"] == "UNCLASSIFIED"
    assert result["checks"]["pass_k_blocker"] == "UNCLASSIFIED"
    assert result["checks"]["neutral_k_blocker"] == "REGRESSION"


def test_joint_policy_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        canaries.classify_joint(float("nan"), (1.0,))
    with pytest.raises(ValueError):
        canaries.classify_joint(0.8, ())
    with pytest.raises(ValueError):
        canaries.classify_joint(0.8, (float("inf"),))


def test_aggregation_reports_per_k_and_joint_ratios() -> None:
    def route(seconds: float) -> dict[str, object]:
        return {"updates": [{"timing_seconds": {"total_update": seconds}}]}

    report = benchmark.aggregate_performance({
        "pytorch-k1": route(10.0),
        "pytorch-k4": route(10.0),
        "native-k1": route(8.0),
        "native-k4": route(8.0),
    })
    assert report["r_k"] == {"K1": 0.8, "K4": 0.8}
    assert report["r_joint"] == 0.8
    assert report["classification"] == "STRONG"
