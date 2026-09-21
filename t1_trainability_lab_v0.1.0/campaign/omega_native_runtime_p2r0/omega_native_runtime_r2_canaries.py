"""Pure synthetic R2 performance-ratio canaries.

This module intentionally imports no torch, corpus, model, DLL, or benchmark
runner. Lower candidate/baseline ratios are better for this performance gate.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


STRONG_LIMIT = 0.80
STOP_LIMIT = 1.25
JOINT_PASS_LIMIT = 0.90
JOINT_NEUTRAL_LIMIT = 1.05
K_PASS_LIMIT = 1.05
K_NEUTRAL_LIMIT = 1.10


def _validate_ratio(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def candidate_baseline_ratio(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        raise ValueError("candidate and baseline must be finite")
    if candidate < 0.0 or baseline <= 0.0:
        raise ValueError("candidate must be non-negative and baseline positive")
    return candidate / baseline


def classify_ratio(ratio: float) -> str:
    """Classify one K using joint policy with neutral per-K ratio 1.0."""
    return classify_joint(ratio, (1.0,))


def classify_joint(r_joint: float, k_ratios: Sequence[float]) -> str:
    """Apply Addendum 240 policy with deterministic precedence.

    The exact rules do not cover every tradeoff combination. Those combinations
    return UNCLASSIFIED instead of being silently assigned a stronger outcome.
    """
    joint = _validate_ratio(r_joint, "r_joint")
    if not k_ratios:
        raise ValueError("k_ratios must not be empty")
    ratios = tuple(_validate_ratio(float(value), "K ratio") for value in k_ratios)
    maximum = max(ratios)
    if joint <= STRONG_LIMIT and maximum <= K_PASS_LIMIT:
        return "STRONG"
    if joint <= JOINT_PASS_LIMIT and maximum <= K_PASS_LIMIT:
        return "PASS"
    if JOINT_PASS_LIMIT < joint <= JOINT_NEUTRAL_LIMIT and maximum <= K_NEUTRAL_LIMIT:
        return "NEUTRAL"
    if joint > JOINT_NEUTRAL_LIMIT or maximum > K_NEUTRAL_LIMIT:
        return "REGRESSION"
    return "UNCLASSIFIED"


def should_stop(ratio: float) -> bool:
    return _validate_ratio(ratio, "ratio") > STOP_LIMIT


def run_canaries() -> dict[str, Any]:
    checks = {
        "ratio_10_8": candidate_baseline_ratio(8.0, 10.0),
        "ratio_10_11": candidate_baseline_ratio(11.0, 10.0),
        "stop_at_1_25": should_stop(1.25),
        "stop_above_1_25": should_stop(1.2500001),
        "joint_strong": classify_joint(0.80, (1.05, 1.00)),
        "joint_pass": classify_joint(0.90, (1.05, 1.00)),
        "joint_neutral": classify_joint(0.91, (1.10, 1.00)),
        "joint_regression": classify_joint(1.05 + 1.0e-6, (1.00, 1.00)),
        "strong_k_blocker": classify_joint(0.80, (1.050001, 1.00)),
        "pass_k_blocker": classify_joint(0.90, (1.050001, 1.00)),
        "neutral_k_blocker": classify_joint(0.95, (1.100001, 1.00)),
        "unclassified_tradeoff": classify_joint(0.85, (1.06, 1.00)),
    }
    expected = {
        "ratio_10_8": 0.8,
        "ratio_10_11": 1.1,
        "stop_at_1_25": False,
        "stop_above_1_25": True,
        "joint_strong": "STRONG",
        "joint_pass": "PASS",
        "joint_neutral": "NEUTRAL",
        "joint_regression": "REGRESSION",
        "strong_k_blocker": "UNCLASSIFIED",
        "pass_k_blocker": "UNCLASSIFIED",
        "neutral_k_blocker": "REGRESSION",
        "unclassified_tradeoff": "UNCLASSIFIED",
    }
    passed = all(checks[name] == value for name, value in expected.items())
    return {"status": "PASS" if passed else "FAIL", "checks": checks, "expected": expected}


if __name__ == "__main__":
    result = run_canaries()
    print(result)
    raise SystemExit(0 if result["status"] == "PASS" else 1)
