from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_omega_tbptt_directional_probe as runner  # noqa: E402


def tiny_factory() -> runner.base.OmegaCoreLMFast:
    return runner.base.fresh_model(runner.SEED, runner.K, vocab_size=31, dimension=4, slots=1)


def test_constants_are_frozen_probe_contract() -> None:
    assert runner.HORIZON == 16
    assert runner.K == 4
    assert runner.SEED == 20260913
    assert runner.SEGMENTS_PER_UPDATE == 16
    assert runner.PHYSICAL_BATCH == 8
    assert runner.WINDOW_TOKENS == 256


def test_tbptt_correctness_gate_compares_incremental_and_reference() -> None:
    gate = runner.tbptt_correctness_gate(model_factory=tiny_factory)
    assert gate["passed"] is True
    assert all(gate["checks"].values())
    assert gate["atol"] == pytest.approx(1e-5)
    assert gate["rtol"] == pytest.approx(1e-5)


def test_tbptt_update_accumulates_sixteen_backward_calls_and_one_step() -> None:
    model = tiny_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=runner.base.BASE_LR, betas=runner.base.ADAMW_BETAS, eps=runner.base.ADAMW_EPS)
    source = torch.randint(0, 31, (8, 257), generator=torch.Generator().manual_seed(3))
    state = model.initial_state(8, device=torch.device("cpu"))
    result = runner._tbptt_update(model, optimizer, source, state, backward_mode="incremental")
    assert result["state"].shape == (8, 1, 4)
    assert result["backward_seconds"] >= 0
    assert runner.SEGMENTS_PER_UPDATE == 16


def test_cost_report_ratio_and_self_hash() -> None:
    rows = [
        {"combination": runner.FULL_COMBINATION, "fresh_process": True, "updates_total": 10, "warmup_updates": 2, "measured_updates": 8, "total_seconds": 10.0, "peak_rss_bytes": 1000, "backward_seconds": 5.0, "student_forward_seconds": 2.0, "vocab_ce_seconds": 1.0, "clip_seconds": 0.1, "adamw_seconds": 0.1},
        {"combination": runner.TBPTT_COMBINATION, "fresh_process": True, "updates_total": 10, "warmup_updates": 2, "measured_updates": 8, "total_seconds": 8.0, "peak_rss_bytes": 700, "backward_seconds": 8.0, "student_forward_seconds": 3.0, "vocab_ce_seconds": 1.0, "clip_seconds": 0.1, "adamw_seconds": 0.1},
    ]
    report = runner.build_cost_report(rows)
    assert report["R_t"] == pytest.approx(0.8)
    assert runner.verify_self_hash(report)


def test_screening_thresholds_are_predeclared() -> None:
    assert runner.classify_screening(rt=0.8, delta_2000=0.04, rss_reduction=0.0)["final"] == "DIRECTIONAL-GO"
    assert runner.classify_screening(rt=0.8, delta_2000=0.08, rss_reduction=0.0)["final"] == "DIRECTIONAL-PROMISING-TRADEOFF"
    assert runner.classify_screening(rt=1.0, delta_2000=0.04, rss_reduction=0.0)["final"] == "DIRECTIONAL-NEUTRAL"
    assert runner.classify_screening(rt=1.1, delta_2000=0.04, rss_reduction=0.0)["final"] == "DIRECTIONAL-STOP"
    assert runner.classify_screening(rt=1.1, delta_2000=0.11, rss_reduction=0.5)["final"] == "DIRECTIONAL-STOP"
    assert runner.classify_screening(rt=0.9, delta_2000=0.08, rss_reduction=0.0)["final"] == "DIRECTIONAL-WEAK-TRADEOFF"
    assert runner.classify_screening(rt=1.0, delta_2000=0.08, rss_reduction=0.0)["final"] == "UNCLASSIFIED"
    assert runner.classify_screening(rt=1.1, delta_2000=0.08, rss_reduction=0.2)["final"] == "UNCLASSIFIED"


def test_rt_formula_canaries_use_tbptt_over_full() -> None:
    def row(full_seconds: float, tbptt_seconds: float) -> list[dict[str, object]]:
        common = {"fresh_process": True, "updates_total": 10, "warmup_updates": 2, "measured_updates": 8, "peak_rss_bytes": 1000, "backward_seconds": 1.0, "student_forward_seconds": 1.0, "vocab_ce_seconds": 1.0, "clip_seconds": 1.0, "adamw_seconds": 1.0}
        return [
            {"combination": runner.FULL_COMBINATION, "total_seconds": full_seconds, **common},
            {"combination": runner.TBPTT_COMBINATION, "total_seconds": tbptt_seconds, **common},
        ]

    fast = runner.build_cost_report(row(10.0, 8.0))
    slow = runner.build_cost_report(row(10.0, 11.0))
    boundary = runner.build_cost_report(row(10.0, 9.0))
    assert fast["R_t"] == pytest.approx(0.80)
    assert runner.classify_screening(rt=fast["R_t"], delta_2000=0.08, rss_reduction=0.0)["final"] == "DIRECTIONAL-PROMISING-TRADEOFF"
    assert slow["R_t"] == pytest.approx(1.10)
    assert runner.classify_screening(rt=slow["R_t"], delta_2000=0.04, rss_reduction=0.0)["technical"] == "TECH_NEGATIVE"
    assert boundary["R_t"] == pytest.approx(0.90)
    assert runner.classify_screening(rt=boundary["R_t"], delta_2000=0.08, rss_reduction=0.0)["final"] == "DIRECTIONAL-WEAK-TRADEOFF"


def test_frozen_full_bptt_curve_is_existing_ce_baseline() -> None:
    curve = runner.load_frozen_full_curve()
    assert [point["update"] for point in curve if point["update"] in runner.BOUNDARIES] == list(runner.BOUNDARIES)
    assert "omega_ce_only_baseline" in str(runner.BASELINE_DIR)


def test_real_execution_entrypoints_are_guarded() -> None:
    with pytest.raises(runner.RealExecutionAuthorizationError):
        runner.run_cost_preflight(Path("unused"), confirm_real_execution=False)
    with pytest.raises(runner.RealExecutionAuthorizationError):
        runner.run_tbptt_training(Path("unused"), confirm_real_execution=False, cost_report_path=Path("unused"))
