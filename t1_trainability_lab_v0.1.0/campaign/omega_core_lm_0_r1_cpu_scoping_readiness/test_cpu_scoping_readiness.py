"""Lightweight contract tests for CPU-only OMEGA R1 scoping."""

from __future__ import annotations

import json
from pathlib import Path

from run_cpu_scoping_readiness import (
    COMPILE_ABS_TOLERANCE,
    COMPILE_MODES,
    CONFIGS,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    EXECUTION_ORDER,
    PROFILE_PHASES,
    STAGE2_UPDATES,
    STAGE3_UPDATES,
    _compile_equivalence,
    _compile_failure_record,
    _pair_records,
    run_stage4,
)


SOURCE = Path(__file__).with_name("run_cpu_scoping_readiness.py")


def test_stage1_uses_inference_mode_and_forbids_training_work() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "with torch.inference_mode():" in text
    assert "forward_token" in text and "forward_window" in text
    assert '"teacher_used": False' in text
    assert '"loss_used": False' in text
    assert '"backward_used": False' in text


def test_stage2_contract_has_all_configs_and_six_update_pair_schedule() -> None:
    assert CONFIGS == (("A", 2), ("B", 4), ("C", 8))
    assert [8 // physical_batch for _, physical_batch in CONFIGS] == [4, 2, 1]
    assert STAGE2_UPDATES == 6
    updates = [{"update": index, "window": index % 2, "elapsed_seconds": 1.0, "valid_tokens": 2048} for index in range(6)]
    pairs = _pair_records(updates)
    assert [pair["updates"] for pair in pairs] == [[0, 1], [2, 3], [4, 5]]
    assert pairs[0]["phase"] == "warmup"
    assert all(pair["phase"] == "measured" for pair in pairs[1:])


def test_stage3_profile_keys_are_present_per_update_and_pair() -> None:
    updates = [{"update": index, "window": index % 2, "elapsed_seconds": 1.0, "valid_tokens": 2048, "phase_timings": {key: float(index + 1) for key in PROFILE_PHASES}} for index in range(4)]
    pairs = _pair_records(updates)
    assert set(updates[0]["phase_timings"]) == set(PROFILE_PHASES)
    assert all(set(pair["phase_timings"]) == set(PROFILE_PHASES) for pair in pairs)
    assert pairs[0]["phase_timings"]["teacher_forward_seconds"] == 3.0
    assert pairs[1]["phase_timings"]["optimizer_step_seconds"] == 7.0


def test_compile_tolerance_policy_is_predeclared_and_not_relaxed() -> None:
    assert COMPILE_MODES == ("eager", "default", "max-autotune")
    assert COMPILE_ABS_TOLERANCE == 1e-5
    assert _compile_equivalence({"loss": COMPILE_ABS_TOLERANCE, "state": 0.0}) == "PASS"
    assert _compile_equivalence({"loss": COMPILE_ABS_TOLERANCE + 1e-7}) == "PERFORMANCE_INTERESTING/SCIENTIFIC_SCOPING_INELIGIBLE"


def test_stage_order_keeps_compile_after_real_stage3_and_before_projection() -> None:
    assert EXECUTION_ORDER == (1, 2, 3, "3-profile", "3-compile", 4)
    text = SOURCE.read_text(encoding="utf-8")
    assert text.index("run_stage3(") < text.index("run_compile_comparison(") < text.index("run_stage4(")


def test_compile_failure_is_typed_and_fail_closed() -> None:
    record = _compile_failure_record(variant_name="shared_K1", mode="default", physical_batch=2, error=RuntimeError("compiler unavailable"), compile_seconds=0.25)
    assert record["status"] == "failed"
    assert record["failure_type"] == "RuntimeError"
    assert record["failure"] == "compiler unavailable"
    assert record["first_execution_seconds"] is None
    assert record["updates_completed"] == 0


def test_stage3_pins_exact_revisions_and_eight_document_contract() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert DATASET_ID == "Salesforce/wikitext"
    assert DATASET_CONFIG == "wikitext-2-raw-v1"
    assert DATASET_REVISION == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert MODEL_ID == "distilbert/distilgpt2"
    assert MODEL_REVISION == "2290a62682d06624634c1f46a6ad5be0f47f38aa"
    assert "DownloadConfig(local_files_only=True)" in text
    assert "local_files_only=True" in text
    assert "APPROVED_SELECTION_MANIFEST[\"selected_documents\"]" in text
    assert "exact approved eight-document manifest" in text
    assert STAGE3_UPDATES == 4


def test_projection_is_blocked_without_real_teacher_times() -> None:
    result = run_stage4({"status": "blocked"})
    assert result["status"] == "blocked"
    assert result["classification"] == "BLOCKED_STAGE3"


def test_projection_math_sums_variant_times_and_counts_8000_updates() -> None:
    stage3 = {
        "status": "completed",
        "variants": [
            {"variant": "shared_K1", "updates": [{"elapsed_seconds": 10.0}] * 4},
            {"variant": "shared_K4", "updates": [{"elapsed_seconds": 20.0}] * 4},
        ],
    }
    result = run_stage4(stage3)
    assert result["total_updates"] == 8000
    assert result["seconds_per_complete_seed_run"] == 60000.0
    assert result["total_seconds"] == 120000.0
    assert result["hours"] == 120000.0 / 3600.0
    assert result["days"] == 120000.0 / 86400.0
    assert "seeds * updates_per_variant * (t_shared_K1 + t_shared_K4) / 3600" in result["formula"]


def test_no_bf16_amp_or_precision_reduction_policy() -> None:
    text = SOURCE.read_text(encoding="utf-8").lower()
    assert "bfloat16" not in text
    assert "autocast" not in text
    assert '"bf16": false' in text
    assert '"amp": false' in text
