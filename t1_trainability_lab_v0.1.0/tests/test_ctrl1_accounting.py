from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_u0c_ctrl1 import ADVANCE, COLLECT, update_causal_accounting, trace_success
from train_u0c_ctrl2 import reference_intervention_pass
from evaluate_u0c_ctrl2_g import comparison_category, diagnostic_counts


def test_forced_wrong_read_is_execution_error_not_controller_error() -> None:
    accounting = {"aligned": True, "first_control_error": None, "first_execution_error": None}
    first = update_causal_accounting(accounting, decision=0, action=ADVANCE, expected_action=ADVANCE, execution_ok=False)

    assert first["action_correct"] is True
    assert first["control_error"] is False
    assert first["execution_error"] is True
    assert accounting["first_control_error"] is None
    assert accounting["first_execution_error"] == {"decision": 0, "instruction": "executor_step"}
    assert accounting["aligned"] is False

    later = update_causal_accounting(accounting, decision=1, action=COLLECT, expected_action=ADVANCE, execution_ok=False)
    assert later["action_correct"] is None
    assert later["control_error"] is False
    assert later["execution_error"] is False
    assert accounting["first_control_error"] is None


def test_trace_success_requires_final_success_and_no_first_divergence() -> None:
    assert trace_success(True, False, None, None)
    assert not trace_success(True, False, {"decision": 1}, None)
    assert not trace_success(True, False, None, {"decision": 1})
    assert not trace_success(True, True, None, None)
    assert not trace_success(False, False, None, None)


def test_reference_intervention_rejects_permutation_of_three_present_classes() -> None:
    entries = [
        {"expected_action": "DECREASE", "predicted_action": "INCREASE"},
        {"expected_action": "KEEP", "predicted_action": "DECREASE"},
        {"expected_action": "INCREASE", "predicted_action": "KEEP"},
    ]
    assert not reference_intervention_pass(entries)


def test_contextual_regression_counts_when_canonical_is_correct() -> None:
    records = [{"canonical_correct": True, "decision_correct": False, "final_success": True}]
    assert comparison_category(True, False) == "contextual_regression"
    counts = diagnostic_counts(records)
    assert counts["contextual_regression_count"] == 1
    assert counts["decision_error_count"] == counts["shared_error_count"] + counts["contextual_regression_count"]


def test_execution_failure_does_not_count_as_comparator_sensitivity() -> None:
    records = [{"canonical_correct": True, "decision_correct": True, "final_success": False}]
    assert comparison_category(True, True) == "agreement_correct"
    counts = diagnostic_counts(records)
    assert counts["contextual_regression_count"] == 0
    assert counts["executor_or_transport_count"] == 1
