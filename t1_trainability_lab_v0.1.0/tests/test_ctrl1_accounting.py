from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_u0c_ctrl1 import ADVANCE, COLLECT, update_causal_accounting, trace_success


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
