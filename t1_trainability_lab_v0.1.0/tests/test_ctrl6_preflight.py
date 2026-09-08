from __future__ import annotations

from pathlib import Path
import inspect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, INCREASE, READ_E, dispatch_unified_action
from evaluate_u0c_ctrl6_preflight import GOALS, oracle_action, target_value


def test_ctrl6_truth_table() -> None:
    assert target_value(10, 12, 20, (0, 0)) == 10
    assert target_value(10, 12, 20, (1, 0)) == 12
    assert target_value(23, 12, 20, (0, 1)) == 20
    assert target_value(10, 12, 20, (1, 1)) == 12
    assert target_value(23, 12, 20, (1, 1)) == 20
    assert set(GOALS) == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_ctrl6_oracle_composes_only_after_copy() -> None:
    assert oracle_action(10, 10, read_e_done=False, copied=False, value=10, lower=12, upper=20, constraints=(1, 1)) == READ_E
    assert oracle_action(10, 10, read_e_done=True, copied=False, value=10, lower=12, upper=20, constraints=(1, 1)) == COPY_E_R
    assert oracle_action(10, 10, read_e_done=True, copied=True, value=10, lower=12, upper=20, constraints=(1, 1)) == INCREASE
    assert oracle_action(10, 10, read_e_done=True, copied=True, value=12, lower=12, upper=20, constraints=(1, 1)) == EMIT


def test_dispatcher_signature_is_constraint_agnostic() -> None:
    parameters = inspect.signature(dispatch_unified_action).parameters
    assert "goal" not in parameters
    assert "constraints" not in parameters
    assert "lower" not in parameters
    assert "upper" not in parameters
