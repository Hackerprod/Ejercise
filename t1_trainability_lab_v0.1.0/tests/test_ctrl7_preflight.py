from __future__ import annotations

from pathlib import Path
import inspect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_u0c_ctrl4_preflight import INCREASE, dispatch_unified_action
from evaluate_u0c_ctrl7_preflight import oracle_action, target_value


def test_ctrl7_terminal_target_does_not_react_to_transient_crossing() -> None:
    assert target_value(10, 14, 12, (1, 1)) == 14
    assert oracle_action(10, 10, read_e_done=True, copied=True, value=12, lower=14, forbidden=12, constraints=(1, 1)) == INCREASE


def test_ctrl7_interaction_requires_second_increase_at_terminal_floor() -> None:
    assert target_value(10, 12, 12, (1, 1)) == 13
    assert oracle_action(10, 10, read_e_done=True, copied=True, value=12, lower=12, forbidden=12, constraints=(1, 1)) == INCREASE


def test_dispatcher_signature_is_ctrl7_agnostic() -> None:
    parameters = inspect.signature(dispatch_unified_action).parameters
    assert not {"constraints", "lower", "forbidden", "floor", "avoid"}.intersection(parameters)
