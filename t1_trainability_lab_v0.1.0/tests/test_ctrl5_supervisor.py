from __future__ import annotations

import inspect
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_u0c_ctrl4_preflight import COPY_E_R, EMIT, FETCH, FETCH_AND_ADJUST, INCREASE, oracle_action, dispatch_unified_action
from train_u0c_ctrl5 import GoalConditionedSupervisor, parameter_count


def test_goal_conditioned_supervisor_has_518_parameters() -> None:
    supervisor = GoalConditionedSupervisor()
    assert parameter_count(supervisor) == 518
    assert supervisor.goal_projection.weight.shape == (32, 2)


def test_zero_goal_projection_makes_both_goals_identical() -> None:
    supervisor = GoalConditionedSupervisor()
    with torch.no_grad():
        supervisor.goal_projection.weight.zero_()
    features = torch.randn(8, 7)
    fetch = torch.tensor([[1.0, 0.0]]).expand(8, -1)
    adjust = torch.tensor([[0.0, 1.0]]).expand(8, -1)
    assert torch.equal(supervisor(features, fetch), supervisor(features, adjust))


def test_dispatcher_signature_is_goal_agnostic() -> None:
    assert "goal" not in inspect.signature(dispatch_unified_action).parameters


def test_oracle_only_changes_after_copy_for_unequal_value() -> None:
    common = {"symbolic_pointer": 10, "goal_key": 10, "read_e_done": True, "copied": True, "symbolic_x": 7, "reference": 9}
    assert oracle_action(**common, goal=FETCH) == EMIT
    assert oracle_action(**common, goal=FETCH_AND_ADJUST) == INCREASE
    assert oracle_action(symbolic_pointer=10, goal_key=10, read_e_done=True, copied=False, symbolic_x=7, reference=9, goal=FETCH) == COPY_E_R
