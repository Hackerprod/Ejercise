from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ctrl2_common import load_ctrl1, load_executor
from train_u0c_ctrl2_o import OrdinalSharedScorer
from train_u0c_ctrl4 import SupervisorMLP, parameter_count, supervisor_features
from train_u0c_ctrl1 import SLOT_COUNT, SLOT_P, SLOT_R
from evaluate_u0c_c1_e_r_alu import DIMENSION

SCORER_CHECKPOINT = Path(__file__).resolve().parents[1] / "campaign" / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"


def test_supervisor_has_exactly_454_parameters() -> None:
    assert parameter_count(SupervisorMLP()) == 454


def test_uninitialized_r_uses_neutral_comparison_features() -> None:
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer = OrdinalSharedScorer()
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = executor.token_embedding(torch.tensor([0]))
    goal = executor.token_embedding(torch.tensor([0]))
    features = supervisor_features(executor, ctrl1, scorer, state, goal, 0, v_e=False, v_r=False)
    assert torch.equal(features[:, 2:5], torch.zeros((1, 3)))
    assert torch.equal(features[:, 5:], torch.zeros((1, 2)))


def test_supervisor_backward_does_not_reach_frozen_recognizers() -> None:
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer = OrdinalSharedScorer()
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = executor.token_embedding(torch.tensor([0]))
    goal = executor.token_embedding(torch.tensor([0]))
    features = supervisor_features(executor, ctrl1, scorer, state, goal, 0, v_e=False, v_r=False)
    supervisor = SupervisorMLP()
    supervisor(features).sum().backward()
    assert all(parameter.grad is None for parameter in executor.parameters())
    assert all(parameter.grad is None for parameter in ctrl1.parameters())
    assert all(parameter.grad is None for parameter in scorer.parameters())
    assert any(parameter.grad is not None for parameter in supervisor.parameters())


def test_supervisor_backward_does_not_reach_frozen_scorer_when_r_is_available() -> None:
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer = OrdinalSharedScorer()
    payload = torch.load(SCORER_CHECKPOINT, map_location="cpu", weights_only=False)
    scorer.load_state_dict(payload["controller"], strict=True)
    state = torch.zeros((1, SLOT_COUNT, DIMENSION))
    state[:, SLOT_P] = executor.token_embedding(torch.tensor([0]))
    state[:, SLOT_R] = executor.token_embedding(torch.tensor([288]))
    goal = executor.token_embedding(torch.tensor([0]))
    features = supervisor_features(executor, ctrl1, scorer, state, goal, 0, v_e=True, v_r=True)
    supervisor = SupervisorMLP()
    supervisor(features).sum().backward()
    assert torch.count_nonzero(features[:, 2:5]).item() > 0
    assert all(parameter.grad is None for parameter in scorer.parameters())
    assert any(parameter.grad is not None for parameter in supervisor.parameters())
