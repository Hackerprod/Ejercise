from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ctrl2_common import load_executor
from train_u0c_ctrl2_o import DIMENSION, DECREASE, INCREASE, KEEP, OrdinalSharedScorer, action_from_difference, codebook_distinctness, parameter_count


def test_parameter_count_is_exactly_4225() -> None:
    assert parameter_count(OrdinalSharedScorer()) == 4225


def test_shared_scorer_inverts_arguments() -> None:
    torch.manual_seed(1)
    scorer = OrdinalSharedScorer()
    register = torch.randn(3, DIMENSION)
    reference = torch.randn(3, DIMENSION)
    torch.testing.assert_close(scorer.difference(register, reference), -scorer.difference(reference, register))


def test_identical_inputs_are_keep() -> None:
    torch.manual_seed(2)
    scorer = OrdinalSharedScorer()
    values = torch.randn(4, DIMENSION)
    assert torch.equal(scorer.predict_action(values, values), torch.full((4,), KEEP, dtype=torch.long))


def test_exact_tau_boundaries_are_keep() -> None:
    tau = torch.tensor(1.0)
    difference = torch.tensor([-1.0, 1.0, -1.000001, 1.000001])
    assert torch.equal(action_from_difference(difference, tau), torch.tensor([KEEP, KEEP, DECREASE, INCREASE]))


def test_same_scorer_receives_gradients_from_both_roles() -> None:
    torch.manual_seed(3)
    scorer = OrdinalSharedScorer()
    register = torch.randn(5, DIMENSION, requires_grad=True)
    reference = torch.randn(5, DIMENSION, requires_grad=True)
    # The full logit sum cancels d because logits are (d, tau, -d). Use a
    # weighted shared-score objective so both role paths contribute gradients.
    (scorer.score(register).sum() + 2.0 * scorer.score(reference).sum()).backward()
    assert scorer.network[0].weight.grad is not None
    assert torch.count_nonzero(scorer.network[0].weight.grad).item() > 0
    assert torch.count_nonzero(register.grad).item() > 0
    assert torch.count_nonzero(reference.grad).item() > 0


def test_optimizer_contains_only_new_scorer_parameters() -> None:
    scorer = OrdinalSharedScorer()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=0.0)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in scorer.parameters()}
    assert len(optimized) == 4


def test_logits_use_exact_d_tau_minus_d_order() -> None:
    torch.manual_seed(4)
    scorer = OrdinalSharedScorer()
    register = torch.randn(2, DIMENSION)
    reference = torch.randn(2, DIMENSION)
    difference = scorer.difference(register, reference)
    logits = scorer.logits(register, reference)
    expected = torch.stack((difference, scorer.tau().expand_as(difference), -difference), dim=-1)
    torch.testing.assert_close(logits, expected)


def test_frozen_executor_value_codebook_rows_are_distinct() -> None:
    result = codebook_distinctness(load_executor())
    assert result["all_distinct"]
    assert result["min_pairwise_l2"] > 0.0
