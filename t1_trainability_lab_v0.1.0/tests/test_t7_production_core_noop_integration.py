from __future__ import annotations

import inspect

import torch

from t1_trainability.t7_production_core_noop_integration import (
    T7ProductionCoreNoopIntegration,
    noop_difference_matrix,
    noop_none_loss,
)


def test_real_production_core_forward_has_34_contexts_and_four_queries() -> None:
    core = T7ProductionCoreNoopIntegration()
    scores = core.noop_candidate_scores()
    assert scores.shape == (4, 34)
    assert torch.unique(scores[0, :32]).numel() > 1


def test_real_production_core_backward_reaches_only_appended_noop_row() -> None:
    core = T7ProductionCoreNoopIntegration()
    thresholds = torch.tensor([0.2, 0.4, 0.6, 0.8])
    scores = core.noop_candidate_scores()
    scores.retain_grad()
    loss = torch.nn.functional.softplus(scores - thresholds[:, None]).mean()
    loss.backward()
    assert core.core.embedding.noop_embedding.grad is not None
    assert float(core.core.embedding.noop_embedding.grad.norm()) > 0.0
    assert all(not parameter.requires_grad for name, parameter in core.core.named_parameters() if name != "embedding.noop_embedding")


def test_real_production_core_old_path_is_bit_exact_after_noop_change() -> None:
    core = T7ProductionCoreNoopIntegration()
    ids = torch.tensor([[0, 1], [4, 33], [35, 2]], dtype=torch.long)
    lengths = torch.full((3,), 2, dtype=torch.long)
    before = tuple(value.detach().clone() for value in core(ids, lengths)["scores"])
    old_ids = torch.arange(37, dtype=torch.long)
    old_snapshot = core.core.embedding.old_embeddings.detach().clone()
    with torch.no_grad():
        core.core.embedding.noop_embedding.add_(19.0)
    after = tuple(value.detach() for value in core(ids, lengths)["scores"])
    assert all(float((left - right).abs().max()) == 0.0 for left, right in zip(before, after))
    assert torch.equal(old_ids, torch.arange(37, dtype=torch.long))
    assert torch.equal(old_snapshot, core.core.embedding.old_embeddings)


def test_integration_uses_hashed_production_forward_without_shortcut() -> None:
    source = inspect.getsource(core_forward := core_forward_source())
    assert "self.embedding(token_ids)" in source
    assert "self.local_norm(self.local_binding" in source
    assert "OP_NOOP" not in source
    assert "return NONE" not in source
    assert "active_roles" not in source


def test_loss_shape_is_exactly_136_terms() -> None:
    core = T7ProductionCoreNoopIntegration()
    thresholds = torch.tensor([0.2, 0.4, 0.6, 0.8])
    differences = noop_difference_matrix(core, thresholds)
    assert differences.numel() == 136
    assert torch.isfinite(noop_none_loss(core, thresholds))


def core_forward_source():
    return type(T7ProductionCoreNoopIntegration().core).forward
