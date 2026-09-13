from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from t1_trainability.noop_none_supervision import (
    SyntheticNoopBinder,
    NOOP_TARGETS,
    noop_contexts,
    noop_difference_matrix,
    noop_none_loss,
)


def test_catalog_and_loss_shape_and_orientation() -> None:
    binder = SyntheticNoopBinder()
    contexts = noop_contexts()
    differences = noop_difference_matrix(binder)
    assert len(contexts) == 34
    assert all(context.score_position == 1 for context in contexts[:32])
    assert contexts[32].tokens == ("OP_NOOP",)
    assert contexts[32].score_position == 0
    assert contexts[33].tokens == ("LINK", "OP_NOOP")
    assert differences.shape == (4, 34)
    assert torch.unique(binder.noop_candidate_scores()[0][:32]).numel() > 1
    assert torch.equal(differences, binder.noop_candidate_scores() - binder.thresholds[:, None])
    assert torch.isfinite(noop_none_loss(binder))


def test_real_backward_reaches_only_noop_embedding_and_matches_sigmoid() -> None:
    binder = SyntheticNoopBinder()
    scores = binder.noop_candidate_scores()
    scores.retain_grad()
    loss = F.softplus(scores - binder.thresholds[:, None]).mean()
    loss.backward()
    assert binder.noop_embedding.grad is not None
    assert float(binder.noop_embedding.grad.norm()) > 0.0
    expected_scaled = torch.sigmoid(scores.detach() - binder.thresholds[:, None])
    measured_scaled = scores.grad * scores.numel()
    assert torch.allclose(measured_scaled, expected_scaled, atol=1e-6, rtol=1e-5)
    assert torch.all(measured_scaled > 0)


def test_only_appended_noop_row_is_trainable() -> None:
    binder = SyntheticNoopBinder()
    trainable = [name for name, parameter in binder.named_parameters() if parameter.requires_grad]
    assert trainable == ["noop_embedding"]
    assert not binder.old_embeddings.requires_grad
    assert not binder.query_bank.requires_grad
    assert not binder.thresholds.requires_grad
    assert all(not parameter.requires_grad for parameter in binder.key_network.parameters())
    assert all(not parameter.requires_grad for parameter in binder.local_norm.parameters())
    assert all(not parameter.requires_grad for parameter in binder.w_v.parameters())


def test_old_route_is_bit_exact_when_noop_changes_and_ids_are_append_only() -> None:
    binder = SyntheticNoopBinder()
    old_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    lengths = torch.full((3,), 2, dtype=torch.long)
    before = binder.score(old_ids, lengths).detach().clone()
    old_id_snapshot = binder.old_embeddings.detach().clone()
    with torch.no_grad():
        binder.noop_embedding.add_(17.0)
    after = binder.score(old_ids, lengths).detach()
    assert float((before - after).abs().max()) == 0.0
    assert torch.equal(old_id_snapshot, binder.old_embeddings)
    assert binder.noop_token_id == binder.old_vocab_size
    assert binder.vocab_size == binder.old_vocab_size + 1


def test_inference_uses_ordinary_lookup_and_scorer_without_noop_shortcut() -> None:
    source = inspect.getsource(SyntheticNoopBinder.score)
    assert "noop_token_id" not in source
    assert "OP_NOOP" not in source
    assert "return torch.zeros" not in source
    assert "active_roles" not in source
    assert "no_grad" not in source
    assert "self.lookup(token_ids)" in source
    assert "self.key_network" in source


def test_null_contract_is_four_none_targets_not_zero_vectors_or_value_zero() -> None:
    assert NOOP_TARGETS == (None, None, None, None)
    assert len(NOOP_TARGETS) == 4
