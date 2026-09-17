"""Phase 1 synthetic Gates I/II for the real CPU fastpath ER32 adapter."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


HERE = Path(__file__).resolve().parent
FASTPATH_DIR = HERE.parent / "omega_core_lm_0_r1_cpu_fastpath_validation"
R1_SCRIPTS_DIR = HERE.parents[1] / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FASTPATH_DIR))
sys.path.insert(0, str(R1_SCRIPTS_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402
from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical  # noqa: E402
from omega_fast_er32 import FactorizedVocabulary, OmegaCoreLMFastER32  # noqa: E402


ATOL = 1e-4
RTOL = 1e-4
TEMPERATURE = 2.0
VOCAB = 50257
RANK = 32
DIMENSION = 128
SLOTS = 8
BATCH = 2
TIME = 8
PARAMETER_REDUCTION = 4_820_576


def _compare(left: torch.Tensor, right: torch.Tensor, category: str, diagnostics: dict[str, tuple[float, float]]) -> None:
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise AssertionError(f"NUMERICAL_EQUIVALENCE_FAIL category={category}: non-finite value")
    difference = (left - right).abs()
    max_abs = float(difference.max().item())
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)
    max_rel = float((difference / denominator).max().item())
    diagnostics[category] = (max_abs, max_rel)
    try:
        torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
    except AssertionError as error:
        raise AssertionError(
            f"NUMERICAL_EQUIVALENCE_FAIL category={category} "
            f"max_abs_error={max_abs:.6g} max_rel_error={max_rel:.6g}"
        ) from error


def _loss_from_logits(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_teacher = teacher_logits.reshape_as(flat_logits)
    weights = valid_mask.reshape(-1).to(logits.dtype)
    denominator = weights.sum().clamp_min(1.0)
    ce_values = F.cross_entropy(flat_logits, targets.reshape(-1), reduction="none")
    student_log_probs = F.log_softmax(flat_logits / TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(flat_teacher / TEMPERATURE, dim=-1)
    kl_values = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1) * TEMPERATURE**2
    ce = (ce_values * weights).sum() / denominator
    kl = (kl_values * weights).sum() / denominator
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}


def _chunked_loss_from_readout(
    model: OmegaCoreLMFastER32,
    readout_states: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    projected = model.project(readout_states).flatten(0, 1)
    flat_teacher = teacher_logits.reshape(-1, VOCAB)
    flat_targets = targets.reshape(-1)
    weights = valid_mask.reshape(-1).to(projected.dtype)
    totals = projected.new_zeros(2)
    for start in range(0, projected.shape[0], 512):
        stop = start + 512
        logits = model.logits_from_projected(projected[start:stop])
        ce = F.cross_entropy(logits, flat_targets[start:stop], reduction="none")
        student_log_probs = F.log_softmax(logits / TEMPERATURE, dim=-1)
        teacher_probs = F.softmax(flat_teacher[start:stop] / TEMPERATURE, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(-1) * TEMPERATURE**2
        totals += torch.stack(((ce * weights[start:stop]).sum(), (kl * weights[start:stop]).sum()))
    denominator = weights.sum().clamp_min(1.0)
    ce, kl = totals / denominator
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}


def test_level_a_factorized_vocabulary_and_logits_equivalence() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260917)
        explicit = FactorizedVocabulary(VOCAB, RANK, DIMENSION, "explicit")
        efficient = FactorizedVocabulary(VOCAB, RANK, DIMENSION, "efficient")
    with torch.no_grad():
        explicit.C.copy_(torch.linspace(-0.05, 0.05, steps=VOCAB * RANK).view(VOCAB, RANK))
        explicit.U.copy_(torch.linspace(-0.02, 0.02, steps=RANK * DIMENSION).view(RANK, DIMENSION))
        efficient.load_state_dict(explicit.state_dict())

    tokens = torch.tensor([[1, 7, 11], [13, 17, 19]], dtype=torch.long)
    projected = torch.linspace(-0.2, 0.2, steps=tokens.numel() * DIMENSION).view(-1, DIMENSION)
    teacher = torch.linspace(-0.1, 0.1, steps=tokens.numel() * VOCAB).view(*tokens.shape, VOCAB)
    targets = torch.tensor([[2, 8, 12], [14, 18, 20]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, False], [True, False, True]])
    diagnostics: dict[str, tuple[float, float]] = {}

    _compare(explicit(tokens), efficient(tokens), "level_a.input_embeddings", diagnostics)
    explicit_logits = (projected @ (explicit.C @ explicit.U).transpose(0, 1) / math.sqrt(DIMENSION)).view(*tokens.shape, VOCAB)
    efficient_logits = ((projected @ efficient.U.transpose(0, 1)) @ efficient.C.transpose(0, 1) / math.sqrt(DIMENSION)).view(*tokens.shape, VOCAB)
    _compare(explicit_logits, efficient_logits, "level_a.logits", diagnostics)
    explicit_loss = _loss_from_logits(explicit_logits, teacher, targets, valid_mask)
    efficient_loss = _loss_from_logits(efficient_logits, teacher, targets, valid_mask)
    for name in ("ce", "kl", "total"):
        _compare(explicit_loss[name], efficient_loss[name], f"level_a.loss.{name}", diagnostics)

    explicit_loss["total"].backward()
    efficient_loss["total"].backward()
    _compare(explicit.C.grad, efficient.C.grad, "level_a.grad.C", diagnostics)
    _compare(explicit.U.grad, efficient.U.grad, "level_a.grad.U", diagnostics)
    assert explicit.C.shape == (VOCAB, RANK)
    assert explicit.U.shape == (RANK, DIMENSION)
    assert 1.0 / math.sqrt(DIMENSION) == 1.0 / math.sqrt(128)
    assert not any(parameter.shape == (VOCAB, DIMENSION) for parameter in explicit.parameters())


def _fresh_f_reference(rounds: int, seed: int) -> OmegaCoreLMFast:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = OmegaCoreLM0R1Technical(
            vocab_size=VOCAB,
            dimension=DIMENSION,
            slots=SLOTS,
            rounds=rounds,
            variant="shared",
        ).float()
        return OmegaCoreLMFast.from_reference(reference).float()


def _make_pair(rounds: int) -> tuple[OmegaCoreLMFast, OmegaCoreLMFastER32, OmegaCoreLMFastER32]:
    f_reference = _fresh_f_reference(rounds, 20261000 + rounds)
    explicit = OmegaCoreLMFastER32.from_f_reference(
        f_reference,
        experimental_seed=20260917,
        implementation="explicit",
    )
    efficient = OmegaCoreLMFastER32.from_f_reference(
        f_reference,
        experimental_seed=20260917,
        implementation="efficient",
    )
    return f_reference, explicit, efficient


def test_factor_seed_is_k_independent_and_factory_rng_isolated() -> None:
    torch.manual_seed(31415)
    rng_before = torch.get_rng_state()
    f_k1 = _fresh_f_reference(1, 20261001)
    f_k4 = _fresh_f_reference(4, 20261004)
    er_k1 = OmegaCoreLMFastER32.from_f_reference(f_k1, experimental_seed=20260917)
    er_k4 = OmegaCoreLMFastER32.from_f_reference(f_k4, experimental_seed=20260917)
    rng_after = torch.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert torch.equal(er_k1.embedding.C, er_k4.embedding.C)
    assert torch.equal(er_k1.embedding.U, er_k4.embedding.U)


def _synthetic_window(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = ((torch.arange(BATCH * TIME).view(BATCH, TIME) * 17 + seed) % VOCAB).long()
    teacher = torch.linspace(-0.2, 0.2, steps=BATCH * TIME * VOCAB).view(BATCH, TIME, VOCAB)
    targets = ((torch.arange(BATCH * TIME).view(BATCH, TIME) * 13 + seed + 3) % VOCAB).long()
    valid_mask = torch.tensor(
        [[True, True, True, False, True, True, False, True], [True, False, True, True, True, False, True, True]],
        dtype=torch.bool,
    )
    return tokens, teacher, targets, valid_mask


def _forward_snapshot(model: OmegaCoreLMFastER32, tokens: torch.Tensor, previous: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    final, continuation, candidate, readout = model.recur_states(tokens, previous, mask)
    projected = model.project(readout)
    logits = model.logits_from_projected(projected)
    window_final, window_logits, window_states = model.forward_window(tokens, previous, mask)
    return {
        "final": final.detach().clone(),
        "continuation": continuation.detach().clone(),
        "candidate": candidate.detach().clone(),
        "readout": readout.detach().clone(),
        "projected": projected.detach().clone(),
        "logits": logits.detach().clone(),
        "window_final": window_final.detach().clone(),
        "window_logits": window_logits.detach().clone(),
        "window_continuation": window_states["continuation_states"].detach().clone(),
        "window_candidate": window_states["candidate_states"].detach().clone(),
        "window_readout": window_states["readout_states"].detach().clone(),
    }


def _run_two_window_cycle(
    model: OmegaCoreLMFastER32,
    window_zero: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    window_one: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    tokens_zero, teacher_zero, targets_zero, mask_zero = window_zero
    tokens_one, teacher_one, targets_one, mask_one = window_one
    state_zero, _, _, readout_zero = model.recur_states(tokens_zero, model.initial_state(BATCH, device=torch.device("cpu")), mask_zero)
    loss_zero = _chunked_loss_from_readout(model, readout_zero, teacher_zero, targets_zero, mask_zero)["total"]
    loss_zero.backward()
    clip_zero = clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    detached_state = state_zero.detach()

    optimizer.zero_grad(set_to_none=True)
    state_one, _, _, readout_one = model.recur_states(tokens_one, detached_state, mask_one)
    losses_one = _chunked_loss_from_readout(model, readout_one, teacher_one, targets_one, mask_one)
    losses_one["total"].backward()
    clip_one = clip_grad_norm_(model.parameters(), max_norm=1.0)
    clipped_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_state = {
        name: {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in optimizer.state[parameter].items()}
        for name, parameter in model.named_parameters()
    }
    return {
        "loss_zero": loss_zero.detach(),
        "loss_one": losses_one["total"].detach(),
        "state_zero": detached_state.clone(),
        "state_one": state_one.detach().clone(),
        "clip_zero": clip_zero.detach().clone(),
        "clip_one": clip_one.detach().clone(),
        "clipped_gradients": clipped_gradients,
        "gradients": gradients,
        "parameters": parameters,
        "optimizer_state": optimizer_state,
    }


@pytest.mark.parametrize("rounds", [1, 4])
def test_level_b_real_f_integration_and_two_window_equivalence(rounds: int) -> None:
    f_reference, explicit, efficient = _make_pair(rounds)
    diagnostics: dict[str, tuple[float, float]] = {}

    assert OmegaCoreLMFastER32.__mro__[1] is OmegaCoreLMFast
    assert OmegaCoreLMFastER32.recur_states is OmegaCoreLMFast.recur_states
    assert OmegaCoreLMFastER32.project is OmegaCoreLMFast.project
    assert OmegaCoreLMFastER32.forward_window is OmegaCoreLMFast.forward_window
    assert explicit.embedding.implementation == "explicit"
    assert efficient.embedding.implementation == "efficient"
    assert [name for name, _ in explicit.embedding.named_parameters()] == ["C", "U"]
    assert [name for name, _ in explicit.named_parameters() if name.startswith("embedding.")] == ["embedding.C", "embedding.U"]
    assert explicit.embedding.C.shape == (VOCAB, RANK)
    assert explicit.embedding.U.shape == (RANK, DIMENSION)
    assert not hasattr(explicit.embedding, "weight")
    assert not any(parameter.shape == (VOCAB, DIMENSION) for parameter in explicit.parameters())
    assert sum(parameter.numel() for parameter in f_reference.parameters()) - sum(parameter.numel() for parameter in explicit.parameters()) == PARAMETER_REDUCTION
    assert len(explicit.blocks) == 1
    assert len(efficient.blocks) == 1
    assert explicit.rounds == rounds
    assert efficient.rounds == rounds

    for name, value in f_reference.state_dict().items():
        if name != "embedding.weight":
            assert torch.equal(value, explicit.state_dict()[name]), name
            assert torch.equal(value, efficient.state_dict()[name]), name
    assert torch.equal(explicit.embedding.C, efficient.embedding.C)
    assert torch.equal(explicit.embedding.U, efficient.embedding.U)

    # Same shared block object is selected once for K1 and once per round for K4.
    calls: list[object] = []
    hook = explicit.blocks[0].register_forward_hook(lambda *_args: calls.append(None))
    explicit.recur_states(torch.tensor([[1]], dtype=torch.long), explicit.initial_state(1, device=torch.device("cpu")))
    hook.remove()
    assert len(calls) == rounds

    tokens, teacher, targets, mask = _synthetic_window(11)
    previous = explicit.initial_state(BATCH, device=torch.device("cpu"))
    with torch.no_grad():
        left = _forward_snapshot(explicit, tokens, previous, mask)
        right = _forward_snapshot(efficient, tokens, previous, mask)
    for name in ("final", "continuation", "candidate", "readout", "projected", "logits", "window_final", "window_logits", "window_continuation", "window_candidate", "window_readout"):
        _compare(left[name], right[name], f"level_b.k{rounds}.{name}", diagnostics)
    left_loss = _chunked_loss_from_readout(explicit, left["readout"], teacher, targets, mask)
    right_loss = _chunked_loss_from_readout(efficient, right["readout"], teacher, targets, mask)
    for name in ("ce", "kl", "total"):
        _compare(left_loss[name], right_loss[name], f"level_b.k{rounds}.loss.{name}", diagnostics)

    window_zero = _synthetic_window(23)
    window_one = _synthetic_window(47)
    expected = _run_two_window_cycle(explicit, window_zero, window_one)
    actual = _run_two_window_cycle(efficient, window_zero, window_one)
    for name in ("loss_zero", "loss_one", "state_zero", "state_one", "clip_zero", "clip_one"):
        _compare(expected[name], actual[name], f"level_b.k{rounds}.cycle.{name}", diagnostics)
    for name, expected_gradient in expected["clipped_gradients"].items():
        _compare(expected_gradient, actual["clipped_gradients"][name], f"level_b.k{rounds}.clip_grad.{name}", diagnostics)
    for name, expected_gradient in expected["gradients"].items():
        _compare(expected_gradient, actual["gradients"][name], f"level_b.k{rounds}.grad.{name}", diagnostics)
    for name, expected_parameter in expected["parameters"].items():
        _compare(expected_parameter, actual["parameters"][name], f"level_b.k{rounds}.parameter.{name}", diagnostics)
    for name, expected_state in expected["optimizer_state"].items():
        actual_state = actual["optimizer_state"][name]
        assert set(expected_state) == {"step", "exp_avg", "exp_avg_sq"}
        assert set(actual_state) == {"step", "exp_avg", "exp_avg_sq"}
        assert expected_state["step"].item() == 2.0
        assert actual_state["step"].item() == 2.0
        _compare(expected_state["exp_avg"], actual_state["exp_avg"], f"level_b.k{rounds}.optimizer.exp_avg.{name}", diagnostics)
        _compare(expected_state["exp_avg_sq"], actual_state["exp_avg_sq"], f"level_b.k{rounds}.optimizer.exp_avg_sq.{name}", diagnostics)

    assert explicit.embedding.C.grad is not None
    assert explicit.embedding.U.grad is not None
    assert efficient.embedding.C.grad is not None
    assert efficient.embedding.U.grad is not None
