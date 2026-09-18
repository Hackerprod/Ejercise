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
ER32_DIR = HERE.parent / "omega_core_lm_0_er32_integration_and_cost_gate"
SCRIPTS_DIR = HERE.parents[1] / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FASTPATH_DIR))
sys.path.insert(0, str(ER32_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402
from omega_fast_er64 import ER64_RANK, FactorizedVocabulary, OmegaCoreLMFastER64  # noqa: E402
from omega_fast_er32 import OmegaCoreLMFastER32  # noqa: E402
from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical  # noqa: E402


VOCAB = 50257
DIMENSION = 128
SLOTS = 8
ATOL = RTOL = 1e-4


def _fresh_f(rounds: int, seed: int, *, vocab_size: int = VOCAB, dimension: int = DIMENSION, slots: int = SLOTS) -> OmegaCoreLMFast:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = OmegaCoreLM0R1Technical(vocab_size=vocab_size, dimension=dimension, slots=slots, rounds=rounds, variant="shared").float()
        return OmegaCoreLMFast.from_reference(reference).float()


def _copy_factors(source: FactorizedVocabulary, destination: FactorizedVocabulary) -> None:
    with torch.no_grad():
        destination.C.copy_(source.C)
        destination.U.copy_(source.U)


def test_rank_shapes_and_no_dense_vocabulary_parameter() -> None:
    model = OmegaCoreLMFastER64(vocab_size=VOCAB, rank=ER64_RANK, dimension=DIMENSION, slots=SLOTS, rounds=1, implementation="efficient")
    assert model.embedding.C.shape == (VOCAB, 64)
    assert model.embedding.U.shape == (64, DIMENSION)
    assert not any(tuple(parameter.shape) == (VOCAB, DIMENSION) for parameter in model.parameters())
    assert not hasattr(model.embedding, "weight")


def test_non_lexical_parameters_match_f_reference_bitwise_and_seed_is_k_independent() -> None:
    f_k1 = _fresh_f(1, 20261001)
    f_k4 = _fresh_f(4, 20261004)
    er_k1 = OmegaCoreLMFastER64.from_f_reference(f_k1, experimental_seed=20260913)
    er_k4 = OmegaCoreLMFastER64.from_f_reference(f_k4, experimental_seed=20260913)
    assert torch.equal(er_k1.embedding.C, er_k4.embedding.C)
    assert torch.equal(er_k1.embedding.U, er_k4.embedding.U)
    for name, value in f_k1.state_dict().items():
        if name != "embedding.weight":
            assert torch.equal(value, er_k1.state_dict()[name]), name


def test_explicit_and_efficient_match_with_frozen_rank64_factors() -> None:
    f_reference = _fresh_f(1, 20261001, vocab_size=31, dimension=8, slots=2)
    explicit = OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=20260913, implementation="explicit")
    efficient = OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=20260913, implementation="efficient")
    tokens = torch.tensor([[1, 7, 11], [13, 17, 19]], dtype=torch.long)
    projected = torch.linspace(-0.2, 0.2, steps=tokens.numel() * 8).view(-1, 8)
    with torch.no_grad():
        assert torch.allclose(explicit.embedding(tokens), efficient.embedding(tokens), atol=ATOL, rtol=RTOL)
        explicit_logits = explicit.logits_from_projected(projected)
        efficient_logits = efficient.logits_from_projected(projected)
        assert torch.allclose(explicit_logits, efficient_logits, atol=ATOL, rtol=RTOL)


def test_full_two_window_adamw_cycle_matches_explicit_and_efficient() -> None:
    f_reference = _fresh_f(4, 20261004, vocab_size=31, dimension=8, slots=2)
    explicit = OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=20260913, implementation="explicit")
    efficient = OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=20260913, implementation="efficient")
    batch, time, vocab = 2, 8, 31
    tokens = (torch.arange(batch * time).view(batch, time) * 7 % vocab).long()
    targets = (tokens + 3) % vocab

    def step(model: OmegaCoreLMFastER64, optimizer: torch.optim.Optimizer, previous: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        state = model.initial_state(batch, device=torch.device("cpu")) if previous is None else previous
        final, logits, _ = model.forward_window(tokens, state)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.detach(), final.detach()

    opt_explicit = torch.optim.AdamW(explicit.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt_efficient = torch.optim.AdamW(efficient.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    first_explicit, state_explicit = step(explicit, opt_explicit, None)
    first_efficient, state_efficient = step(efficient, opt_efficient, None)
    assert torch.allclose(first_explicit, first_efficient, atol=ATOL, rtol=RTOL)
    assert torch.allclose(state_explicit, state_efficient, atol=ATOL, rtol=RTOL)
    opt_explicit.zero_grad(set_to_none=True)
    opt_efficient.zero_grad(set_to_none=True)
    second_explicit, _ = step(explicit, opt_explicit, state_explicit)
    second_efficient, _ = step(efficient, opt_efficient, state_efficient)
    assert torch.allclose(second_explicit, second_efficient, atol=ATOL, rtol=RTOL)


def test_parameter_counts_are_derived_from_real_tensors() -> None:
    model = OmegaCoreLMFastER64(vocab_size=VOCAB, rank=ER64_RANK, dimension=DIMENSION, slots=SLOTS, rounds=1, implementation="efficient")
    lexical = model.embedding.C.numel() + model.embedding.U.numel()
    dense_f = VOCAB * DIMENSION
    assert lexical == model.embedding.C.shape[0] * model.embedding.C.shape[1] + model.embedding.U.shape[0] * model.embedding.U.shape[1]
    assert lexical == 3_224_640
    assert dense_f - lexical == 3_208_256
    er32 = OmegaCoreLMFastER32(vocab_size=VOCAB, rank=32, dimension=DIMENSION, slots=SLOTS, rounds=1, implementation="efficient", _initialize_factors=False)
    er32_lexical = er32.embedding.C.numel() + er32.embedding.U.numel()
    assert dense_f - er32_lexical == 4_820_576
    assert lexical - er32_lexical == 1_612_320
    assert dense_f - lexical == 3_208_256
