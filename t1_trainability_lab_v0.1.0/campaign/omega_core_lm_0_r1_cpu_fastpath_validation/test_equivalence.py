"""Executable Step 1 equivalence gate for the isolated CPU fast candidate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


UNIT_DIR = Path(__file__).resolve().parent
LAB_ROOT = UNIT_DIR.parents[1]
SCRIPTS_DIR = LAB_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(UNIT_DIR))

from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    OmegaCoreLM0R1Technical,
    distillation_loss,
)
from omega_fast_candidate import (  # noqa: E402
    GATE_ATOL,
    OmegaCoreLMFast,
    distillation_loss_lse,
    parameter_mapping,
    teacher_targets,
)


ATOL = 1e-5
RTOL = 0.0
SEED = 20260914
METRICS: dict[str, dict[str, float | str]] = {}
FAILURES: list[str] = []


def _record_failure(component: str) -> None:
    if component not in FAILURES:
        FAILURES.append(component)


def gate_assert(condition: bool, component: str) -> None:
    if not condition:
        _record_failure(component)
        pytest.fail(component)


def assert_close(component: str, actual: torch.Tensor | float, expected: torch.Tensor | float) -> None:
    actual_tensor = torch.as_tensor(actual).detach().to(dtype=torch.float64, device="cpu")
    expected_tensor = torch.as_tensor(expected).detach().to(dtype=torch.float64, device="cpu")
    max_error = float((actual_tensor - expected_tensor).abs().max().item())
    margin = ATOL - max_error
    METRICS[component] = {"max_abs_error": max_error, "margin_to_1e-5": margin}
    if max_error > ATOL:
        _record_failure(component)
        pytest.fail(f"{component}: max_abs_error={max_error:.9g}, margin_to_1e-5={margin:.9g}")


def make_pair(rounds: int, *, variant: str = "shared") -> tuple[OmegaCoreLM0R1Technical, OmegaCoreLMFast]:
    torch.manual_seed(SEED + rounds)
    reference = OmegaCoreLM0R1Technical(vocab_size=19, dimension=8, slots=3, rounds=rounds, variant=variant).float()
    fast = OmegaCoreLMFast.from_reference(reference).float()
    return reference, fast


def fixtures(batch: int = 2, time: int = 4, vocab_size: int = 19) -> tuple[torch.Tensor, ...]:
    tokens = torch.tensor([[1, 4, 2, 7], [3, 5, 6, 8]], dtype=torch.long)[:batch, :time]
    targets = torch.tensor([[4, 2, 7, 3], [5, 6, 8, 1]], dtype=torch.long)[:batch, :time] % vocab_size
    input_mask = torch.tensor([[True, False, True, True], [True, True, False, True]])[:batch, :time]
    loss_mask = torch.tensor([[False, True, True, False], [True, False, True, True]])[:batch, :time]
    teacher_logits = torch.linspace(-2.0, 2.0, steps=batch * time * vocab_size, dtype=torch.float32).view(batch, time, vocab_size)
    return tokens, targets, input_mask, loss_mask, teacher_logits


def mapped_tensor(reference: torch.nn.Module, fast: OmegaCoreLMFast, mapping: tuple[str, ...], candidate_name: str, *, gradient: bool = False, optimizer_state: str | None = None, optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] | None = None) -> torch.Tensor:
    reference_values = dict(reference.named_parameters())
    fast_values = dict(fast.named_parameters())
    if gradient:
        candidate_value = fast_values[candidate_name].grad
        reference_parts = [reference_values[name].grad for name in mapping]
    elif optimizer_state is not None:
        assert optimizers is not None
        candidate_value = optimizers[1].state[fast_values[candidate_name]][optimizer_state]
        reference_parts = [optimizers[0].state[reference_values[name]][optimizer_state] for name in mapping]
    else:
        candidate_value = fast_values[candidate_name]
        reference_parts = [reference_values[name] for name in mapping]
    gate_assert(candidate_value is not None and all(part is not None for part in reference_parts), f"{candidate_name}:{optimizer_state or 'value'}:present")
    expected = reference_parts[0] if len(reference_parts) == 1 else torch.cat(reference_parts, dim=0)
    return candidate_value, expected  # type: ignore[return-value]


def compare_mapped_parameters(reference: torch.nn.Module, fast: OmegaCoreLMFast, *, stage: str, gradient: bool = False, optimizer_state: str | None = None, optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] | None = None) -> None:
    mapping = parameter_mapping(reference, fast)
    reference_names = set(dict(reference.named_parameters()))
    mapped_reference_names = [name for names in mapping.values() for name in names]
    gate_assert(set(mapping) == set(dict(fast.named_parameters())), f"{stage}:candidate_mapping_complete")
    gate_assert(len(mapped_reference_names) == len(set(mapped_reference_names)), f"{stage}:reference_mapping_unique")
    gate_assert(set(mapped_reference_names) == reference_names, f"{stage}:reference_mapping_complete")
    for candidate_name, reference_names_for_candidate in mapping.items():
        candidate_value, expected = mapped_tensor(
            reference,
            fast,
            reference_names_for_candidate,
            candidate_name,
            gradient=gradient,
            optimizer_state=optimizer_state,
            optimizers=optimizers,
        )
        assert_close(f"{stage}:{candidate_name}:{optimizer_state or ('gradient' if gradient else 'parameter')}", candidate_value, expected)


def run_window(reference: OmegaCoreLM0R1Technical, fast: OmegaCoreLMFast, tokens: torch.Tensor, previous_state: torch.Tensor, input_mask: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    reference_final, reference_logits = reference.forward_window(tokens, previous_state, input_mask)
    fast_final, fast_logits, trace = fast.forward_window(tokens, previous_state.clone(), input_mask)
    assert_close("window:final_state", fast_final, reference_final)
    assert_close("window:logits", fast_logits, reference_logits)
    assert_close("window:continuation_states", trace["continuation_states"], _reference_continuation_states(reference, tokens, previous_state, input_mask))
    assert_close("window:candidate_states", trace["candidate_states"], _reference_candidate_states(reference, tokens, previous_state, input_mask))
    assert_close("window:readout_states", trace["readout_states"], trace["candidate_states"])
    return (fast_final, fast_logits, trace), (reference_final, reference_logits)


def _reference_candidate_states(reference: OmegaCoreLM0R1Technical, tokens: torch.Tensor, previous_state: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    state = previous_state
    states = []
    for position in range(tokens.shape[1]):
        candidate, _ = reference.forward_token(tokens[:, position], state)
        states.append(candidate)
        state = torch.where(input_mask[:, position].view(-1, 1, 1), candidate, state)
    return torch.stack(states, dim=1)


def _reference_continuation_states(reference: OmegaCoreLM0R1Technical, tokens: torch.Tensor, previous_state: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
    state = previous_state
    states = []
    for position in range(tokens.shape[1]):
        candidate, _ = reference.forward_token(tokens[:, position], state)
        state = torch.where(input_mask[:, position].view(-1, 1, 1), candidate, state)
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("rounds", [1, 4])
def test_parameter_mapping_and_initial_states(rounds: int) -> None:
    reference, fast = make_pair(rounds)
    compare_mapped_parameters(reference, fast, stage=f"k{rounds}:initial")
    zero = fast.initial_state(2, device=torch.device("cpu"))
    gate_assert(torch.equal(zero, torch.zeros_like(zero)), f"k{rounds}:initial_state_zero")
    deterministic = torch.arange(zero.numel(), dtype=torch.float32).view_as(zero) / 17.0
    deterministic_tokens = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    _, logits_a, _ = fast.forward_window(deterministic_tokens, deterministic, torch.ones(2, 2, dtype=torch.bool))
    _, logits_b, _ = fast.forward_window(deterministic_tokens, deterministic, torch.ones(2, 2, dtype=torch.bool))
    gate_assert(torch.equal(logits_a, logits_b), f"k{rounds}:deterministic_nonzero_state")


@pytest.mark.parametrize("rounds", [1, 4])
def test_window_equivalence_with_distinct_input_and_loss_masks(rounds: int) -> None:
    reference, fast = make_pair(rounds)
    tokens, targets, input_mask, loss_mask, teacher_logits = fixtures()
    fast_result, reference_result = run_window(reference, fast, tokens, fast.initial_state(tokens.shape[0], device=torch.device("cpu")), input_mask)
    fast_final, fast_logits, trace = fast_result
    reference_final, reference_logits = reference_result
    assert_close(f"k{rounds}:explicit_reference_final", fast_final, reference_final)
    assert_close(f"k{rounds}:explicit_reference_logits", fast_logits, reference_logits)
    reference_losses = distillation_loss(reference_logits, teacher_logits, targets, loss_mask)
    teacher_probs, teacher_neg_entropy = teacher_targets(teacher_logits)
    local_losses = distillation_loss_lse(fast, trace["readout_states"], teacher_probs, teacher_neg_entropy, targets, loss_mask)
    for component in ("ce", "kl", "total"):
        assert_close(f"k{rounds}:loss:{component}", local_losses[component], reference_losses[component])


def test_masked_position_reads_candidate_not_reverted_continuation() -> None:
    reference, fast = make_pair(4)
    tokens, _, input_mask, _, _ = fixtures()
    previous = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8) / 11.0
    _, reference_logits = reference.forward_window(tokens, previous, input_mask)
    final, candidate_logits, trace = fast.forward_window(tokens, previous, input_mask)
    del final
    assert_close("mask_regression:reference_logits", candidate_logits, reference_logits)
    candidate_readout_logits = fast.logits_from_states(trace["candidate_states"])
    reverted_logits = fast.logits_from_states(trace["continuation_states"])
    assert_close("mask_regression:candidate_readout", candidate_logits, candidate_readout_logits)
    masked = ~input_mask
    difference = (candidate_logits[masked] - reverted_logits[masked]).abs().max().item()
    gate_assert(difference > ATOL, "mask_regression:old_reverted_readout_would_fail")


@pytest.mark.parametrize("teacher_case", ["concentrated", "low_kl"])
def test_loss_equivalence_concentrated_and_low_kl_teachers(teacher_case: str) -> None:
    reference, fast = make_pair(4)
    tokens, targets, input_mask, loss_mask, teacher_logits = fixtures()
    _, reference_logits = reference.forward_window(tokens, reference.initial_state(2, device=torch.device("cpu")), input_mask)
    _, _, trace = fast.forward_window(tokens, fast.initial_state(2, device=torch.device("cpu")), input_mask)
    if teacher_case == "concentrated":
        teacher_logits = torch.full_like(reference_logits, -18.0)
        teacher_logits[..., 0] = 18.0
    else:
        teacher_logits = reference_logits.detach()
    reference_losses = distillation_loss(reference_logits, teacher_logits, targets, loss_mask)
    teacher_probs, teacher_neg_entropy = teacher_targets(teacher_logits)
    local_losses = distillation_loss_lse(fast, trace["readout_states"], teacher_probs, teacher_neg_entropy, targets, loss_mask)
    for component in ("ce", "kl", "total"):
        assert_close(f"loss_case:{teacher_case}:{component}", local_losses[component], reference_losses[component])


def test_sdpa_contract_is_explicit() -> None:
    reference, fast = make_pair(1)
    del reference
    original = F.scaled_dot_product_attention
    calls: list[tuple[float, bool]] = []

    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((float(kwargs["dropout_p"]), bool(kwargs["is_causal"])))
        return original(*args, **kwargs)

    F.scaled_dot_product_attention = wrapped  # type: ignore[assignment]
    try:
        fast.forward_window(torch.tensor([[1, 2]], dtype=torch.long), fast.initial_state(1, device=torch.device("cpu")))
    finally:
        F.scaled_dot_product_attention = original  # type: ignore[assignment]
    gate_assert(calls and all(dropout == 0.0 and not causal for dropout, causal in calls), "sdpa:dropout_zero_noncausal")


def test_round_constants_recompute_after_adamw_and_keep_gradients() -> None:
    _, fast = make_pair(4)
    tokens, _, input_mask, _, _ = fixtures()
    parameters = [fast.blocks[0].fc1.weight, fast.depth_embedding.weight, fast.gate_logits]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    before_biases, before_gates = fast._round_constants()
    _, _, trace = fast.forward_window(tokens, fast.initial_state(2, device=torch.device("cpu")), input_mask)
    loss = trace["candidate_states"].square().mean()
    loss.backward()
    for parameter in parameters:
        gate_assert(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"round_constants:gradient_connected:{parameter.shape}")
    optimizer.step()
    after_biases, after_gates = fast._round_constants()
    gate_assert(any(not torch.equal(a, b) for a, b in zip(before_biases, after_biases)), "round_constants:updated_w1_or_depth_used")
    gate_assert(not torch.equal(before_gates, after_gates), "round_constants:updated_gate_used")
    _, _, next_trace = fast.forward_window(tokens, fast.initial_state(2, device=torch.device("cpu")), input_mask)
    gate_assert(not torch.equal(trace["candidate_states"], next_trace["candidate_states"]), "round_constants:next_forward_uses_updated_values")


def adamw_cycle(rounds: int) -> None:
    reference, fast = make_pair(rounds)
    tokens, targets, input_mask, loss_mask, teacher_logits = fixtures()
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    fast_optimizer = torch.optim.AdamW(fast.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    state_ref = reference.initial_state(tokens.shape[0], device=torch.device("cpu"))
    state_fast = fast.initial_state(tokens.shape[0], device=torch.device("cpu"))
    for window_index, window_mask in enumerate((input_mask, torch.logical_not(input_mask))):
        ref_optimizer.zero_grad(set_to_none=True)
        fast_optimizer.zero_grad(set_to_none=True)
        state_ref, ref_logits = reference.forward_window(tokens, state_ref, window_mask)
        state_fast, fast_logits, trace = fast.forward_window(tokens, state_fast, window_mask)
        assert_close(f"k{rounds}:cycle{window_index}:logits", fast_logits, ref_logits)
        assert_close(f"k{rounds}:cycle{window_index}:final", state_fast, state_ref)
        ref_losses = distillation_loss(ref_logits, teacher_logits, targets, loss_mask)
        teacher_probs, teacher_neg_entropy = teacher_targets(teacher_logits)
        fast_losses = distillation_loss_lse(fast, trace["readout_states"], teacher_probs, teacher_neg_entropy, targets, loss_mask)
        for component in ("ce", "kl", "total"):
            assert_close(f"k{rounds}:cycle{window_index}:loss:{component}", fast_losses[component], ref_losses[component])
        ref_losses["total"].backward()
        fast_losses["total"].backward()
        compare_mapped_parameters(reference, fast, stage=f"k{rounds}:cycle{window_index}:before_clip", gradient=True)
        ref_norm = clip_grad_norm_(reference.parameters(), max_norm=0.25)
        fast_norm = clip_grad_norm_(fast.parameters(), max_norm=0.25)
        assert_close(f"k{rounds}:cycle{window_index}:clip_norm", fast_norm, ref_norm)
        compare_mapped_parameters(reference, fast, stage=f"k{rounds}:cycle{window_index}:after_clip", gradient=True)
        ref_optimizer.step()
        fast_optimizer.step()
        compare_mapped_parameters(reference, fast, stage=f"k{rounds}:cycle{window_index}:after_adamw")
        compare_mapped_parameters(reference, fast, stage=f"k{rounds}:cycle{window_index}:exp_avg", optimizer_state="exp_avg", optimizers=(ref_optimizer, fast_optimizer))
        compare_mapped_parameters(reference, fast, stage=f"k{rounds}:cycle{window_index}:exp_avg_sq", optimizer_state="exp_avg_sq", optimizers=(ref_optimizer, fast_optimizer))
        state_ref = state_ref.detach()
        state_fast = state_fast.detach()
    ref_optimizer.zero_grad(set_to_none=True)
    fast_optimizer.zero_grad(set_to_none=True)
    reset_mask = torch.ones_like(input_mask)
    reset_state_ref, reset_logits_ref = reference.forward_window(tokens, reference.initial_state(tokens.shape[0], device=torch.device("cpu")), reset_mask)
    reset_state_fast, reset_logits_fast, reset_trace = fast.forward_window(tokens, fast.initial_state(tokens.shape[0], device=torch.device("cpu")), reset_mask)
    assert_close(f"k{rounds}:reset_after_weight_change:state", reset_state_fast, reset_state_ref)
    assert_close(f"k{rounds}:reset_after_weight_change:logits", reset_logits_fast, reset_logits_ref)
    assert_close(f"k{rounds}:reset_after_weight_change:readout", reset_trace["readout_states"], fast.recur_states(tokens, fast.initial_state(tokens.shape[0], device=torch.device("cpu")), reset_mask)[3])


@pytest.mark.parametrize("rounds", [1, 4])
def test_complete_two_window_adamw_cycle(rounds: int) -> None:
    adamw_cycle(rounds)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(session: pytest.Session, exitstatus: int) -> None:
    reference_path = SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    proposal_path = LAB_ROOT / "campaign" / "omega_core_lm_0_r1_fable_optimization_proposal" / "omega_fast.py"
    report: dict[str, Any] = {
        "schema": "omega-core-lm-0-r1-cpu-fastpath-validation-step1-v1",
        "status": "passed" if exitstatus == 0 and not FAILURES else "failed",
        "step": 1,
        "step_name": "complete equivalence gate",
        "step2_authorized": False,
        "benchmarks_run": False,
        "speed_measurement_run": False,
        "test_count": session.testscollected,
        "failed_components": FAILURES,
        "max_errors_and_margins": METRICS,
        "gate": {"atol": ATOL, "rtol": RTOL, "assertion_policy": "absolute-only; fail closed"},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "execution": "eager",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "source_identity": {
            "candidate_sha256": _sha256(UNIT_DIR / "omega_fast_candidate.py"),
            "test_sha256": _sha256(UNIT_DIR / "test_equivalence.py"),
            "conftest_sha256": _sha256(UNIT_DIR / "conftest.py"),
            "readme_sha256": _sha256(UNIT_DIR / "README.md"),
            "reference_baseline_sha256": _sha256(reference_path),
            "proposal_read_only_reference_sha256": _sha256(proposal_path),
            "identity_note": "Own Step 1 source hashes; no old freeze or T0 hashes claimed.",
        },
    }
    unsigned = dict(report)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    report["artifact_self_hash"] = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    (UNIT_DIR / "step1_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
