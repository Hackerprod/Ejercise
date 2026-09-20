from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_omega_native_runtime_p0 as profiler  # noqa: E402


def test_real_execution_requires_explicit_confirmation() -> None:
    try:
        profiler.require_real_authorization(False)
    except profiler.RealExecutionAuthorizationError:
        return
    raise AssertionError("real profiling must require explicit authorization")


def test_instrumented_forward_matches_production_forward() -> None:
    model = profiler.ce.fresh_model(20260913, 1, vocab_size=31, dimension=8, slots=2)
    tokens = torch.randint(0, 31, (2, 4), dtype=torch.long)
    state = model.initial_state(2, device=torch.device("cpu"))
    expected_state, expected_logits, expected_extra = model.forward_window(tokens, state)
    actual_state, actual_logits, boundaries, timings = profiler._student_forward_timed(model, tokens, state)
    assert torch.equal(expected_state, actual_state)
    assert torch.equal(expected_logits, actual_logits)
    assert torch.equal(expected_extra["readout_states"], boundaries["recurrent_readout_boundary"])
    assert set(("student_embedding_prelude", "recurrent_forward", "student_readout_projection", "student_vocab_projection")) <= set(timings)


def test_single_backward_hooks_segment_without_second_backward() -> None:
    model = profiler.ce.fresh_model(20260913, 1, vocab_size=23, dimension=8, slots=2)
    tokens = torch.randint(0, 23, (2, 3), dtype=torch.long)
    targets = torch.randint(0, 23, (2, 3), dtype=torch.long)
    state = model.initial_state(2, device=torch.device("cpu"))
    _, logits, boundaries, _ = profiler._student_forward_timed(model, tokens, state)
    teacher_logits = torch.randn_like(logits)
    loss = profiler.hidden.base.distillation_loss(logits, teacher_logits, targets)["total"]
    timing = profiler._backward_timed(loss, boundaries)
    assert timing["method"] == "boundary_hooks_single_backward"
    assert timing["status"] == "segmented"
    assert timing["backward_total"] >= 0.0
    assert timing["backward_vocab_readout"] >= 0.0
    assert timing["backward_recurrent"] >= 0.0
    assert timing["backward_embedding_prelude"] >= 0.0


def test_backward_hook_instrumentation_preserves_gradients() -> None:
    left = profiler.ce.fresh_model(20260913, 1, vocab_size=19, dimension=8, slots=2)
    right = profiler.ce.fresh_model(20260913, 1, vocab_size=19, dimension=8, slots=2)
    tokens = torch.randint(0, 19, (2, 3), dtype=torch.long)
    targets = torch.randint(0, 19, (2, 3), dtype=torch.long)
    teacher_logits = torch.randn(2, 3, 19)
    state = left.initial_state(2, device=torch.device("cpu"))
    _, left_logits, left_boundaries, _ = profiler._student_forward_timed(left, tokens, state)
    left_loss = profiler.hidden.base.distillation_loss(left_logits, teacher_logits, targets)["total"]
    profiler._backward_timed(left_loss, left_boundaries)
    right_state, right_logits = right.forward_window(tokens, right.initial_state(2, device=torch.device("cpu")))[:2]
    del right_state
    right_loss = profiler.hidden.base.distillation_loss(right_logits, teacher_logits, targets)["total"]
    right_loss.backward()
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        assert left_parameter.grad is not None
        assert right_parameter.grad is not None
        assert torch.equal(left_parameter.grad, right_parameter.grad)


def test_aggregate_excludes_warmup_and_reports_percentages() -> None:
    records = [{"timings": {**{name: 1.0 for name in profiler.STAGE_NAMES}, "total": 1.0}} for _ in range(4)]
    aggregate = profiler._aggregate(records, warmup_updates=1)
    assert aggregate["updates_total"] == 4
    assert aggregate["measured_updates"] == 3
    assert aggregate["mean_seconds"]["backward_total"] == 1.0
    assert aggregate["percent_of_total"]["adamw"] == 100.0
