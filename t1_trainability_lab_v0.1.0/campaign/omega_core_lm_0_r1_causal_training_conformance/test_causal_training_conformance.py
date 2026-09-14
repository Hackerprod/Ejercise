"""CPU-only conformance tests for the authorized R1 causal training fixes."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

RUNNER_DIR = Path(__file__).parents[1] / "omega_core_lm_0_gpu_environment_preparation"
SCRIPTS_DIR = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(RUNNER_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from omega_nominal_microbatch_runner import (  # noqa: E402
    APPROVED_SELECTION_MANIFEST,
    OmegaCoreLM0R1Technical,
    _preflight_source,
    backend_options,
    construct_adamw,
    state_source_update_for_update,
    update_window_schedule,
    validate_selected_documents,
    validate_window_schedule,
    window_for_update,
)
from run_omega_core_lm_0_r1_training_technical_preflight import select_documents  # noqa: E402


class SyntheticTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [index for index, _ in enumerate(text.split())]


def synthetic_dataset() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(8):
        rows.append({"text": f"= Synthetic Document {index} ="})
        rows.append({"text": " ".join(f"word_{index}_{token}" for token in range(512))})
    return rows


def synthetic_source() -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], dict[str, object]]:
    tokenizer = SyntheticTokenizer()
    _, manifest = select_documents(synthetic_dataset(), tokenizer)
    return _preflight_source(synthetic_dataset(), tokenizer, torch.device("cpu"), approved_manifest=manifest)


def test_old_calendar_is_rejected_and_window_one_never_reuses_window_one_state() -> None:
    old_schedule = [0, 1, 1, 1, 1]
    corrected = update_window_schedule(len(old_schedule))
    assert corrected == [0, 1, 0, 1, 0]
    with pytest.raises(ValueError, match="invalid causal window schedule"):
        validate_window_schedule(old_schedule)
    assert window_for_update(2) == 0
    assert state_source_update_for_update(2) is None
    assert state_source_update_for_update(3) == 2


def test_embedded_pin_contains_approved_document_set_and_windows() -> None:
    assert [item["document_index"] for item in APPROVED_SELECTION_MANIFEST["selected_documents"]] == [0, 1, 2, 3, 4, 5, 12, 13]
    assert all(item["selected_token_count"] == 513 for item in APPROVED_SELECTION_MANIFEST["selected_documents"])
    assert APPROVED_SELECTION_MANIFEST["windows"] == [
        {"window": 0, "input_range": [0, 256], "target_range": [1, 257], "teacher_context_range": [0, 256]},
        {"window": 1, "input_range": [256, 512], "target_range": [257, 513], "teacher_context_range": [0, 512]},
    ]


def test_continuity_matches_full_real_sequence_without_optimizer_step() -> None:
    source, _, _, _ = synthetic_source()
    tokens = source[:1, :512] % 17
    student = OmegaCoreLM0R1Technical(vocab_size=17, dimension=4, slots=2, rounds=1).eval()
    initial = student.initial_state(1, device=torch.device("cpu"))
    full_state, full_logits = student.forward_window(tokens, initial)
    first_state, first_logits = student.forward_window(tokens[:, :256], initial)
    second_state, second_logits = student.forward_window(tokens[:, 256:], first_state.detach())
    assert torch.equal(first_logits, full_logits[:, :256])
    assert torch.equal(second_logits, full_logits[:, 256:])
    assert torch.equal(second_state, full_state)


def test_window_zero_reset_matches_from_zero_run() -> None:
    source, _, _, _ = synthetic_source()
    tokens = source[:1, :513] % 17
    student = OmegaCoreLM0R1Technical(vocab_size=17, dimension=4, slots=2, rounds=1).eval()
    zero = student.initial_state(1, device=torch.device("cpu"))
    prior_state, _ = student.forward_window(tokens[:, :256], zero)
    prior_state, _ = student.forward_window(tokens[:, 256:512], prior_state.detach())
    reset_state, reset_logits = student.forward_window(tokens[:, :256], student.initial_state(1, device=torch.device("cpu")))
    from_zero_state, from_zero_logits = student.forward_window(tokens[:, :256], student.initial_state(1, device=torch.device("cpu")))
    assert torch.equal(reset_state, from_zero_state)
    assert torch.equal(reset_logits, from_zero_logits)
    assert not torch.equal(prior_state, zero)
    # Production calendar discards prior state at update 2 before window 0.
    assert state_source_update_for_update(2) is None


def test_optimizer_param_group_has_all_explicit_r1_values() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = construct_adamw(model)
    group = optimizer.param_groups[0]
    assert group["lr"] == 3e-4
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.0


def test_source_manifest_is_eight_distinct_documents_with_two_full_windows() -> None:
    selected, manifest = select_documents(synthetic_dataset(), SyntheticTokenizer())
    assert len({item["document_index"] for item in selected}) == 8
    source, input_masks, target_masks, enriched = _preflight_source(synthetic_dataset(), SyntheticTokenizer(), torch.device("cpu"), approved_manifest=manifest)
    assert tuple(source.shape) == (8, 513)
    assert all(int(mask.sum()) == 2048 for mask in target_masks)
    assert all(int(mask.sum()) == 2048 for mask in input_masks)
    assert len(enriched["selected_document_ids"]) == 8
    assert len(set(enriched["selected_document_ids"])) == 8
    assert [item["document_id"] for item in enriched["selected_documents"]] == enriched["selected_document_ids"]
    assert [item["row_range"] for item in enriched["selected_documents"]] == [item["row_range"] for item in selected]
    assert manifest["windows"] == enriched["windows"]


def test_default_source_path_rejects_unapproved_synthetic_manifest() -> None:
    with pytest.raises(ValueError, match="approved selection manifest"):
        _preflight_source(synthetic_dataset(), SyntheticTokenizer(), torch.device("cpu"))


def test_source_manifest_hash_and_length_mismatch_abort_before_model_work() -> None:
    selected, manifest = select_documents(synthetic_dataset(), SyntheticTokenizer())
    tampered = copy.deepcopy(selected)
    tampered[0]["tokens"][0] += 1
    with pytest.raises(ValueError, match="token hash mismatch"):
        validate_selected_documents(tampered, copy.deepcopy(manifest))
    shortened = copy.deepcopy(selected)
    shortened[0]["tokens"] = shortened[0]["tokens"][:-1]
    with pytest.raises(ValueError, match="exactly 513"):
        validate_selected_documents(shortened, copy.deepcopy(manifest))


def test_backend_report_contains_effective_numeric_options() -> None:
    options = backend_options(seed=123)
    assert options["torch_version"] == torch.__version__
    assert options["torch_cuda_available"] is torch.cuda.is_available()
    assert isinstance(options["deterministic_algorithms"], bool)
    assert options["float32_matmul_precision"] in {"highest", "high", "medium"}
    assert isinstance(options["threads"], int)
    assert isinstance(options["interop_threads"], int)
