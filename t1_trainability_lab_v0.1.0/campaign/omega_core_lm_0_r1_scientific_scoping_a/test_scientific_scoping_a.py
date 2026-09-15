from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_scientific_scoping_a import (  # noqa: E402
    FULL_BOUNDARIES,
    FULL_PAIRS,
    FULL_UPDATES,
    MIN_AVAILABLE_BYTES,
    PHYSICAL_BATCH,
    SMOKE_BOUNDARIES,
    VARIANTS,
    TinyTeacher,
    WINDOW_TOKENS,
    _batch_loss,
    _checkpoint_payload,
    _config_payload,
    _finite_gradients,
    build_manifest,
    build_pair_manifest,
    checkpoint_boundaries,
    classify_results,
    collect_all_eligible_documents,
    forward_window_adapter,
    implementation_identity,
    is_level_one_header,
    make_f_model,
    memory_guard,
    NumericalSafetyError,
    save_checkpoint,
    run_single,
    schedule,
    synthetic_documents,
    validate_policy,
    validate_resume_boundary,
)
from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    OmegaCoreLM0R1Technical,
    distillation_loss,
    teacher_window_logits,
)
from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402


class TokenizerFixture:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        values: list[int] = []
        for line in text.splitlines():
            if line.startswith("= "):
                continue
            values.extend(int(value) for value in line.split(",") if value)
        return values


def row(header: str, start: int, length: int, duplicate: str | None = None) -> dict[str, str]:
    values = ",".join(str((start + index) % 17) for index in range(length))
    return {"text": f"{header}\n{values}" if duplicate is None else duplicate}


def test_all_eligible_documents_are_stable_and_deduped() -> None:
    dataset = [
        {"text": "= One ="}, row("", 0, 514),
        {"text": "= Two ="}, row("", 1, 512),
        {"text": "= Three ="}, row("", 2, 514),
        {"text": "= One ="}, row("", 0, 514),
    ]
    selected = collect_all_eligible_documents(dataset, TokenizerFixture())
    assert [item["header"] for item in selected] == ["= One =", "= Three ="]
    assert len(selected) == 2


def test_level_one_reconstruction_rule_is_exact() -> None:
    assert is_level_one_header("= Title =")
    assert not is_level_one_header("= = Subsection = =")
    assert not is_level_one_header("== malformed ==")


def test_cycle_manifest_counts_wraps_and_evidence() -> None:
    documents = synthetic_documents(10, 17)
    manifest = build_pair_manifest(documents, FULL_PAIRS)
    assert manifest["pair_count"] == 1000
    assert manifest["documents_consumed"] == 8000
    assert manifest["cycle_count"] == 800
    assert manifest["wraps"] == 600
    assert len(manifest["pairs"]) == FULL_PAIRS
    assert manifest["pairs"][0]["document_indices"] == list(range(8))
    assert manifest["pairs"][1]["document_indices"] == [8, 9, 0, 1, 2, 3, 4, 5]


def test_validation_manifest_is_deterministic_and_separate() -> None:
    train = synthetic_documents(8, 17)
    validation = synthetic_documents(4, 17)
    train, train_manifest = build_manifest(train, split_name="train", pair_count=2)
    keys = {(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in train}
    with pytest.raises(RuntimeError):
        build_manifest(validation, split_name="validation", excluded_keys=keys)
    first, manifest_a = build_manifest(synthetic_documents(4, 17), split_name="validation")
    second, manifest_b = build_manifest(synthetic_documents(4, 17), split_name="validation")
    assert [item["retained_513_token_sha256"] for item in first] == [item["retained_513_token_sha256"] for item in second]
    assert manifest_a["manifest_sha256"] == manifest_b["manifest_sha256"]
    assert train_manifest["split"] == "train"


def test_schedule_and_checkpoint_boundaries() -> None:
    assert [item["window"] for item in schedule(FULL_UPDATES)[:5]] == [0, 1, 0, 1, 0]
    assert [item["pair"] for item in schedule(5)] == [0, 0, 1, 1, 2]
    assert checkpoint_boundaries(FULL_UPDATES, 500) == FULL_BOUNDARIES
    assert checkpoint_boundaries(4, 2) == SMOKE_BOUNDARIES
    with pytest.raises(ValueError):
        checkpoint_boundaries(7, 500)
    with pytest.raises(ValueError, match="boundary"):
        validate_resume_boundary(501, 500)


def test_classification_rule_has_no_gate_claim() -> None:
    runs = [
        {"seed": 20260913, "variant": "shared_K1", "validation_curve": [{"nll": 4.0}]},
        {"seed": 20260913, "variant": "shared_K4", "validation_curve": [{"nll": 3.0}]},
        {"seed": 20260914, "variant": "shared_K1", "validation_curve": [{"nll": 5.0}]},
        {"seed": 20260914, "variant": "shared_K4", "validation_curve": [{"nll": 4.0}]},
    ]
    result = classify_results(runs)
    assert result["per_seed_delta"] == [{"seed": 20260913, "delta_k1_minus_k4": 1.0}, {"seed": 20260914, "delta_k1_minus_k4": 1.0}]
    assert result["classification"] == "PROMISING"
    assert "PASS" not in result["classification"]


def test_new_runner_has_no_external_split_or_accelerator_path() -> None:
    source = Path(__file__).with_name("run_scientific_scoping_a.py").read_text(encoding="utf-8")
    assert not re.search(r"split\s*=\s*[\"']test[\"']", source)
    assert "torch.cuda" not in source
    assert "torch.compile" not in source


def test_fp32_cpu_policy_is_explicit() -> None:
    policy = validate_policy()
    assert policy == {"device": "cpu", "dtype": "float32", "execution": "eager", "physical_batch": PHYSICAL_BATCH, "effective_batch": 8, "microbatches": 1, "optimizer": "AdamW", "teacher_frozen": True, "intraop_threads": 4, "interop_threads": 1}


def test_f_model_is_converted_from_reference_and_identity_is_pinned() -> None:
    torch.manual_seed(20260914)
    reference = OmegaCoreLM0R1Technical(vocab_size=17, dimension=4, slots=1, rounds=1, variant="shared").float()
    model = OmegaCoreLMFast.from_reference(reference)
    assert isinstance(model, OmegaCoreLMFast)
    assert implementation_identity()["implementation"] == "F"
    assert implementation_identity()["candidate_source_path"].endswith("omega_fast_candidate.py")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {id(parameter) for parameter in model.parameters()}


def test_forward_adapter_discards_candidate_trace() -> None:
    class TripleOutput(torch.nn.Module):
        vocab_size = 3

        def forward_window(self, tokens: torch.Tensor, state: torch.Tensor):
            return state + 1, torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab_size), {"trace": state}

    state = torch.zeros(2, 1, 1)
    next_state, logits = forward_window_adapter(TripleOutput(), torch.zeros(2, 4, dtype=torch.long), state)
    assert next_state.shape == state.shape
    assert logits.shape == (2, 4, 3)


def test_checkpoint_rejects_non_f_identity(tmp_path: Path) -> None:
    model = make_f_model(vocab_size=17, dimensions=(4, 1), variant="shared_K1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    config = {"seed": 1, "implementation_identity": implementation_identity()}
    payload = _checkpoint_payload(model, optimizer, update=0, data_position={"next_update": 0, "pair": 0}, run_id="x", config=config, identity_hash="x", data_hashes={}, teacher_hash="x", execution_segment=0)
    payload["implementation_identity"] = {"implementation": "original"}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)
    documents = synthetic_documents(8, 17)
    train, train_manifest = build_manifest(documents, split_name="synthetic_train", pair_count=2)
    validation, validation_manifest = build_manifest(synthetic_documents(2, 17), split_name="synthetic_validation")
    with pytest.raises(ValueError, match="implementation identity"):
        from run_scientific_scoping_a import run_single

        run_single(run_dir=tmp_path / "run", run_id="x", seed=1, variant="shared_K1", train_documents=train, validation_documents=validation, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=TinyTeacher(17), total_updates=2, checkpoint_interval=2, smoke=True, dimensions=(4, 1), resume_checkpoint=path)


def test_memory_guard_hard_stops_below_one_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("run_scientific_scoping_a.psutil.virtual_memory", lambda: type("Memory", (), {"available": MIN_AVAILABLE_BYTES - 1})())
    with pytest.raises(Exception, match="hard stop"):
        memory_guard("test")


def test_non_finite_loss_error_is_typed() -> None:
    from run_scientific_scoping_a import _finite_loss

    with pytest.raises(NumericalSafetyError, match="optimizer step aborted"):
        _finite_loss(torch.tensor(float("nan")), "test")


def test_non_finite_gradient_error_is_typed() -> None:
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    with pytest.raises(NumericalSafetyError, match="optimizer step aborted"):
        _finite_gradients(model, "test")


def test_f_identity_is_in_config_and_integration_smoke_is_bounded() -> None:
    config = _config_payload(variant="shared_K1", seed=20260913, smoke=False, dimensions=(128, 8), integration_smoke=True)
    assert config["implementation_identity"] == implementation_identity()
    assert config["integration_smoke"] is True
    source = Path(__file__).with_name("run_scientific_scoping_a.py").read_text(encoding="utf-8")
    assert "--integration-smoke" in source
    assert '"updates_per_variant": SMOKE_UPDATES' in source
    assert "total_updates=2 if phase == \"initial\" else 4" in source


def _reference_batch_loss(
    model: OmegaCoreLM0R1Technical,
    teacher: torch.nn.Module,
    documents: list[dict[str, object]],
    window: int,
    previous_states: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    losses: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    valid_tokens = 0
    for index, document in enumerate(documents):
        source = torch.tensor(document["tokens"], dtype=torch.long).unsqueeze(0)
        start = window * WINDOW_TOKENS
        inputs = source[:, start : start + WINDOW_TOKENS]
        targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
        if window == 0:
            state = model.initial_state(1, device=torch.device("cpu"))
        else:
            assert previous_states is not None
            state = previous_states[index]
        next_state, logits = forward_window_adapter(model, inputs, state)
        teacher_logits = teacher_window_logits(teacher, source, window)
        mask = torch.ones_like(targets, dtype=torch.bool)
        losses.append(distillation_loss(logits, teacher_logits[:, :WINDOW_TOKENS], targets, mask)["total"])
        next_states.append(next_state.detach())
        valid_tokens += int(mask.sum().item())
    return torch.stack(losses).mean(), next_states, valid_tokens


def test_batched_loss_matches_reference_loop_through_optimizer_step() -> None:
    torch.manual_seed(20260914)
    documents = [
        {"tokens": [((index + 1) * 3 + position) % 17 for position in range(513)]}
        for index in range(PHYSICAL_BATCH)
    ]
    reference_source = OmegaCoreLM0R1Technical(vocab_size=17, dimension=4, slots=1, rounds=1, variant="shared").to(dtype=torch.float32)
    reference = OmegaCoreLMFast.from_reference(reference_source).to(dtype=torch.float32)
    batched = OmegaCoreLMFast.from_reference(reference_source).to(dtype=torch.float32)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4)
    batched_optimizer = torch.optim.AdamW(batched.parameters(), lr=3e-4)
    teacher = TinyTeacher(17)
    reference_states: list[torch.Tensor] | None = None
    batched_states: list[torch.Tensor] | None = None

    for window in (0, 1):
        reference_optimizer.zero_grad(set_to_none=True)
        batched_optimizer.zero_grad(set_to_none=True)
        reference_loss, reference_states_next, reference_valid = _reference_batch_loss(reference, teacher, documents, window, reference_states)
        batched_loss, batched_states_next, batched_valid = _batch_loss(batched, teacher, documents, window, batched_states)
        assert batched_valid == reference_valid == PHYSICAL_BATCH * WINDOW_TOKENS
        assert torch.allclose(batched_loss, reference_loss, atol=1e-5, rtol=1e-5)
        assert all(not state.requires_grad for state in batched_states_next)
        for reference_state, batched_state in zip(reference_states_next, batched_states_next):
            assert torch.allclose(batched_state, reference_state, atol=1e-5, rtol=1e-5)
        reference_loss.backward()
        batched_loss.backward()
        for reference_parameter, batched_parameter in zip(reference.parameters(), batched.parameters()):
            assert reference_parameter.grad is not None
            assert batched_parameter.grad is not None
            assert torch.allclose(batched_parameter.grad, reference_parameter.grad, atol=1e-5, rtol=1e-5)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(batched.parameters(), 1.0)
        reference_optimizer.step()
        batched_optimizer.step()
        # Fused parameter layout differs, so equivalence is established at logits and loss above.
        reference_states = reference_states_next if window == 0 else None
        batched_states = batched_states_next if window == 0 else None


def test_curve_is_persisted_at_each_checkpoint_and_resume_deduplicates(tmp_path: Path) -> None:
    train, train_manifest = build_manifest(synthetic_documents(16, 17), split_name="synthetic_train", pair_count=2)
    validation, validation_manifest = build_manifest(synthetic_documents(2, 17), split_name="synthetic_validation")
    run_dir = tmp_path / "curve"
    first = run_single(run_dir=run_dir, run_id="curve", seed=20260913, variant="shared_K1", train_documents=train, validation_documents=validation, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=TinyTeacher(17), total_updates=2, checkpoint_interval=2, smoke=True, dimensions=(4, 1))
    assert [point["update"] for point in json.loads((run_dir / "validation_curve.json").read_text())] == [0, 2]
    resumed = run_single(run_dir=run_dir, run_id="curve", seed=20260913, variant="shared_K1", train_documents=train, validation_documents=validation, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=TinyTeacher(17), total_updates=4, checkpoint_interval=2, smoke=True, dimensions=(4, 1), resume_checkpoint=run_dir / "checkpoint_00002.pt")
    assert [point["update"] for point in resumed["validation_curve"]] == [0, 2, 4]
    assert len([json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines() if json.loads(line).get("record_type") == "update"]) == 4
    assert first["implementation_identity"]["implementation"] == "F"
