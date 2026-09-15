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
    SCOPE_B_PAIRS,
    SCOPE_B_UPDATES,
    VARIANTS,
    TinyTeacher,
    WINDOW_TOKENS,
    _validate_scope_b_continue_checkpoint,
    _batch_loss,
    _checkpoint_payload,
    _config_payload,
    _finite_gradients,
    build_manifest,
    build_pair_manifest,
    canonical_hash,
    checkpoint_boundaries,
    classify_results,
    collect_all_eligible_documents,
    forward_window_adapter,
    implementation_identity,
    is_level_one_header,
    main,
    make_f_model,
    memory_guard,
    NumericalSafetyError,
    save_checkpoint,
    run_single,
    run_scope_b_continue,
    schedule,
    synthetic_documents,
    validate_policy,
    validate_resume_boundary,
    verify_manifest_extension,
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


def test_scope_b_manifest_extension_preserves_canonical_prefix() -> None:
    documents = synthetic_documents(10, 17)
    _, old_manifest = build_manifest(documents, split_name="train", pair_count=FULL_PAIRS)
    _, new_manifest = build_manifest(documents, split_name="train", pair_count=SCOPE_B_PAIRS)
    evidence = verify_manifest_extension(
        {"data_hashes": {"train_manifest": old_manifest["manifest_sha256"]}},
        old_manifest,
        new_manifest,
    )
    assert evidence["verified"]
    assert evidence["old_pair_count"] == FULL_PAIRS
    assert evidence["new_pair_count"] == SCOPE_B_PAIRS
    assert evidence["prefix"]["byte_identical"]
    assert evidence["prefix"]["old_pair_bytes_sha256"] == evidence["prefix"]["new_prefix_bytes_sha256"]


def test_scope_b_manifest_extension_rejects_corrupted_prefix() -> None:
    documents = synthetic_documents(10, 17)
    _, old_manifest = build_manifest(documents, split_name="train", pair_count=FULL_PAIRS)
    _, new_manifest = build_manifest(documents, split_name="train", pair_count=SCOPE_B_PAIRS)
    new_manifest["cyclic_pairs"]["pairs"][0]["document_indices"][0] += 1
    new_manifest["manifest_sha256"] = canonical_hash({key: value for key, value in new_manifest.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError, match="prefix"):
        verify_manifest_extension(
            {"data_hashes": {"train_manifest": old_manifest["manifest_sha256"]}},
            old_manifest,
            new_manifest,
        )


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


def test_scope_b_first_resumed_schedule_is_pair_1000() -> None:
    assert schedule(SCOPE_B_UPDATES)[FULL_UPDATES] == {"update": 2000, "window": 0, "pair": FULL_PAIRS}


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


def test_scope_b_resume_accepts_extended_manifest_and_restores_state(tmp_path: Path) -> None:
    documents = synthetic_documents(8, 17)
    train, old_manifest = build_manifest(documents, split_name="train", pair_count=FULL_PAIRS)
    _, extended_manifest = build_manifest(documents, split_name="train", pair_count=SCOPE_B_PAIRS)
    validation, validation_manifest = build_manifest(synthetic_documents(2, 17), split_name="validation")
    run_dir = tmp_path / "scope-b"
    run_single(
        run_dir=run_dir,
        run_id="scope-b",
        seed=20260913,
        variant="shared_K1",
        train_documents=train,
        validation_documents=validation,
        train_manifest=old_manifest,
        validation_manifest=validation_manifest,
        teacher=TinyTeacher(17),
        total_updates=2,
        checkpoint_interval=2,
        smoke=True,
        dimensions=(4, 1),
    )
    resumed = run_single(
        run_dir=run_dir,
        run_id="scope-b",
        seed=20260913,
        variant="shared_K1",
        train_documents=train,
        validation_documents=validation,
        train_manifest=extended_manifest,
        validation_manifest=validation_manifest,
        teacher=TinyTeacher(17),
        total_updates=4,
        checkpoint_interval=2,
        smoke=True,
        dimensions=(4, 1),
        resume_checkpoint=run_dir / "checkpoint_00002.pt",
        old_train_manifest=old_manifest,
    )
    assert resumed["manifest_extension_evidence"]["verified"]
    assert resumed["restoration_evidence"]["exact_equality"] == {"model": True, "optimizer": True, "torch_rng": True, "python_rng": True}
    assert resumed["restoration_evidence"]["first_resumed_schedule"] == {"update": 2, "window": 0, "pair": 1}
    assert [point["update"] for point in resumed["validation_curve"]] == [0, 2, 4]
    assert [point["update"] for point in resumed["previous_validation_curve"]] == [0, 2]


def test_scope_b_continue_uses_exact_b_identity_and_no_extension_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    documents = synthetic_documents(8, 17)
    train, b_manifest = build_manifest(documents, split_name="train", pair_count=SCOPE_B_PAIRS)
    validation, validation_manifest = build_manifest(synthetic_documents(2, 17), split_name="validation")
    checkpoint_payload = {
        "update": FULL_UPDATES + 500,
        "data_position": {"next_update": FULL_UPDATES + 500, "pair": (FULL_UPDATES + 500) // 2},
        "data_hashes": {
            "train_manifest": b_manifest["manifest_sha256"],
            "validation_manifest": validation_manifest["manifest_sha256"],
        },
    }
    loaded = []
    run_calls: list[dict[str, object]] = []

    def fake_load_real_documents(*, pair_count: int | None = None):
        loaded.append(pair_count)
        return train, validation, b_manifest, validation_manifest, TinyTeacher(17)

    def fake_run_single(**kwargs: object) -> dict[str, object]:
        run_calls.append(kwargs)
        return {
            "parent_checkpoint_hash": "run-parent-hash",
            "restoration_evidence": {"exact_equality": True},
            "previous_validation_curve": [],
        }

    monkeypatch.setattr("run_scientific_scoping_a._load_real_documents", fake_load_real_documents)
    monkeypatch.setattr("run_scientific_scoping_a.load_checkpoint", lambda path: (checkpoint_payload, "parent-hash"))
    monkeypatch.setattr("run_scientific_scoping_a.verify_manifest_extension", lambda *args, **kwargs: pytest.fail("extension verification was rerun"))
    monkeypatch.setattr("run_scientific_scoping_a.run_single", fake_run_single)

    report = run_scope_b_continue(tmp_path / "report", tmp_path / "checkpoint_02500.pt", "scope-b", 20260913, "shared_K1")

    assert loaded == [SCOPE_B_PAIRS]
    assert len(run_calls) == 1
    assert run_calls[0]["total_updates"] == SCOPE_B_UPDATES
    assert run_calls[0]["resume_checkpoint"] == (tmp_path / "checkpoint_02500.pt").resolve()
    assert run_calls[0]["old_train_manifest"] is None
    assert report["mode"] == "authorized_scope_b_continue"
    assert report["parent_checkpoint_hash"] == "parent-hash"
    assert report["checkpoint_update"] == 2500
    assert report["manifest_identity_evidence"]["train_manifest_exact_match"]
    assert report["manifest_identity_evidence"]["validation_manifest_exact_match"]
    assert report["extension_verification"]["old_to_new_extension_verification_rerun"] is False
    assert report["extension_verification"]["normal_exact_hash_identity_validation_active"] is True


@pytest.mark.parametrize(
    "wrong_field",
    [
        "train_manifest",
        "validation_manifest",
    ],
)
def test_scope_b_continue_rejects_non_b_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_field: str,
) -> None:
    train, b_manifest = build_manifest(synthetic_documents(8, 17), split_name="train", pair_count=SCOPE_B_PAIRS)
    _, a_manifest = build_manifest(synthetic_documents(8, 17), split_name="train", pair_count=FULL_PAIRS)
    validation, validation_manifest = build_manifest(synthetic_documents(2, 17), split_name="validation")
    checkpoint_train_hash = a_manifest["manifest_sha256"] if wrong_field == "train_manifest" else b_manifest["manifest_sha256"]
    checkpoint_validation_hash = "wrong-validation-hash" if wrong_field == "validation_manifest" else validation_manifest["manifest_sha256"]
    payload = {
        "update": 2500,
        "data_position": {"next_update": 2500, "pair": 1250},
        "data_hashes": {"train_manifest": checkpoint_train_hash, "validation_manifest": checkpoint_validation_hash},
    }
    monkeypatch.setattr("run_scientific_scoping_a._load_real_documents", lambda *, pair_count: (train, validation, b_manifest, validation_manifest, TinyTeacher(17)))
    monkeypatch.setattr("run_scientific_scoping_a.load_checkpoint", lambda path: (payload, "parent-hash"))

    with pytest.raises(ValueError, match=wrong_field):
        run_scope_b_continue(tmp_path / "report", tmp_path / "checkpoint_02500.pt", "scope-b", 20260913, "shared_K1")


@pytest.mark.parametrize(
    ("update", "message"),
    [(2000, "update >= 2500"), (2501, "boundary")],
)
def test_scope_b_continue_requires_b_update_and_boundary(update: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_scope_b_continue_checkpoint(
            {"update": update, "data_position": {"next_update": update, "pair": update // 2}}
        )


def test_scope_b_continue_cli_routes_explicit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_continue(*args: object) -> dict[str, object]:
        calls.append(args)
        return {"mode": "authorized_scope_b_continue"}

    monkeypatch.setattr("run_scientific_scoping_a.run_scope_b_continue", fake_continue)
    checkpoint = tmp_path / "checkpoint_02500.pt"
    assert main([
        "--scope-b-continue",
        "--full",
        "--confirm-smoke",
        "--resume-checkpoint",
        str(checkpoint),
        "--run-id",
        "scope-b",
        "--seed",
        "20260913",
        "--variant",
        "shared_K1",
        "--output-dir",
        str(tmp_path / "output"),
    ]) == 0
    assert len(calls) == 1
    assert calls[0][1] == checkpoint
    assert json.loads(capsys.readouterr().out)["mode"] == "authorized_scope_b_continue"


def test_fresh_single_run_selects_one_run_and_keeps_per_run_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls = 0
    run_calls: list[dict[str, object]] = []

    def fake_load_real_documents() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object], object]:
        nonlocal load_calls
        load_calls += 1
        return [], [], {"manifest_sha256": "train"}, {"manifest_sha256": "validation"}, object()

    def fake_run_single(**kwargs: object) -> dict[str, object]:
        run_calls.append(kwargs)
        return {
            "run_id": kwargs["run_id"],
            "seed": kwargs["seed"],
            "variant": kwargs["variant"],
            "run_dir": str(kwargs["run_dir"]),
        }

    monkeypatch.setattr("run_scientific_scoping_a._load_real_documents", fake_load_real_documents)
    monkeypatch.setattr("run_scientific_scoping_a.run_single", fake_run_single)
    monkeypatch.setattr("run_scientific_scoping_a.run_full", lambda output_dir: pytest.fail("fresh CLI selected full campaign"))

    output_dir = tmp_path / "results"
    k1_id = "shared_K1_seed_20260913"
    k4_id = "shared_K4_seed_20260913"
    assert main(["--run-id", k1_id, "--seed", "20260913", "--variant", "shared_K1", "--full", "--confirm-smoke", "--output-dir", str(output_dir)]) == 0
    assert main(["--run-id", k4_id, "--seed", "20260913", "--variant", "shared_K4", "--full", "--confirm-smoke", "--output-dir", str(output_dir)]) == 0

    assert load_calls == 2
    assert len(run_calls) == 2
    assert all(call["resume_checkpoint"] is None for call in run_calls)
    assert [(call["run_id"], call["seed"], call["variant"], call["run_dir"]) for call in run_calls] == [
        (k1_id, 20260913, "shared_K1", output_dir / "runs" / k1_id),
        (k4_id, 20260913, "shared_K4", output_dir / "runs" / k4_id),
    ]
    k1_report = json.loads((output_dir / f"fresh_report_{k1_id}.json").read_text())
    k4_report = json.loads((output_dir / f"fresh_report_{k4_id}.json").read_text())
    assert k1_report["run"]["run_id"] == k1_id
    assert k4_report["run"]["run_id"] == k4_id
