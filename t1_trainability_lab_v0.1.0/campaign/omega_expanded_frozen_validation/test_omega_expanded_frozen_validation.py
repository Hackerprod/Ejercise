from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_omega_expanded_frozen_validation as runner  # noqa: E402


class TokenizerFixture:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        values: list[int] = []
        for line in text.splitlines():
            if line.startswith("= "):
                continue
            values.extend(int(value) for value in line.split(",") if value)
        return values


def row(header: str, start: int, length: int) -> list[dict[str, str]]:
    values = ",".join(str((start + index) % 17) for index in range(length))
    return [{"text": header}, {"text": values}]


def dataset_rows(*rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for group in rows for item in group]


def test_expanded_selection_keeps_all_eligible_documents_in_source_order() -> None:
    dataset = dataset_rows(
        row("= First =", 0, 513),
        row("= Too Short =", 1, 512),
        row("= Second =", 2, 514),
        row("= Third =", 3, 513),
    )
    selected = runner.collect_all_eligible_documents(dataset, TokenizerFixture())
    assert [item["header"] for item in selected] == ["= First =", "= Second =", "= Third ="]
    assert [item["document_index"] for item in selected] == [0, 2, 3]
    assert len(selected) == 3
    assert all(len(item["tokens"]) == runner.RETAINED_TOKENS for item in selected)


def test_expanded_selection_deduplicates_first_occurrence_only() -> None:
    duplicate = row("= Duplicate =", 4, 513)
    dataset = dataset_rows(duplicate, row("= Other =", 5, 513), duplicate)
    selected = runner.collect_all_eligible_documents(dataset, TokenizerFixture())
    assert [item["header"] for item in selected] == ["= Duplicate =", "= Other ="]
    assert [item["document_index"] for item in selected] == [0, 1]


def test_manifest_declares_no_eight_document_cap_and_is_hashed() -> None:
    manifest = runner.build_validation_manifest(runner._synthetic_documents(10))
    assert manifest["document_count"] == 10
    assert manifest["selection_scope"] == "all eligible validation documents; no eight-document cap"
    assert manifest["original_validation_document_count"] == 8
    assert manifest["manifest_sha256"] == runner.canonical_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def test_frozen_matrix_has_exact_r1_er32_k1_k4_seed13_seed14_update_2000() -> None:
    matrix = runner.frozen_target_matrix()
    assert len(matrix) == 8
    assert {(item["family"], item["variant"], item["seed"], item["update"]) for item in matrix} == {
        (family, variant, seed, 2000)
        for family in ("R1", "ER32")
        for variant in ("K1", "K4")
        for seed in (20260913, 20260914)
    }
    assert all(item["checkpoint_path"].endswith("checkpoint_02000.pt") for item in matrix)


def test_checkpoint_paths_match_historical_frozen_layout() -> None:
    assert runner.frozen_checkpoint_path("R1", "K1", 20260913).as_posix().endswith(
        "omega_core_lm_0_r1_scientific_scoping_a/results/full_campaign/runs/shared_K1_seed_20260913/checkpoint_02000.pt"
    )
    assert runner.frozen_checkpoint_path("ER32", "K4", 20260914).as_posix().endswith(
        "omega_core_lm_0_er32_quality_scoping_a/results/quality_scoping_a/runs/ER32-K4_seed_20260914/checkpoint_02000.pt"
    )


def test_primary_gate_is_explicitly_untouched_in_smoke_report(tmp_path: Path) -> None:
    report = runner.run_smoke(tmp_path)
    assert report["validation_role"] == "SECONDARY_FROZEN_VALIDATION"
    assert report["primary_gate"] == {
        "status": "UNTOUCHED",
        "reclassification": False,
        "statement": "Secondary evidence never replaces or reclassifies primary R1/ER32 gate results.",
    }
    assert "classification" not in report
    assert report["real_data_loaded"] is False
    assert report["real_checkpoints_loaded"] is False


def test_smoke_artifact_self_hash_is_valid_and_contains_eight_targets(tmp_path: Path) -> None:
    runner.run_smoke(tmp_path)
    artifact = tmp_path / "secondary_frozen_validation_smoke.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert runner.verify_self_hashed(artifact)
    assert payload["validation_role"] == "SECONDARY_FROZEN_VALIDATION"
    assert len(payload["frozen_targets"]) == 8


def test_real_evaluation_requires_explicit_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        runner.main([])
    assert "--confirm-secondary-frozen-validation" in capsys.readouterr().err


def test_full_without_confirmation_is_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        runner.main(["--full"])
    assert "requires --full --confirm-secondary-frozen-validation" in capsys.readouterr().err


def test_explicit_confirmation_routes_real_path_without_running_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[Path] = []

    def fake_run_real(output_dir: Path) -> dict[str, object]:
        calls.append(output_dir)
        return {"validation_role": "SECONDARY_FROZEN_VALIDATION", "mode": "real_frozen_evaluation"}

    monkeypatch.setattr(runner, "run_real", fake_run_real)
    assert runner.main(["--full", "--confirm-secondary-frozen-validation", "--output-dir", str(tmp_path)]) == 0
    assert calls == [tmp_path.resolve()]
    assert json.loads(capsys.readouterr().out)["validation_role"] == "SECONDARY_FROZEN_VALIDATION"


def test_smoke_does_not_load_historical_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("historical checkpoint loader called during smoke")

    monkeypatch.setattr(runner, "_load_frozen_model", fail)
    runner.run_smoke(tmp_path)
