from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_omega_rank_svd_diagnostic as runner  # noqa: E402


def test_checkpoint_paths_are_exact_frozen_r1_update_2000_paths() -> None:
    specs = runner.checkpoint_specs()
    assert [(spec.seed, spec.k) for spec in specs] == [
        (20260913, 1),
        (20260913, 4),
        (20260914, 1),
        (20260914, 4),
    ]
    assert all(spec.path.as_posix().endswith(f"shared_K{spec.k}_seed_{spec.seed}/checkpoint_02000.pt") for spec in specs)


def test_primary_protocol_is_frozen_eight_document_nll_protocol() -> None:
    protocol = runner.primary_validation_protocol()
    assert protocol["document_count"] == 8
    assert protocol["document_tokens"] == 513
    assert protocol["windows"] == [0, 1]
    assert protocol["test_split"] is False
    assert protocol["evaluator"] == "run_scientific_scoping_a.evaluate_validation"


def test_full_svd_retains_all_singular_directions_and_reconstructs() -> None:
    weight = torch.arange(7 * 5, dtype=torch.float64).reshape(7, 5) / 10
    svd = runner.full_svd(weight)
    assert svd.left.shape == (7, 5)
    assert svd.singular_values.shape == (5,)
    assert svd.right_transpose.shape == (5, 5)
    assert torch.allclose(svd.left @ torch.diag(svd.singular_values) @ svd.right_transpose, weight, atol=1e-10, rtol=1e-10)


def test_rank_factors_use_symmetric_square_root_formula() -> None:
    weight = torch.randn(70, 128, generator=torch.Generator().manual_seed(7))
    svd = runner.full_svd(weight)
    c, u = runner.svd_factors(svd, 32)
    expected = svd.left[:, :32] * svd.singular_values[:32].sqrt().unsqueeze(0)
    expected_u = svd.singular_values[:32].sqrt().unsqueeze(1) * svd.right_transpose[:32]
    assert torch.equal(c, expected)
    assert torch.equal(u, expected_u)
    assert torch.allclose(c @ u, svd.left[:, :32] @ torch.diag(svd.singular_values[:32]) @ svd.right_transpose[:32], atol=1e-5, rtol=1e-5)


def test_frozen_vocab_replaces_tied_input_and_output_without_dense_weight() -> None:
    weight = torch.randn(70, 128, generator=torch.Generator().manual_seed(8))
    svd = runner.full_svd(weight)
    c, u = runner.svd_factors(svd, 64)
    vocabulary = runner.FrozenSVDVocabulary(c, u)
    tokens = torch.tensor([[1, 4, 9]])
    assert vocabulary(tokens).shape == (1, 3, 128)
    assert torch.allclose(vocabulary(tokens), torch.nn.functional.embedding(tokens, c @ u), atol=1e-5, rtol=1e-5)
    assert not hasattr(vocabulary, "weight")
    assert list(vocabulary.parameters()) == []


def test_model_replacement_copies_only_nonlexical_state_and_freezes_all() -> None:
    source = runner.OmegaCoreLMFast(vocab_size=70, dimension=128, slots=1, rounds=1, variant="shared").float()
    svd = runner.full_svd(source.embedding.weight)
    c, u = runner.svd_factors(svd, 32)
    replacement = runner.replace_tied_embedding(source, c, u)
    assert replacement.vocab_size == source.vocab_size
    assert replacement.embedding.C.shape == (70, 32)
    assert replacement.embedding.U.shape == (32, 128)
    assert not any(parameter.requires_grad for parameter in replacement.parameters())
    for name, value in source.state_dict().items():
        if name != "embedding.weight":
            assert torch.equal(value, replacement.state_dict()[name]), name


def test_self_hashed_artifact_is_write_once_and_verifiable(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic_report.json"
    digest = runner.write_self_hashed_json(path, {"status": "DIAGNOSTIC_COMPLETE", "results": []})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_self_hash"] == digest
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = runner.SELF_HASH_PLACEHOLDER
    assert runner.sha256_bytes(runner.canonical_json(unsigned)) == digest
    with pytest.raises(FileExistsError):
        runner.write_self_hashed_json(path, {"status": "DIAGNOSTIC_COMPLETE"})


def test_synthetic_smoke_is_complete_but_never_real(tmp_path: Path) -> None:
    report = runner.run_smoke(tmp_path)
    assert report["status"] == "DIAGNOSTIC_COMPLETE"
    assert report["real_checkpoint_evaluation"] is False
    assert report["training_executed"] is False
    assert report["checkpoint_count"] == 1
    assert {item["rank"] for item in report["results"]} == {32, 64}
    artifact = json.loads((tmp_path / "diagnostic_report.json").read_text(encoding="utf-8"))
    assert artifact["artifact_self_hash"] == report["artifact_self_hash"]


def test_real_cli_is_blocked_without_explicit_confirmation() -> None:
    with pytest.raises(SystemExit):
        runner.main([])
    with pytest.raises(SystemExit):
        runner.main(["--full"])


def test_real_loader_is_not_reached_by_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail() -> None:
        raise AssertionError("real context must not load during smoke")

    monkeypatch.setattr(runner, "run_real", fail)
    assert runner.main(["--smoke", "--output-dir", str(tmp_path)]) == 0
