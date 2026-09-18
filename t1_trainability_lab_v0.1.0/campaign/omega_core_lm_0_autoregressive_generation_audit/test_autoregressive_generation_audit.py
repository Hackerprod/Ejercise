from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_autoregressive_generation_audit as audit  # noqa: E402


def _prompt(document_id: str = "doc-1") -> audit.PromptSpec:
    return audit.PromptSpec(document_id, f"hash-{document_id}", tuple(range(audit.PROMPT_LENGTH)))


def test_generation_metrics_detect_repetition_and_cycles() -> None:
    metrics = audit.generation_metrics([1, 1, 2, 1, 1, 2], None)
    assert metrics["unique_token_proportion"] == pytest.approx(2 / 6)
    assert metrics["distinct_1"] == pytest.approx(2 / 6)
    assert metrics["distinct_2"] == pytest.approx(3 / 5)
    assert metrics["max_same_token_run"] == 2
    assert 3 in metrics["exact_cycle_lengths_1_to_8"]
    assert metrics["repeated_ngram_frequencies"]["2"]["1 1"] == 2


def test_greedy_free_running_stops_on_eos_and_preserves_ids() -> None:
    model = audit.TinyAuditModel([101, 102, audit.EOS_TOKEN_ID, 999])
    result = audit.generate_one(model, _prompt(), audit.TinyTokenizer(), max_new_tokens=8)
    assert result["generated_token_ids"] == [101, 102, audit.EOS_TOKEN_ID]
    assert result["generated_length"] == 3
    assert result["stop_reason"] == "EOS"
    assert result["metrics"]["eos_position"] == 3


def test_greedy_free_running_uses_max_length_without_sampling() -> None:
    model = audit.TinyAuditModel([7, 8, 9])
    result = audit.generate_one(model, _prompt(), audit.TinyTokenizer(), max_new_tokens=4)
    assert result["generated_token_ids"] == [7, 8, 9, 9]
    assert result["generated_length"] == 4
    assert result["stop_reason"] == "MAX_LENGTH"


@pytest.mark.parametrize("architecture", ["R1", "ER32"])
def test_checkpoint_identity_verification_requires_update_seed_k_and_architecture(architecture: str, tmp_path: Path) -> None:
    spec = audit.CheckpointSpec(architecture, 1, 20260913, tmp_path / "checkpoint.pt")
    identity = {"implementation": "F"} if architecture == "R1" else {"er32": "omega_fast_er32.OmegaCoreLMFastER32"}
    payload = {
        "update": 2000,
        "seed": 20260913,
        "config": {"seed": 20260913, "variant": "shared_K1" if architecture == "R1" else "ER32-K1", "experimental_seed": 20260913},
        "implementation_identity": identity,
        "model": {"weight": torch.tensor([1.0])},
    }
    metadata = audit.verify_checkpoint_payload(payload, spec, "abc")
    assert metadata["checkpoint_update"] == 2000
    payload["update"] = 1500
    with pytest.raises(ValueError, match="update mismatch"):
        audit.verify_checkpoint_payload(payload, spec, "abc")


def test_audit_writes_provenance_and_separate_blind_identity(tmp_path: Path) -> None:
    specs = [audit.CheckpointSpec("R1", 1, 20260913, Path("synthetic"))]

    def loader(spec: audit.CheckpointSpec) -> tuple[torch.nn.Module, dict[str, object]]:
        return audit.TinyAuditModel([4, audit.EOS_TOKEN_ID]), {
            "architecture": spec.architecture,
            "K": spec.k,
            "seed": spec.seed,
            "checkpoint_sha256": "synthetic-sha",
            "checkpoint_update": 2000,
        }

    report = audit.run_audit(specs, [_prompt("doc-1"), _prompt("doc-2")], tmp_path, audit.TinyTokenizer(), loader, max_new_tokens=8)
    assert report["generation_count"] == 2
    assert report["audit_status"] is None
    generation = json.loads((tmp_path / "generation_results.json").read_text(encoding="utf-8"))
    identity = json.loads((tmp_path / "blind_identity_map.json").read_text(encoding="utf-8"))
    blind = (tmp_path / "blind_review.md").read_text(encoding="utf-8")
    assert generation["generations"][0]["prompt_token_ids"] == list(range(32))
    assert set(identity) == {"generation-01", "generation-02"}
    assert "R1" not in blind
    assert "20260913" not in blind


def test_cli_requires_explicit_real_confirmation() -> None:
    with pytest.raises(SystemExit):
        audit.main([])


def test_real_path_configures_cpu_before_loading_context() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    real_body = source[source.index("def run_real"):source.index("class TinyTokenizer")]
    assert real_body.index("configure_cpu_runtime()") < real_body.index("load_real_context()")
    audit.configure_cpu_runtime()
    audit.configure_cpu_runtime()
