"""Focused invariant checks for Step 2 runner; does not launch benchmark."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from run_step2_benchmark import SELF_HASH_PLACEHOLDER, canonical_report_bytes, sha256_bytes, write_report


UNIT_DIR = Path(__file__).resolve().parent
RUNNER = UNIT_DIR / "run_step2_benchmark.py"


def test_runner_isolated_and_has_required_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert '"R", "F", "L", "C"' in source
    assert "WINDOW_SCHEDULE = [0, 1, 0, 1, 0, 1]" in source
    assert '"total_updates": 48' in source
    assert '"projection_calculated": False' in source
    assert '"scope_a_relaunched": False' in source
    assert '"test_split_loaded": False' in source
    assert "from run_scientific_scoping_a import configure_cpu_runtime" in source
    assert "configure_cpu_runtime()" in source
    assert not any(isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "set_num_threads" for node in ast.walk(tree) if getattr(node, "lineno", 0) > 75)


def test_shared_runtime_policy_is_four_intraop_one_interop() -> None:
    import torch

    assert torch.get_num_threads() == 4
    assert torch.get_num_interop_threads() == 1


def test_runner_does_not_import_protected_fable_or_scope_a_runner() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "from omega_fast import" not in source
    assert "scope_a" not in source.lower().replace("scope_a_relaunched", "")
    assert "torch.compile" not in source
    assert "cuda" not in source.lower()


def test_candidate_exposes_fast_readout_and_lse_paths() -> None:
    candidate = (UNIT_DIR / "omega_fast_candidate.py").read_text(encoding="utf-8")
    assert "def recur_states" in candidate
    assert "def logits_from_states" in candidate
    assert "def distillation_loss_lse" in candidate


def test_write_report_roundtrip_self_hash_with_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = {
        "status": "completed",
        "path": report_path,
        "nested": {"ledger": Path("relative/ledger.json"), "values": (1, 2)},
        "artifact_self_hash": "stale",
    }

    digest, file_hash = write_report(report_path, report)
    persisted = report_path.read_bytes()
    parsed = json.loads(persisted.decode("utf-8"))
    stored_hash = parsed["artifact_self_hash"]
    parsed["artifact_self_hash"] = SELF_HASH_PLACEHOLDER

    assert stored_hash == digest
    assert canonical_report_bytes(parsed) == persisted.replace(stored_hash.encode("ascii"), SELF_HASH_PLACEHOLDER.encode("ascii"), 1)
    assert sha256_bytes(canonical_report_bytes(parsed)) == stored_hash
    assert file_hash == sha256_bytes(persisted)
    assert json.loads(persisted.decode("utf-8"))["path"] == report_path.as_posix()
    assert json.loads(persisted.decode("utf-8"))["nested"]["ledger"] == "relative/ledger.json"


def test_write_report_rejects_mutation_after_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_step2_benchmark as runner

    original_canonical_report_bytes = runner.canonical_report_bytes
    calls = 0

    def mutate_on_final_serialization(snapshot: dict[str, object]) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            snapshot["status"] = "mutated-after-digest"
        return original_canonical_report_bytes(snapshot)

    monkeypatch.setattr(runner, "canonical_report_bytes", mutate_on_final_serialization)
    with pytest.raises(RuntimeError, match="self-hash verification failed"):
        runner.write_report(tmp_path / "report.json", {"status": "before"})
