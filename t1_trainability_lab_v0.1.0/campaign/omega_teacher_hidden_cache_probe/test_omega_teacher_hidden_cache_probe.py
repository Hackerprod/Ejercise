from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_omega_teacher_hidden_cache_probe as runner  # noqa: E402


def _documents(count: int = 602) -> list[dict[str, object]]:
    return [{"document_index": 1000 + index * 7, "full_text_sha256": f"text-{index}", "retained_513_token_sha256": f"tokens-{index}"} for index in range(count)]


def _manifest(count: int = 602) -> dict[str, object]:
    pairs = [{"pair": pair, "document_indices": [(pair * 8 + offset) % count for offset in range(8)]} for pair in range(1000)]
    return {"document_count": count, "manifest_sha256": "frozen", "cyclic_pairs": {"pair_count": 1000, "documents_available": count, "pairs": pairs}, "documents": _documents(count)}


def test_phase_a_hidden_footprint_exact() -> None:
    assert runner.ENTRY_BYTES == 786_432
    assert runner.EXPECTED_CACHE_BYTES == 946_864_128
    report = runner.phase_a_report(HERE / "results")
    assert report["expected_entries"] == 1204
    assert report["layout"]["order"] == ["window", "document", "token", "hidden"]


def test_hidden_storage_margin_and_default_nvme_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.base, "disk_free_bytes", lambda _path: runner.EXPECTED_CACHE_BYTES)
    result = runner.storage_feasibility(tmp_path)
    assert result["status"] == "INCONCLUSIVE_STORAGE"
    assert runner.DEFAULT_BACKING_STORE.drive.upper() == "C:"


def test_noncontiguous_document_identity_does_not_change_position_schedule() -> None:
    plan = runner.build_entry_plan(_manifest())
    assert len(plan) == 1204
    assert len({(row["document_index"], row["window"]) for row in plan}) == 1204
    selected = runner.base.scheduled_documents(_documents(8), {"document_indices": [6, 2, 7, 0]})
    assert [item["document_index"] for item in selected] == [1042, 1014, 1049, 1000]


def test_hidden_key_excludes_k_seed_student_and_temperature() -> None:
    key = runner.hidden_cache_key(_documents(1)[0], 1, teacher_parameter_sha256="teacher", tokenizer_hash="tokenizer")
    fields = key["fields"]
    assert fields["representation"] == "raw_fp32_hidden"
    assert fields["hidden_size"] == 768
    assert "temperature" not in fields and "K" not in json.dumps(fields) and "seed" not in json.dumps(fields) and "student" not in json.dumps(fields)


def test_hidden_offsets_and_tiny_readonly_mmap(tmp_path: Path) -> None:
    assert runner.hidden_offset_bytes(0, 0) == 0
    assert runner.hidden_offset_bytes(1, 601) + runner.ENTRY_BYTES == runner.EXPECTED_CACHE_BYTES
    shape = (2, 2, 3, 4)
    path = tmp_path / "hidden.fp32"
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    values.tofile(path)
    manifest = {"status": "CACHE_SEALED", "access": "READ_ONLY", "shape": list(shape)}
    # Tiny fixture uses a local reader contract equivalent to production shape.
    with pytest.raises(runner.HiddenCacheContractError):
        runner.HiddenMMapCache(path, manifest)


class _TinyTeacher(nn.Module):
    def __init__(self, hidden: int = 3, vocab: int = 5) -> None:
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.transformer = self
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(vocab * hidden, dtype=torch.float32).view(vocab, hidden))

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


class _TinyTransformer(nn.Module):
    def __init__(self, hidden: int = 3) -> None:
        super().__init__()
        self.hidden = hidden
        self.contexts: list[Tensor] = []

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        self.contexts.append(input_ids.detach().clone())
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_hidden_window_context_and_lm_head_reconstruction() -> None:
    teacher = _TinyTeacher()
    transformer = _TinyTransformer()
    teacher.transformer = transformer
    source = torch.arange(513, dtype=torch.long).view(1, -1)
    hidden0 = runner.hidden_window_states(teacher, source, 0, expected_hidden=3)
    hidden1 = runner.hidden_window_states(teacher, source, 1, expected_hidden=3)
    assert torch.equal(transformer.contexts[0], source[:, :256])
    assert torch.equal(transformer.contexts[1], source[:, :512])
    assert torch.equal(hidden0, source[:, :256].float().unsqueeze(-1).expand(-1, -1, 3))
    assert torch.equal(hidden1, source[:, 256:512].float().unsqueeze(-1).expand(-1, -1, 3))
    direct = teacher.lm_head(hidden1)
    assert torch.equal(runner.hidden_to_logits(hidden1, teacher.lm_head.weight), direct)


def test_hidden_to_logits_is_bit_exact_for_same_weight() -> None:
    hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    weight = torch.arange(15, dtype=torch.float32).view(5, 3)
    expected = torch.nn.functional.linear(hidden, weight)
    assert torch.equal(runner.hidden_to_logits(hidden, weight), expected)
    assert runner.tensor_hash(runner.hidden_to_logits(hidden, weight)) == runner.tensor_hash(expected)


def test_ratio_canaries_and_new_hidden_canaries() -> None:
    assert runner.benchmark_gates(100, 90, 100, 90)["individual_pass"] is True
    assert runner.benchmark_gates(100, 110, 100, 90)["individual_pass"] is False
    assert runner.benchmark_gates(100, 80, 100, 80)["joint_pass"] is True
    assert runner.benchmark_gates(100, 90, 100, 90)["joint_pass"] is False
    assert runner.benchmark_ratio(2.84, 0.10) == pytest.approx(0.0352112676056338)
    assert runner.benchmark_ratio(2.84, 0.10) <= 0.90
    assert runner.benchmark_ratio(2.84, 3.20) == pytest.approx(1.1267605633802817)
    assert runner.benchmark_ratio(2.84, 3.20) > 0.90


def test_protocol_is_preload_ram_and_timing_split() -> None:
    protocol = runner.benchmark_protocol()
    assert protocol["routes"] == ["DIRECT-K1", "DIRECT-K4", "HIDDEN-RAM-K1", "HIDDEN-RAM-K4"]
    assert protocol["updates_total"] == 152 and protocol["primary_update_count"] == 112
    assert protocol["cost_ratio_formula"] == "R_hidden=T_HIDDEN_RAM/T_DIRECT"


def test_self_hash_and_real_guard(tmp_path: Path) -> None:
    report = runner.write_self_hashed(tmp_path / "report.json", {"schema": "test", "value": 1})
    assert runner.verify_self_hash(report)
    with pytest.raises(runner.RealExecutionAuthorizationError):
        runner.build_hidden_cache(output_root=tmp_path / "out", backing_store=tmp_path / "hidden.fp32", confirm_real_execution=False)
