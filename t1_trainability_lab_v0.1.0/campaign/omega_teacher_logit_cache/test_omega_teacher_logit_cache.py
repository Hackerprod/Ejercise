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
import run_omega_teacher_logit_cache as runner  # noqa: E402


def _documents(count: int = 602) -> list[dict[str, object]]:
    return [
        {"document_index": index, "full_text_sha256": f"text-{index}", "retained_513_token_sha256": f"tokens-{index}"}
        for index in range(count)
    ]


def _manifest(count: int = 602) -> dict[str, object]:
    pairs = [{"pair": pair, "document_indices": [(pair * 8 + offset) % count for offset in range(8)]} for pair in range(1000)]
    return {"document_count": count, "manifest_sha256": "frozen", "cyclic_pairs": {"pair_count": 1000, "documents_available": count, "pairs": pairs}, "documents": _documents(count)}


def test_phase_a_exact_footprint_and_layout(tmp_path: Path) -> None:
    assert runner.ENTRY_BYTES == 51_463_168
    assert runner.EXPECTED_CACHE_BYTES == 61_961_654_272
    report = runner.phase_a_report(tmp_path / "output")
    assert report["layout"]["order"] == ["window", "document", "token", "vocab"]
    assert report["expected_entries"] == 1204


def test_storage_margin_can_stop_without_construction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "disk_free_bytes", lambda _path: runner.EXPECTED_CACHE_BYTES)
    result = runner.storage_feasibility(tmp_path)
    assert result["status"] == "INCONCLUSIVE_STORAGE"
    assert result["will_construct"] is False
    assert not list(tmp_path.iterdir())


def test_build_schedule_covers_each_document_window_once_and_skips_wrap_duplicates() -> None:
    plan = runner.build_entry_plan(_manifest())
    assert len(plan) == 1204
    assert len({(row["document_index"], row["window"]) for row in plan}) == 1204
    assert all(row["pair"] <= 75 for row in plan)


def test_cache_key_has_identity_and_excludes_student_or_temperature() -> None:
    key = runner.cache_key(_documents(1)[0], 1, teacher_parameter_sha256="teacher", tokenizer_hash="tokenizer")
    fields = key["fields"]
    assert fields["window"] == 1
    assert fields["teacher_context_range"] == [0, 512]
    assert fields["representation"] == "raw_fp32_logits"
    assert "K" not in json.dumps(fields)
    assert "seed" not in json.dumps(fields)
    assert "student" not in json.dumps(fields)
    assert "temperature" not in fields
    assert runner.cache_key(_documents(1)[0], 0)["digest"] != runner.cache_key(_documents(1)[0], 1)["digest"]
    assert runner.cache_key(_documents(2)[1], 0)["digest"] != runner.cache_key(_documents(2)[0], 0)["digest"]


def test_entry_offset_and_shape_are_exact() -> None:
    assert runner.entry_offset_bytes(0, 0) == 0
    assert runner.entry_offset_bytes(1, 0) == runner.DOCUMENT_COUNT * runner.ENTRY_BYTES
    assert runner.entry_offset_bytes(1, 601) + runner.ENTRY_BYTES == runner.EXPECTED_CACHE_BYTES
    with pytest.raises(IndexError):
        runner.entry_offset_bytes(2, 0)


def test_mmap_cache_is_sealed_read_only_and_rejects_bad_lookup(tmp_path: Path) -> None:
    shape = (2, 2, 3, 5)
    cache_path = tmp_path / "teacher_logits.fp32"
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    source.tofile(cache_path)
    manifest = {"status": "CACHE_SEALED", "access": "READ_ONLY", "shape": list(shape)}
    with runner.MMapTeacherCache(cache_path, manifest) as cache:
        assert torch.equal(cache.lookup(1, 0), torch.from_numpy(source[0, 1]))
        with pytest.raises(IndexError):
            cache.lookup(2, 0)
        with pytest.raises(IndexError):
            cache.lookup(0, 2)
    with pytest.raises(runner.CacheContractError):
        runner.MMapTeacherCache(cache_path, {"status": "CACHE_SEALED", "access": "READ_WRITE", "shape": list(shape)})


class _RecordingTeacher(nn.Module):
    def __init__(self, vocab: int = 5) -> None:
        super().__init__()
        self.vocab = vocab
        self.contexts: list[Tensor] = []

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        self.contexts.append(input_ids.detach().clone())
        values = input_ids.to(torch.float32).unsqueeze(-1) + torch.arange(self.vocab, dtype=torch.float32)
        return SimpleNamespace(logits=values)


def test_window_contexts_are_distinct_and_sliced_correctly() -> None:
    teacher = _RecordingTeacher()
    source = torch.arange(513, dtype=torch.long).view(1, -1)
    window0 = runner.teacher_window_logits(teacher, source, 0, expected_vocab=5)
    window1 = runner.teacher_window_logits(teacher, source, 1, expected_vocab=5)
    assert teacher.contexts[0].shape == (1, 256)
    assert teacher.contexts[1].shape == (1, 512)
    assert torch.equal(teacher.contexts[0], source[:, :256])
    assert torch.equal(teacher.contexts[1], source[:, :512])
    assert torch.equal(window0, (source[:, :256].float().unsqueeze(-1) + torch.arange(5)).float())
    assert torch.equal(window1, (source[:, 256:512].float().unsqueeze(-1) + torch.arange(5)).float())


def test_cached_route_never_loads_or_calls_teacher() -> None:
    class Cache:
        def lookup(self, document_index: int, window: int) -> Tensor:
            return torch.full((2, 3), document_index + window, dtype=torch.float32)

    route = runner.CachedTeacherRoute(Cache())
    values = route.logits([1, 2], 1)
    assert values.shape == (2, 2, 3)
    assert runner.cache_route_contract(route) == {"teacher_loaded": False, "teacher_forward_calls": 0}


class _TinyStudent(nn.Module):
    def __init__(self, vocab: int = 5) -> None:
        super().__init__()
        self.projection = nn.Linear(3, vocab, bias=False)

    def forward_window(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        return state + inputs[:, -1:, :].mean(dim=-1, keepdim=True), self.projection(inputs)


def test_training_update_is_bit_exact_direct_vs_cached() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(2, 4, 3)
    targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    teacher = torch.randn(2, 4, 5)
    state = torch.zeros(2, 1, 1)
    result = runner.compare_training_update(
        _TinyStudent,
        lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3),
        inputs,
        targets,
        state,
        teacher,
        teacher.clone(),
    )
    assert result["passed"] is True


def test_raw_logit_gate_requires_torch_equal_and_matching_hash() -> None:
    left = torch.tensor([[1.0, 2.0]])
    assert runner.raw_logits_equal(left, left.clone())
    assert not runner.raw_logits_equal(left, left + 1e-7)


def test_benchmark_formula_canaries_are_not_inverted() -> None:
    assert runner.benchmark_gates(100, 90, 100, 90)["individual_pass"] is True
    assert runner.benchmark_gates(100, 110, 100, 90)["individual_pass"] is False
    assert runner.benchmark_gates(100, 80, 100, 80)["joint_pass"] is True
    assert runner.benchmark_gates(100, 90, 100, 90)["joint_pass"] is False
    assert runner.benchmark_gates(100, 90, 100, 90)["R_joint"] == pytest.approx(0.90)


def test_break_even_formula_and_first_run_amortization() -> None:
    assert runner.break_even_runs(20, 10, 5) == pytest.approx(4.0)
    assert runner.break_even_runs(20, 5, 5) is None
    assert 20 + 5 < 30


def test_benchmark_protocol_matches_authorized_scope() -> None:
    protocol = runner.benchmark_protocol()
    assert protocol["routes"] == ["DIRECT-K1", "CACHE-K1", "DIRECT-K4", "CACHE-K4"]
    assert protocol["updates_total"] == 152
    assert protocol["warmup_updates"] == [0, 39]
    assert protocol["primary_updates"] == [40, 151]
    assert protocol["primary_update_count"] == 112
    assert protocol["cost_ratio_formula"] == "R_cost=T_CANDIDATE/T_BASELINE"


def test_self_hashed_report_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = runner.write_self_hashed(path, {"schema": "test", "value": 3})
    assert runner.verify_self_hash(report)
    assert runner.verify_self_hash(json.loads(path.read_text(encoding="utf-8")))


def test_real_entrypoints_are_guarded() -> None:
    with pytest.raises(SystemExit):
        runner.main([])
    with pytest.raises(runner.RealExecutionAuthorizationError):
        runner.build_cache(output_root=HERE / "never-build", confirm_real_execution=False)
