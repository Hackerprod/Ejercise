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
import run_omega_hidden_cache_production_integration as runner  # noqa: E402


def test_smoke_schedule_covers_windows_and_pair_transition() -> None:
    schedule = runner.smoke_schedule()
    assert schedule[0] == {"update": 0, "pair": 0, "window": 0}
    assert schedule[1] == {"update": 1, "pair": 0, "window": 1}
    assert schedule[2] == {"update": 2, "pair": 1, "window": 0}
    assert schedule[-1] == {"update": 7, "pair": 3, "window": 1}


def test_provenance_fail_closed_on_any_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = {"manifest_self_hash": "bad"}
    path = tmp_path / "cache_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.ProvenanceError, match="manifest self-hash"):
        runner.verify_provenance(path, expected={})


def test_provenance_shape_mismatch_fails_before_cache_use(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = runner.hidden.verify_self_hash
    monkeypatch.setattr(runner.hidden, "verify_self_hash", lambda *_args, **_kwargs: True)
    manifest = {"manifest_self_hash": "ok", "shape": [1, 2], "dtype": "float16", "representation": "wrong", "status": "CACHE_SEALED", "access": "READ_ONLY", "cache_file": str(tmp_path / "cache")}
    path = tmp_path / "cache_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.ProvenanceError, match="shape"):
        runner.verify_provenance(path, expected={})
    monkeypatch.setattr(runner.hidden, "verify_self_hash", original)


def test_cached_route_has_no_teacher_transformer_and_reads_ram_only() -> None:
    payload = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    weight = torch.eye(2)
    route = runner.ProductionHiddenRoute(payload, weight, None)
    result = route.logits([1], 0)
    assert result.shape == (1, 2, 2)
    assert route.accounting() == {"teacher_transformer_loaded": False, "teacher_transformer_forward_calls": 0}


def test_memory_gate_fails_closed_when_available_memory_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "memory_snapshot", lambda: {"rss_bytes": 10, "available_bytes": 99})
    with pytest.raises(runner.MemorySafetyError):
        runner.memory_safety_gate(minimum_available_bytes=100)


def test_lock_rejects_parallel_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache.lock"
    first = runner.CacheLock(lock_path)
    first.__enter__()
    try:
        with pytest.raises(runner.IntegrationLockError):
            with runner.CacheLock(lock_path):
                pass
    finally:
        first.__exit__(None, None, None)
    assert not lock_path.exists()


def test_checkpoint_resume_payload_preserves_schedule_and_state(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = torch.randn(1, 1, 2)
    payload = runner.checkpoint_payload(model, optimizer, state, 4, accounting={"teacher_transformer_loaded": False, "teacher_transformer_forward_calls": 0})
    path = tmp_path / "checkpoint.pt"
    runner.save_checkpoint(path, payload)
    resumed = runner.load_checkpoint(path)
    assert resumed["update"] == 4
    assert resumed["schedule_cursor"] == {"next_update": 4, "next_pair": 2, "next_window": 0}
    assert torch.equal(resumed["state"], state)
    assert resumed["accounting"]["teacher_transformer_loaded"] is False


def test_checkpoint_save_is_immutable(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    payload = runner.checkpoint_payload(model, optimizer, torch.zeros(1, 1, 1), 1, accounting={})
    runner.save_checkpoint(path, payload)
    with pytest.raises(FileExistsError):
        runner.save_checkpoint(path, payload)


def test_load_student_from_checkpoint_reaches_adamw_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

    fake_module = SimpleNamespace(fresh_model=lambda _seed, _k: Tiny())
    monkeypatch.setitem(sys.modules, "run_omega_ce_only_baseline", fake_module)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=runner.BASE_LR, betas=runner.ADAMW_BETAS, eps=runner.ADAMW_EPS, weight_decay=runner.WEIGHT_DECAY)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": torch.zeros(1, 1, 1), "update": 4}
    resumed_model, resumed_optimizer, state, update = runner._load_student_from_checkpoint(payload, 1)
    assert isinstance(resumed_model, Tiny)
    assert isinstance(resumed_optimizer, torch.optim.AdamW)
    assert torch.equal(state, payload["state"])
    assert update == 4


def test_compare_results_requires_exact_values() -> None:
    a = {"model_state_hash": "m", "optimizer_state_hash": "o", "next_state": torch.ones(1), "losses": {"total": torch.tensor(1.0)}, "clip_norm": 2.0}
    b = {"model_state_hash": "m", "optimizer_state_hash": "o", "next_state": torch.ones(1), "losses": {"total": torch.tensor(1.0)}, "clip_norm": 2.0}
    assert runner.compare_results(a, b)["passed"] is True
    b["losses"]["total"] = torch.tensor(1.0 + 1e-7)
    assert runner.compare_results(a, b)["passed"] is False


class _TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 2, bias=False)

    def initial_state(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, 1, 1, device=device)

    def forward_window(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        values = inputs.to(torch.float32).unsqueeze(-1)
        return state + values[:, -1:].mean(dim=1, keepdim=True), self.projection(values)


class _TinyTeacher(nn.Module):
    def forward(self, input_ids: Tensor):
        logits = input_ids.to(torch.float32).unsqueeze(-1) + torch.arange(2, dtype=torch.float32)
        return type("Output", (), {"logits": logits})()


def test_short_oracle_direct_vs_production_route_is_bit_exact() -> None:
    documents = [{"tokens": [index % 2 for index in range(513)], "document_index": index} for index in range(8)]
    pairs = [{"document_indices": list(range(8))}, {"document_indices": list(range(8))}]
    payload = np.zeros((2, 8, 256, 2), dtype=np.float32)
    for position, document in enumerate(documents):
        tokens = torch.tensor(document["tokens"], dtype=torch.long).view(1, -1)
        for window in (0, 1):
            payload[window, position] = runner.hidden.teacher_direct_logits(_TinyTeacher(), tokens, window)[0].numpy()
    route = runner.ProductionHiddenRoute(payload, torch.eye(2), None)

    def run(mode: str) -> dict[str, object]:
        torch.manual_seed(91)
        model = _TinyStudent()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = model.initial_state(8, device=torch.device("cpu"))
        result, final_state = runner._run_sequence(model=model, optimizer=optimizer, state=state, start_update=0, end_update=4, documents=documents, pairs=pairs, mode=mode, teacher=_TinyTeacher() if mode == "direct" else None, route=route if mode == "cached" else None)
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": final_state, "rows": result["rows"]}

    direct = run("direct")
    cached = run("cached")
    assert runner._state_dict_exact(direct["model"], cached["model"])
    assert runner._nested_exact(direct["optimizer"], cached["optimizer"])
    assert torch.equal(direct["state"], cached["state"])
    assert all(left["model_state_hash"] == right["model_state_hash"] for left, right in zip(direct["rows"], cached["rows"]))


def test_main_never_runs_without_explicit_authorization() -> None:
    with pytest.raises(runner.RealExecutionAuthorizationError):
        runner.main(["--smoke"])
