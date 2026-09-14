"""Small synthetic checks for the prepared nominal microbatch runner."""

from __future__ import annotations

import tempfile
import copy
from pathlib import Path

import torch

from omega_nominal_microbatch_runner import EFFECTIVE_BATCH, OmegaCoreLM0R1Technical, RunLedger, ledger_summary, microbatch_count_for_physical_batch, microbatch_update, read_ledger


class TinyTeacher(torch.nn.Module):
    def __init__(self, vocab: int = 11) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.arange(vocab * vocab, dtype=torch.float32).reshape(vocab, vocab) / 100.0, requires_grad=False)

    def forward(self, input_ids: torch.Tensor):
        return type("Output", (), {"logits": self.table[input_ids % self.table.shape[0]]})()


def test_masked_accumulation_and_state() -> None:
    torch.manual_seed(4101)
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run-1"
        ledger = RunLedger(run_dir, "fixture-run")
        student = OmegaCoreLM0R1Technical(vocab_size=11, dimension=4, slots=2, rounds=1)
        teacher = TinyTeacher()
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
        source = torch.arange(8 * 513, dtype=torch.long).reshape(8, 513) % 11
        mask = torch.ones(8, 256, dtype=torch.bool)
        mask[0, 3:] = False
        mask[1, 9:] = False
        mask[2:, 17:] = False
        input_mask = torch.ones(8, 256, dtype=torch.bool)
        input_mask[0, 3:] = False
        input_mask[1, 11:] = False
        before = [parameter.detach().clone() for parameter in student.parameters()]
        reference = copy.deepcopy(student)
        state, metrics = microbatch_update(ledger=ledger, run_id="fixture-run", variant="shared_K1", update=1, student=student, teacher=teacher, source=source, valid_mask=mask, input_valid_mask=input_mask, optimizer=optimizer, window=0, persistent_state=None)
        events = ledger_summary(ledger.events_path)
        assert tuple(state.shape) == (8, 2, 4)
        assert metrics["physical_batch"] == 2 and metrics["microbatches"] == 4 and metrics["effective_batch"] == 8
        assert metrics["valid_tokens"] == int(mask.sum())
        assert len(events["applied_updates"]) == 1 and len(events["completed_updates"]) == 1
        events = read_ledger(ledger.events_path)
        assert sum(event["phase"] == "backward_completed" for event in events) == 4
        ledger_fields = {"document_id", "input_range", "target_range", "teacher_context_range", "window", "state_reset", "state_source_update", "valid_tokens"}
        assert all(ledger_fields <= event.keys() for event in events)
        assert all(len(event["document_id"]) == (8 if event["microbatch"] == "all" else 2) for event in events)
        assert all(event["input_range"] == [0, 256] and event["target_range"] == [1, 257] for event in events)
        weights = [event["metrics"]["weight_from_real_mask"] for event in events if event["phase"] == "loss_completed"]
        assert abs(sum(weights) - 1.0) < 1e-12 and len(set(weights)) > 1
        step_events = [event for event in events if event["phase"] == "optimizer_step_started"]
        assert len(step_events) == 1 and step_events[0]["metrics"]["clip_max_norm"] == 1.0
        assert step_events[0]["metrics"]["pre_clip_grad_norm"] >= 0.0
        assert any(not torch.equal(before_item, after_item) for before_item, after_item in zip(before, student.parameters(), strict=True))

        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4)
        all_input_state, _ = microbatch_update(ledger=RunLedger(Path(directory) / "run-reference", "fixture-reference"), run_id="fixture-reference", variant="shared_K1", update=1, student=reference, teacher=teacher, source=source, valid_mask=mask, input_valid_mask=torch.ones_like(input_mask), optimizer=reference_optimizer, window=0, persistent_state=None)
        assert not torch.equal(state[0], all_input_state[0]), "input mask must control final persistent state"


def test_memory_guard_aborts_before_backward() -> None:
    with tempfile.TemporaryDirectory() as directory:
        ledger = RunLedger(Path(directory) / "run-memory", "fixture-memory")
        student = OmegaCoreLM0R1Technical(vocab_size=11, dimension=4, slots=2, rounds=1)
        optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
        source = torch.arange(8 * 513, dtype=torch.long).reshape(8, 513) % 11
        mask = torch.ones(8, 256, dtype=torch.bool)
        try:
            microbatch_update(ledger=ledger, run_id="fixture-memory", variant="shared_K1", update=1, student=student, teacher=TinyTeacher(), source=source, valid_mask=mask, input_valid_mask=mask, optimizer=optimizer, window=0, persistent_state=None, memory_limit_rss_bytes=1)
        except MemoryError:
            pass
        else:
            raise AssertionError("memory guard did not abort")
        events = read_ledger(ledger.events_path)
        assert any(event["phase"] == "memory_guard_failed" and event["status"] == "ABORTED" for event in events)
        assert not any(event["phase"] == "backward_completed" for event in events)


def test_physical_batch_partitions_are_cpu_numerically_equivalent() -> None:
    torch.manual_seed(4110)
    source = torch.arange(EFFECTIVE_BATCH * 513, dtype=torch.long).reshape(EFFECTIVE_BATCH, 513) % 11
    mask = torch.ones(EFFECTIVE_BATCH, 256, dtype=torch.bool)
    input_mask = torch.ones_like(mask)
    template = OmegaCoreLM0R1Technical(vocab_size=11, dimension=4, slots=2, rounds=1)
    template_optimizer = torch.optim.AdamW(template.parameters(), lr=3e-4)
    initial_model = copy.deepcopy(template.state_dict())
    initial_optimizer = copy.deepcopy(template_optimizer.state_dict())
    results: dict[int, tuple[torch.Tensor, dict[str, object], torch.nn.Module, torch.optim.Optimizer]] = {}
    with tempfile.TemporaryDirectory() as directory:
        for physical_batch in (2, 4, 8):
            student = OmegaCoreLM0R1Technical(vocab_size=11, dimension=4, slots=2, rounds=1)
            student.load_state_dict(copy.deepcopy(initial_model))
            optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
            optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            state, metrics = microbatch_update(
                ledger=RunLedger(Path(directory) / f"run-{physical_batch}", f"fixture-{physical_batch}"),
                run_id=f"fixture-{physical_batch}",
                variant="shared_K1",
                update=0,
                student=student,
                teacher=TinyTeacher(),
                source=source,
                valid_mask=mask,
                input_valid_mask=input_mask,
                optimizer=optimizer,
                window=0,
                persistent_state=None,
                physical_batch=physical_batch,
            )
            events = read_ledger(Path(directory) / f"run-{physical_batch}" / "events.jsonl")
            assert sum(event["phase"] == "backward_completed" for event in events) == microbatch_count_for_physical_batch(physical_batch)
            assert all(event["input_shape"] == [physical_batch, 256] for event in events if event["microbatch"] != "all")
            assert all(event["input_range"] == [0, 256] and event["target_range"] == [1, 257] for event in events)
            assert all(event["document_id"] == [f"sequence-{index}" for index in range(start, start + physical_batch)] for start in range(0, 8, physical_batch) for event in events if event["microbatch"] == start // physical_batch)
            results[physical_batch] = (state, metrics, student, optimizer)
    baseline_state, baseline_metrics, baseline_student, baseline_optimizer = results[2]
    for physical_batch in (4, 8):
        state, metrics, student, optimizer = results[physical_batch]
        assert torch.allclose(state, baseline_state, atol=1e-5, rtol=0.0)
        for key in ("ce", "kl", "total_loss_mean"):
            assert abs(metrics["loss_summary"][key] - baseline_metrics["loss_summary"][key]) <= 1e-5
        assert abs(metrics["pre_clip_grad_norm"] - baseline_metrics["pre_clip_grad_norm"]) <= 1e-5
        for left, right in zip(baseline_student.parameters(), student.parameters(), strict=True):
            assert torch.allclose(left, right, atol=1e-5, rtol=0.0)
        assert baseline_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
        for left_state, right_state in zip(baseline_optimizer.state_dict()["state"].values(), optimizer.state_dict()["state"].values(), strict=True):
            for key in left_state:
                left_value, right_value = left_state[key], right_state[key]
                if torch.is_tensor(left_value):
                    assert torch.allclose(left_value, right_value, atol=1e-5, rtol=0.0)
                else:
                    assert left_value == right_value


def test_directory_protection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run-1"
        first = RunLedger(run_dir, "fixture-run")
        first.append({"run_id": "fixture-run", "variant": "tiny", "update": 1, "microbatch": 0, "phase": "update_started", "status": "completed", "elapsed_seconds": 0.0, "memory": {}, "document_id": ["sequence-0"], "input_range": [0, 256], "target_range": [1, 257], "teacher_context_range": [0, 256], "window": 0, "state_reset": True, "state_source_update": None, "valid_tokens": 0})
        try:
            RunLedger(run_dir, "fixture-run")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing ledger was silently reused")
        assert len(first.events_path.read_bytes()) > 0


def main() -> int:
    test_masked_accumulation_and_state()
    test_memory_guard_aborts_before_backward()
    test_directory_protection()
    print("prepared runner synthetic tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
