"""Small synthetic checks for the prepared nominal microbatch runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from omega_nominal_microbatch_runner import OmegaCoreLM0R1Technical, RunLedger, ledger_summary, microbatch_update


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
        before = [parameter.detach().clone() for parameter in student.parameters()]
        state, metrics = microbatch_update(ledger=ledger, run_id="fixture-run", variant="shared_K1", update=1, student=student, teacher=teacher, source=source, valid_mask=mask, optimizer=optimizer, window=0, persistent_state=None)
        events = ledger_summary(ledger.events_path)
        assert tuple(state.shape) == (8, 2, 4)
        assert metrics["physical_batch"] == 2 and metrics["microbatches"] == 4 and metrics["effective_batch"] == 8
        assert metrics["valid_tokens"] == int(mask.sum())
        assert len(events["applied_updates"]) == 1 and len(events["completed_updates"]) == 1
        assert sum(event["phase"] == "backward_completed" for event in __import__("omega_nominal_microbatch_runner").read_ledger(ledger.events_path)) == 4
        weights = [event["metrics"]["weight_from_real_mask"] for event in __import__("omega_nominal_microbatch_runner").read_ledger(ledger.events_path) if event["phase"] == "loss_completed"]
        assert abs(sum(weights) - 1.0) < 1e-12 and len(set(weights)) > 1
        assert any(not torch.equal(before_item, after_item) for before_item, after_item in zip(before, student.parameters(), strict=True))


def test_directory_protection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run-1"
        first = RunLedger(run_dir, "fixture-run")
        first.append({"run_id": "fixture-run", "variant": "tiny", "update": 1, "microbatch": 0, "phase": "update_started", "status": "completed", "elapsed_seconds": 0.0, "memory": {}})
        try:
            RunLedger(run_dir, "fixture-run")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing ledger was silently reused")
        assert len(first.events_path.read_bytes()) > 0


def main() -> int:
    test_masked_accumulation_and_state()
    test_directory_protection()
    print("prepared runner synthetic tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
