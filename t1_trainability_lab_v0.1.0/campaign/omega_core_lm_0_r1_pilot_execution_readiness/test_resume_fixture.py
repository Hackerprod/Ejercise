import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resume_fixture import (
    AppendOnlySegmentLedger,
    CANONICAL_UPDATE,
    INTERRUPTED_UPDATE,
    TinyModel,
    canonical_resume_plan,
    create_boundary_checkpoint,
    run_updates,
    state_digest,
    update_window,
    validate_update_boundary,
)


def test_interrupted_update_selects_canonical_boundary_and_window() -> None:
    plan = canonical_resume_plan(INTERRUPTED_UPDATE)
    assert plan == {
        "interrupted_update": 12347,
        "canonical_update": CANONICAL_UPDATE,
        "discarded_start": 12001,
        "discarded_end": 12347,
        "resume_update": 12000,
        "resume_window": 0,
    }
    assert update_window(500) == 0
    validate_update_boundary(12000)
    with pytest.raises(ValueError):
        validate_update_boundary(12347)


def test_checkpoint_restart_in_fresh_process_matches_continuous_execution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scientific.pt"
    ledger_path = tmp_path / "events.jsonl"
    checkpoint_info = create_boundary_checkpoint(checkpoint, ledger_path)

    torch.manual_seed(20260914)
    continuous_model = TinyModel()
    continuous_optimizer = torch.optim.AdamW(continuous_model.parameters(), lr=0.01)
    continuous_position = run_updates(continuous_model, continuous_optimizer, 0, 504, 0)
    continuous_digest = state_digest(continuous_model, continuous_optimizer, continuous_position)

    module = Path(__file__).with_name("resume_fixture.py")
    completed = subprocess.run(
        [sys.executable, str(module), "continue", str(checkpoint), "504", str(ledger_path), checkpoint_info["checkpoint_hash"]],
        check=True,
        capture_output=True,
        text=True,
    )
    resumed = json.loads(completed.stdout)
    assert resumed["state_digest"] == continuous_digest
    assert resumed["data_position"] == continuous_position
    assert resumed["update"] == 504
    assert resumed["parent_checkpoint_hash"]

    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 505
    assert events[500]["status"] == "interrupted_evidence"
    assert events[500]["canonical"] is False
    assert events[500]["discarded_range"] == [12001, 12347]
    assert events[-1]["execution_segment"] == 2
    assert events[-1]["parent_checkpoint_hash"] == resumed["parent_checkpoint_hash"]


def test_resumed_segment_requires_parent_checkpoint_hash(tmp_path: Path) -> None:
    ledger = AppendOnlySegmentLedger(tmp_path / "events.jsonl")
    with pytest.raises(ValueError, match="parent_checkpoint_hash"):
        ledger.append(
            {
                "run_id": "resume-fixture",
                "execution_segment": 1,
                "update": 500,
                "status": "completed",
                "canonical": True,
                "parent_checkpoint_hash": None,
            }
        )
