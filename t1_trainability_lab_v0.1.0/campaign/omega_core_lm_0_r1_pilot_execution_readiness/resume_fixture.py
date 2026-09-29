"""Tiny CPU-only checkpoint/restart fixture for OMEGA readiness semantics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


UPDATE_INTERVAL = 500
INTERRUPTED_UPDATE = 12_347
CANONICAL_UPDATE = 12_000


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value).squeeze(-1)


class AppendOnlySegmentLedger:
    """Minimal JSONL ledger that requires parent hash on resumed segments."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: dict[str, Any]) -> None:
        required = {"run_id", "execution_segment", "update", "status", "canonical", "parent_checkpoint_hash"}
        missing = required.difference(event)
        if missing:
            raise ValueError(f"ledger event missing fields: {sorted(missing)}")
        if int(event["execution_segment"]) > 0 and not event["parent_checkpoint_hash"]:
            raise ValueError("resumed execution segment requires parent_checkpoint_hash")
        encoded = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def update_window(update: int) -> int:
    if update < 0:
        raise ValueError("update must be non-negative")
    return update % 2


def validate_update_boundary(update: int) -> None:
    if update < 0 or update % UPDATE_INTERVAL != 0:
        raise ValueError(f"canonical update must be a non-negative multiple of {UPDATE_INTERVAL}")


def canonical_resume_plan(interrupted_update: int) -> dict[str, int]:
    if interrupted_update < 0:
        raise ValueError("interrupted update must be non-negative")
    canonical_update = (interrupted_update // UPDATE_INTERVAL) * UPDATE_INTERVAL
    validate_update_boundary(canonical_update)
    return {
        "interrupted_update": interrupted_update,
        "canonical_update": canonical_update,
        "discarded_start": canonical_update + 1,
        "discarded_end": interrupted_update,
        "resume_update": canonical_update,
        "resume_window": update_window(canonical_update),
    }


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_payload(model: TinyModel, optimizer: torch.optim.Optimizer, update: int, data_position: int) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "rng_state": torch.get_rng_state(),
        "data_position": data_position,
        "provenance": {
            "seed": 20260914,
            "logical_run": "resume-fixture",
            "image_digest": "fixture-cpu-only",
            "gpu_backend": "cpu",
            "code": "resume_fixture.py",
            "config": "tiny-linear-adamw-v1",
            "data": "inline-fixture-v1",
        },
    }


def save_checkpoint(path: Path, model: TinyModel, optimizer: torch.optim.Optimizer, update: int, data_position: int) -> str:
    validate_update_boundary(update)
    torch.save(_checkpoint_payload(model, optimizer, update, data_position), path)
    return checkpoint_sha256(path)


def record_interrupted_evidence(ledger_path: Path, parent_checkpoint_hash: str) -> None:
    plan = canonical_resume_plan(INTERRUPTED_UPDATE)
    AppendOnlySegmentLedger(ledger_path).append(
        {
            "run_id": "resume-fixture",
            "execution_segment": 1,
            "update": INTERRUPTED_UPDATE,
            "status": "interrupted_evidence",
            "canonical": False,
            "parent_checkpoint_hash": parent_checkpoint_hash,
            "discarded_range": [plan["discarded_start"], plan["discarded_end"]],
            "resume_update": plan["resume_update"],
            "resume_window": plan["resume_window"],
        }
    )


def restore_checkpoint(path: Path, expected_hash: str | None = None) -> tuple[TinyModel, torch.optim.Optimizer, dict[str, Any], str]:
    actual_hash = checkpoint_sha256(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(f"checkpoint hash mismatch: expected {expected_hash}, got {actual_hash}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng_state"])
    return model, optimizer, payload, actual_hash


def _fixture_batch(data_position: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.75, 0.5], [1.25, 0.75]])
    targets = torch.tensor([0.25, 0.5, -0.25, 0.75])
    index = data_position % len(values)
    return values[index], targets[index]


def run_updates(
    model: TinyModel,
    optimizer: torch.optim.Optimizer,
    start_update: int,
    end_update: int,
    data_position: int,
    *,
    ledger: AppendOnlySegmentLedger | None = None,
    execution_segment: int = 0,
    parent_checkpoint_hash: str | None = None,
) -> int:
    for update in range(start_update, end_update):
        value, target = _fixture_batch(data_position)
        prediction = model(value)
        loss = (prediction - target).square() + torch.rand(()) * 0.01
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ledger is not None:
            ledger.append(
                {
                    "run_id": "resume-fixture",
                    "execution_segment": execution_segment,
                    "update": update,
                    "status": "completed",
                    "canonical": True,
                    "parent_checkpoint_hash": parent_checkpoint_hash,
                    "window": update_window(update),
                    "data_position": data_position,
                }
            )
        data_position += 1
    return data_position


def state_digest(model: TinyModel, optimizer: torch.optim.Optimizer, data_position: int) -> str:
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "data_position": data_position}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def create_boundary_checkpoint(path: Path, ledger_path: Path, checkpoint_update: int = 500) -> dict[str, Any]:
    validate_update_boundary(checkpoint_update)
    torch.manual_seed(20260914)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ledger = AppendOnlySegmentLedger(ledger_path)
    data_position = run_updates(model, optimizer, 0, checkpoint_update, 0, ledger=ledger)
    checkpoint_hash = save_checkpoint(path, model, optimizer, checkpoint_update, data_position)
    record_interrupted_evidence(ledger_path, checkpoint_hash)
    return {"checkpoint_hash": checkpoint_hash, "update": checkpoint_update, "data_position": data_position}


def continue_from_checkpoint(path: Path, ledger_path: Path, total_updates: int, expected_hash: str | None = None) -> dict[str, Any]:
    model, optimizer, payload, checkpoint_hash = restore_checkpoint(path, expected_hash)
    start_update = int(payload["update"])
    data_position = run_updates(
        model,
        optimizer,
        start_update,
        total_updates,
        int(payload["data_position"]),
        ledger=AppendOnlySegmentLedger(ledger_path),
        execution_segment=2,
        parent_checkpoint_hash=checkpoint_hash,
    )
    return {
        "state_digest": state_digest(model, optimizer, data_position),
        "update": total_updates,
        "data_position": data_position,
        "parent_checkpoint_hash": checkpoint_hash,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("continue",))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("total_updates", type=int)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("expected_hash")
    args = parser.parse_args()
    print(json.dumps(continue_from_checkpoint(args.checkpoint, args.ledger, args.total_updates, args.expected_hash), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
