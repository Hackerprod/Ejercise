"""SU-2 single-operation runs using the current tied sequential architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.sequential_update import SequentialUpdateCore, SequentialUpdateHead


SEED = 101
VALUE_COUNT = 32
OPERATION_IDS = {"ADD": SequentialUpdateCore.OP_ADD, "SUB": SequentialUpdateCore.OP_SUB, "MUL": SequentialUpdateCore.OP_MUL}


def truth_table(operation: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(VALUE_COUNT).repeat_interleave(VALUE_COUNT)
    operands = torch.arange(VALUE_COUNT).repeat(VALUE_COUNT)
    operations = torch.full_like(values, operation)
    if operation == SequentialUpdateCore.OP_ADD:
        targets = (values + operands) % VALUE_COUNT
    elif operation == SequentialUpdateCore.OP_SUB:
        targets = (values - operands) % VALUE_COUNT
    else:
        targets = (values * operands) % VALUE_COUNT
    operation_types = operations.unsqueeze(1)
    operand_rows = operands.unsqueeze(1)
    step_mask = torch.ones((len(values), 1), dtype=torch.bool)
    return values, operation_types, operand_rows, step_mask, targets


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=tuple(OPERATION_IDS), required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("campaign") / f"sequential_update_su2_{args.operation.lower()}_seed101"
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    operation = OPERATION_IDS[args.operation]
    initial, operation_types, operands, step_mask, targets = truth_table(operation)
    if len(targets) != 1024:
        raise AssertionError(f"unexpected SU-2 table size: {len(targets)}")
    core = SequentialUpdateCore(64, 32, 3, 6)
    head = SequentialUpdateHead(64, core.value_embedding)
    modules = nn.ModuleList((core, head))
    optimizer = torch.optim.AdamW(modules.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    config = {
        "phase": "SU-2 single-operation sequential-update run",
        "seed": SEED,
        "operation": args.operation,
        "table_size": 1024,
        "input": "shared value_embedding(register)+operand_embedding+operation_embedding",
        "model": "SequentialUpdateCore(dimension=64, max_rounds=6) + tied SequentialUpdateHead",
        "operator": "shared MLP; softmax @ value_embedding.weight",
        "rounds_executed": 1,
        "recurrence": False,
        "readout_tying": True,
        "optimizer": "AdamW(lr=3e-4, weight_decay=1e-4)",
        "max_steps": args.max_steps,
        "modulus": 32,
    }
    save_json(output_dir / "config.json", config)
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = core(initial, operation_types, operands, step_mask, rounds=1)
        loss = criterion(head(value), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            with torch.no_grad():
                predictions = head(value).argmax(-1)
                accuracy = float((predictions == targets).float().mean())
            metric = {"step": step, "loss": float(loss.detach()), "accuracy": accuracy, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
    with torch.no_grad():
        value = core(initial, operation_types, operands, step_mask, rounds=1)
        logits = head(value)
        predictions = logits.argmax(-1)
        correct = int((predictions == targets).sum())
    torch.save({"config": config, "core": core.state_dict(), "head": head.state_dict()}, output_dir / "final.pt")
    final = {
        "status": "completed",
        "finite": True,
        "seed": SEED,
        "operation": args.operation,
        "steps": args.max_steps,
        "correct": correct,
        "total": len(targets),
        "accuracy": correct / len(targets),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
