"""Train SU-4 jointly on the complete sequential-update truth table."""

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

from t1_trainability.sequential_update_su4 import SequentialUpdateSU4Core  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
VALUE_COUNT = 32
OPERATION_COUNT = 3
OPERATION_NAMES = ("ADD", "SUB", "MUL")


def truth_table() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(VALUE_COUNT).repeat_interleave(VALUE_COUNT * OPERATION_COUNT)
    operands = torch.arange(VALUE_COUNT).repeat(VALUE_COUNT * OPERATION_COUNT)
    operations = torch.arange(OPERATION_COUNT).repeat(VALUE_COUNT * VALUE_COUNT)
    targets = torch.where(
        operations == 0,
        (values + operands) % VALUE_COUNT,
        torch.where(operations == 1, (values - operands) % VALUE_COUNT, (values * operands) % VALUE_COUNT),
    )
    return values, operands, operations, targets


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("campaign") / "sequential_update_su4_seed101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    register, operands, operations, targets = truth_table()
    if len(targets) != 3072:
        raise AssertionError(f"unexpected SU-4 table size: {len(targets)}")
    core = SequentialUpdateSU4Core(64, 32, 3)
    optimizer = torch.optim.AdamW(core.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    config = {
        "phase": "SU-4 shared trunk plus operation-specific heads",
        "seed": args.seed,
        "table_size": 3072,
        "input": "left=value_embedding(register), right=operand_embedding(operand), op_embedding, left-right",
        "left_projection": "Linear(64,64)",
        "right_projection": "Linear(64,64)",
        "trunk": "Linear(256,256)->SiLU->Linear(256,64)",
        "heads": "three Linear(64,32), selected by op_id",
        "canonicalization": "softmax(head_logits) @ value_embedding.weight",
        "recurrence": False,
        "optimizer": "AdamW(lr=3e-4, weight_decay=1e-4)",
        "max_steps": args.max_steps,
        "modulus": 32,
    }
    save_json(args.output_dir / "config.json", config)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, logits, _ = core(register, operands, operations)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            with torch.no_grad():
                predictions = logits.argmax(-1)
                by_operation = {
                    OPERATION_NAMES[operation]: float((predictions[operations == operation] == targets[operations == operation]).float().mean())
                    for operation in range(OPERATION_COUNT)
                }
            metric = {"step": step, "loss": float(loss.detach()), "overall_accuracy": float((predictions == targets).float().mean()), "accuracy_by_operation": by_operation, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
    with torch.no_grad():
        _, logits, _ = core(register, operands, operations)
        predictions = logits.argmax(-1)
        by_operation = {
            OPERATION_NAMES[operation]: {
                "correct": int((predictions[operations == operation] == targets[operations == operation]).sum()),
                "total": int((operations == operation).sum()),
                "accuracy": float((predictions[operations == operation] == targets[operations == operation]).float().mean()),
            }
            for operation in range(OPERATION_COUNT)
        }
        overall = float((predictions == targets).float().mean())
    torch.save({"config": config, "core": core.state_dict()}, args.output_dir / "final.pt")
    final = {"status": "completed", "finite": True, "seed": args.seed, "steps": args.max_steps, "table_size": len(targets), "overall_accuracy": overall, "accuracy_by_operation": by_operation, "elapsed_seconds": time.perf_counter() - started}
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
