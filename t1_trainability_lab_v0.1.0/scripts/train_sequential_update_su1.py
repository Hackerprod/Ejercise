"""SU-1 isolated one-hot baseline over the complete 32x32x3 truth table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch import nn


SEED = 101
VALUE_COUNT = 32
OPERATION_COUNT = 3
TABLE_SIZE = VALUE_COUNT * VALUE_COUNT * OPERATION_COUNT
OPERATION_NAMES = ("ADD", "SUB", "MUL")


def truth_table() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = []
    operands = []
    operations = []
    targets = []
    for operation in range(OPERATION_COUNT):
        for value in range(VALUE_COUNT):
            for operand in range(VALUE_COUNT):
                values.append(value)
                operands.append(operand)
                operations.append(operation)
                if operation == 0:
                    target = (value + operand) % VALUE_COUNT
                elif operation == 1:
                    target = (value - operand) % VALUE_COUNT
                else:
                    target = (value * operand) % VALUE_COUNT
                targets.append(target)
    value_tensor = torch.tensor(values, dtype=torch.long)
    operand_tensor = torch.tensor(operands, dtype=torch.long)
    operation_tensor = torch.tensor(operations, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    inputs = torch.cat(
        (
            F.one_hot(value_tensor, VALUE_COUNT),
            F.one_hot(operand_tensor, VALUE_COUNT),
            F.one_hot(operation_tensor, OPERATION_COUNT),
        ),
        dim=-1,
    ).to(torch.float32)
    return inputs, target_tensor, operation_tensor, torch.stack((value_tensor, operand_tensor), dim=-1)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("campaign") / "sequential_update_su1_seed101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    inputs, targets, operations, pairs = truth_table()
    if inputs.shape != (TABLE_SIZE, 67):
        raise AssertionError(f"unexpected SU-1 input shape: {tuple(inputs.shape)}")
    model = nn.Sequential(
        nn.Linear(67, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 32),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    config = {
        "phase": "SU-1 isolated sequential-update baseline",
        "seed": SEED,
        "table_size": TABLE_SIZE,
        "input": "one-hot register(32)+operand(32)+operation(3)=67",
        "model": "Linear(67,128)->SiLU->Linear(128,128)->SiLU->Linear(128,32)",
        "embeddings": False,
        "recurrence": False,
        "readout_tying": False,
        "optimizer": "Adam(lr=1e-3)",
        "max_steps": args.max_steps,
        "modulus": 32,
    }
    save_json(args.output_dir / "config.json", config)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            with torch.no_grad():
                predictions = logits.argmax(-1)
                by_operation = {
                    OPERATION_NAMES[operation]: float((predictions[operations == operation] == targets[operations == operation]).float().mean())
                    for operation in range(OPERATION_COUNT)
                }
                overall = float((predictions == targets).float().mean())
            metric = {"step": step, "loss": float(loss.detach()), "overall_accuracy": overall, "accuracy_by_operation": by_operation, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
    with torch.no_grad():
        logits = model(inputs)
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
    torch.save({"config": config, "model": model.state_dict(), "pairs": pairs, "targets": targets}, args.output_dir / "final.pt")
    final = {
        "status": "completed",
        "finite": True,
        "seed": SEED,
        "steps": args.max_steps,
        "table_size": TABLE_SIZE,
        "overall_accuracy": overall,
        "accuracy_by_operation": by_operation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
