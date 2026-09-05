"""Train U0-A canaries one task at a time through the unified model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from train_u0a import (
    BATCH_SIZE,
    TASKS,
    ExampleDataset,
    build_canonical_data,
    build_optimizer,
    collate,
    evaluate_accuracy,
    evaluate_matrix,
    evaluate_workspace_error,
    run_rounds,
    run_rounds_with_trace,
    save_json,
    task_loss,
)
from t1_trainability.unified import UnifiedT1U0


CANARIES = ("workspace_accumulation", "variable_binding", "pointer_chasing")
STEPS = 5_000


def train_canary(task: str, datasets: dict[str, dict[str, list[object]]], output_dir: Path, seed: int, steps: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = UnifiedT1U0(64)
    optimizer = build_optimizer(model)
    loader = DataLoader(ExampleDataset(datasets[task]["train"]), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=collate)
    iterator = iter(loader)
    best_validation = -float("inf")
    best_step = 0
    metrics_path = output_dir / "metrics.jsonl"
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        if task == "sequential_update":
            state, logits_trace = run_rounds_with_trace(model, batch, batch["opcodes"].shape[1])
            loss = task_loss(model, task, state, batch, logits_trace)
        else:
            loss = task_loss(model, task, run_rounds(model, batch, batch["opcodes"].shape[1]), batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {task} loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 500 == 0 or step == steps:
            model.eval()
            validation = evaluate_accuracy(model, task, datasets[task]["val"], rounds=6)
            metric = {"step": step, "loss": float(loss.detach()), "gradient_norm": grad_norm, "validation_accuracy": validation, "learning_rate": optimizer.param_groups[0]["lr"]}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if validation > best_validation:
                best_validation = validation
                best_step = step
                torch.save({"step": step, "model": model.state_dict()}, output_dir / "best.pt")
            model.train()
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if task == "workspace_accumulation":
        test = evaluate_workspace_error(model, datasets[task]["test"])
    elif task == "pointer_chasing":
        test = evaluate_matrix(model, task, datasets[task]["test"], (1, 2, 4), (1, 2, 3, 4))
    elif task == "multi_hop":
        test = evaluate_matrix(model, task, datasets[task]["test"], (1, 2, 3, 4), (1, 2, 3, 4))
    elif task == "variable_binding":
        test = evaluate_matrix(model, task, datasets[task]["test"], (1, 2, 4), (2,))
    elif task == "associative_recall":
        test = evaluate_matrix(model, task, datasets[task]["test"], (1, 2, 4), (1,))
    else:
        test = evaluate_matrix(model, task, datasets[task]["test"], (1, 2, 4, 6), (3, 4, 5, 6))
    result = {"status": "completed", "task": task, "seed": seed, "steps": steps, "best_step": best_step, "best_validation_accuracy": best_validation, "test": test, "elapsed_seconds": time.perf_counter() - started, "finite": True}
    save_json(output_dir / "final.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "campaign" / "u0a_canaries_seed101")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets = build_canonical_data(args.output_root)
    results = {}
    for task in CANARIES:
        results[task] = train_canary(task, datasets, args.output_root / task, args.seed, args.steps)
    save_json(args.output_root / "summary.json", {"status": "completed", "seed": args.seed, "steps_per_canary": args.steps, "results": results})
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
