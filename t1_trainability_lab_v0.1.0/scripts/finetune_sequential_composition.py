"""Fine-tune H1-pretrained UnifiedT1U0 on H3-H6 composition sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    ExampleDataset,
    build_canonical_data,
    build_optimizer,
    collate,
    evaluate_accuracy,
    run_rounds_with_trace,
    save_json,
    task_loss,
)
from t1_trainability.unified import UnifiedT1U0  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets = build_canonical_data(args.output_dir)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    optimizer = build_optimizer(model)
    loader = DataLoader(ExampleDataset(datasets["sequential_update"]["train"]), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    iterator = iter(loader)
    best_validation = -1.0
    best_step = 0
    started = time.perf_counter()
    metrics_path = args.output_dir / "metrics.jsonl"
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        state, trace = run_rounds_with_trace(model, batch, batch["opcodes"].shape[1])
        loss = task_loss(model, "sequential_update", state, batch, trace)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite composition loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            model.eval()
            validation = evaluate_accuracy(model, "sequential_update", datasets["sequential_update"]["val"], rounds=6)
            metric = {"step": step, "loss": float(loss.detach()), "gradient_norm": grad_norm, "validation_accuracy": validation}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if validation > best_validation:
                best_validation = validation
                best_step = step
                torch.save({"step": step, "model": model.state_dict()}, args.output_dir / "best.pt")
            model.train()
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    result = {"status": "completed", "seed": args.seed, "steps": args.steps, "source_checkpoint": str(args.checkpoint), "best_step": best_step, "best_validation_accuracy": best_validation, "elapsed_seconds": time.perf_counter() - started, "finite": True}
    save_json(args.output_dir / "final.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
