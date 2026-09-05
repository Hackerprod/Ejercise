"""Replay full-table H1 batches while fine-tuning H3-H6 composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from pretrain_unified_alu import make_table  # noqa: E402
from train_u0a import (  # noqa: E402
    BATCH_SIZE,
    ExampleDataset,
    build_canonical_data,
    build_optimizer,
    collate,
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
    h1_examples = make_table()
    composition = build_canonical_data(args.output_dir)["sequential_update"]["train"]
    h1_loader = DataLoader(ExampleDataset(h1_examples), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    composition_loader = DataLoader(ExampleDataset(composition), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1), collate_fn=collate)
    h1_iterator = iter(h1_loader)
    composition_iterator = iter(composition_loader)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedT1U0(64)
    model.load_state_dict(payload["model"], strict=False)
    optimizer = build_optimizer(model)
    started = time.perf_counter()
    metrics_path = args.output_dir / "metrics.jsonl"
    model.train()
    for step in range(1, args.steps + 1):
        if step % 2:
            try:
                batch = next(h1_iterator)
            except StopIteration:
                h1_iterator = iter(h1_loader)
                batch = next(h1_iterator)
            task = "h1"
            rounds = 1
        else:
            try:
                batch = next(composition_iterator)
            except StopIteration:
                composition_iterator = iter(composition_loader)
                batch = next(composition_iterator)
            task = "composition"
            rounds = batch["opcodes"].shape[1]
        optimizer.zero_grad(set_to_none=True)
        state, trace = run_rounds_with_trace(model, batch, rounds)
        loss = task_loss(model, "sequential_update", state, batch, trace)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite replay loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            metric = {"step": step, "batch": task, "loss": float(loss.detach()), "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
    torch.save({"step": args.steps, "model": model.state_dict()}, args.output_dir / "best.pt")
    result = {"status": "completed", "seed": args.seed, "steps": args.steps, "h1_batches": args.steps // 2, "composition_batches": args.steps // 2, "source_checkpoint": str(args.checkpoint), "elapsed_seconds": time.perf_counter() - started, "finite": True}
    save_json(args.output_dir / "final.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
