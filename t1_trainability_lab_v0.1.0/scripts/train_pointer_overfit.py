"""Overfit a small fixed pointer-chasing dataset for Fase D capacity diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import generate_pointer_examples, write_jsonl  # noqa: E402
from t1_trainability.pointer import PointerCore, PointerHead  # noqa: E402
from train_pointer_chasing import EVAL_ROUNDS, HOPS, evaluate_matrix  # noqa: E402
from generate_pointer_dataset import solve  # noqa: E402


def as_dataset(examples) -> TensorDataset:
    starts = torch.tensor([int(row.metadata["start_key"]) for row in examples], dtype=torch.long)
    sources = torch.tensor([[int(value) for value in str(row.metadata["memory_sources"]).split(",")] for row in examples], dtype=torch.long)
    destinations = torch.tensor([[int(value) for value in str(row.metadata["memory_destinations"]).split(",")] for row in examples], dtype=torch.long)
    hops = torch.tensor([int(row.metadata["hop_count"]) for row in examples], dtype=torch.long)
    targets = torch.tensor([row.target for row in examples], dtype=torch.long)
    return TensorDataset(starts, sources, destinations, hops, targets)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--heldout-count", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "pointer_overfit_seed101")
    args = parser.parse_args()
    if not 128 <= args.count <= 512:
        raise ValueError("count must be between 128 and 512")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    train_examples = generate_pointer_examples("train", args.count, args.seed)
    heldout_examples = generate_pointer_examples("val", args.heldout_count, args.seed + 1)
    if any(solve(example) != example.target for example in train_examples + heldout_examples):
        raise AssertionError("independent pointer solver failed")
    write_jsonl(output_dir / "train_fixed.jsonl", train_examples)
    write_jsonl(output_dir / "heldout_fixed.jsonl", heldout_examples)
    config = {
        "phase": "D capacity probe",
        "task": "pointer_chasing",
        "dimension": 64,
        "slots": 1,
        "rounds": 4,
        "seed": args.seed,
        "alpha": 0.5,
        "fixed_train_count": args.count,
        "fixed_heldout_count": args.heldout_count,
        "max_steps": args.max_steps,
        "early_stopping": False,
        "execution": "training exact H transitions; evaluation min(H,R) transitions",
        "experiment_reason": "Fase D fixed-set overfit capacity probe before procedural generalization",
    }
    save_json(output_dir / "config.json", config)
    train_data = as_dataset(train_examples)
    heldout_data = as_dataset(heldout_examples)
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    core = PointerCore(64, 256, 4)
    head = PointerHead(64, core.key_embedding)
    modules = nn.ModuleList((core, head))
    optimizer = torch.optim.AdamW(modules.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    metrics_path = output_dir / "metrics.jsonl"
    best_score = -1.0
    best_step = 0
    started = time.perf_counter()
    iterator = iter(loader)
    last_loss = None
    for step in range(1, args.max_steps + 1):
        try:
            starts, sources, destinations, hops, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            starts, sources, destinations, hops, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        logits = head(core(starts, sources, destinations, required_hops=hops))
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        optimizer.step()
        last_loss = float(loss.detach())
        if step % 100 == 0 or step == args.max_steps:
            train_matrix = evaluate_matrix(core, head, train_data)
            score = sum(train_matrix[str(hop)]["4"] for hop in HOPS) / len(HOPS)
            metric = {"step": step, "train_loss": last_loss, "train_matrix": train_matrix, "train_r4_mean": score, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if score > best_score:
                best_score = score
                best_step = step
                torch.save({"config": config, "step": step, "core": core.state_dict(), "head": head.state_dict()}, output_dir / "best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    final = {
        "status": "completed",
        "finite": True,
        "seed": args.seed,
        "steps": args.max_steps,
        "best_step": best_step,
        "best_train_r4_mean": best_score,
        "train_matrix": evaluate_matrix(core, head, train_data),
        "heldout_matrix": evaluate_matrix(core, head, heldout_data),
        "last_train_loss": last_loss,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
