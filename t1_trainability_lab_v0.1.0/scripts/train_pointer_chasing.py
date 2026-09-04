"""Train Fase B fixed-alpha pointer-chasing model and evaluate accuracy[H][R]."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import load_jsonl  # noqa: E402
from t1_trainability.pointer import PointerCore, PointerHead  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
HOPS = (1, 2, 3, 4)
EVAL_ROUNDS = (1, 2, 4)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split(split: str) -> TensorDataset:
    examples = load_jsonl(ROOT / "datasets" / "pointer_chasing" / f"{split}.jsonl")
    starts = torch.tensor([int(row.metadata["start_key"]) for row in examples], dtype=torch.long)
    sources = torch.tensor([[int(value) for value in str(row.metadata["memory_sources"]).split(",")] for row in examples], dtype=torch.long)
    destinations = torch.tensor([[int(value) for value in str(row.metadata["memory_destinations"]).split(",")] for row in examples], dtype=torch.long)
    hops = torch.tensor([int(row.metadata["hop_count"]) for row in examples], dtype=torch.long)
    targets = torch.tensor([row.target for row in examples], dtype=torch.long)
    return TensorDataset(starts, sources, destinations, hops, targets)


@torch.no_grad()
def evaluate_matrix(core: PointerCore, head: PointerHead, dataset: TensorDataset) -> dict[str, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = {(hop, rounds): 0 for hop in HOPS for rounds in EVAL_ROUNDS}
    counts = {(hop, rounds): 0 for hop in HOPS for rounds in EVAL_ROUNDS}
    for starts, sources, destinations, hops, targets in loader:
        for hop in HOPS:
            selected = hops == hop
            if not selected.any():
                continue
            selected_starts = starts[selected]
            selected_sources = sources[selected]
            selected_destinations = destinations[selected]
            selected_targets = targets[selected]
            for rounds in EVAL_ROUNDS:
                execution_hops = torch.minimum(torch.full_like(selected_targets, rounds), torch.full_like(selected_targets, hop))
                state = core(selected_starts, selected_sources, selected_destinations, rounds=rounds, required_hops=execution_hops)
                predictions = head(state).argmax(dim=-1)
                correct[(hop, rounds)] += int((predictions == selected_targets).sum())
                counts[(hop, rounds)] += int(selected_targets.numel())
    return {str(hop): {str(rounds): correct[(hop, rounds)] / counts[(hop, rounds)] for rounds in EVAL_ROUNDS} for hop in HOPS}


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transition_description = (
        "pointer slot replacement: pointer_next=reader_output; no residual, alpha, or learned gate"
        if args.pointer_transition == "pointer_replacement"
        else "pre-norm residual: z=RMSNorm(h); delta=F(z,M); h_next=h+alpha*delta; no post-sum norm"
    )
    experiment_reason = (
        "T1.2 P2 pointer-slot replacement diagnostic"
        if args.pointer_transition == "pointer_replacement"
        else "Fase B fixed-alpha plus Fase C pointer-chasing causal-depth test"
    )
    alpha = None if args.pointer_transition == "pointer_replacement" else 0.5
    alpha_definition = "not used for pointer replacement" if alpha is None else "1/sqrt(R=4)"
    config = {
        "phase": "B+C",
        "task": "pointer_chasing",
        "dimension": 64,
        "slots": 1,
        "rounds": 4,
        "seed": args.seed,
        "alpha": alpha,
        "alpha_definition": alpha_definition,
        "residual": transition_description,
        "memory_reader": "differentiable key-addressed source-key mask, one weighted destination read per round",
        "execution": "training executes exactly H required transitions per sample; evaluation uses min(H,R) transitions",
        "transition": args.pointer_transition,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_steps": args.max_steps,
        "early_stopping": False,
        "experiment_reason": experiment_reason,
        "dataset_splits": "10000/2000/2000 fixed generated mappings",
    }
    save_json(output_dir / "config.json", config)
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    core = PointerCore(64, 256, 4, transition=args.pointer_transition)
    head = PointerHead(64, core.key_embedding)
    modules = nn.ModuleList((core, head))
    optimizer = torch.optim.AdamW(modules.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    best_val = -math.inf
    best_step = 0
    started = time.perf_counter()
    step = 0
    iterator = iter(train_loader)
    while step < args.max_steps:
        try:
            starts, sources, destinations, hops, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            starts, sources, destinations, hops, targets = next(iterator)
        step += 1
        optimizer.zero_grad(set_to_none=True)
        state = core(starts, sources, destinations, required_hops=hops)
        logits = head(state)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            val_matrix = evaluate_matrix(core, head, val_data)
            val_mean = sum(val_matrix[str(hop)]["4"] for hop in HOPS) / len(HOPS)
            metric = {
                "step": step,
                "train_loss": float(loss.detach()),
                "val_mean_r4": val_mean,
                "val_matrix": val_matrix,
                "gradient_norm": grad_norm,
            }
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if val_mean > best_val:
                best_val = val_mean
                best_step = step
                torch.save({"config": config, "step": step, "core": core.state_dict(), "head": head.state_dict()}, output_dir / "best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    matrix = evaluate_matrix(core, head, test_data)
    final = {
        "status": "completed",
        "task": "pointer_chasing",
        "seed": args.seed,
        "steps": step,
        "best_step": best_step,
        "best_val_mean_r4": best_val,
        "accuracy_matrix": matrix,
        "elapsed_seconds": time.perf_counter() - started,
        "finite": True,
    }
    save_json(output_dir / "final.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--pointer-transition", choices=("residual_pre_norm", "pointer_replacement"), default="residual_pre_norm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "pointer_chasing_seed101")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
