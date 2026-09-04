"""Train minimal P1-only typed multi-hop pointer model on existing T1 splits."""

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

from t1_trainability.data import load_jsonl  # noqa: E402
from t1_trainability.pointer import PointerCore, PointerHead  # noqa: E402


SEED = 101
HOPS = (1, 2, 3, 4)
EVAL_ROUNDS = (1, 2, 3, 4)


def parse_symbol(token: str) -> int:
    return int(token.split(":", 1)[1])


def load_split(split: str) -> TensorDataset:
    examples = load_jsonl(ROOT / "datasets" / "multi_hop" / f"{split}.jsonl")
    parsed = []
    for example in examples:
        sources: list[int] = []
        destinations: list[int] = []
        for index, token in enumerate(example.tokens):
            if token == "REL":
                sources.append(parse_symbol(example.tokens[index + 1]))
                destinations.append(parse_symbol(example.tokens[index + 2]))
        parsed.append((parse_symbol(example.query_token), sources, destinations, int(example.metadata["hop_count"]), example.target))
    max_rows = max(len(item[1]) for item in parsed)
    starts = torch.tensor([item[0] for item in parsed], dtype=torch.long)
    sources = torch.zeros((len(parsed), max_rows), dtype=torch.long)
    destinations = torch.zeros((len(parsed), max_rows), dtype=torch.long)
    memory_mask = torch.zeros((len(parsed), max_rows), dtype=torch.bool)
    hops = torch.tensor([item[3] for item in parsed], dtype=torch.long)
    targets = torch.tensor([item[4] for item in parsed], dtype=torch.long)
    for row, (_, row_sources, row_destinations, _, _) in enumerate(parsed):
        length = len(row_sources)
        sources[row, :length] = torch.tensor(row_sources, dtype=torch.long)
        destinations[row, :length] = torch.tensor(row_destinations, dtype=torch.long)
        memory_mask[row, :length] = True
    return TensorDataset(starts, sources, destinations, memory_mask, hops, targets)


@torch.no_grad()
def evaluate_matrix(core: PointerCore, head: PointerHead, dataset: TensorDataset) -> dict[str, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = {(hop, rounds): 0 for hop in HOPS for rounds in EVAL_ROUNDS}
    counts = {(hop, rounds): 0 for hop in HOPS for rounds in EVAL_ROUNDS}
    for starts, sources, destinations, memory_mask, hops, targets in loader:
        for hop in HOPS:
            selected = hops == hop
            if not selected.any():
                continue
            for rounds in EVAL_ROUNDS:
                selected_targets = targets[selected]
                execution_hops = torch.minimum(torch.full_like(selected_targets, rounds), torch.full_like(selected_targets, hop))
                state = core(
                    starts[selected],
                    sources[selected],
                    destinations[selected],
                    memory_mask[selected],
                    rounds=rounds,
                    required_hops=execution_hops,
                )
                correct[(hop, rounds)] += int((head(state).argmax(-1) == selected_targets).sum())
                counts[(hop, rounds)] += int(selected_targets.numel())
    return {str(hop): {str(rounds): correct[(hop, rounds)] / counts[(hop, rounds)] for rounds in EVAL_ROUNDS} for hop in HOPS}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "multihop_pointer_seed101")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.seed not in (101, 202, 303, 404, 505):
        raise ValueError("seed must be one of 101, 202, 303, 404, 505")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "phase": "typed multi-hop P1-only",
        "task": "multi_hop",
        "seed": args.seed,
        "dimension": 64,
        "slots": {"pointer": 1, "total": 1},
        "rounds": 4,
        "transition": "pointer_replacement",
        "reader": "P2 differentiable key-addressed reader adapted to REL rows",
        "slot_mix": "disabled",
        "alpha": None,
        "early_stopping": False,
        "max_steps": args.max_steps,
        "dataset": "existing corrected multi_hop 10000/2000/2000 splits; no regeneration",
        "experiment_reason": "Typed multi-hop P1-only transfer of validated P2 pointer mechanism",
    }
    save_json(output_dir / "config.json", config)
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    core = PointerCore(64, 32, 4, transition="pointer_replacement")
    head = PointerHead(64, core.key_embedding)
    modules = nn.ModuleList((core, head))
    optimizer = torch.optim.AdamW(modules.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    metrics_path = output_dir / "metrics.jsonl"
    best_val = -1.0
    best_step = 0
    started = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, args.max_steps + 1):
        try:
            starts, sources, destinations, memory_mask, hops, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            starts, sources, destinations, memory_mask, hops, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        logits = head(core(starts, sources, destinations, memory_mask, required_hops=hops))
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            val_matrix = evaluate_matrix(core, head, val_data)
            val_score = sum(val_matrix[str(hop)]["4"] for hop in HOPS) / len(HOPS)
            metric = {"step": step, "train_loss": float(loss.detach()), "val_matrix": val_matrix, "val_r4_mean": val_score, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if val_score > best_val:
                best_val = val_score
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
        "best_val_r4_mean": best_val,
        "test_matrix": evaluate_matrix(core, head, test_data),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
