"""Train typed associative recall with stable query and replacement value slots."""

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

from t1_trainability.associative import AssociativeCore, AssociativeHead  # noqa: E402
from t1_trainability.data import load_jsonl  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
ROUNDS = (1, 2, 4)


def parse_symbol(token: str) -> int:
    return int(token.split(":", 1)[1])


def load_split(split: str) -> TensorDataset:
    examples = load_jsonl(ROOT / "datasets" / "associative_recall" / f"{split}.jsonl")
    parsed = []
    for example in examples:
        keys: list[int] = []
        values: list[int] = []
        for index, token in enumerate(example.tokens):
            if token == "PAIR":
                keys.append(parse_symbol(example.tokens[index + 1]))
                values.append(int(example.tokens[index + 2].split(":", 1)[1]))
        parsed.append((parse_symbol(example.query_token), keys, values, example.target))
    max_pairs = max(len(row[1]) for row in parsed)
    query_keys = torch.tensor([row[0] for row in parsed], dtype=torch.long)
    pair_keys = torch.zeros((len(parsed), max_pairs), dtype=torch.long)
    pair_values = torch.zeros((len(parsed), max_pairs), dtype=torch.long)
    pair_mask = torch.zeros((len(parsed), max_pairs), dtype=torch.bool)
    targets = torch.tensor([row[3] for row in parsed], dtype=torch.long)
    for row, (_, keys, values, _) in enumerate(parsed):
        length = len(keys)
        pair_keys[row, :length] = torch.tensor(keys, dtype=torch.long)
        pair_values[row, :length] = torch.tensor(values, dtype=torch.long)
        pair_mask[row, :length] = True
    return TensorDataset(query_keys, pair_keys, pair_values, pair_mask, targets)


@torch.no_grad()
def evaluate(core: AssociativeCore, head: AssociativeHead, dataset: TensorDataset, rounds: int, value_round: int | None = None) -> float:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = 0
    count = 0
    for query_keys, pair_keys, pair_values, pair_mask, targets in loader:
        _, value, _, value_states = core(query_keys, pair_keys, pair_values, pair_mask, rounds=max(rounds, value_round or 0), return_states=True)
        selected_value = value_states[value_round] if value_round is not None else value
        correct += int((head(selected_value).argmax(-1) == targets).sum())
        count += int(targets.numel())
    return correct / count


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "associative_typed_seed101")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "phase": "typed associative recall",
        "task": "associative_recall",
        "seed": args.seed,
        "dimension": 64,
        "slots": {"query_pointer": 1, "retrieved_value": 1, "total": 2},
        "rounds": 4,
        "query_transition": "identity; query key remains stable",
        "value_transition": "replacement with reader output each round",
        "reader": "P2 differentiable key-addressed reader adapted to PAIR rows",
        "slot_mix": "disabled",
        "head": "Norm(V_R) only",
        "early_stopping": False,
        "max_steps": args.max_steps,
        "dataset": "existing associative_recall 10000/2000/2000 splits; no regeneration",
        "experiment_reason": "Typed associative query/value slot migration; one direct read should suffice",
    }
    save_json(output_dir / "config.json", config)
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    core = AssociativeCore(64, 32, 32, 4)
    head = AssociativeHead(64, core.value_embedding)
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
            query_keys, pair_keys, pair_values, pair_mask, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            query_keys, pair_keys, pair_values, pair_mask, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        _, value = core(query_keys, pair_keys, pair_values, pair_mask, rounds=4)
        loss = criterion(head(value), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            val_accuracy = evaluate(core, head, val_data, rounds=4)
            metric = {"step": step, "train_loss": float(loss.detach()), "val_accuracy_r4": val_accuracy, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_step = step
                torch.save({"config": config, "step": step, "core": core.state_dict(), "head": head.state_dict()}, output_dir / "best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    matrix = {str(rounds): evaluate(core, head, test_data, rounds=rounds) for rounds in ROUNDS}
    per_round = {str(rounds): evaluate(core, head, test_data, rounds=4, value_round=rounds) for rounds in range(1, 5)}
    final = {"status": "completed", "finite": True, "seed": args.seed, "steps": args.max_steps, "best_step": best_step, "best_val_accuracy_r4": best_val, "accuracy_direct": matrix, "value_accuracy_by_round": per_round, "elapsed_seconds": time.perf_counter() - started}
    save_json(output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
