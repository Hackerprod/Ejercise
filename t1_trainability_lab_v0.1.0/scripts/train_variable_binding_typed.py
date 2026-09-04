"""Train and diagnose typed variable binding with reference/value slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import load_jsonl  # noqa: E402
from t1_trainability.variable_binding import VariableBindingCore, VariableBindingHead  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
ROUNDS = (1, 2, 4)
VAR_REFERENCE_ID = 32


def parse_reference(token: str) -> int:
    if token == "VAR:X":
        return VAR_REFERENCE_ID
    if token.startswith("OBJECT:"):
        return int(token.split(":", 1)[1])
    raise ValueError(f"not a reference token: {token}")


def parse_color(token: str) -> int:
    return int(token.split(":", 1)[1])


def load_split(split: str) -> TensorDataset:
    examples = load_jsonl(ROOT / "datasets" / "variable_binding" / f"{split}.jsonl")
    parsed: list[tuple[int, list[int], list[int], list[int], int, int]] = []
    for example in examples:
        sources: list[int] = []
        destinations: list[int] = []
        row_types: list[int] = []
        tokens = example.tokens
        for index, token in enumerate(tokens):
            if token == "ASSIGN":
                sources.append(parse_reference(tokens[index + 1]))
                destinations.append(parse_reference(tokens[index + 2]))
                row_types.append(VariableBindingCore.ASSIGN)
            elif token == "ATTR":
                sources.append(parse_reference(tokens[index + 1]))
                destinations.append(parse_color(tokens[index + 3]))
                row_types.append(VariableBindingCore.ATTR)
        target_object = destinations[row_types.index(VariableBindingCore.ASSIGN)]
        parsed.append((VAR_REFERENCE_ID, sources, destinations, row_types, target_object, example.target))
    max_rows = max(len(row[1]) for row in parsed)
    query_references = torch.tensor([row[0] for row in parsed], dtype=torch.long)
    sources = torch.zeros((len(parsed), max_rows), dtype=torch.long)
    destinations = torch.zeros((len(parsed), max_rows), dtype=torch.long)
    row_types = torch.zeros((len(parsed), max_rows), dtype=torch.long)
    row_mask = torch.zeros((len(parsed), max_rows), dtype=torch.bool)
    reference_targets = torch.zeros(len(parsed), dtype=torch.long)
    color_targets = torch.zeros(len(parsed), dtype=torch.long)
    for row_index, (_, row_sources, row_destinations, row_kinds, reference_target, color_target) in enumerate(parsed):
        length = len(row_sources)
        sources[row_index, :length] = torch.tensor(row_sources, dtype=torch.long)
        destinations[row_index, :length] = torch.tensor(row_destinations, dtype=torch.long)
        row_types[row_index, :length] = torch.tensor(row_kinds, dtype=torch.long)
        row_mask[row_index, :length] = True
        reference_targets[row_index] = reference_target
        color_targets[row_index] = color_target
    return TensorDataset(query_references, sources, destinations, row_types, row_mask, reference_targets, color_targets)


@torch.no_grad()
def evaluate(
    core: VariableBindingCore,
    head: VariableBindingHead,
    dataset: TensorDataset,
    rounds: int,
    value_round: int | None = None,
) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    reference_correct = 0
    value_correct = 0
    count = 0
    for query, sources, destinations, row_types, row_mask, reference_targets, color_targets in loader:
        _, _, reference_states, value_states = core(
            query,
            sources,
            destinations,
            row_types,
            row_mask,
            rounds=max(rounds, value_round or 0),
            return_states=True,
        )
        reference_state = reference_states[rounds][:, 0, :]
        reference_embeddings = F.normalize(core.reference_embedding.weight[:32], dim=-1)
        reference_prediction = torch.matmul(F.normalize(reference_state, dim=-1), reference_embeddings.transpose(0, 1)).argmax(-1)
        selected_value = value_states[value_round] if value_round is not None else value_states[rounds]
        value_prediction = head(selected_value).argmax(-1)
        reference_correct += int((reference_prediction == reference_targets).sum())
        value_correct += int((value_prediction == color_targets).sum())
        count += int(color_targets.numel())
    return reference_correct / count, value_correct / count


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "variable_binding_typed_seed101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "phase": "typed variable binding",
        "task": "variable_binding",
        "seed": args.seed,
        "dimension": 64,
        "slots": {"reference": 1, "value": 1, "workspace": 0, "total": 2},
        "rounds": 4,
        "reference_transition": "ASSIGN at r1; replacement; stable thereafter",
        "value_transition": "ATTR at r2+; overwrite each round",
        "reader": "one shared key-addressed reader with ASSIGN/ATTR row mask",
        "row_schedule": {"1": "ASSIGN", "2+": "ATTR"},
        "reference_embedding": "shared VAR:X/OBJECT:N source/destination space",
        "slot_mix": "disabled",
        "workspace": "omitted; no constraints to combine in benchmark",
        "head": "Norm(V) only",
        "early_stopping": False,
        "max_steps": args.max_steps,
        "dataset": "existing variable_binding 10000/2000/2000 splits; attributes shuffled; unresolved VAR:X query; no regeneration",
    }
    save_json(args.output_dir / "config.json", config)
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    core = VariableBindingCore(64, 33, 8, 4)
    head = VariableBindingHead(64, core.value_embedding)
    modules = nn.ModuleList((core, head))
    optimizer = torch.optim.AdamW(modules.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    metrics_path = args.output_dir / "metrics.jsonl"
    best_val = -1.0
    best_step = 0
    started = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, args.max_steps + 1):
        try:
            query, sources, destinations, row_types, row_mask, _, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            query, sources, destinations, row_types, row_mask, _, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        _, value = core(query, sources, destinations, row_types, row_mask, rounds=4)
        loss = criterion(head(value), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            _, val_value = evaluate(core, head, val_data, rounds=4)
            metric = {"step": step, "train_loss": float(loss.detach()), "val_accuracy_r4": val_value, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if val_value > best_val:
                best_val = val_value
                best_step = step
                torch.save({"config": config, "step": step, "core": core.state_dict(), "head": head.state_dict()}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    matrix = {str(rounds): evaluate(core, head, test_data, rounds=rounds)[1] for rounds in ROUNDS}
    reference_by_round = {str(rounds): evaluate(core, head, test_data, rounds=rounds)[0] for rounds in range(1, 5)}
    value_by_round = {str(rounds): evaluate(core, head, test_data, rounds=4, value_round=rounds)[1] for rounds in range(1, 5)}
    final = {
        "status": "completed",
        "finite": True,
        "seed": args.seed,
        "steps": args.max_steps,
        "best_step": best_step,
        "best_val_accuracy_r4": best_val,
        "accuracy_direct": matrix,
        "reference_accuracy_by_round": reference_by_round,
        "value_accuracy_by_round": value_by_round,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
