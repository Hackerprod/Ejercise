"""Train and diagnose typed sequential update with replacement-only register."""

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
from t1_trainability.sequential_update import SequentialUpdateCore, SequentialUpdateHead  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
ROUNDS = (1, 2, 4, 6)
HOPS = (3, 4, 5, 6)
OPERATION_IDS = {"OP:ADD": SequentialUpdateCore.OP_ADD, "OP:SUB": SequentialUpdateCore.OP_SUB, "OP:MUL": SequentialUpdateCore.OP_MUL}


def parse_value(token: str) -> int:
    return int(token.split(":", 1)[1])


def apply_operation(value: int, operation: int, operand: int) -> int:
    if operation == SequentialUpdateCore.OP_ADD:
        return (value + operand) % 32
    if operation == SequentialUpdateCore.OP_SUB:
        return (value - operand) % 32
    if operation == SequentialUpdateCore.OP_MUL:
        return (value * operand) % 32
    raise ValueError(f"unknown operation id: {operation}")


def load_split(split: str) -> TensorDataset:
    examples = load_jsonl(ROOT / "datasets" / "sequential_update" / f"{split}.jsonl")
    parsed: list[tuple[int, list[int], list[int], int, list[int]]] = []
    for example in examples:
        initial = 0
        operation_types: list[int] = []
        operands: list[int] = []
        trajectory: list[int] = []
        for index, token in enumerate(example.tokens):
            if token == "INITIAL":
                initial = parse_value(example.tokens[index + 1])
            elif token == "STEP":
                operation = OPERATION_IDS[example.tokens[index + 1]]
                operand = parse_value(example.tokens[index + 2])
                operation_types.append(operation)
                operands.append(operand)
        current = initial
        trajectory.append(current)
        for operation, operand in zip(operation_types, operands):
            current = apply_operation(current, operation, operand)
            trajectory.append(current)
        if current != example.target:
            raise ValueError(f"solver mismatch: {example.tokens}")
        parsed.append((initial, operation_types, operands, example.target, trajectory))
    max_steps = 6
    initial_values = torch.tensor([row[0] for row in parsed], dtype=torch.long)
    operation_types = torch.zeros((len(parsed), max_steps), dtype=torch.long)
    operands = torch.zeros((len(parsed), max_steps), dtype=torch.long)
    step_mask = torch.zeros((len(parsed), max_steps), dtype=torch.bool)
    targets = torch.tensor([row[3] for row in parsed], dtype=torch.long)
    trajectories = torch.zeros((len(parsed), max_steps + 1), dtype=torch.long)
    hop_counts = torch.zeros(len(parsed), dtype=torch.long)
    for row_index, (_, row_operations, row_operands, _, row_trajectory) in enumerate(parsed):
        length = len(row_operations)
        operation_types[row_index, :length] = torch.tensor(row_operations, dtype=torch.long)
        operands[row_index, :length] = torch.tensor(row_operands, dtype=torch.long)
        step_mask[row_index, :length] = True
        trajectories[row_index, : len(row_trajectory)] = torch.tensor(row_trajectory, dtype=torch.long)
        hop_counts[row_index] = length
    return TensorDataset(initial_values, operation_types, operands, step_mask, targets, trajectories, hop_counts)


@torch.no_grad()
def evaluate_matrix(core: SequentialUpdateCore, head: SequentialUpdateHead, dataset: TensorDataset) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, dict[str, float]]]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    by_h_and_round: dict[str, dict[str, list[bool]]] = {str(hops): {str(rounds): [] for rounds in ROUNDS} for hops in HOPS}
    round_hits: dict[str, list[bool]] = {str(rounds): [] for rounds in range(1, 7)}
    round_hits_by_h: dict[str, dict[str, list[bool]]] = {str(hops): {str(rounds): [] for rounds in range(1, 7)} for hops in HOPS}
    for initial, operation_types, operands, step_mask, targets, trajectories, hop_counts in loader:
        _, states = core(initial, operation_types, operands, step_mask, rounds=6, return_states=True)
        for rounds in ROUNDS:
            final = states[rounds]
            predictions = head(final).argmax(-1)
            for hops in HOPS:
                selected = hop_counts == hops
                by_h_and_round[str(hops)][str(rounds)].extend((predictions[selected] == targets[selected]).tolist())
        for rounds in range(1, 7):
            predictions = head(states[rounds]).argmax(-1)
            expected_indices = torch.minimum(hop_counts, torch.full_like(hop_counts, rounds))
            expected = trajectories.gather(1, expected_indices.unsqueeze(1)).squeeze(1)
            round_hits[str(rounds)].extend((predictions == expected).tolist())
            for hops in HOPS:
                selected = hop_counts == hops
                round_hits_by_h[str(hops)][str(rounds)].extend((predictions[selected] == expected[selected]).tolist())
    matrix = {hops: {rounds: sum(values) / len(values) for rounds, values in rounds_map.items()} for hops, rounds_map in by_h_and_round.items()}
    by_round = {rounds: sum(values) / len(values) for rounds, values in round_hits.items()}
    by_round_h = {hops: {rounds: sum(values) / len(values) for rounds, values in rounds_map.items()} for hops, rounds_map in round_hits_by_h.items()}
    return matrix, by_round, by_round_h


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "sequential_update_typed_seed101")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = {
        "phase": "typed sequential update",
        "task": "sequential_update",
        "seed": args.seed,
        "dimension": 64,
        "slots": {"register": 1, "workspace": 0, "total": 1},
        "rounds": 6,
        "step_reader": "hard indexed selection step_index == round_index",
        "operator": "shared MLP over Norm(V), typed operation and operand; softmax @ register embedding",
        "register_transition": "complete candidate replacement v_(r+1)=Op_r(v_r)",
        "operations": ["ADD", "SUB", "MUL"],
        "modulus": 32,
        "residual": "disabled; no accumulator",
        "head": "Norm(V) only",
        "early_stopping": False,
        "max_steps": args.max_steps,
        "dataset": "existing sequential_update 10000/2000/2000 splits; no regeneration",
    }
    save_json(args.output_dir / "config.json", config)
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    core = SequentialUpdateCore(64, 32, 3, 6)
    head = SequentialUpdateHead(64, core.value_embedding)
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
            initial, operation_types, operands, step_mask, targets, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            initial, operation_types, operands, step_mask, targets, _, _ = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        register = core(initial, operation_types, operands, step_mask, rounds=6)
        loss = criterion(head(register), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            val_matrix, _, _ = evaluate_matrix(core, head, val_data)
            val_accuracy = sum(val_matrix[str(hops)]["6"] for hops in HOPS) / len(HOPS)
            metric = {"step": step, "train_loss": float(loss.detach()), "val_accuracy_r6": val_accuracy, "gradient_norm": grad_norm}
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_step = step
                torch.save({"config": config, "step": step, "core": core.state_dict(), "head": head.state_dict()}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    matrix, register_by_round, register_by_round_h = evaluate_matrix(core, head, test_data)
    final = {
        "status": "completed",
        "finite": True,
        "seed": args.seed,
        "steps": args.max_steps,
        "best_step": best_step,
        "best_val_accuracy_r6": best_val,
        "accuracy_by_h_and_round": matrix,
        "register_accuracy_by_round": register_by_round,
        "register_accuracy_by_round_and_h": register_by_round_h,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
