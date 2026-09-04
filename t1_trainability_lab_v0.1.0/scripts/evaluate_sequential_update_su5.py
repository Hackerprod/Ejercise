"""SU-5 teacher-forcing versus free-running evaluation of frozen SU-4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.sequential_update_su4 import SequentialUpdateSU4Core  # noqa: E402


SEEDS = (101, 202, 303, 404, 505)
VALUE_COUNT = 32
OPERATION_COUNT = 3
HOPS = (1, 2, 3, 4)
ROUNDS = (1, 2, 4)


def apply_operation(value: int, operation: int, operand: int) -> int:
    if operation == SequentialUpdateSU4Core.OP_ADD:
        return (value + operand) % VALUE_COUNT
    if operation == SequentialUpdateSU4Core.OP_SUB:
        return (value - operand) % VALUE_COUNT
    return (value * operand) % VALUE_COUNT


def generate_fixed_h(hops: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed + 100_003 * hops)
    rows: list[tuple[int, list[int], list[int], list[int]]] = []
    for _ in range(count):
        if hops < 3:
            initial = rng.randrange(VALUE_COUNT)
            operations: list[int] = []
            operands: list[int] = []
            trajectory = [initial]
            current = initial
            for _step in range(hops):
                candidates: list[tuple[int, int]] = []
                for operation in range(3):
                    for operand in range(VALUE_COUNT):
                        next_value = apply_operation(current, operation, operand)
                        if next_value != current and (operation != SequentialUpdateSU4Core.OP_MUL or current * operand < VALUE_COUNT):
                            candidates.append((operation, operand))
                operation, operand = rng.choice(candidates)
                current = apply_operation(current, operation, operand)
                operations.append(operation)
                operands.append(operand)
                trajectory.append(current)
            rows.append((initial, operations, operands, trajectory))
            continue
        target = rng.randrange(VALUE_COUNT)
        for _attempt in range(1000):
            multiply_position = rng.randrange(hops)
            reverse_values = [target]
            reverse_operations: list[int] = []
            reverse_operands: list[int] = []
            possible = True
            for reverse_step in range(hops):
                forward_step = hops - reverse_step - 1
                operation = SequentialUpdateSU4Core.OP_MUL if forward_step == multiply_position else rng.choice((SequentialUpdateSU4Core.OP_ADD, SequentialUpdateSU4Core.OP_SUB))
                current = reverse_values[-1]
                if operation == SequentialUpdateSU4Core.OP_ADD:
                    candidates = list(range(1, current + 1)) or [0]
                    operand = rng.choice(candidates)
                    previous = current - operand
                elif operation == SequentialUpdateSU4Core.OP_SUB:
                    candidates = list(range(1, VALUE_COUNT - current)) or [0]
                    operand = rng.choice(candidates)
                    previous = current + operand
                else:
                    factors = [factor for factor in range(2, VALUE_COUNT) if current % factor == 0]
                    if not factors:
                        possible = False
                        break
                    operand = rng.choice(factors)
                    previous = current // operand
                reverse_values.append(previous)
                reverse_operations.append(operation)
                reverse_operands.append(operand)
            if not possible:
                continue
            operations = list(reversed(reverse_operations))
            operands = list(reversed(reverse_operands))
            initial = reverse_values[-1]
            trajectory = [initial]
            current = initial
            for operation, operand in zip(operations, operands):
                current = apply_operation(current, operation, operand)
                trajectory.append(current)
            if current == target:
                rows.append((initial, operations, operands, trajectory))
                break
        else:
            raise RuntimeError(f"unable to generate fixed-H sequence: H={hops}")
    initial = torch.tensor([row[0] for row in rows], dtype=torch.long)
    operations = torch.tensor([row[1] for row in rows], dtype=torch.long)
    operands = torch.tensor([row[2] for row in rows], dtype=torch.long)
    trajectories = torch.tensor([row[3] for row in rows], dtype=torch.long)
    targets = trajectories[:, -1]
    return initial, operations, operands, trajectories, targets


def step(core: SequentialUpdateSU4Core, state: torch.Tensor, operands: torch.Tensor, operations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # SU-4 forward normally receives a discrete register id. Free-running
    # chaining feeds its continuous softmax@embedding candidate back in, so
    # replay the same feature construction directly from the state vector.
    left = core.left_projection(core.input_norm(state[:, 0, :]))
    right = core.right_projection(core.operand_embedding(operands))
    op_state = core.operation_embedding(operations)
    features = torch.cat((left, right, op_state, left - right), dim=-1)
    hidden = core.trunk(features)
    all_logits = torch.stack([head(hidden) for head in core.operation_heads], dim=1)
    logits = all_logits[torch.arange(state.shape[0]), operations]
    candidate = torch.softmax(logits, dim=-1) @ core.value_embedding.weight
    return candidate.unsqueeze(1), logits


@torch.no_grad()
def evaluate_mode(core: SequentialUpdateSU4Core, mode: str, sequences: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    round_accuracy: dict[str, list[bool]] = {str(rounds): [] for rounds in range(1, 5)}
    for hops in HOPS:
        initial, operations, operands, trajectories, targets = sequences[hops]
        if mode == "teacher_forcing":
            state = core.value_embedding(initial).unsqueeze(1)
        else:
            state = core.value_embedding(initial).unsqueeze(1)
        predictions: list[torch.Tensor] = []
        for step_index in range(hops):
            input_state = core.value_embedding(trajectories[:, step_index]).unsqueeze(1) if mode == "teacher_forcing" else state
            state, logits = step(core, input_state, operands[:, step_index], operations[:, step_index])
            predictions.append(logits.argmax(-1))
        matrix[str(hops)] = {}
        for rounds in ROUNDS:
            selected_index = min(rounds, hops) - 1
            matrix[str(hops)][str(rounds)] = float((predictions[selected_index] == targets).float().mean())
        for rounds in range(1, 5):
            selected_index = min(rounds, hops) - 1
            expected = trajectories[:, min(rounds, hops)]
            round_accuracy[str(rounds)].extend((predictions[selected_index] == expected).tolist())
    return matrix, {rounds: sum(values) / len(values) for rounds, values in round_accuracy.items()}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "sequential_update_su4_seed101" / "final.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaign" / "sequential_update_su5_seed101")
    parser.add_argument("--samples-per-h", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    core = SequentialUpdateSU4Core(64, 32, 3)
    core.load_state_dict(checkpoint["core"])
    core.eval()
    sequences = {hops: generate_fixed_h(hops, args.samples_per_h, args.seed) for hops in HOPS}
    started = time.perf_counter()
    teacher_matrix, teacher_round = evaluate_mode(core, "teacher_forcing", sequences)
    free_matrix, free_round = evaluate_mode(core, "free_running", sequences)
    final = {
        "status": "completed",
        "finite": True,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "samples_per_h": args.samples_per_h,
        "modes": {
            "teacher_forcing": {"accuracy_by_h_and_round": teacher_matrix, "one_step_register_accuracy_by_round": teacher_round},
            "free_running": {"accuracy_by_h_and_round": free_matrix, "one_step_register_accuracy_by_round": free_round},
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output_dir / "final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
