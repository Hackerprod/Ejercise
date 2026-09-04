"""P0 benchmark oracle: exact one-hot pointer transitions, no learned code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch


N = 16
HMAX = 4
LAYERS = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15))
HOPS = (1, 2, 3, 4)
ROUNDS = (1, 2, 4)


def build_graph(seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    adjacency = torch.zeros((N, N), dtype=torch.float64)
    for source_layer, destination_layer in zip(LAYERS[:-1], LAYERS[1:]):
        for source in source_layer:
            destination = rng.choice(destination_layer)
            adjacency[source, destination] = 1.0
    return adjacency


def solve_target(adjacency: torch.Tensor, start: int, hops: int) -> int:
    pointer = torch.zeros(N, dtype=torch.float64)
    pointer[start] = 1.0
    for _ in range(hops):
        pointer = pointer @ adjacency
    return int(pointer.argmax())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, default=Path("campaign/p0_oracle.json"))
    args = parser.parse_args()
    adjacency = build_graph(args.seed)
    starts = list(LAYERS[0])
    matrix: dict[str, dict[str, float]] = {}
    exact_targets: dict[str, dict[str, int]] = {}
    for hops in HOPS:
        matrix[str(hops)] = {}
        exact_targets[str(hops)] = {}
        targets = [solve_target(adjacency, start, hops) for start in starts]
        for rounds in ROUNDS:
            correct = 0
            for start, target in zip(starts, targets):
                predicted = solve_target(adjacency, start, min(hops, rounds))
                correct += int(predicted == target)
            matrix[str(hops)][str(rounds)] = correct / len(starts)
        exact_targets[str(hops)] = {str(start): target for start, target in zip(starts, targets)}

    topological_order = [node for layer in LAYERS for node in layer]
    rank = {node: index for index, node in enumerate(topological_order)}
    edges = [(source, destination) for source, destination in zip(*torch.where(adjacency == 1))]
    acyclic = all(rank[int(source)] < rank[int(destination)] for source, destination in edges)
    graph_checks = {
        "node_count": N,
        "hmax": HMAX,
        "self_loops": int(torch.diagonal(adjacency).sum()),
        "edges": len(edges),
        "acyclic_topological_order": acyclic,
        "cycles_at_or_below_hmax": not acyclic,
        "one_hot_rows": bool(torch.all(adjacency.sum(dim=1)[:13] == 1) and torch.all(adjacency.sum(dim=1)[13:] == 0)),
    }
    result = {
        "phase": "P0",
        "training_performed": False,
        "reader": "exact hand-coded p_next = p @ A",
        "matrix_accuracy": matrix,
        "random_baseline_1_over_N": 1.0 / N,
        "graph_checks": graph_checks,
        "exact_targets_by_hop": exact_targets,
        "adjacency": adjacency.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
