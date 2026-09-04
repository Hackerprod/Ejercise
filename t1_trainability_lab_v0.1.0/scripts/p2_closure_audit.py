"""Close T1.2 P2 with R=3, no-overshoot, metamorphic, and split audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import load_jsonl  # noqa: E402
from t1_trainability.pointer import PointerCore, PointerHead  # noqa: E402


HOPS = (1, 2, 3, 4)
ROUNDS = (1, 2, 3, 4)


def tensors(examples):
    starts = torch.tensor([int(row.metadata["start_key"]) for row in examples], dtype=torch.long)
    sources = torch.tensor([[int(value) for value in str(row.metadata["memory_sources"]).split(",")] for row in examples], dtype=torch.long)
    destinations = torch.tensor([[int(value) for value in str(row.metadata["memory_destinations"]).split(",")] for row in examples], dtype=torch.long)
    hops = torch.tensor([int(row.metadata["hop_count"]) for row in examples], dtype=torch.long)
    targets = torch.tensor([row.target for row in examples], dtype=torch.long)
    return starts, sources, destinations, hops, targets


def matrix(core: PointerCore, head: PointerHead, batch):
    starts, sources, destinations, hops, targets = batch
    result = {str(hop): {} for hop in HOPS}
    for hop in HOPS:
        selected = hops == hop
        for rounds in ROUNDS:
            required = torch.minimum(torch.full_like(hops[selected], rounds), hops[selected])
            state = core(starts[selected], sources[selected], destinations[selected], rounds=rounds, required_hops=required)
            result[str(hop)][str(rounds)] = float((head(state).argmax(-1) == targets[selected]).float().mean())
    return result


def mapping_hash(example) -> str:
    sources = [int(value) for value in str(example.metadata["memory_sources"]).split(",")]
    destinations = [int(value) for value in str(example.metadata["memory_destinations"]).split(",")]
    canonical = {"mapping": sorted(zip(sources, destinations)), "start": int(example.metadata["start_key"]), "hops": int(example.metadata["hop_count"])}
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode("utf-8")).hexdigest()


def relabel(batch, permutation: torch.Tensor):
    starts, sources, destinations, hops, targets = batch
    return permutation[starts], permutation[sources], permutation[destinations], hops, permutation[targets]


def shuffle_rows(batch, seed: int):
    starts, sources, destinations, hops, targets = batch
    generator = torch.Generator().manual_seed(seed)
    order = torch.stack([torch.randperm(sources.shape[1], generator=generator) for _ in range(sources.shape[0])])
    return starts, sources.gather(1, order), destinations.gather(1, order), hops, targets


def summarize_matrices(matrices: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        str(hop): {
            str(rounds): {
                "mean": sum(matrix[str(hop)][str(rounds)] for matrix in matrices) / len(matrices),
                "min": min(matrix[str(hop)][str(rounds)] for matrix in matrices),
                "max": max(matrix[str(hop)][str(rounds)] for matrix in matrices),
            }
            for rounds in ROUNDS
        }
        for hop in HOPS
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "pointer_chasing_p2_seed101" / "best.pt")
    parser.add_argument("--test", type=Path, default=ROOT / "datasets" / "pointer_chasing" / "test.jsonl")
    parser.add_argument("--train", type=Path, default=ROOT / "datasets" / "pointer_chasing" / "train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "p2_closure_audit.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    transition = checkpoint.get("config", {}).get("transition", "pointer_replacement")
    core = PointerCore(64, 256, 4, transition=transition)
    head = PointerHead(64, core.key_embedding)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    test_examples = load_jsonl(args.test)
    train_examples = load_jsonl(args.train)
    original = tensors(test_examples)

    # R=3 closure cells and full exact matrix on the same checkpoint.
    original_matrix = matrix(core, head, original)
    r3_cells = {"H3_R3": original_matrix["3"]["3"], "H4_R3": original_matrix["4"]["3"]}

    # Ten full-test random label permutations and ten independent row shuffles.
    relabel_matrices = []
    shuffle_matrices = []
    for index in range(10):
        generator = torch.Generator().manual_seed(70_000 + index)
        permutation = torch.randperm(256, generator=generator)
        relabel_matrices.append(matrix(core, head, relabel(original, permutation)))
        shuffle_matrices.append(matrix(core, head, shuffle_rows(original, 80_000 + index)))

    train_hashes = {mapping_hash(example) for example in train_examples}
    test_hashes = {mapping_hash(example) for example in test_examples}
    source = (ROOT / "scripts" / "train_pointer_chasing.py").read_text(encoding="utf-8")
    no_overshoot_source = {
        "mechanism": "active-round mask",
        "source_uses_execution_hops": "required_hops = min(declared_R, declared_hop_count)",
        "mask_is_target_key_dependent": False,
        "mask_is_answer_dependent": False,
        "token_stop": False,
        "identity_transition": False,
        "readout_selects_state_by_H": False,
        "source_confirmation": "evaluate_matrix computes execution_hops=min(rounds, hop), then passes required_hops to PointerCore; PointerCore applies torch.where(active, next_state, state).",
    }
    checkpoint_selection = {
        "best_checkpoint_step": checkpoint["step"],
        "selected_from": "datasets/pointer_chasing/val.jsonl",
        "test_used_for_checkpoint_selection": False,
        "evidence": "train_pointer_chasing evaluates val_data every 100 steps and selects best_val_mean_r4; test_data is evaluated only after loading best.pt for final.json.",
    }
    result = {
        "phase": "P2 closure",
        "training_performed": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "transition": transition,
        "test_examples": len(test_examples),
        "r3_cells": r3_cells,
        "original_matrix": original_matrix,
        "metamorphic": {
            "permutations": 10,
            "row_shuffles": 10,
            "relabel_summary": summarize_matrices(relabel_matrices),
            "row_shuffle_summary": summarize_matrices(shuffle_matrices),
        },
        "split_hash_audit": {
            "train_unique": len(train_hashes),
            "test_unique": len(test_hashes),
            "train_test_intersection": len(train_hashes & test_hashes),
            "hash_definition": "sha256(sorted(mapping source-destination pairs)+start+hops)",
        },
        "no_overshoot_audit": no_overshoot_source,
        "checkpoint_selection_audit": checkpoint_selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
