"""P1 per-round pointer-reader diagnostics; performs no training or optimizer step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.data import load_jsonl  # noqa: E402
from t1_trainability.pointer import PointerCore, PointerHead  # noqa: E402


def examples_to_tensors(examples) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    starts = torch.tensor([int(row.metadata["start_key"]) for row in examples], dtype=torch.long)
    sources = torch.tensor([[int(value) for value in str(row.metadata["memory_sources"]).split(",")] for row in examples], dtype=torch.long)
    destinations = torch.tensor([[int(value) for value in str(row.metadata["memory_destinations"]).split(",")] for row in examples], dtype=torch.long)
    paths = []
    for row in examples:
        mapping = dict(zip(sources[len(paths)].tolist(), destinations[len(paths)].tolist()))
        current = int(row.metadata["start_key"])
        path = [current]
        for _ in range(4):
            current = mapping[current]
            path.append(current)
        paths.append(path)
    return starts, sources, destinations, torch.tensor(paths, dtype=torch.long)


@torch.no_grad()
def diagnose(core: PointerCore, head: PointerHead, examples) -> dict[str, object]:
    starts, sources, destinations, paths = examples_to_tensors(examples)
    state = core.initial_state(starts)
    rows = []
    for round_index in range(4):
        normalized = core.input_norm(state)
        query = core.query(normalized[:, 0, :])
        current_logits = torch.matmul(query, core.key_embedding.weight.transpose(0, 1)) * core.scale
        source_log_mask = torch.log_softmax(current_logits, dim=-1).gather(1, sources)
        attention = torch.softmax(source_log_mask, dim=-1)
        reader_distribution = torch.zeros((len(examples), core.key_count), dtype=attention.dtype)
        reader_distribution.scatter_add_(1, destinations, attention)
        correct_keys = paths[:, round_index + 1]
        old_keys = paths[:, round_index]
        correct_probability = reader_distribution.gather(1, correct_keys[:, None]).squeeze(1)
        old_probability = reader_distribution.gather(1, old_keys[:, None]).squeeze(1)
        masked = reader_distribution.clone()
        masked.scatter_(1, correct_keys[:, None], -1.0)
        second_probability = masked.max(dim=1).values
        entropy = -(reader_distribution * reader_distribution.clamp_min(1e-12).log()).sum(dim=1)
        retrieved = torch.matmul(attention.unsqueeze(1), core.key_embedding(destinations)).squeeze(1).unsqueeze(1)
        if core.transition == "pointer_replacement":
            state = retrieved
        else:
            delta = core.core(normalized + retrieved)
            state = state + core.alpha * delta
        rows.append(
            {
                "round": round_index + 1,
                "pointer_acc": float((reader_distribution.argmax(dim=1) == correct_keys).float().mean()),
                "reader_entropy": float(entropy.mean()),
                "reader_margin": float((correct_probability - second_probability).mean()),
                "old_pointer_mass": float(old_probability.mean()),
                "correct_key_probability": float(correct_probability.mean()),
                "second_candidate_probability": float(second_probability.mean()),
                "attention_row_entropy": float((-(attention * attention.clamp_min(1e-12).log()).sum(dim=1)).mean()),
            }
        )
    return {"examples": len(examples), "rounds": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "pointer_chasing_seed101" / "best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "pointer_chasing" / "test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "p1_round_diagnostics.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    transition = checkpoint.get("config", {}).get("transition", "residual_pre_norm")
    core = PointerCore(64, 256, 4, transition=transition)
    head = PointerHead(64, core.key_embedding)
    core.load_state_dict(checkpoint["core"])
    head.load_state_dict(checkpoint["head"])
    examples = load_jsonl(args.dataset)
    result = {
        "phase": "P1",
        "training_performed": False,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "checkpoint_step": checkpoint["step"],
        "metric_definitions": {
            "pointer_acc": "argmax over reader destination-key distribution equals true key after this round",
            "reader_entropy": "entropy over 256 destination-key probabilities",
            "reader_margin": "correct-key probability minus highest non-correct-key probability",
            "old_pointer_mass": "reader probability mass assigned to previous true-round key",
        },
        "diagnostics": diagnose(core, head, examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
