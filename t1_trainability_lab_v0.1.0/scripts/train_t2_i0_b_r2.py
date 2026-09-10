"""T2-I0-B-R2 shared clause encoder with additive conditioning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_u0c_ctrl2_o import sha256
from t2_i0_instruction_r1 import parse_instruction_r1
from t2_i0_instruction_r1_1 import instruction_for_r1_1, r1_1_corpus


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"
CTRL7_CHECKPOINT = SOURCE_ROOT / "final.pt"
def output_for_seed(seed: int) -> Path:
    return CAMPAIGN_ROOT / f"t2_i0_b_r2_seed{seed}"


class SharedClauseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.embedding = nn.Embedding(37, 16); self.recurrence = nn.GRU(16, 32, batch_first=True)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        sequence, _ = self.recurrence(self.embedding(token_ids)); return sequence[torch.arange(token_ids.shape[0]), lengths - 1]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def clause_condition(encoder: SharedClauseEncoder, instructions: list[str]) -> Tensor:
    clauses = [[tuple(token) for token in parse_instruction_r1(text).tokens] for text in instructions]
    first = [tuple(token for token in parse_instruction_r1(text).tokens[:next((index for index, token in enumerate(parse_instruction_r1(text).tokens) if token == "AND"), len(parse_instruction_r1(text).tokens))]) for text in instructions]
    second = [tuple(parse_instruction_r1(text).tokens[next((index for index, token in enumerate(parse_instruction_r1(text).tokens) if token == "AND"), len(parse_instruction_r1(text).tokens)) + 1:]) for text in instructions]
    def encode_clauses(rows: list[tuple[str, ...]]) -> Tensor:
        lengths = torch.tensor([len(row) for row in rows]); tokens = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
        for index, row in enumerate(rows): tokens[index, :len(row)] = torch.tensor(parse_instruction_r1(" ".join(row)).token_ids)
        return encoder(tokens, lengths)
    result = encode_clauses(first); has_second = torch.tensor([len(row) > 0 for row in second])
    if bool(has_second.any()): result = result + encode_clauses([row if len(row) else ("NOOP", "VALUE_0") for row in second]) * has_second.unsqueeze(1)
    return result


def source_data(split: str, corpus: list[tuple[str, tuple[int, int]]]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    observations = torch.load(SOURCE_ROOT / split / "observations.pt", weights_only=False); labels = torch.load(SOURCE_ROOT / split / "labels.pt", weights_only=False); lookup = {text: index for index, (text, _) in enumerate(corpus)}; ids = []
    for index, (bits, lower, forbidden) in enumerate(zip(labels["constraints"].tolist(), labels["lower"].tolist(), labels["forbidden"].tolist())):
        key = tuple(bits); value = lower if key == (1, 0) else forbidden if key == (0, 1) else 0; variant = index % (2 if key == (0, 0) else 3); dummy = value if variant == 0 else (value + (7, 13, 23)[index % 3]) % 32; ids.append(lookup[instruction_for_r1_1(key, value, variant=variant, dummy=dummy, dummy2=(dummy + 11) % 32)])
    return observations, {**labels, "instruction_id": torch.tensor(ids)}


def train(encoder: SharedClauseEncoder, supervisor: LatentConditionedSupervisor, observations: dict[str, Tensor], labels: dict[str, Tensor], corpus: list[tuple[str, tuple[int, int]]], seed: int) -> dict[str, int | float]:
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); buckets = {key: {action: torch.where((labels["constraints"][:, 0] == key[0]) & (labels["constraints"][:, 1] == key[1]) & (labels["action"] == action))[0] for action in (0, 1, 2, 3, 5)} for key in ((0, 0), (1, 0), (0, 1))}; counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}
    for step in range(1, 5001):
        progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress; selected = torch.cat([buckets[key][action][torch.randint(len(buckets[key][action]), (count,), generator=generator)] for key, action_counts in counts.items() for action, count in action_counts.items()]); ids = labels["instruction_id"][selected]; optimizer.zero_grad(set_to_none=True); condition = clause_condition(encoder, [corpus[index][0] for index in ids.tolist()]); loss = F.cross_entropy(supervisor(observations["features"][selected], condition), labels["action"][selected]); loss.backward(); optimizer.step()
    return {"updates": 5000, "seed": seed, "trainable_parameters": parameter_count(encoder), "frozen_supervisor_trainable": 0, "final_loss": float(loss.detach())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=5701); args = parser.parse_args(); corpus = r1_1_corpus(); torch.manual_seed(args.seed); random.seed(args.seed); encoder = SharedClauseEncoder(); observations, labels = source_data("train", corpus); supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); training = train(encoder, supervisor, observations, labels, corpus, args.seed); output_root = output_for_seed(args.seed); output_root.mkdir(parents=True, exist_ok=True); torch.save({"encoder": encoder.state_dict(), "seed": args.seed, "updates": 5000, "and_trained": False, "trainable_parameters": parameter_count(encoder)}, output_root / "final.pt"); result = {"status": "trained", "task": "T2-I0-B-R2", "seed": args.seed, "corpus_rows": len(corpus), "training": training, "checkpoint": sha256(output_root / "final.pt"), "heldout_orders_excluded": True}; (output_root / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
