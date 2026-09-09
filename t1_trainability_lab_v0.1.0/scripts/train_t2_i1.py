"""Train T2-I1 shared clause encoder with lexical operator aliases."""

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
from t2_i1_instruction import parse_instruction_i1

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN_ROOT = ROOT / "campaign"; SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"; CTRL7 = SOURCE_ROOT / "final.pt"; OUTPUT = CAMPAIGN_ROOT / "t2_i1_b_seed5801"


class SharedClauseEncoderI1(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.embedding = nn.Embedding(39, 16); self.recurrence = nn.GRU(16, 32, batch_first=True)
    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        sequence, _ = self.recurrence(self.embedding(token_ids)); return sequence[torch.arange(token_ids.shape[0]), lengths - 1]


def clause_condition(encoder: SharedClauseEncoderI1, instructions: list[str]) -> Tensor:
    first_rows = []; second_rows = []
    for text in instructions:
        tokens = parse_instruction_i1(text).tokens; split = tokens.index("AND") if "AND" in tokens else len(tokens); first_rows.append(tokens[:split]); second_rows.append(tokens[split + 1:] if split < len(tokens) else ())
    def encode(rows):
        actual = [row if row else ("NOOP", "VALUE_0") for row in rows]; lengths = torch.tensor([len(row) for row in actual]); token_ids = torch.zeros((len(actual), int(lengths.max())), dtype=torch.long)
        for index, row in enumerate(actual): token_ids[index, :len(row)] = torch.tensor(parse_instruction_i1(" ".join(row)).token_ids)
        return encoder(token_ids, lengths)
    result = encode(first_rows); second_mask = torch.tensor([bool(row) for row in second_rows]).unsqueeze(1)
    return result + encode(second_rows) * second_mask


def dummy_values(value: int) -> tuple[int, ...]: return tuple((value + offset) % 32 for offset in (0, 7, 13, 23))


def instruction_for(constraints: tuple[int, int], value: int, variant: int, dummy: int = 0, alias: str | None = None, dummy2: int | None = None) -> str:
    if constraints == (0, 0):
        first = f"NOOP VALUE_{dummy}"; return first if variant == 0 else f"{first} AND NOOP VALUE_{dummy if dummy2 is None else dummy2}"
    operator = f"{alias} VALUE_{value}"
    if variant == 0: return operator
    noop = f"NOOP VALUE_{dummy}"; return f"{operator} AND {noop}" if variant == 1 else f"{noop} AND {operator}"


def corpus() -> list[tuple[str, tuple[int, int]]]:
    rows = [(instruction_for((0, 0), 0, 0, dummy=d), (0, 0)) for d in range(32)]; rows.extend((instruction_for((0, 0), 0, 1, dummy=d, dummy2=(d + 11) % 32), (0, 0)) for d in range(32))
    for constraints, aliases in (((1, 0), ("AT_LEAST", "MINIMUM")), ((0, 1), ("AVOID", "EXCLUDE"))):
        for alias in aliases:
            for value in range(32):
                for dummy in dummy_values(value):
                    rows.extend((instruction_for(constraints, value, variant, dummy=dummy, alias=alias), constraints) for variant in (0, 1, 2))
    return rows


def tensorize(rows):
    parsed = [parse_instruction_i1(text) for text, _ in rows]; lengths = torch.tensor([len(item.token_ids) for item in parsed]); token_ids = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for index, item in enumerate(parsed): token_ids[index, :len(item.token_ids)] = torch.tensor(item.token_ids)
    return {"token_ids": token_ids, "lengths": lengths, "bits": torch.tensor([bits for _, bits in rows])}


def source_data(split, rows):
    observations = torch.load(SOURCE_ROOT / split / "observations.pt", weights_only=False); labels = torch.load(SOURCE_ROOT / split / "labels.pt", weights_only=False); lookup = {text: index for index, (text, _) in enumerate(rows)}; ids = []
    for index, (bits, lower, forbidden) in enumerate(zip(labels["constraints"].tolist(), labels["lower"].tolist(), labels["forbidden"].tolist())):
        key = tuple(bits); value = lower if key == (1, 0) else forbidden if key == (0, 1) else 0; alias = ("AT_LEAST", "MINIMUM")[index % 2] if key == (1, 0) else ("AVOID", "EXCLUDE")[index % 2] if key == (0, 1) else None; variant = index % (2 if key == (0, 0) else 3); dummy = value if variant == 0 else (value + (7, 13, 23)[index % 3]) % 32; ids.append(lookup[instruction_for(key, value, variant, dummy=dummy, alias=alias, dummy2=(dummy + 11) % 32)])
    return observations, {**labels, "instruction_id": torch.tensor(ids)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=5801); args = parser.parse_args(); rows = corpus(); data_rows = tensorize(rows); torch.manual_seed(args.seed); random.seed(args.seed); encoder = SharedClauseEncoderI1(); supervisor = LatentConditionedSupervisor(CTRL7); observations, labels = source_data("train", rows); optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(args.seed + 1); buckets = {key: {action: torch.where((labels["constraints"][:, 0] == key[0]) & (labels["constraints"][:, 1] == key[1]) & (labels["action"] == action))[0] for action in (0, 1, 2, 3, 5)} for key in ((0, 0), (1, 0), (0, 1))}; counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}
    for step in range(1, 5001):
        progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress; selected = torch.cat([buckets[key][action][torch.randint(len(buckets[key][action]), (count,), generator=generator)] for key, action_counts in counts.items() for action, count in action_counts.items()]); ids = labels["instruction_id"][selected]; optimizer.zero_grad(set_to_none=True); condition = clause_condition(encoder, [rows[index][0] for index in ids.tolist()]); loss = F.cross_entropy(supervisor(observations["features"][selected], condition), labels["action"][selected]); loss.backward(); optimizer.step()
    OUTPUT.mkdir(parents=True, exist_ok=True); torch.save({"encoder": encoder.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": sum(p.numel() for p in encoder.parameters()), "and_trained": False}, OUTPUT / "final.pt"); result = {"status": "trained", "task": "T2-I1", "seed": args.seed, "corpus_rows": len(rows), "training": {"updates": 5000, "trainable_parameters": sum(p.numel() for p in encoder.parameters()), "frozen_supervisor_trainable": 0, "final_loss": float(loss.detach())}, "heldout_real_operator_combinations": True, "checkpoint": sha256(OUTPUT / "final.pt")}; (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
