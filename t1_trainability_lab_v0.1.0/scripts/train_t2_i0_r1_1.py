"""Train fresh R1.1 Baseline A/B models on deterministic NOOP variation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from train_t2_i0_r1 import BaselineAR1Classifier, SharedInstructionEncoderR1
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_u0c_ctrl2_o import sha256
from t2_i0_instruction_r1_1 import instruction_for_r1_1, r1_1_corpus


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701"
CTRL7_CHECKPOINT = SOURCE_ROOT / "final.pt"
OUTPUT_A = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_a_seed5501"
OUTPUT_B = CAMPAIGN_ROOT / "t2_i0_r1_1_baseline_b_seed5601"


def tensorize(rows):
    from t2_i0_instruction_r1 import parse_instruction_r1
    parsed = [parse_instruction_r1(text) for text, _ in rows]; lengths = torch.tensor([len(item.token_ids) for item in parsed]); tokens = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for index, item in enumerate(parsed): tokens[index, :len(item.token_ids)] = torch.tensor(item.token_ids)
    return {"token_ids": tokens, "lengths": lengths, "bits": torch.tensor([bits for _, bits in rows])}


def train_a(model, data, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); buckets = {key: torch.where((data["bits"][:, 0] == key[0]) & (data["bits"][:, 1] == key[1]))[0] for key in ((0, 0), (1, 0), (0, 1))}
    for step in range(1, 5001):
        progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress; indices = torch.cat([bucket[torch.randint(len(bucket), (43,), generator=generator)] for bucket in buckets.values()]); optimizer.zero_grad(set_to_none=True); floor, avoid = model(data["token_ids"][indices], data["lengths"][indices]); loss = (F.cross_entropy(floor, data["bits"][indices, 0]) + F.cross_entropy(avoid, data["bits"][indices, 1])) / 2; loss.backward(); optimizer.step()
    return {"updates": 5000, "seed": seed, "trainable_parameters": sum(p.numel() for p in model.parameters()), "final_loss": float(loss.detach())}


def source_data(split, corpus):
    observations = torch.load(SOURCE_ROOT / split / "observations.pt", weights_only=False); labels = torch.load(SOURCE_ROOT / split / "labels.pt", weights_only=False); lookup = {text: index for index, (text, _) in enumerate(corpus)}; ids = []
    for index, (bits, lower, forbidden) in enumerate(zip(labels["constraints"].tolist(), labels["lower"].tolist(), labels["forbidden"].tolist())):
        key = tuple(bits); value = lower if key == (1, 0) else forbidden if key == (0, 1) else 0; variant = index % (2 if key == (0, 0) else 3); dummy = value if variant == 0 else (value + (7, 13, 23)[index % 3]) % 32; dummy2 = (dummy + 11) % 32; text = instruction_for_r1_1(key, value, variant=variant, dummy=dummy, dummy2=dummy2); ids.append(lookup[text])
    return observations, {**labels, "instruction_id": torch.tensor(ids)}


def train_b(encoder, supervisor, observations, labels, corpus_data, seed):
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); buckets = {key: {action: torch.where((labels["constraints"][:, 0] == key[0]) & (labels["constraints"][:, 1] == key[1]) & (labels["action"] == action))[0] for action in (0, 1, 2, 3, 5)} for key in ((0, 0), (1, 0), (0, 1))}; counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}
    for step in range(1, 5001):
        progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress; selected = torch.cat([buckets[key][action][torch.randint(len(buckets[key][action]), (count,), generator=generator)] for key, actions in counts.items() for action, count in actions.items()]); optimizer.zero_grad(set_to_none=True); ids = labels["instruction_id"][selected]; condition = encoder(corpus_data["token_ids"][ids], corpus_data["lengths"][ids]); loss = F.cross_entropy(supervisor(observations["features"][selected], condition), labels["action"][selected]); loss.backward(); optimizer.step()
    return {"updates": 5000, "seed": seed, "trainable_parameters": sum(p.numel() for p in encoder.parameters()), "frozen_supervisor_trainable": 0, "final_loss": float(loss.detach())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed-a", type=int, default=5501); parser.add_argument("--seed-b", type=int, default=5601); args = parser.parse_args(); corpus = r1_1_corpus(); corpus_data = tensorize(corpus); torch.manual_seed(args.seed_a); random.seed(args.seed_a); model_a = BaselineAR1Classifier(); a_training = train_a(model_a, corpus_data, args.seed_a); OUTPUT_A.mkdir(parents=True, exist_ok=True); torch.save({"model": model_a.state_dict(), "seed": args.seed_a, "updates": 5000, "and_trained": False}, OUTPUT_A / "final.pt"); torch.manual_seed(args.seed_b); random.seed(args.seed_b); model_b = SharedInstructionEncoderR1(); core = LatentConditionedSupervisor(CTRL7_CHECKPOINT); observations, labels = source_data("train", corpus); b_training = train_b(model_b, core, observations, labels, corpus_data, args.seed_b); OUTPUT_B.mkdir(parents=True, exist_ok=True); torch.save({"encoder": model_b.state_dict(), "seed": args.seed_b, "updates": 5000, "and_trained": False}, OUTPUT_B / "final.pt")
    result = {"status": "trained", "task": "T2-I0-R1.1", "corpus_rows": len(corpus), "heldout_orders_excluded": True, "baseline_a": a_training, "baseline_b": b_training, "checkpoints": {"a": sha256(OUTPUT_A / "final.pt"), "b": sha256(OUTPUT_B / "final.pt")}}
    for root, baseline in ((OUTPUT_A, "A"), (OUTPUT_B, "B")): (root / "results.json").write_text(json.dumps({**result, "baseline": baseline}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
