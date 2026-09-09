"""Train XF-CLAUSE on frozen R1.1 curriculum, seed 6001."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from t2_i0_instruction_r1_1 import instruction_for_r1_1, r1_1_corpus
from t2_xf_transformer import MatchedTransformerEncoder, encode_clauses, parameter_count
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_xf_seq_r1_1 import source_data
from train_u0c_ctrl2_o import sha256

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN_ROOT = ROOT / "campaign"; CTRL7_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"; OUTPUT = CAMPAIGN_ROOT / "t2_xf_clause_r1_1_seed6001"


def train(model, supervisor, observations, labels, texts, seed: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1)
    buckets = {key: {action: torch.where((labels["constraints"][:, 0] == key[0]) & (labels["constraints"][:, 1] == key[1]) & (labels["action"] == action))[0] for action in (0, 1, 2, 3, 5)} for key in ((0, 0), (1, 0), (0, 1))}; counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}; batch_times = []
    for step in range(1, 5001):
        progress = (step - 1) / 4999; optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress; selected = torch.cat([buckets[key][action][torch.randint(len(buckets[key][action]), (count,), generator=generator)] for key, actions in counts.items() for action, count in actions.items()]); start = time.perf_counter(); optimizer.zero_grad(set_to_none=True); ids = labels["instruction_id"][selected]; condition = encode_clauses(model, [texts[index] for index in ids.tolist()]); loss = F.cross_entropy(supervisor(observations["features"][selected], condition), labels["action"][selected]); loss.backward(); optimizer.step(); batch_times.append(time.perf_counter() - start)
    return {"updates": 5000, "seed": seed, "trainable_parameters": parameter_count(model), "frozen_supervisor_trainable": 0, "final_loss": float(loss.detach()), "batch_time_seconds": {"mean": sum(batch_times) / len(batch_times), "min": min(batch_times), "max": max(batch_times), "samples": len(batch_times)}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6001); args = parser.parse_args(); torch.manual_seed(args.seed); random.seed(args.seed); rows = r1_1_corpus(); texts = [text for text, _ in rows]; model = MatchedTransformerEncoder(); supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); observations, labels = source_data("train", rows); training = train(model, supervisor, observations, labels, texts, args.seed); OUTPUT.mkdir(parents=True, exist_ok=True); checkpoint = OUTPUT / "final.pt"; torch.save({"encoder": model.state_dict(), "seed": args.seed, "updates": 5000, "and_trained": False, "trainable_parameters": parameter_count(model), "configuration": "XF-CLAUSE R1.1"}, checkpoint); result = {"status": "trained", "task": "T2-XF", "phase": "XF-CLAUSE_R1.1_training", "curriculum_rows": len(rows), "heldout_orders_excluded": True, "seed": args.seed, "training": training, "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)}}; (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
