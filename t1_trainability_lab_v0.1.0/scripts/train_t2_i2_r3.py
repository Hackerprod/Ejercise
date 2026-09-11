"""Train T2-I2-R3 with frozen R2 curriculum and additive writer pooling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

import train_t2_i2_r2 as r2
from t2_i2_r3_semantic_writer import CompetitiveSemanticWriter, architecture_report, parameter_count
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl2_o import sha256

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"


def output_for_seed(seed: int) -> Path: return CAMPAIGN_ROOT / f"t2_i2_r3_seed{seed}"
def checkpoint_for_seed(seed: int) -> Path: return output_for_seed(seed) / "final.pt"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=6301); parser.add_argument("--output-root", type=Path, default=None); args = parser.parse_args(); output = output_for_seed(args.seed) if args.output_root is None else args.output_root; output.mkdir(parents=True, exist_ok=True); torch.manual_seed(args.seed); random.seed(args.seed)
    observations, labels = r2.load_source(); panels = r2.build_panels(labels); manifest = {"task": "T2-I2-R3", "source_split": "train", "actions": list(r2.ACTIONS), "panels": panels}; manifest_path = output / "panel_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"); manifest_sha = sha256(manifest_path); writer = CompetitiveSemanticWriter(); supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT); supervisor.eval()
    with torch.no_grad(): p_empty_train = torch.softmax(supervisor(observations["features"], torch.zeros((len(observations["features"]), 32))), dim=-1).detach()
    training = r2.train(writer, supervisor, observations, labels, panels, p_empty_train, args.seed); checkpoint = checkpoint_for_seed(args.seed) if args.output_root is None else output / "final.pt"; torch.save({"writer": writer.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(writer), "supervisor_trainable_parameters": 0, "lambda_card": 1.0, "panel_manifest_sha256": manifest_sha, "pooling": "additive per-token writes"}, checkpoint); report = architecture_report(writer) | {"task": "T2-I2-R3", "lambda_card": 1.0, "panel_manifest_sha256": manifest_sha, "panel_J": {key: item["J"] for key, item in panels.items()}, "p_empty": "per train state, precomputed detached before optimizer", "seeds": [6301, 6302, 6303]}; (output / "architecture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); result = {"status": "trained", "task": "T2-I2-R3", "seed": args.seed, "training": training, "checkpoint": sha256(checkpoint), "panel_manifest_sha256": manifest_sha, "architecture_report": report}; (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
