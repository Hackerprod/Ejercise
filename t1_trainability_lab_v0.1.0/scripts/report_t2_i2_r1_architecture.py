"""Write immutable T2-I2-R1 architecture report before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, architecture_report
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl7 import GoalConditionedSupervisor614

ROOT = Path(__file__).resolve().parents[1]; DEFAULT_OUTPUT = ROOT / "campaign" / "t2_i2_r1_architecture_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(); writer = CompetitiveSemanticWriter(); supervisor = GoalConditionedSupervisor614(); supervisor.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True); weight = supervisor.goal_projection.weight.detach(); c_gt = float(F.cosine_similarity(weight[:, 0], weight[:, 1], dim=0)); report = architecture_report(writer); report.update({"task": "T2-I2-R1", "ctrl7_goal_projection_cosine_c_gt": c_gt, "frozen_components": ["CTRL-7 supervisor", "additive slot composition", "dispatcher", "I2 v0 evaluators"], "audit_matching_scope": "128 real + 32 interaction + 1 crossing per realization/order, both permutations; no 31744 audit rollouts", "curriculum": "identical to I2 v0", "seeds": [6101, 6102, 6103], "output_pattern": "campaign/t2_i2_r1_seed{seed}/", "v0_output_pattern": "campaign/t2_i2_seed{seed}/", "collision_policy": "R1 checkpoint path must differ from v0 path and any existing R1 checkpoint", "reconstructed_conditioning": "sum of both semantic slots from each atomic encoding; NULL has no output projection and is excluded"}); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(args.output), "report": report}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
