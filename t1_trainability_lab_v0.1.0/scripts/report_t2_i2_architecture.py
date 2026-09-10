"""Write immutable T2-I2 architecture report before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from t2_i2_semantic_writer import SemanticWriter, architecture_report
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl7 import GoalConditionedSupervisor614

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "campaign" / "t2_i2_architecture_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    writer = SemanticWriter()
    supervisor = GoalConditionedSupervisor614()
    supervisor.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True)
    weight = supervisor.goal_projection.weight.detach()
    c_gt = float(F.cosine_similarity(weight[:, 0], weight[:, 1], dim=0))
    report = architecture_report(writer)
    report.update({
        "task": "T2-I2",
        "ctrl7_checkpoint": str(CTRL7_CHECKPOINT),
        "ctrl7_goal_projection_cosine_c_gt": c_gt,
        "reconstruction_cosine_threshold": c_gt + 0.05,
        "frozen_components": ["CTRL-7 supervisor", "additive slot composition", "dispatcher", "R2/I1 evaluators"],
        "learned_change_only": "full sequence to content-addressed two-slot writer",
        "oracle_isolation": "writer and dispatcher receive no symbolic segments or constraints",
        "heldout_reconstruction_conditions": ["JOINT", "RECONSTRUCTED"],
        "seeds": [6101, 6102, 6103],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "report": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
