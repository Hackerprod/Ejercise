"""Frozen T2-I1 held-out battery, independently evaluated for all eight forms."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ctrl2_common import load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl4_preflight import dispatch_unified_action, load_fixed_manifest
from evaluate_u0c_ctrl7_heldout import run_canonical_learned
from evaluate_u0c_ctrl7_preflight import real_cases
from evaluate_u0c_ctrl7_trained import run_learned
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i1 import SharedClauseEncoderI1, clause_condition
from train_u0c_ctrl2_o import OrdinalSharedScorer, sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
CHECKPOINT = CAMPAIGN_ROOT / "t2_i1_b_seed5801" / "final.pt"
CTRL7 = CAMPAIGN_ROOT / "u0c_ctrl7_pilot_seed4701" / "final.pt"
SCORER = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt"
MANIFEST = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
MANIFEST_SHA = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
OUTPUT = CAMPAIGN_ROOT / "t2_i1_heldout_seed5801"
REALIZATIONS = (
    ("AT_LEAST_AVOID", "AT_LEAST", "AVOID"),
    ("AT_LEAST_EXCLUDE", "AT_LEAST", "EXCLUDE"),
    ("MINIMUM_AVOID", "MINIMUM", "AVOID"),
    ("MINIMUM_EXCLUDE", "MINIMUM", "EXCLUDE"),
)
ORDERS = ("NORMAL", "INVERTED")


class Adapter(nn.Module):
    def __init__(self, core: LatentConditionedSupervisor, condition: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction(floor: str, avoid: str, order: str, lower: int, forbidden: int) -> str:
    clauses = (f"{floor} VALUE_{lower}", f"{avoid} VALUE_{forbidden}")
    if order == "INVERTED":
        clauses = clauses[::-1]
    return " AND ".join(clauses)


def post_copy(result: dict[str, Any]) -> list[str]:
    actions = [event["action_name"] for event in result["events"]]
    return actions[next(index for index, action in enumerate(actions) if action == "COPY_E_R") + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(name in inspect.signature(dispatch_unified_action).parameters for name in ("constraints", "lower", "forbidden", "floor", "avoid")):
        raise RuntimeError("dispatcher received forbidden inputs")

    encoder = SharedClauseEncoderI1()
    encoder.load_state_dict(torch.load(CHECKPOINT, weights_only=False)["encoder"], strict=True)
    encoder.eval()
    core = LatentConditionedSupervisor(CTRL7)
    core.eval()
    manifest = load_base_manifests()["test"]
    fixed = load_fixed_manifest(MANIFEST, MANIFEST_SHA)
    executor = load_executor()
    ctrl1 = load_ctrl1()
    scorer_payload = torch.load(SCORER, weights_only=False)
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(scorer_payload["controller"], strict=True)
    scorer.eval()
    episode_by_x = {int(entry["x"]): manifest["episodes"][entry["episode"]] for entry in fixed["entries"]}

    results: dict[str, dict[str, Any]] = {}
    for realization, floor, avoid in REALIZATIONS:
        for order in ORDERS:
            key = f"{realization}_{order}"
            cache: dict[str, Adapter] = {}

            def adapter(instruction_text: str) -> Adapter:
                if instruction_text not in cache:
                    cache[instruction_text] = Adapter(core, clause_condition(encoder, [instruction_text]))
                return cache[instruction_text]

            canonical_success = 0
            real_rows = []
            for x0 in range(32):
                for lower in range(32):
                    for forbidden in range(31):
                        text = instruction(floor, avoid, order, lower, forbidden)
                        candidate = run_canonical_learned(executor, ctrl1, scorer, adapter(text), manifest, episode_by_x[x0], x0, lower, forbidden, (1, 1))
                        canonical_success += int(candidate["success"])
            for category, x0, lower, forbidden in real_cases():
                text = instruction(floor, avoid, order, lower, forbidden)
                candidate = run_learned(executor, ctrl1, scorer, adapter(text), manifest, episode_by_x[x0], lower, forbidden, (1, 1))
                real_rows.append({"category": category, "result": candidate})
            interaction = [row for row in real_rows if row["category"] == "x_lt_L_eq_F"]
            interaction_gate = sum(1 for row in interaction if post_copy(row["result"]) == ["INCREASE"] * (row["result"]["forbidden"] - row["result"]["x0"] + 1) + ["EMIT"])
            crossing = next(row for row in real_rows if row["result"]["x0"] == 10 and row["result"]["lower"] == 14 and row["result"]["forbidden"] == 12)
            crossing_actions = post_copy(crossing["result"])
            crossing_gate = int(crossing_actions == ["INCREASE"] * 4 + ["EMIT"])
            results[key] = {
                "realization": realization,
                "order": order,
                "canonical": {"samples": 31744, "success": canonical_success},
                "real": {
                    "samples": 128,
                    "success": sum(row["result"]["success"] for row in real_rows),
                    "categories": {
                        category: {
                            "samples": sum(row["category"] == category for row in real_rows),
                            "success": sum(row["category"] == category and row["result"]["success"] for row in real_rows),
                        }
                        for category in sorted({row["category"] for row in real_rows})
                    },
                },
                "interaction_gate": {"samples": 32, "success": interaction_gate},
                "crossing_literal": {"samples": 1, "success": crossing_gate, "post_copy_actions": crossing_actions},
            }
            print(json.dumps({key: results[key]}, sort_keys=True))

    passed = all(item["canonical"]["success"] == 31744 and item["real"]["success"] == 128 and item["interaction_gate"]["success"] == 32 and item["crossing_literal"]["success"] == 1 for item in results.values())
    result = {
        "status": "passed" if passed else "failed",
        "task": "T2-I1",
        "phase": "heldout_eight_realizations",
        "training": False,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT)},
        "ctrl7_checkpoint": sha256(CTRL7),
        "dispatcher_guard": "passed",
        "realizations": results,
    }
    output = args.output_root / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": file_sha(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
