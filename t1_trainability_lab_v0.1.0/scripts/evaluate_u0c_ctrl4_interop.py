"""T1-CTRL-4-I frozen scorer interoperability evaluation.

Only live free execution is compared: stored seed-2201 features are not
reused as if they were new observations for other scorers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl3 import sha256
from evaluate_u0c_ctrl4 import run_controls, run_learned
from evaluate_u0c_ctrl4_preflight import load_fixed_manifest
from train_u0c_ctrl2_o import OrdinalSharedScorer
from train_u0c_ctrl4 import SupervisorMLP


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SUPERVISOR_CHECKPOINT = CAMPAIGN_ROOT / "u0c_ctrl4_pilot_seed4401" / "final.pt"
MANIFEST_PATH = CAMPAIGN_ROOT / "u0c_ctrl3_real_r_seed2201" / "real_r_manifest.json"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl4_interop"
MANIFEST_SHA256 = "d62e30833b5b8cbbfb618800828e5bea4609cd8ad1ad860608cf2279d0953df3"
SCORER_PATHS = {2201: CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen" / "final.pt", **{seed: CAMPAIGN_ROOT / f"u0c_ctrl2_o_replica_seed{seed}_frozen" / "final.pt" for seed in (2202, 2203, 2204, 2205)}}


def compact(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trajectory.items() if key != "state"}


def compare_to_reference(reference: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    feature_deltas: list[float] = []
    feature_changed = 0
    action_changed = 0
    state_hash_changed = 0
    first_decision_difference: dict[str, Any] | None = None
    for index, (left, right) in enumerate(zip(reference, current)):
        left_events, right_events = left["events"], right["events"]
        if len(left_events) != len(right_events) and first_decision_difference is None:
            first_decision_difference = {"trajectory_index": index, "reason": "event_count", "reference": len(left_events), "current": len(right_events)}
        for event_index, (left_event, right_event) in enumerate(zip(left_events, right_events)):
            delta = max(abs(float(a) - float(b)) for a, b in zip(left_event["features"], right_event["features"]))
            feature_deltas.append(delta)
            feature_changed += delta > 1e-12
            action_changed += left_event["action"] != right_event["action"]
            state_hash_changed += left_event["r_after_sha256"] != right_event["r_after_sha256"]
            if first_decision_difference is None and left_event["action"] != right_event["action"]:
                first_decision_difference = {"trajectory_index": index, "decision": event_index, "reference_action": left_event["action_name"], "current_action": right_event["action_name"]}
    return {"feature_event_count": len(feature_deltas), "feature_changed_event_count": feature_changed, "feature_max_abs_delta": max(feature_deltas, default=0.0), "action_changed_event_count": action_changed, "state_hash_changed_event_count": state_hash_changed, "first_decision_difference": first_decision_difference}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--supervisor-checkpoint", type=Path, default=SUPERVISOR_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed_manifest(args.manifest, args.manifest_sha256)
    manifest = load_base_manifests()["test"]
    model = load_executor(); ctrl1 = load_ctrl1()
    supervisor_payload = torch.load(args.supervisor_checkpoint, map_location="cpu", weights_only=False); supervisor = SupervisorMLP(); supervisor.load_state_dict(supervisor_payload["supervisor"], strict=True); supervisor.eval()
    episode_by_x: dict[int, dict[str, Any]] = {}
    for entry in fixed["entries"]:
        episode_by_x.setdefault(int(entry["x"]), manifest["episodes"][entry["episode"]])
    per_seed: dict[str, Any] = {}
    trajectories_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed, scorer_path in SCORER_PATHS.items():
        scorer_payload = torch.load(scorer_path, map_location="cpu", weights_only=False); scorer = OrdinalSharedScorer(); scorer.load_state_dict(scorer_payload["controller"], strict=True); scorer.eval()
        trajectories = [run_learned(model, ctrl1, scorer, supervisor, manifest, episode_by_x[x], reference) for x in range(32) for reference in range(32)]
        trajectories_by_seed[seed] = trajectories
        per_seed[str(seed)] = {"scorer": {"path": str(scorer_path), "sha256": sha256(scorer_path), "training_seed": scorer_payload.get("controller_seed")}, "summary": {"samples": len(trajectories), "trajectory_success": sum(item["trajectory_success"] for item in trajectories), "final_success": sum(item["final_success"] for item in trajectories), "timeouts": sum(item["timeout"] for item in trajectories), "first_bad_count": sum(item["first_bad"] is not None for item in trajectories), "max_decisions": max(item["decisions"] for item in trajectories)}, "controls": run_controls(model, ctrl1, scorer, supervisor, manifest, episode_by_x), "trajectories": [compact(item) for item in trajectories]}
    reference = trajectories_by_seed[2201]
    comparisons = {str(seed): compare_to_reference(reference, trajectories) for seed, trajectories in trajectories_by_seed.items() if seed != 2201}
    result = {"status": "completed", "task": "T1-CTRL-4-I", "training": False, "scope": "free_execution_only_for_interoperability", "protocol": {"supervisor_frozen": True, "scorer_checkpoints_fixed": True, "features_recomputed_live_per_scorer": True, "stored_seed2201_features_reused": False, "trace_metrics_omitted": True, "no_recalibration": True, "manifest_reused_existing": True}, "checkpoint_supervisor": {"path": str(args.supervisor_checkpoint), "sha256": sha256(args.supervisor_checkpoint), "seed": supervisor_payload.get("seed")}, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "samples": len(fixed["entries"])}, "per_seed": per_seed, "comparisons_to_2201": comparisons}
    output = args.output_root / "results.json"; output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "per_seed": {seed: data["summary"] for seed, data in per_seed.items()}, "comparisons": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
