"""Frozen train/validation score-consistency diagnosis for CTRL-2-O."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from ctrl2_common import load_executor
from train_u0c_ctrl2_o import OrdinalSharedScorer


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "campaign" / "u0c_ctrl2_o_pilot_seed2201_frozen"
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_o_consistency_diagnostic"
CHECKPOINT = PILOT_ROOT / "final.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_controller() -> OrdinalSharedScorer:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    controller = OrdinalSharedScorer()
    controller.load_state_dict(payload["controller"], strict=True)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def diagnose_split(controller: OrdinalSharedScorer, split: str, root: Path, q_scores: list[float], tau: float) -> dict[str, Any]:
    observations = torch.load(root / split / "observations.pt", map_location="cpu", weights_only=False)
    labels = torch.load(root / split / "labels.pt", map_location="cpu", weights_only=False)
    with torch.inference_mode():
        z_scores = controller.score(observations["register"])
    records: list[dict[str, Any]] = []
    abs_deltas: list[float] = []
    for index, (x_tensor, z_tensor) in enumerate(zip(labels["x"], z_scores)):
        x = int(x_tensor.item())
        z_r = float(z_tensor.item())
        delta = z_r - q_scores[x]
        abs_delta = abs(delta)
        keep_ok = abs_delta <= tau
        lower_ok = None if x == 0 else z_r > q_scores[x - 1] + tau
        upper_ok = None if x == VALUE_COUNT - 1 else z_r < q_scores[x + 1] - tau
        abs_deltas.append(abs_delta)
        records.append({"index": index, "x": x, "z_R": z_r, "q_x": q_scores[x], "delta_R": delta, "abs_delta_R": abs_delta, "keep_condition": {"applicable": True, "pass": keep_ok, "inequality": "abs(z_R-q_x)<=tau"}, "lower_neighbor_condition": {"applicable": x != 0, "pass": lower_ok, "inequality": "z_R>q_(x-1)+tau" if x != 0 else None, "neighbor": x - 1 if x != 0 else None}, "upper_neighbor_condition": {"applicable": x != VALUE_COUNT - 1, "pass": upper_ok, "inequality": "z_R<q_(x+1)-tau" if x != VALUE_COUNT - 1 else None, "neighbor": x + 1 if x != VALUE_COUNT - 1 else None}, "all_applicable_conditions_pass": bool(keep_ok and (lower_ok is None or lower_ok) and (upper_ok is None or upper_ok)), "label_action": int(labels["action"][index].item())})
    return {"split": split, "samples": len(records), "source_hashes": {"observations": sha256(root / split / "observations.pt"), "labels": sha256(root / split / "labels.pt")}, "condition_counts": {"keep_applicable": len(records), "keep_pass": sum(item["keep_condition"]["pass"] for item in records), "lower_applicable": sum(item["lower_neighbor_condition"]["applicable"] for item in records), "lower_pass": sum(item["lower_neighbor_condition"]["pass"] is True for item in records), "upper_applicable": sum(item["upper_neighbor_condition"]["applicable"] for item in records), "upper_pass": sum(item["upper_neighbor_condition"]["pass"] is True for item in records), "all_applicable_conditions_pass": sum(item["all_applicable_conditions_pass"] for item in records)}, "abs_delta_percentiles": {"p50": percentile(abs_deltas, 0.50), "p90": percentile(abs_deltas, 0.90), "p95": percentile(abs_deltas, 0.95), "p99": percentile(abs_deltas, 0.99), "p999": percentile(abs_deltas, 0.999), "max": max(abs_deltas)}, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    controller = load_controller()
    model = load_executor()
    with torch.inference_mode():
        q_scores = [float(value) for value in controller.score(model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))).tolist()]
        tau = float(controller.tau().item())
    result = {"status": "completed", "task": "T1-CTRL-2-O-consistency-diagnostic", "training": False, "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT)}, "tau": tau, "q_x": q_scores, "source_root": str(PILOT_ROOT), "protocol": {"observations_reused": True, "data_regenerated": False, "weights_changed": False, "contextual_64000_rerun": False}, "splits": {split: diagnose_split(controller, split, PILOT_ROOT, q_scores, tau) for split in ("train", "val")}}
    output_path = args.output_root / "diagnosis.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output_path), "sha256": sha256(output_path), "checkpoint_sha256": sha256(CHECKPOINT), "tau": tau, "train": result["splits"]["train"]["condition_counts"], "val": result["splits"]["val"]["condition_counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
