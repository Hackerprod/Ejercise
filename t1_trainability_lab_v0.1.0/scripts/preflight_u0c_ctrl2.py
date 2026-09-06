"""CTRL-2 oracle adjustment preflight on the frozen CTRL-1 executor path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import ADJUSTMENT_NAMES, CTRL1_CHECKPOINT, CTRL1_PILOT, augment_manifest, decode_value, dispatch_adjustment, load_ctrl1, load_executor, load_base_manifests, navigate_collect, trace_success
from train_u0c_ctrl1 import DISTANCES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_preflight_seed101_frozen"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    by_extreme: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [result for result in results if result["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected), "timeouts": sum(result["timeout"] for result in selected), "first_control_error_count": sum(result["first_control_error"] is not None for result in selected), "first_execution_error_count": sum(result["first_execution_error"] is not None for result in selected)}
    for action in range(3):
        selected = [result for result in results if result["action"] == action]
        by_action[ADJUSTMENT_NAMES[action]] = {"samples": len(selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected)}
    for extreme in (0, 31):
        selected = [result for result in results if result["x"] == extreme]
        by_extreme[str(extreme)] = {"samples": len(selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected)}
    return {"samples": len(results), "final_success_count": sum(result["final_success"] for result in results), "trace_success_count": sum(result["trace_success"] for result in results), "final_success_rate": sum(result["final_success"] for result in results) / max(1, len(results)), "trace_success_rate": sum(result["trace_success"] for result in results) / max(1, len(results)), "by_distance": by_distance, "by_action": by_action, "by_extreme_x": by_extreme}


def run_preflight(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    test_manifest = manifests["test"]
    augmented = augment_manifest(test_manifest)
    manifest_path = output_root / "test_manifest.json"
    manifest_path.write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    executor = load_executor()
    ctrl1 = load_ctrl1()
    examples_by_episode: dict[int, list[dict[str, Any]]] = {}
    for example in augmented["ctrl2_examples"]:
        examples_by_episode.setdefault(example["episode"], []).append(example)
    results: list[dict[str, Any]] = []
    for episode in test_manifest["episodes"]:
        navigation = navigate_collect(executor, ctrl1, test_manifest, episode)
        for example in examples_by_episode[episode["episode"]]:
            final_predicted: int | None = None
            operation = None
            trace = list(navigation["trace"])
            first_execution_error = navigation["first_execution_error"]
            if navigation["collected"]:
                state = navigation["state"].clone()
                state, operation = dispatch_adjustment(executor, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], state, navigation["presence"], example["action"])
                final_predicted = decode_value(executor, state)
                trace.append({"stage": "ORACLE-CTRL2", "action": ADJUSTMENT_NAMES[example["action"]], "operation": operation, "r_decoded_before": decode_value(executor, navigation["state"]), "r_decoded_after": final_predicted})
            final_success = final_predicted == example["target_value"]
            trace_ok = trace_success(final_success, navigation["timeout"], navigation["first_control_error"], first_execution_error)
            results.append({"example": example["example"], "episode": episode["episode"], "graph": episode["graph"], "distance": episode["distance"], "x": example["x"], "reference": example["reference"], "action": example["action"], "action_name": ADJUSTMENT_NAMES[example["action"]], "target_value": example["target_value"], "predicted_value": final_predicted, "final_success": final_success, "trace_success": trace_ok, "timeout": navigation["timeout"], "first_control_error": navigation["first_control_error"], "first_execution_error": first_execution_error, "operation": operation, "trace": trace})
    summary = summarize(results)
    status = "passed" if summary["final_success_rate"] >= 0.999 and summary["trace_success_rate"] >= 0.999 and all(summary["by_extreme_x"][str(extreme)]["samples"] > 0 for extreme in (0, 31)) else "failed"
    output = {"status": status, "task": "T1-CTRL-2-preflight", "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": hashlib.sha256(CTRL1_CHECKPOINT.read_bytes()).hexdigest()}, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "class_counts": {name: sum(result["action_name"] == name for result in results) for name in ADJUSTMENT_NAMES}, "summary": summary}
    (output_root / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    if status != "passed":
        raise SystemExit(1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run_preflight(args.output_root)


if __name__ == "__main__":
    main()
