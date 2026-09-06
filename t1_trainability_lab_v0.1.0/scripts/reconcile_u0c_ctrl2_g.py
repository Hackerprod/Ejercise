"""Reconcile CTRL-2-G diagnostics without retraining or rewriting original results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ctrl2_common import CTRL1_CHECKPOINT, BASE_CHECKPOINT, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_ctrl2_g import PILOT_ROOT, aggregate_contextual, diagnostic_counts, layer_b, pair_table, load_ctrl2
from train_u0c_ctrl2 import generate_dataset


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = ROOT / "campaign" / "u0c_ctrl2_g_coverage_frozen"
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_g_coverage_corrected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--ctrl2-checkpoint", type=Path, default=PILOT_ROOT / "final.pt")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    original_results_path = ORIGINAL_ROOT / "results.json"
    original_manifest_path = ORIGINAL_ROOT / "manifest.json"
    original_results = json.loads(original_results_path.read_text(encoding="utf-8"))
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    copied_manifest_path = args.output_root / "manifest.json"
    copied_manifest_path.write_bytes(original_manifest_path.read_bytes())
    model = load_executor()
    ctrl1 = load_ctrl1()
    ctrl2 = load_ctrl2(args.ctrl2_checkpoint)
    base_manifest = load_base_manifests()["test"]
    _, _, runtime = generate_dataset(model, ctrl1, base_manifest, keep_runtime=True)
    canonical_records = {(record["x"], record["reference"]): record for record in original_results["layer_a_records"]}
    contextual_records = layer_b(model, ctrl2, ctrl1, base_manifest, runtime, canonical_records, capture_logits=True)
    summary = aggregate_contextual(contextual_records)
    diagnostics = diagnostic_counts(contextual_records)
    diagnostics["representation_sensitivity_count"] = diagnostics["contextual_regression_count"]
    diagnostics["comparator_generalization_count"] = diagnostics["shared_error_count"]
    failures = [record for record in contextual_records if not record["decision_correct"] or not record["final_success"] or not record["trace_success"]]
    failure_path = args.output_root / "contextual_failures.jsonl"
    failure_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in failures), encoding="utf-8")
    corrected = {"status": "completed", "task": "T1-CTRL-2-G-corrected", "training": False, "original_results": {"path": str(original_results_path), "sha256": hashlib.sha256(original_results_path.read_bytes()).hexdigest()}, "original_manifest": {"path": str(original_manifest_path), "sha256": hashlib.sha256(original_manifest_path.read_bytes()).hexdigest()}, "manifest_copy": {"path": str(copied_manifest_path), "sha256": hashlib.sha256(copied_manifest_path.read_bytes()).hexdigest()}, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": hashlib.sha256(CTRL1_CHECKPOINT.read_bytes()).hexdigest()}, "checkpoint_ctrl2": {"path": str(args.ctrl2_checkpoint), "sha256": hashlib.sha256(args.ctrl2_checkpoint.read_bytes()).hexdigest()}, "layer_a_canonical_reference": {"samples": original_results["layer_a_canonical"]["samples"], "correct": original_results["layer_a_canonical"]["correct"], "exact": original_results["layer_a_canonical"]["exact"], "source": str(original_results_path)}, "layer_b_contextual": summary, "layer_b_pair_table": pair_table(contextual_records), "diagnostics": diagnostics, "diagnostic_invariant": {"decision_error_count": diagnostics["decision_error_count"], "shared_error_count_plus_contextual_regression_count": diagnostics["shared_error_count"] + diagnostics["contextual_regression_count"], "holds": diagnostics["decision_error_count"] == diagnostics["shared_error_count"] + diagnostics["contextual_regression_count"]}, "failed_case_count": len(failures), "failed_case_artifact": {"path": str(failure_path), "sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest()}, "protocol": {"retrained": False, "weights_changed": False, "original_results_untouched": True}}
    corrected_path = args.output_root / "results_corrected.json"
    corrected_path.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(corrected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
