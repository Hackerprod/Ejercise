"""T1-CTRL-2-G frozen numeric coverage evaluation; no training or weight changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import ADJUSTMENT_NAMES, BASE_CHECKPOINT, CTRL1_CHECKPOINT, INCREASE, KEEP, DECREASE, adjusted_target, adjustment_action, build_examples, decode_value, dispatch_adjustment, load_base_manifests, load_ctrl1, load_executor, navigate_collect, reference_curriculum_metadata, reference_values
from train_u0c_ctrl2 import AdjustmentMLP, generate_dataset
from train_u0c_ctrl1 import DISTANCES, SLOT_R, trace_success
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "campaign" / "u0c_ctrl2_pilot_seed2201_frozen"
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_g_coverage_frozen"


def load_ctrl2(checkpoint: Path = PILOT_ROOT / "final.pt") -> AdjustmentMLP:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    controller = AdjustmentMLP()
    controller.load_state_dict(payload["controller"], strict=True)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def expected_action_name(action: int) -> str:
    return ADJUSTMENT_NAMES[action]


def make_manifest(test_manifest: dict[str, Any], train_pair_set: set[tuple[int, int]], formula_pair_set: set[tuple[int, int]]) -> dict[str, Any]:
    all_pairs = [{"x": x, "reference": reference, "action": adjustment_action(x, reference), "action_name": expected_action_name(adjustment_action(x, reference)), "target_value": adjusted_target(x, reference)} for x in range(VALUE_COUNT) for reference in range(VALUE_COUNT)]
    contextual = [{**episode, "x": next(row["value"] for row in test_manifest["graphs"][episode["graph"]]["rows"] if row["kind"] == 1 and row["key"] == episode["goal_key"]), "reference": reference} for episode in test_manifest["episodes"] for reference in range(VALUE_COUNT)]
    for example in contextual:
        example["action"] = adjustment_action(example["x"], example["reference"])
        example["action_name"] = expected_action_name(example["action"])
        example["target_value"] = adjusted_target(example["x"], example["reference"])
    curriculum = reference_curriculum_metadata()
    return {"task": "T1-CTRL-2-G", "frozen": {"executor": str(BASE_CHECKPOINT), "ctrl1": str(CTRL1_CHECKPOINT), "memory_rows": 32, "distances": list(DISTANCES), "read_set": "explicit", "diagnostic_read_e_select": False, "training": False}, "reference_curriculum": curriculum, "source_pilot_test_manifest": str(PILOT_ROOT / "test_manifest.json"), "source_train_pair_partition": {"label_pair_count": len(train_pair_set), "formula_pair_count": len(formula_pair_set), "label_formula_match": train_pair_set == formula_pair_set, "labels_only": True}, "canonical_pairs": all_pairs, "contextual_cases": contextual}


def confusion_matrix(records: list[dict[str, Any]]) -> list[list[int]]:
    matrix = [[0, 0, 0] for _ in range(3)]
    for record in records:
        matrix[record["expected_action"]][record["predicted_action"]] += 1
    return matrix


def comparison_category(canonical_correct: bool, decision_correct: bool) -> str:
    if canonical_correct and decision_correct:
        return "agreement_correct"
    if not canonical_correct and not decision_correct:
        return "shared_error"
    if canonical_correct and not decision_correct:
        return "contextual_regression"
    return "contextual_recovery"


def diagnostic_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    decision_error_count = sum(not record["decision_correct"] for record in records)
    shared_error_count = sum(not record["canonical_correct"] and not record["decision_correct"] for record in records)
    contextual_regression_count = sum(record["canonical_correct"] and not record["decision_correct"] for record in records)
    contextual_recovery_count = sum(not record["canonical_correct"] and record["decision_correct"] for record in records)
    executor_or_transport_count = sum(record["decision_correct"] and not record["final_success"] for record in records)
    assert decision_error_count == shared_error_count + contextual_regression_count
    return {"decision_error_count": decision_error_count, "shared_error_count": shared_error_count, "contextual_regression_count": contextual_regression_count, "contextual_recovery_count": contextual_recovery_count, "executor_or_transport_count": executor_or_transport_count}


def layer_a(model: Any, controller: AdjustmentMLP, train_pairs: set[tuple[int, int]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    class_ids = torch.tensor([VALUE_BASE + value for value in range(VALUE_COUNT)])
    embeddings = model.token_embedding(class_ids)
    for x in range(VALUE_COUNT):
        r = embeddings[x:x + 1]
        for reference in range(VALUE_COUNT):
            b = embeddings[reference:reference + 1]
            logits = controller(r, b).squeeze(0)
            expected = adjustment_action(x, reference)
            other = torch.cat((logits[:expected], logits[expected + 1:]))
            records.append({"x": x, "reference": reference, "expected_action": expected, "expected_action_name": expected_action_name(expected), "predicted_action": int(logits.argmax().item()), "predicted_action_name": expected_action_name(int(logits.argmax().item())), "gamma": float((logits[expected] - other.max()).item()), "seen_in_train_labels": (x, reference) in train_pairs})
    groups = {"seen": [record for record in records if record["seen_in_train_labels"]], "unseen": [record for record in records if not record["seen_in_train_labels"]]}
    group_summary = {name: {"samples": len(selected), "correct": sum(record["expected_action"] == record["predicted_action"] for record in selected), "accuracy": sum(record["expected_action"] == record["predicted_action"] for record in selected) / max(1, len(selected)), "by_expected_action": {action_name: {"samples": sum(record["expected_action_name"] == action_name for record in selected), "correct": sum(record["expected_action_name"] == action_name and record["expected_action"] == record["predicted_action"] for record in selected)} for action_name in ADJUSTMENT_NAMES}} for name, selected in groups.items()}
    keep_ood = [record for record in groups["unseen"] if record["expected_action_name"] == "KEEP"]
    return {"samples": len(records), "correct": sum(record["expected_action"] == record["predicted_action"] for record in records), "exact": sum(record["expected_action"] == record["predicted_action"] for record in records) == VALUE_COUNT * VALUE_COUNT, "confusion_matrix_expected_rows_predicted_columns": confusion_matrix(records), "margin": {"min": min(record["gamma"] for record in records), "max": max(record["gamma"] for record in records), "mean": sum(record["gamma"] for record in records) / len(records)}, "groups": group_summary, "keep_ood": {"samples": len(keep_ood), "note": "No KEEP-OOD pair group exists because all x=b equalities are admitted by the curriculum."}, "records": records}


def aggregate_contextual(records: list[dict[str, Any]], *, action_key: str = "expected_action") -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    by_extreme: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [record for record in records if record["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "decision_correct": sum(record["decision_correct"] for record in selected), "oracle_final_success": sum(record["oracle_final_success"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected), "timeouts": sum(record["timeout"] for record in selected)}
    for action in range(3):
        selected = [record for record in records if record[action_key] == action]
        by_action[ADJUSTMENT_NAMES[action]] = {"samples": len(selected), "decision_correct": sum(record["decision_correct"] for record in selected), "oracle_final_success": sum(record["oracle_final_success"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected)}
    for x in (0, 31):
        selected = [record for record in records if record["x"] == x]
        by_extreme[str(x)] = {"samples": len(selected), "decision_correct": sum(record["decision_correct"] for record in selected), "oracle_final_success": sum(record["oracle_final_success"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected)}
    total = max(1, len(records))
    return {"samples": len(records), "decision_correct": sum(record["decision_correct"] for record in records), "oracle_final_success": sum(record["oracle_final_success"] for record in records), "final_success": sum(record["final_success"] for record in records), "trace_success": sum(record["trace_success"] for record in records), "decision_accuracy": sum(record["decision_correct"] for record in records) / total, "oracle_final_success_rate": sum(record["oracle_final_success"] for record in records) / total, "final_success_rate": sum(record["final_success"] for record in records) / total, "trace_success_rate": sum(record["trace_success"] for record in records) / total, "by_distance": by_distance, "by_action": by_action, "by_extreme_x": by_extreme}


def layer_b(model: Any, controller: AdjustmentMLP, ctrl1: Any, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]], canonical_records: dict[tuple[int, int], dict[str, Any]], *, capture_logits: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        navigation = runtime[episode["episode"]]
        x = navigation["symbolic_x"]
        canonical_x_records = {(x, reference): canonical_records[(x, reference)] for reference in range(VALUE_COUNT)}
        for reference in range(VALUE_COUNT):
            expected = adjustment_action(x, reference)
            target = adjusted_target(x, reference)
            oracle_predicted = None
            predicted_action = None
            predicted = None
            controller_logits = None
            if navigation["collected"]:
                base_state = navigation["state"].clone()
                oracle_state, oracle_operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], base_state.clone(), navigation["presence"], expected)
                oracle_predicted = decode_value(model, oracle_state)
                logits = controller(navigation["state"][:, SLOT_R], model.token_embedding(torch.tensor([VALUE_BASE + reference])))
                predicted_action = int(logits.argmax(-1).item())
                if capture_logits:
                    controller_logits = logits.squeeze(0).tolist()
                predicted_state, predicted_operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], base_state.clone(), navigation["presence"], predicted_action)
                predicted = decode_value(model, predicted_state)
            oracle_final_success = oracle_predicted == target
            decision_correct = predicted_action == expected
            final_success = predicted == target
            first_control_error = navigation["first_control_error"]
            first_execution_error = navigation["first_execution_error"]
            if navigation["aligned"] and not decision_correct:
                first_control_error = {"stage": "CTRL-2", "reference": reference, "expected_action": expected, "predicted_action": predicted_action}
            elif navigation["aligned"] and decision_correct and not final_success:
                first_execution_error = {"stage": "CTRL-2", "reference": reference, "instruction": "adjustment_dispatch"}
            canonical_correct = canonical_x_records[(x, reference)]["expected_action"] == canonical_x_records[(x, reference)]["predicted_action"]
            record = {"episode": episode["episode"], "graph": episode["graph"], "distance": episode["distance"], "x": x, "reference": reference, "expected_action": expected, "predicted_action": predicted_action, "target_value": target, "oracle_predicted_value": oracle_predicted, "predicted_value": predicted, "oracle_final_success": oracle_final_success, "decision_correct": decision_correct, "final_success": final_success, "trace_success": trace_success(final_success, navigation["timeout"], first_control_error, first_execution_error), "timeout": navigation["timeout"], "first_control_error": first_control_error, "first_execution_error": first_execution_error, "canonical_correct": canonical_correct, "comparison_category": comparison_category(canonical_correct, decision_correct), "diagnostic_representation_sensitivity": canonical_correct and not decision_correct, "diagnostic_comparator_generalization": not canonical_correct and not decision_correct, "diagnostic_executor_or_transport": decision_correct and not final_success}
            if capture_logits and controller_logits is not None:
                record["controller_logits"] = controller_logits
            records.append(record)
    return records


def pair_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for x in range(VALUE_COUNT):
        for reference in range(VALUE_COUNT):
            selected = [record for record in records if record["x"] == x and record["reference"] == reference]
            table.append({"x": x, "reference": reference, "expected_action": expected_action_name(adjustment_action(x, reference)), "samples": len(selected), "decision_correct": sum(record["decision_correct"] for record in selected), "oracle_final_success": sum(record["oracle_final_success"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected), "canonical_correct": sum(record["canonical_correct"] for record in selected)})
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--ctrl2-checkpoint", type=Path, default=PILOT_ROOT / "final.pt")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    train_labels = torch.load(PILOT_ROOT / "train" / "labels.pt", map_location="cpu", weights_only=False)
    train_pair_set = set(zip(train_labels["x"].tolist(), train_labels["reference_value"].tolist()))
    formula_pair_set = {(x, reference) for x in range(VALUE_COUNT) for reference in reference_values(x)}
    manifest = make_manifest(manifests["test"], train_pair_set, formula_pair_set)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_correction = {"source_artifact": str(PILOT_ROOT), "correction": "Published pilot manifests used stale curriculum description; this structured metadata is additive and does not overwrite them.", "reference_curriculum": reference_curriculum_metadata()}
    correction_path = args.output_root / "source_metadata_correction.json"
    correction_path.write_text(json.dumps(metadata_correction, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = load_executor()
    ctrl1 = load_ctrl1()
    ctrl2 = load_ctrl2(args.ctrl2_checkpoint)
    test_observations, _, runtime = generate_dataset(model, ctrl1, manifests["test"], keep_runtime=True)
    canonical = layer_a(model, ctrl2, train_pair_set)
    canonical_records = {(record["x"], record["reference"]): record for record in canonical["records"]}
    contextual_records = layer_b(model, ctrl2, ctrl1, manifests["test"], runtime, canonical_records)
    contextual_summary = aggregate_contextual(contextual_records)
    diagnostics = diagnostic_counts(contextual_records)
    diagnostics.update({"representation_sensitivity_count": diagnostics["contextual_regression_count"], "comparator_generalization_count": diagnostics["shared_error_count"], "executor_or_transport_count": diagnostics["executor_or_transport_count"]})
    table = pair_table(contextual_records)
    gate = {"decision": contextual_summary["decision_accuracy"] >= 0.999, "oracle_final": contextual_summary["oracle_final_success_rate"] >= 0.999, "final": contextual_summary["final_success_rate"] >= 0.999, "trace": contextual_summary["trace_success_rate"] >= 0.999, "by_action": all(item["samples"] == 0 or item["decision_correct"] / item["samples"] >= 0.999 and item["final_success"] / item["samples"] >= 0.999 and item["trace_success"] / item["samples"] >= 0.999 for item in contextual_summary["by_action"].values()), "by_distance": all(item["samples"] == 0 or item["decision_correct"] / item["samples"] >= 0.999 and item["final_success"] / item["samples"] >= 0.999 and item["trace_success"] / item["samples"] >= 0.999 for item in contextual_summary["by_distance"].values()), "all_contextual_pairs": all(item["samples"] == item["decision_correct"] == item["oracle_final_success"] == item["final_success"] == item["trace_success"] for item in table)}
    output = {"status": "completed", "task": "T1-CTRL-2-G", "training": False, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": hashlib.sha256(CTRL1_CHECKPOINT.read_bytes()).hexdigest()}, "checkpoint_ctrl2": {"path": str(args.ctrl2_checkpoint), "sha256": hashlib.sha256(args.ctrl2_checkpoint.read_bytes()).hexdigest()}, "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "canonical_pairs": 1024, "contextual_cases": 64000, "contextual_base_episodes": 2000, "references_per_episode": 32}, "source_metadata_correction": {"path": str(correction_path), "sha256": hashlib.sha256(correction_path.read_bytes()).hexdigest()}, "train_pair_partition": {"from_labels_count": len(train_pair_set), "formula_count": len(formula_pair_set), "label_formula_match": train_pair_set == formula_pair_set, "seen": sum(record["seen_in_train_labels"] for record in canonical["records"]), "unseen": sum(not record["seen_in_train_labels"] for record in canonical["records"]), "expected_formula_seen": 270, "expected_formula_unseen": 754}, "layer_a_canonical": {key: value for key, value in canonical.items() if key != "records"}, "layer_a_records": canonical["records"], "layer_b_contextual": contextual_summary, "layer_b_pair_table": table, "diagnostics": diagnostics, "gate": gate, "policy_inputs": ["real_register_state_after_COPY", "direct_numeric_reference_embedding"], "policy_excluded": ["x", "reference_relation", "target_value", "action_label", "distance", "memory", "symbolic_pointer", "trace"], "protocol": {"executor_frozen": True, "ctrl1_frozen": True, "ctrl2_frozen": True, "fine_tuning": False, "new_weights": False, "read_set": "explicit", "diagnostic_read_e_select": False}}
    result_path = args.output_root / "results.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
