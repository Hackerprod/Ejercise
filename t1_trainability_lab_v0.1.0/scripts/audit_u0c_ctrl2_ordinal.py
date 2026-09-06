"""Frozen CTRL-2 ordinal, geometry, and ideal-to-real-R audit.

This script only loads existing checkpoints and artifacts.  It does not train,
modify weights, regenerate datasets, alter temperature, or add observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import ADJUSTMENT_NAMES, adjustment_action, load_base_manifests, load_ctrl1, load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from train_u0c_ctrl1 import SLOT_R
from train_u0c_ctrl2 import AdjustmentMLP, generate_dataset


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
PILOT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_pilot_seed2201_frozen"
AUDIT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_ordinal_audit"
SEEDS = (2201, 2202, 2203, 2204, 2205)
ANCHORS = frozenset((0, 8, 16, 20, 24, 31))
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
ACTION_ORDER = {2: 0, 1: 1, 0: 2}  # DECREASE -> KEEP -> INCREASE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(seed: int) -> Path:
    if seed == 2201:
        return PILOT_ROOT / "final.pt"
    return CAMPAIGN_ROOT / f"u0c_ctrl2_replica_seed{seed}" / "final.pt"


def prior_g_results_path(seed: int) -> Path:
    if seed == 2201:
        return CAMPAIGN_ROOT / "u0c_ctrl2_g_coverage_corrected" / "results_corrected.json"
    return CAMPAIGN_ROOT / f"u0c_ctrl2_replica_seed{seed}_g" / "results.json"


def load_controller(checkpoint: Path) -> AdjustmentMLP:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    controller = AdjustmentMLP()
    controller.load_state_dict(payload["controller"], strict=True)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def action_name(action: int | None) -> str | None:
    return None if action is None else ADJUSTMENT_NAMES[action]


def margins(logits: torch.Tensor, expected: int) -> dict[str, Any]:
    values = [float(value) for value in logits.tolist()]
    order = values[0] - values[2]
    keep = values[1] - max(values[0], values[2])
    ordered = sorted(values, reverse=True)
    return {
        "logits": values,
        "d_order": order,
        "d_keep": keep,
        "expected_margin": values[expected] - max(values[index] for index in range(3) if index != expected),
        "top2_margin": ordered[0] - ordered[1],
        "predicted_action": int(logits.argmax().item()),
        "predicted_action_name": action_name(int(logits.argmax().item())),
    }


def load_train_pairs() -> set[tuple[int, int]]:
    labels = torch.load(PILOT_ROOT / "train" / "labels.pt", map_location="cpu", weights_only=False)
    return set(zip(labels["x"].tolist(), labels["reference_value"].tolist()))


def pair_stratum(x: int, reference: int, seen: bool) -> tuple[Any, ...]:
    return (seen, abs(x - reference), adjustment_action(x, reference), x in ANCHORS, reference in ANCHORS)


def canonical_audit(model: Any, controller: AdjustmentMLP, train_pairs: set[tuple[int, int]]) -> dict[str, Any]:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    embeddings = model.token_embedding(class_ids).detach()
    normalized = torch.nn.functional.normalize(embeddings, dim=-1)
    feature_vectors: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for x in range(VALUE_COUNT):
            for reference in range(VALUE_COUNT):
                ux = normalized[x]
                ub = normalized[reference]
                feature = torch.cat((ux, ub, ux - ub, ux * ub))
                logits = controller(embeddings[x:x + 1], embeddings[reference:reference + 1]).squeeze(0)
                expected = adjustment_action(x, reference)
                feature_vectors.append(feature)
                record = {
                    "x": x,
                    "reference": reference,
                    "expected_action": expected,
                    "expected_action_name": action_name(expected),
                    "seen_in_train_labels": (x, reference) in train_pairs,
                    "distance": abs(x - reference),
                    "x_is_anchor": x in ANCHORS,
                    "reference_is_anchor": reference in ANCHORS,
                    "anchor_position": "both" if x in ANCHORS and reference in ANCHORS else "x" if x in ANCHORS else "reference" if reference in ANCHORS else "neither",
                }
                record.update(margins(logits, expected))
                record["canonical_correct"] = record["predicted_action"] == expected
                records.append(record)

    features = torch.stack(feature_vectors)
    distances = torch.cdist(features, features, p=2)
    expected_actions = [record["expected_action"] for record in records]
    correct_indices = [index for index, record in enumerate(records) if record["canonical_correct"]]
    failed_indices = [index for index, record in enumerate(records) if not record["canonical_correct"]]

    # Part A: relations are measured exactly as predicted. No rule is applied
    # to alter predicted_action.
    inverse_violations: list[dict[str, Any]] = []
    for x in range(VALUE_COUNT):
        for reference in range(x + 1, VALUE_COUNT):
            left = records[x * VALUE_COUNT + reference]
            right = records[reference * VALUE_COUNT + x]
            expected_inverse = {0: 2, 1: 1, 2: 0}[left["predicted_action"]]
            if right["predicted_action"] != expected_inverse:
                inverse_violations.append({"x": x, "reference": reference, "forward_predicted": action_name(left["predicted_action"]), "reverse_predicted": action_name(right["predicted_action"]), "expected_reverse": action_name(expected_inverse)})

    equality_violations: list[dict[str, Any]] = []
    for record in records:
        should_keep = record["x"] == record["reference"]
        is_keep = record["predicted_action"] == 1
        if should_keep != is_keep:
            equality_violations.append({"x": record["x"], "reference": record["reference"], "predicted_action": record["predicted_action_name"], "violation": "missed_diagonal_keep" if should_keep else "off_diagonal_keep"})

    transitivity_violations: list[dict[str, int]] = []
    transitive_antecedents = 0
    for a in range(VALUE_COUNT):
        for b in range(VALUE_COUNT):
            if b == a or records[a * VALUE_COUNT + b]["predicted_action"] != 2:
                continue
            for c in range(VALUE_COUNT):
                if c in (a, b):
                    continue
                if records[b * VALUE_COUNT + c]["predicted_action"] == 2:
                    transitive_antecedents += 1
                    if records[a * VALUE_COUNT + c]["predicted_action"] != 2:
                        if len(transitivity_violations) < 256:
                            transitivity_violations.append({"a": a, "b": b, "c": c, "predicted_ac": records[a * VALUE_COUNT + c]["predicted_action"]})
    transitivity_violation_count = 0
    for a in range(VALUE_COUNT):
        for b in range(VALUE_COUNT):
            if b == a or records[a * VALUE_COUNT + b]["predicted_action"] != 2:
                continue
            for c in range(VALUE_COUNT):
                if c in (a, b):
                    continue
                if records[b * VALUE_COUNT + c]["predicted_action"] == 2 and records[a * VALUE_COUNT + c]["predicted_action"] != 2:
                    transitivity_violation_count += 1

    monotonic_rows: list[dict[str, Any]] = []
    retrograde_transition_count = 0
    for x in range(VALUE_COUNT):
        sequence = [records[x * VALUE_COUNT + reference]["predicted_action"] for reference in range(VALUE_COUNT)]
        retrogrades = [reference for reference in range(VALUE_COUNT - 1) if ACTION_ORDER[sequence[reference]] > ACTION_ORDER[sequence[reference + 1]]]
        if retrogrades:
            retrograde_transition_count += len(retrogrades)
            monotonic_rows.append({"x": x, "predicted_sequence": [action_name(action) for action in sequence], "retrograde_reference_boundaries": retrogrades})

    # Part B: all distances use the controller's actual normalized feature
    # space.  Correct controls are selected by exact stratum, then nearest
    # feature distance inside that stratum.
    strata: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(records):
        strata.setdefault(pair_stratum(record["x"], record["reference"], record["seen_in_train_labels"]), []).append(index)
    feature_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        same_candidates = [candidate for candidate in correct_indices if candidate != index and pair_stratum(records[candidate]["x"], records[candidate]["reference"], records[candidate]["seen_in_train_labels"]) == pair_stratum(record["x"], record["reference"], record["seen_in_train_labels"])]
        opposite_candidates = [candidate for candidate in range(len(records)) if candidate != index and records[candidate]["expected_action"] != record["expected_action"]]
        admitted_opposite = [candidate for candidate in opposite_candidates if records[candidate]["seen_in_train_labels"]]
        feature_record = {
            "x": record["x"],
            "reference": record["reference"],
            "feature_l2_from_origin": float(features[index].norm().item()),
            "normalized_embedding_l2": float((normalized[record["x"]] - normalized[record["reference"]]).norm().item()),
            "nearest_correct_same_stratum": None,
            "nearest_canonical_opposite": None,
            "nearest_admitted_opposite": None,
        }
        for key, candidates in (("nearest_correct_same_stratum", same_candidates), ("nearest_canonical_opposite", opposite_candidates), ("nearest_admitted_opposite", admitted_opposite)):
            if candidates:
                candidate = min(candidates, key=lambda item: (float(distances[index, item].item()), records[item]["x"], records[item]["reference"]))
                feature_record[key] = {"x": records[candidate]["x"], "reference": records[candidate]["reference"], "distance": float(distances[index, candidate].item()), "expected_action": records[candidate]["expected_action_name"], "canonical_correct": records[candidate]["canonical_correct"]}
        feature_records.append(feature_record)

    def stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}

    failed_geometry = [feature_records[index] for index in failed_indices]
    failed_to_controls = [item for item in failed_geometry if item["nearest_correct_same_stratum"] is not None]
    b_summary = {
        "feature_definition": "z=[Normalize(E_x),Normalize(E_b),Normalize(E_x)-Normalize(E_b),Normalize(E_x)*Normalize(E_b)]",
        "space_dimension": int(features.shape[1]),
        "canonical_pair_count": len(records),
        "failed_pair_count": len(failed_indices),
        "correct_control_match_count": len(failed_to_controls),
        "failed_feature_l2": stats([item["feature_l2_from_origin"] for item in failed_geometry]),
        "failed_nearest_correct_same_stratum_l2": stats([item["nearest_correct_same_stratum"]["distance"] for item in failed_to_controls]),
        "failed_nearest_canonical_opposite_l2": stats([item["nearest_canonical_opposite"]["distance"] for item in failed_geometry if item["nearest_canonical_opposite"] is not None]),
        "failed_nearest_admitted_opposite_l2": stats([item["nearest_admitted_opposite"]["distance"] for item in failed_geometry if item["nearest_admitted_opposite"] is not None]),
        "matching_rule": "same seen/unseen, |x-b|, expected orientation/action, x-anchor flag, reference-anchor flag; nearest correct feature in normalized z-space",
        "distance_to_admitted_note": "nearest_admitted_opposite is distance to admitted canonical training pairs only; it is not distance to the real-R training set",
    }

    return {
        "records": records,
        "feature_records": feature_records,
        "summary": {
            "samples": len(records),
            "correct": len(correct_indices),
            "failed": len(failed_indices),
            "accuracy": len(correct_indices) / len(records),
            "part_a": {
                "argument_inversion": {"domain": 496, "violation_count": len(inverse_violations), "violations": inverse_violations},
                "equality_keep_reserved": {"domain": 1024, "violation_count": len(equality_violations), "off_diagonal_keep_count": sum(item["violation"] == "off_diagonal_keep" for item in equality_violations), "missed_diagonal_keep_count": sum(item["violation"] == "missed_diagonal_keep" for item in equality_violations), "violations": equality_violations},
                "strict_transitivity": {"ordered_distinct_triples": VALUE_COUNT * (VALUE_COUNT - 1) * (VALUE_COUNT - 2), "antecedent_count": transitive_antecedents, "violation_count": transitivity_violation_count, "note": "A violated triple is not an independent model failure; one predicted edge can participate in many violated triples.", "sample_violations": transitivity_violations},
                "reference_monotonicity": {"rows": VALUE_COUNT, "rows_with_retrogression": len(monotonic_rows), "retrograde_transition_count": retrograde_transition_count, "violations": monotonic_rows},
                "raw_logits_fields": ["logits", "d_order=z_INCREASE-z_DECREASE", "d_keep=z_KEEP-max(z_INCREASE,z_DECREASE)", "expected_margin", "top2_margin"],
                "rule_application": "Predictions were not corrected; properties are measured only.",
            },
            "part_b": b_summary,
        },
    }


def contextual_rows(model: Any, controller: AdjustmentMLP, manifests: dict[str, dict[str, Any]], runtime: dict[int, dict[str, Any]], canonical_records: list[dict[str, Any]], train_pairs: set[tuple[int, int]]) -> list[dict[str, Any]]:
    canonical_by_pair = {(record["x"], record["reference"]): record for record in canonical_records}
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    embeddings = model.token_embedding(class_ids).detach()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for episode in manifests["test"]["episodes"]:
            navigation = runtime[episode["episode"]]
            if not navigation["collected"]:
                continue
            x = int(navigation["symbolic_x"])
            r_real = navigation["state"][:, SLOT_R].detach().clone()
            for reference in range(VALUE_COUNT):
                expected = adjustment_action(x, reference)
                logits = controller(r_real, embeddings[reference:reference + 1]).squeeze(0)
                output = margins(logits, expected)
                canonical = canonical_by_pair[(x, reference)]
                decision_correct = output["predicted_action"] == expected
                canonical_correct = canonical["canonical_correct"]
                category = "agreement_correct" if canonical_correct and decision_correct else "shared_error" if not canonical_correct and not decision_correct else "contextual_regression" if canonical_correct else "contextual_recovery"
                rows.append({"episode": episode["episode"], "graph": episode["graph"], "x": x, "reference": reference, "distance": abs(x - reference), "expected_action": expected, "expected_action_name": action_name(expected), "seen_in_train_labels": (x, reference) in train_pairs, "canonical_predicted_action": canonical["predicted_action"], "canonical_predicted_action_name": canonical["predicted_action_name"], "canonical_correct": canonical_correct, "decision_correct": decision_correct, "comparison_category": category, "r_real": [float(value) for value in r_real.squeeze(0).tolist()], **{key: value for key, value in output.items()}})
    return rows


def choose_path_cases(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Fixed rule: one lexicographically first contextual episode per
    # (category, x, reference), retaining every distinct failing canonical
    # region observed in a seed. Controls: first two agreement-correct rows
    # per (seen/unseen, distance, expected action), sorted by episode/x/b.
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, int]] = set()
    for row in sorted(rows, key=lambda item: (item["comparison_category"], item["x"], item["reference"], item["episode"])):
        if row["comparison_category"] in ("contextual_regression", "contextual_recovery"):
            key = (row["comparison_category"], row["x"], row["reference"])
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append({**row, "selection": "first_episode_for_category_x_reference"})
    controls: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["comparison_category"] == "agreement_correct":
            key = (row["seen_in_train_labels"], row["distance"], row["expected_action"])
            grouped.setdefault(key, []).append(row)
    for key, candidates in sorted(grouped.items(), key=lambda item: item[0]):
        for row in sorted(candidates, key=lambda item: (item["episode"], item["x"], item["reference"]))[:2]:
            controls.append({**row, "selection": "first_two_agreement_correct_per_seen_distance_orientation"})
    return selected + controls, {"failure_selection": "one lexicographically first episode per category/x/reference", "control_selection": "first two agreement-correct episodes per (seen, |x-b|, expected orientation)", "selected_failures": len(selected), "selected_controls": len(controls)}


def path_audit(model: Any, controller: AdjustmentMLP, cases: list[dict[str, Any]]) -> dict[str, Any]:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    embeddings = model.token_embedding(class_ids).detach()
    trajectories: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in cases:
            x = case["x"]
            reference = case["reference"]
            ideal = embeddings[x:x + 1]
            real = torch.tensor(case["r_real"], dtype=ideal.dtype).reshape(1, -1)
            points: list[dict[str, Any]] = []
            for t in TS:
                r = (1.0 - t) * ideal + t * real
                logits = controller(r, embeddings[reference:reference + 1]).squeeze(0)
                point = {"t": t, **margins(logits, case["expected_action"])}
                point["action_name"] = point["predicted_action_name"]
                points.append(point)
            actions = [point["predicted_action"] for point in points]
            switch_count = sum(actions[index] != actions[index - 1] for index in range(1, len(actions)))
            trajectories.append({"seed_case": {key: case[key] for key in ("episode", "graph", "x", "reference", "expected_action", "expected_action_name", "canonical_predicted_action", "canonical_predicted_action_name", "predicted_action", "predicted_action_name", "comparison_category", "canonical_correct", "decision_correct", "selection")}, "trajectory": points, "endpoint_match": {"t0_matches_canonical": actions[0] == case["canonical_predicted_action"], "t1_matches_contextual": actions[-1] == case["predicted_action"]}, "switch_count": switch_count, "multiple_action_changes": switch_count > 1, "action_sequence": [action_name(action) for action in actions]})
    return {"intervention_formula": "R(t)=(1-t)*E_x+t*R_real", "t_values": list(TS), "cases": trajectories, "summary": {"case_count": len(trajectories), "multiple_action_change_count": sum(item["multiple_action_changes"] for item in trajectories), "max_switch_count": max((item["switch_count"] for item in trajectories), default=0), "endpoint_t0_match_count": sum(item["endpoint_match"]["t0_matches_canonical"] for item in trajectories), "endpoint_t1_match_count": sum(item["endpoint_match"]["t1_matches_contextual"] for item in trajectories), "note": "Intermediate points are diagnostic interventions, not training examples; no monotonicity or single boundary is assumed."}}


def audit_seed(model: Any, ctrl1: Any, manifests: dict[str, dict[str, Any]], runtime: dict[int, dict[str, Any]], seed: int, train_pairs: set[tuple[int, int]]) -> dict[str, Any]:
    checkpoint = checkpoint_path(seed)
    controller = load_controller(checkpoint)
    canonical = canonical_audit(model, controller, train_pairs)
    contextual = contextual_rows(model, controller, manifests, runtime, canonical["records"], train_pairs)
    cases, selection = choose_path_cases(contextual)
    path = path_audit(model, controller, cases)
    return {
        "status": "completed",
        "task": "T1-CTRL-2-frozen-ordinal-audit",
        "seed": seed,
        "training": False,
        "checkpoint_ctrl2": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "checkpoint_executor": {"path": str(CAMPAIGN_ROOT / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt"), "sha256": sha256(CAMPAIGN_ROOT / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")},
        "checkpoint_ctrl1": {"path": str(CAMPAIGN_ROOT / "u0c_ctrl1_pilot_seed101_frozen" / "selected.pt"), "sha256": sha256(CAMPAIGN_ROOT / "u0c_ctrl1_pilot_seed101_frozen" / "selected.pt")},
        "prior_g_results": {"path": str(prior_g_results_path(seed)), "sha256": sha256(prior_g_results_path(seed))},
        "protocol": {"weights_frozen": True, "data_frozen": True, "temperature_changed": False, "representation_changed": False, "new_weights": False, "new_data": False, "prediction_rules_applied": False},
        "part_a": canonical["summary"]["part_a"],
        "part_b": {"summary": canonical["summary"]["part_b"], "records": canonical["feature_records"]},
        "part_a_records": canonical["records"],
        "part_c": {"selection": selection, **path},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=AUDIT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    train_pairs = load_train_pairs()
    model = load_executor()
    ctrl1 = load_ctrl1()
    _, _, runtime = generate_dataset(model, ctrl1, manifests["test"], keep_runtime=True)
    seed_outputs: list[dict[str, Any]] = []
    output_paths: dict[str, str] = {}
    for seed in SEEDS:
        result = audit_seed(model, ctrl1, manifests, runtime, seed, train_pairs)
        path = args.output_root / f"seed_{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_paths[str(seed)] = str(path)
        seed_outputs.append({"seed": seed, "path": str(path), "sha256": sha256(path), "summary": {"part_a": result["part_a"], "part_b": result["part_b"]["summary"], "part_c": result["part_c"]["summary"]}})
    summary = {"status": "completed", "task": "T1-CTRL-2-frozen-ordinal-audit", "training": False, "seed_count": len(SEEDS), "seeds": seed_outputs, "output_files": output_paths, "audit_script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))}, "protocol": {"all_checkpoints_existing": True, "all_data_existing": True, "weights_frozen": True, "no_new_data_or_weights": True}, "part_a_note": "Violation counts over relational triples are counts of violated triples, not independent model failures; one predicted edge may occur in many triples.", "part_b_note": "Distances are in normalized controller feature space; admitted-pair distances are not distances to real-R training examples.", "part_c_note": "Intermediate R(t) points are diagnostic interventions only; endpoint actions must match canonical at t=0 and contextual at t=1."}
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"task": summary["task"], "summary": {"path": str(summary_path), "sha256": sha256(summary_path)}, "seed_files": {str(seed): {"path": str(args.output_root / f"seed_{seed}.json"), "sha256": sha256(args.output_root / f"seed_{seed}.json")} for seed in SEEDS}, "audit_script": summary["audit_script"]}
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "summary_sha256": sha256(summary_path), "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "seed_outputs": [{"seed": item["seed"], "path": item["path"], "sha256": item["sha256"]} for item in seed_outputs]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
