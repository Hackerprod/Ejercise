"""Frozen T1-CTRL-2-O-CANON interface diagnostic; no training or new weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import ADJUSTMENT_NAMES, BASE_CHECKPOINT, CTRL1_CHECKPOINT, KEEP, adjusted_target, adjustment_action, build_examples, decode_value, dispatch_adjustment, load_base_manifests, load_ctrl1, load_executor, navigate_collect
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from train_u0c_ctrl1 import DISTANCES, SLOT_R, trace_success
from train_u0c_ctrl2 import generate_dataset
from train_u0c_ctrl2_o import OrdinalSharedScorer, action_from_difference, build_evaluation_gate, load_dataset, sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SCORER_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_o_pilot_seed2201_frozen"
SOURCE_DATA_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_pilot_seed2201_frozen"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_o_canon_diagnostic"


@torch.no_grad()
def canonical_value_view(executor, values):
    if values.ndim != 2 or values.shape[-1] != executor.dimension:
        raise ValueError("Se esperaba un tensor [B, D]")
    if not torch.isfinite(values).all():
        raise ValueError("La entrada contiene valores no finitos")
    ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, device=values.device)
    codebook = executor.token_embedding(ids)
    logits = executor.register_decoder(values, codebook)
    local_indices = logits.argmax(dim=-1)
    canonical = codebook.index_select(0, local_indices)
    return canonical, local_indices


@torch.no_grad()
def decoder_details(executor, values: torch.Tensor) -> dict[str, Any]:
    ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, device=values.device)
    codebook = executor.token_embedding(ids)
    logits = executor.register_decoder(values, codebook)
    top = logits.topk(k=2, dim=-1)
    return {"predicted_local_indices": logits.argmax(dim=-1), "logits": logits, "margins": top.values[:, 0] - top.values[:, 1]}


def score_detail(scorer: OrdinalSharedScorer, register: torch.Tensor, reference: torch.Tensor, expected: int) -> dict[str, Any]:
    difference = scorer.difference(register, reference)
    tau = scorer.tau()
    logits = scorer.logits(register, reference).squeeze(0)
    action = int(action_from_difference(difference, tau).item())
    values = [float(value) for value in logits.tolist()]
    return {"scores": {"s_R": float(scorer.score(register).item()), "s_b": float(scorer.score(reference).item())}, "difference": float(difference.item()), "tau": float(tau.item()), "logits": values, "d_order": values[0] - values[2], "d_keep": values[1] - max(values[0], values[2]), "action": action, "action_name": ADJUSTMENT_NAMES[action], "expected_action": expected, "expected_action_name": ADJUSTMENT_NAMES[expected], "action_correct": action == expected}


def simple_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [record for record in records if record["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "action_correct": sum(record["action_correct"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected)}
    for action in range(3):
        selected = [record for record in records if record["expected_action"] == action]
        by_action[ADJUSTMENT_NAMES[action]] = {"samples": len(selected), "action_correct": sum(record["action_correct"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected)}
    return {"samples": len(records), "action_correct": sum(record["action_correct"] for record in records), "final_success": sum(record["final_success"] for record in records), "trace_success": sum(record["trace_success"] for record in records), "action_accuracy": sum(record["action_correct"] for record in records) / max(1, len(records)), "final_success_rate": sum(record["final_success"] for record in records) / max(1, len(records)), "trace_success_rate": sum(record["trace_success"] for record in records) / max(1, len(records)), "by_action": by_action, "by_distance": by_distance}


def prototype_check(executor: Any) -> dict[str, Any]:
    embeddings = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))
    details = decoder_details(executor, embeddings)
    local = details["predicted_local_indices"].tolist()
    expected = list(range(VALUE_COUNT))
    records = [{"value": value, "token_id": VALUE_BASE + value, "predicted_local_index": local[value], "decoder_margin": float(details["margins"][value].item()), "identity_preserved": local[value] == value} for value in range(VALUE_COUNT)]
    return {"samples": VALUE_COUNT, "all_identity_preserved": local == expected, "min_decoder_margin": min(record["decoder_margin"] for record in records), "records": records}


def canonical_scorer_check(executor: Any, scorer: OrdinalSharedScorer) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    embeddings = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))
    canonical, local = canonical_value_view(executor, embeddings)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for x in range(VALUE_COUNT):
            for reference in range(VALUE_COUNT):
                detail = score_detail(scorer, canonical[x:x + 1], canonical[reference:reference + 1], adjustment_action(x, reference))
                records.append({"x": x, "reference": reference, **detail})
    return {"samples": len(records), "correct": sum(record["action_correct"] for record in records), "exact": all(record["action_correct"] for record in records), "accuracy": sum(record["action_correct"] for record in records) / len(records), "scorer_tau": float(scorer.tau().item())}, records


def real_r_check(executor: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for episode in manifest["episodes"]:
            navigation = runtime[episode["episode"]]
            raw = navigation["state"][:, SLOT_R].clone()
            decoded = decoder_details(executor, raw)
            canonical, local = canonical_value_view(executor, raw)
            predicted = int(local.item())
            margin = float(decoded["margins"].item())
            raw_hash = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
            canonical_hash = hashlib.sha256(canonical.numpy().tobytes()).hexdigest()
            x = int(navigation["symbolic_x"])
            records.append({"episode": episode["episode"], "graph": episode["graph"], "x": x, "raw_r_sha256": raw_hash, "q_r_value": predicted, "q_r_token_id": VALUE_BASE + predicted, "selected_embedding_sha256": canonical_hash, "q_r_score": float(scorer.score(canonical).item()), "decoder_margin": margin, "decoder_correct_against_symbolic_x": predicted == x, "decoder_logits": [float(value) for value in decoded["logits"].squeeze(0).tolist()]})
    margins = [record["decoder_margin"] for record in records]
    return {"samples": len(records), "correct_against_symbolic_x": sum(record["decoder_correct_against_symbolic_x"] for record in records), "all_correct": all(record["decoder_correct_against_symbolic_x"] for record in records), "decoder_margin": {"min": min(margins), "max": max(margins), "mean": sum(margins) / len(margins)}, "records": records}, records


def evaluate_original_pilot(executor: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for example in build_examples(manifest):
            navigation = runtime[example["episode"]]
            raw = navigation["state"][:, SLOT_R].clone()
            q_r, _ = canonical_value_view(executor, raw)
            q_b, _ = canonical_value_view(executor, executor.token_embedding(torch.tensor([VALUE_BASE + example["reference"]])))
            detail = score_detail(scorer, q_r, q_b, example["action"])
            state, operation = dispatch_adjustment(executor, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], detail["action"])
            predicted_value = decode_value(executor, state)
            final_success = predicted_value == example["target_value"]
            records.append({"example": example["example"], "episode": example["episode"], "distance": example["distance"], "x": example["x"], "reference": example["reference"], "expected_action": example["action"], "action_correct": detail["action_correct"], "final_success": final_success, "trace_success": trace_success(final_success, navigation["timeout"], navigation["first_control_error"], None if final_success else {"stage": "CTRL-2-O-CANON", "instruction": operation}), "predicted_value": predicted_value, "target_value": example["target_value"], "operation": operation})
    return simple_metrics(records), records


def evaluate_contextual(executor: Any, scorer: OrdinalSharedScorer, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    epsilon_by_episode: list[dict[str, Any]] = []
    with torch.inference_mode():
        for episode in manifest["episodes"]:
            navigation = runtime[episode["episode"]]
            raw = navigation["state"][:, SLOT_R].clone()
            q_r, q_index = canonical_value_view(executor, raw)
            q_r_hash = hashlib.sha256(q_r.numpy().tobytes()).hexdigest()
            x = int(navigation["symbolic_x"])
            epsilon_by_episode.append({"episode": episode["episode"], "graph": episode["graph"], "x": x, "raw_r_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(), "q_r_value": int(q_index.item()), "q_r_embedding_sha256": q_r_hash})
            for reference in range(VALUE_COUNT):
                q_b, _ = canonical_value_view(executor, executor.token_embedding(torch.tensor([VALUE_BASE + reference])))
                expected = adjustment_action(x, reference)
                detail = score_detail(scorer, q_r, q_b, expected)
                oracle_state, _ = dispatch_adjustment(executor, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], expected)
                predicted_state, operation = dispatch_adjustment(executor, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], detail["action"])
                target = adjusted_target(x, reference)
                oracle_value = decode_value(executor, oracle_state)
                predicted_value = decode_value(executor, predicted_state)
                records.append({"episode": episode["episode"], "graph": episode["graph"], "distance": episode["distance"], "x": x, "reference": reference, "expected_action": expected, "predicted_action": detail["action"], "predicted_action_name": detail["action_name"], "action_correct": detail["action_correct"], "oracle_final_success": oracle_value == target, "final_success": predicted_value == target, "trace_success": trace_success(predicted_value == target, navigation["timeout"], None if detail["action_correct"] else {"stage": "CTRL-2-O-CANON", "reference": reference}, None if predicted_value == target else {"stage": "CTRL-2-O-CANON", "instruction": operation}), "canonicalized_r_value": int(q_index.item()), "predicted_value": predicted_value, "target_value": target, "operation": operation})
    summary = simple_metrics(records)
    summary.update({"decision_correct": summary["action_correct"], "decision_accuracy": summary["action_accuracy"], "oracle_final_success": sum(record["oracle_final_success"] for record in records), "oracle_final_success_rate": sum(record["oracle_final_success"] for record in records) / len(records), "by_action": {name: {**values, "decision_correct": values["action_correct"]} for name, values in summary["by_action"].items()}})
    summary["by_distance"] = {distance: {**values, "decision_correct": values["action_correct"]} for distance, values in summary["by_distance"].items()}
    pair_table = [{"x": x, "reference": reference, "expected_action": ADJUSTMENT_NAMES[adjustment_action(x, reference)], "samples": sum(record["x"] == x and record["reference"] == reference for record in records), "decision_correct": sum(record["x"] == x and record["reference"] == reference and record["action_correct"] for record in records), "oracle_final_success": sum(record["x"] == x and record["reference"] == reference and record["oracle_final_success"] for record in records), "final_success": sum(record["x"] == x and record["reference"] == reference and record["final_success"] for record in records), "trace_success": sum(record["x"] == x and record["reference"] == reference and record["trace_success"] for record in records)} for x in range(VALUE_COUNT) for reference in range(VALUE_COUNT)]
    return summary, records, pair_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = load_base_manifests()
    model = load_executor()
    ctrl1 = load_ctrl1()
    scorer_payload = torch.load(SCORER_ROOT / "final.pt", map_location="cpu", weights_only=False)
    scorer = OrdinalSharedScorer()
    scorer.load_state_dict(scorer_payload["controller"], strict=True)
    scorer.eval()
    _, _, runtime = generate_dataset(model, ctrl1, manifests["test"], keep_runtime=True)
    prototype = prototype_check(model)
    real_r_summary, real_r_records = real_r_check(model, scorer, manifests["test"], runtime)
    original_summary, original_records = evaluate_original_pilot(model, scorer, manifests["test"], runtime)
    contextual_summary, contextual_records, pair_table = evaluate_contextual(model, scorer, manifests["test"], runtime)
    canonical_summary, canonical_records = canonical_scorer_check(model, scorer)
    pilot_gate_input = {"samples": original_summary["samples"], "correct_count": original_summary["action_correct"], "by_action": {name: {"samples": value["samples"], "correct_count": value["action_correct"], "final_success_count": value["final_success"]} for name, value in original_summary["by_action"].items()}, "by_distance": {name: {"samples": value["samples"], "final_success_count": value["final_success"]} for name, value in original_summary["by_distance"].items()}}
    gate = build_evaluation_gate(canonical_summary, pilot_gate_input, contextual_summary, pair_table)
    per_r = {record["episode"]: record for record in real_r_records}
    for record in contextual_records:
        summary = per_r[record["episode"]]
        summary.setdefault("action_counts", {name: 0 for name in ADJUSTMENT_NAMES})[record["predicted_action_name"]] += 1
        if not record["action_correct"] or not record["final_success"] or not record["oracle_final_success"]:
            summary.setdefault("failed_references", []).append({"reference": record["reference"], "action": record["action_correct"], "final_success": record["final_success"], "oracle_final_success": record["oracle_final_success"]})
    for summary in real_r_records:
        summary.setdefault("failed_references", [])
    full_context_failures = [record for record in contextual_records if not record["action_correct"] or not record["final_success"] or not record["oracle_final_success"]]
    decoder_failures = [record for record in real_r_records if not record["decoder_correct_against_symbolic_x"]]
    tail = {"selection_rule_fixed_before_evaluation": "all decoder-misclassified R or contextual rows with action/final/oracle failure; counts computed before output limiting", "decoder_failure_count": len(decoder_failures), "contextual_failure_count": len(full_context_failures), "decoder_failures": decoder_failures, "contextual_failures": full_context_failures[:512]}
    result = {"status": "completed", "task": "T1-CTRL-2-O-CANON", "training": False, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": sha256(BASE_CHECKPOINT)}, "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": sha256(CTRL1_CHECKPOINT)}, "checkpoint_ctrl2_original": {"path": str(SCORER_ROOT / "final.pt"), "sha256": sha256(SCORER_ROOT / "final.pt")}, "protocol": {"diagnostic_read_e_select": False, "r_real_preserved_for_alu": True, "scorer_input": "Q(R) and Q(reference)", "alu_input": "raw R", "new_weights": False, "new_data": False, "training": False}, "bridge": "canonical_value_view", "prototype_check": prototype, "canonical_scorer_check": canonical_summary, "canonical_scorer_records": canonical_records, "real_r_check": {key: value for key, value in real_r_summary.items() if key != "records"}, "real_r_records": real_r_records, "original_pilot": original_summary, "original_pilot_records": original_records, "contextual": contextual_summary, "contextual_pair_table": pair_table, "contextual_records": contextual_records, "tail": tail, "gate": gate}
    output_path = args.output_root / "results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output_path), "sha256": sha256(output_path), "checkpoint_ctrl2_sha256": sha256(SCORER_ROOT / "final.pt"), "prototype": {key: value for key, value in prototype.items() if key != "records"}, "real_r": {key: value for key, value in real_r_summary.items() if key != "records"}, "original_pilot": {key: value for key, value in original_summary.items() if key not in ("by_action", "by_distance")}, "contextual": {key: value for key, value in contextual_summary.items() if key not in ("by_action", "by_distance")}, "gate": gate, "tail": {key: value for key, value in tail.items() if key not in ("decoder_failures", "contextual_failures")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
