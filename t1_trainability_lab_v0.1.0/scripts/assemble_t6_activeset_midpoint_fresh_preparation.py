"""Assemble T6 fresh preparation freeze; never trains, calibrates, or creates checkpoints."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREP_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation"
OUTPUT = PREP_ROOT / "t6_activeset_midpoint_fresh_preparation.json"


def file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    path.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return digest


def run_json(script: str) -> dict[str, Any]:
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(completed.stdout)


def main() -> int:
    checker_output = run_json("check_t6_activeset_midpoint_fresh_manifests.py")
    static_output = run_json("audit_t6_activeset_midpoint_fresh_static.py")
    checker_artifact = json.loads((ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "checker.json").read_text(encoding="utf-8"))
    static_artifact = json.loads((ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "static_audit.json").read_text(encoding="utf-8"))
    manifest_records = [file_record(f"campaign/t6_activeset_midpoint_fresh_preparation/manifests/manifest_{seed}_v1.json") for seed in (7601, 7602, 7603, 7604, 7605)]
    components = {
        "generic_core": {**file_record("scripts/t5_nrole_design_audit.py"), "symbols": ["GenericNRoleBinder"]},
        "trainer": {**file_record("scripts/execute_t6_activeset_midpoint_development.py"), "symbols": ["objective_n4", "train_seed"], "status": "future recipe only; no fresh training executed"},
        "calibrator": {**file_record("scripts/execute_t6_activeset_midpoint_development.py"), "symbols": ["d2_calibrate"], "status": "procedure frozen; no fresh calibration executed"},
        "null_wrapper": {**file_record("scripts/execute_t6_nrole_activeset_design_audit.py"), "symbols": ["NullCapableBinder"]},
        "corrected_evaluator": {**file_record("scripts/execute_t6_activeset_midpoint_development.py"), "symbols": ["d3_evaluate", "decode_states"], "padding_contract": "variable token rows padded with LINK and lengths delimit valid positions"},
        "fresh_generator": {**file_record("scripts/generate_t6_activeset_midpoint_fresh_manifests.py"), "symbols": ["make_assignments", "make_manifest"]},
        "fresh_checker": {**file_record("scripts/check_t6_activeset_midpoint_fresh_manifests.py"), "symbols": ["check_manifest", "expected_assignments"]},
        "static_audit": file_record("scripts/audit_t6_activeset_midpoint_fresh_static.py"),
        "decoder_loader": file_record("scripts/ctrl2_common.py"),
        "decoder_model_definition": file_record("scripts/evaluate_u0c_c1_e_r_alu.py"),
        "decoder_codebook_checkpoint": file_record("campaign/u0c_c1_lossnorm_anneal_seed101_12000/final.pt"),
    }
    correct_development = file_record("campaign/t6_activeset_midpoint_development_closed2/results.json")
    invalid_harness_roots = [f"campaign/t6_activeset_midpoint_{name}" for name in ("development", "development_closed", "development_complete", "development_corrected", "development_corrected2", "development_final")]
    artifact: dict[str, Any] = {
        "status": "passed" if checker_output["status"] == "passed" and static_output["status"] == "passed" else "failed",
        "task": "T6-ACTIVESET-MIDPOINT-FRESH-PREPARATION",
        "schema": "T6-activeset-midpoint-fresh-preparation-v1",
        "scope": {"fresh_seeds": [7601, 7602, 7603, 7604, 7605], "reserved_future_seeds": [], "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False},
        "development_provenance": {"correct_execution": {**correct_development, "classification": "T6-ACTIVESET-MIDPOINT DEVELOPMENT: CLOSED/PASS", "scientific_status": "valid"}, "invalid_harness_bug_attempts": [{"path": path, "classification": "INVALID/HARNESS BUG", "historical": True, "modified": False, "bugs": ["decoder invocation omitted codebook", "variable-length evaluation batch padding"]} for path in invalid_harness_roots], "commit": "d6060e796b819bbf7e69a8b61ab1ad12c5bc44c8"},
        "role_assignment": {"batch_seed": 7600, "formula": "random.Random(7600).sample(all_24_role_permutations_of_FAMH, 5)", "operators": ["OP_W", "OP_X", "OP_Y", "OP_Z"], "roles": ["FLOOR", "AVOID", "MATCH", "ANCHOR"], "all_assignment_count": 24, "selected_mappings": checker_artifact["selected_mappings"], "omitted_mappings": checker_artifact["omitted_mappings"], "selection_order_frozen": True},
        "recipe_freeze": {"source": "T6-ACTIVESET-MIDPOINT DEVELOPMENT CLOSED/PASS", "Q_shape": [4, 16], "roles": ["FLOOR", "AVOID", "MATCH", "ANCHOR"], "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_sep": {"role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "reduction_denominator": 16, "coefficient": 1.0}, "background_contexts": {"START→OP": 4, "ARG→LINK": 32, "LINK→OP": 4, "total": 40}, "shared_w_v": True, "pure_HARDPTR": True, "masks": False, "gate": False, "stage_b": False, "soft_mixture": False, "updates": 5000, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "batch_rows": {role: 32 for role in ["FLOOR", "AVOID", "MATCH", "ANCHOR"]} | {"total": 128}, "fresh_init_total": True, "prior_binder_weights_loaded": False, "decoder_codebook_frozen": True, "multi_clause_train": 0, "test_rows_used": 0},
        "null_calibration_procedure_freeze": {"has_four_role_thresholds_per_future_core": True, "formula": "b_r=N_r^max+(P_r^min-N_r^max)/2", "P_r_min": "min_i s_r(OP_r,ARG_i)", "N_r_max": "max(max_{r'!=r,j}s_r(OP_r',ARG_j), max_{z in S_4}s_r(z))", "allowed_sources": ["128 atomic relations", "40 BG contexts"], "forbidden_sources": ["multi-clause rows", "active-set labels", "evaluator results", "other-seed thresholds"], "acceptance": ["P_r^min>N_r^max", "P_r^min-b_r>0 after REAL dtype storage", "b_r-N_r^max>0 after REAL dtype storage"], "one_vector_per_model": True, "no_threshold_training": True, "no_inference_adaptation": True, "fresh_calibration_executed": False},
        "active_set_protocol": {"subsets": 15, "ordered_configurations": 64, "forward_inputs": ["token_ids", "lengths"], "active_roles_model_input": False, "active_roles_evaluator_only": True, "four_queries_always_run": True, "absence": {"present": False, "pointer": None, "value_state": None, "decoded_value": None, "RAW": None, "CANON": None}, "value_zero_is_valid": True, "canonicalize_only_present": True, "tie_policy": {"NULL_vs_real": "NULL wins ties", "real_vs_real": "minimum valid position wins"}},
        "evaluation_plan": {"per_seed_counts_by_subset_size": {"1": 128, "2": 11904, "3": 23808, "4": 23808}, "per_seed_total": 59648, "per_seed_multi_clause": 59520, "all_seed_total": 298240, "all_seed_multi_clause": 297600, "singletons_are_atomic_controls": True, "derived_value_formulas": {"v2": "(seed+v0+v1+1)%32; increment modulo 32 until distinct", "v3": "(seed+v0+v1+v2+2)%32; increment modulo 32 until distinct"}},
        "manifests": manifest_records,
        "checker": {"artifact": file_record("campaign/t6_activeset_midpoint_fresh_preparation/checker.json"), "result": checker_artifact},
        "static_audit": {"artifact": file_record("campaign/t6_activeset_midpoint_fresh_preparation/static_audit.json"), "result": static_artifact},
        "components": components,
        "training_performed": False,
        "model_constructed": False,
        "calibration_performed": False,
        "new_checkpoints": False,
    }
    digest = write_self_hashed(OUTPUT, artifact)
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "training_performed": False, "calibration_performed": False, "new_checkpoints": False, "manifest_count": len(manifest_records), "checker_status": checker_artifact["status"], "static_status": static_artifact["status"]}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
