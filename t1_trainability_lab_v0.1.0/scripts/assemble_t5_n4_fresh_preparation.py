"""Assemble T5 N=4 fresh-preparation receipt; no model/training path."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import check_t5_n4_fresh_manifests as checker


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "t5_n4_fresh_preparation" / "t5_n4_fresh_preparation.json"


def file_record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_self_hashed(path: Path, artifact: dict) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(written)
    placeholder = written.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if hashlib.sha256(placeholder).hexdigest() != digest:
        raise RuntimeError("fresh preparation self-hash verification failed")
    return digest


def static_audit() -> dict:
    script = ROOT / "scripts" / "audit_t5_n4_static.py"
    completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(completed.stdout)


def main() -> int:
    all_assignments = [dict(zip(checker.OPERATORS, permutation)) for permutation in itertools.permutations(checker.ROLES)]
    selected = __import__("random").Random(7400).sample(all_assignments, 5)
    omitted = [item for item in all_assignments if item not in selected]
    reports = [checker.check(checker.MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, selected, omitted) for seed in checker.SEEDS]
    current_ids = set()
    for report in reports:
        current_ids.update(json.loads(Path(report["path"]).read_text(encoding="utf-8")).get("token_ids", {}).values())
    prior_ids = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t5_n4_fresh_preparation" in path.parts:
            continue
        try:
            checker.collect_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    checker_failures = [failure for report in reports for failure in report["failures"]]
    if len(current_ids) != 185:
        checker_failures.append("cross_seed_physical_id_overlap")
    if current_ids.intersection(prior_ids):
        checker_failures.append("prior_physical_id_overlap")
    total_rows = sum(report["test_rows"] for report in reports)
    if total_rows != checker.EXPECTED_TOTAL:
        checker_failures.append("total_test_rows")
    checker_result = {"status": "passed" if not checker_failures and all(report["status"] == "passed" for report in reports) else "failed", "failure_count": len(checker_failures), "failures": checker_failures, "selected_mappings": selected, "omitted_mappings": omitted, "manifests": reports, "total_atomic_training_rows": 640, "total_test_rows": total_rows, "expected_total_test_rows": checker.EXPECTED_TOTAL, "prior_physical_id_overlap": sorted(current_ids.intersection(prior_ids)), "cross_seed_physical_id_overlap": len(current_ids) != 185, "model_constructed": False, "training_performed": False, "new_checkpoints": False}
    static = static_audit()
    recipe_files = {
        "model_generic_nrole": file_record("scripts/t5_nrole_design_audit.py"),
        "trainer": file_record("scripts/execute_t5_n4_development.py"),
        "generic_rcsep_n_bg": file_record("scripts/t5_nrole_design_audit.py"),
        "evaluator_d1_d2_d3": file_record("scripts/execute_t5_n4_development.py"),
        "development_manifest_generator": file_record("scripts/generate_t5_n4_manifests.py"),
        "development_manifest_checker": file_record("scripts/check_t5_n4_manifests.py"),
        "fresh_manifest_generator": file_record("scripts/generate_t5_n4_fresh_manifests.py"),
        "fresh_manifest_checker": file_record("scripts/check_t5_n4_fresh_manifests.py"),
        "static_audit": file_record("scripts/audit_t5_n4_static.py"),
        "decoder_loader": file_record("scripts/ctrl2_common.py"),
        "decoder_constants": file_record("scripts/evaluate_u0c_c1_e_r_alu.py"),
        "supervisor_runtime": file_record("scripts/train_t2_i0_baseline_b.py"),
        "decoder_checkpoint": file_record("campaign/u0c_c1_lossnorm_anneal_seed101_12000/final.pt"),
        "supervisor_checkpoint": file_record("campaign/u0c_ctrl7_pilot_seed4701/final.pt"),
    }
    artifact = {"status": "passed" if checker_result["status"] == "passed" and static["status"] == "passed" else "failed", "task": "T5-N4-FRESH-PREPARATION", "scope": {"training_performed": False, "new_checkpoints": False, "model_constructed": False, "development_seeds": [7301, 7302, 7303, 7304, 7305], "fresh_seeds": list(checker.SEEDS), "reserved_future_seeds": []}, "recipe_freeze": {"status": "frozen_exactly_from_T5_N4_DEVELOPMENT_CLOSURE_PASS", "Q_shape": [4, 16], "roles": list(checker.ROLES), "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "behavior_roles": ["FLOOR", "AVOID"], "reference_roles": list(checker.ROLES), "L_sep": {"role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "reduction_denominator": 16, "coefficient": 1.0}, "background_contexts": {"START→OP": 4, "ARG→LINK": 32, "LINK→OP": 4, "total": 40}, "shared_w_v": True, "pure_HARDPTR": True, "masks": False, "gate": False, "stage_b": False, "soft_mixture": False, "updates": 5000, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "batch_rows": {role: 32 for role in checker.ROLES} | {"total": 128}, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "files": recipe_files}, "role_assignment": {"batch_seed": 7400, "formula": "random.Random(7400).sample(all_24_role_permutations_of_FAMH, 5)", "operators": list(checker.OPERATORS), "roles": list(checker.ROLES), "selected": selected, "omitted": omitted, "all_assignment_count": 24, "no_changes_after_sample": True}, "rules": {"E": "E=(seed+L+F+1)%32; increment by 1 modulo 32 until E not in {L,F}", "H": "H=(seed+L+F+E+2)%32; increment by 1 modulo 32 until H not in {L,F,E}", "quadruples": 992, "orders_per_quadruple": 24, "test_rows_per_seed": 23808, "test_rows_total": 119040}, "checker": checker_result, "static_audit": static, "training_performed": False, "new_checkpoints": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    digest = write_self_hashed(OUTPUT, artifact)
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "training_performed": False, "new_checkpoints": False, "checker": checker_result, "static_audit": static}, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
