"""Assemble T6 active-set midpoint preparation receipt; no model/training path."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import check_t6_activeset_midpoint_manifests as checker


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "t6_activeset_midpoint_preparation" / "t6_activeset_midpoint_preparation.json"


def file_record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_self_hashed(path: Path, artifact: dict[str, object]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(written)
    placeholder = written.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if hashlib.sha256(placeholder).hexdigest() != digest:
        raise RuntimeError("T6 preparation self-hash verification failed")
    return digest


def run_checker() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_t6_activeset_midpoint_manifests.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def main() -> int:
    checker_result = run_checker()
    manifest_records = [
        file_record(f"campaign/t6_activeset_midpoint_preparation/manifests/manifest_{seed}_v1.json")
        for seed in checker.SEEDS
    ]
    files = {
        "manifest_generator": file_record("scripts/generate_t6_activeset_midpoint_manifests.py"),
        "manifest_checker": file_record("scripts/check_t6_activeset_midpoint_manifests.py"),
        "design_reference": file_record("campaign/t6_nrole_activeset_design/t6_nrole_activeset_design.json"),
        "manifests": manifest_records,
    }
    artifact: dict[str, object] = {
        "status": "passed" if checker_result["status"] == "passed" else "failed",
        "task": "T6-ACTIVESET-MIDPOINT-PREPARATION",
        "schema": "T6-activeset-midpoint-preparation-v1",
        "scope": {
            "fresh_seeds": list(checker.SEEDS),
            "reserved_future_seeds": [7601, 7602, 7603, 7604, 7605],
            "training_performed": False,
            "model_constructed": False,
            "new_checkpoints": False,
        },
        "role_assignment": {
            "batch_seed": 7500,
            "formula": "random.Random(7500).sample(all_24_role_permutations_of_FAMH, 5)",
            "operators": list(checker.OPERATORS),
            "roles": list(checker.ROLES),
            "all_assignment_count": 24,
            "selected_and_omitted_frozen": True,
        },
        "recipe_freeze": {
            "source": "T5-N4-DEVELOPMENT-CLOSURE-PASS",
            "Q_shape": [4, 16],
            "roles": list(checker.ROLES),
            "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)",
            "L_sep": {"role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "reduction_denominator": 16, "coefficient": 1.0},
            "background_contexts": {"START→OP": 4, "ARG→LINK": 32, "LINK→OP": 4, "total": 40},
            "shared_w_v": True,
            "pure_HARDPTR": True,
            "masks": False,
            "gate": False,
            "stage_b": False,
            "soft_mixture": False,
            "updates": 5000,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "batch_rows": {role: 32 for role in checker.ROLES} | {"total": 128},
            "multi_clause_train": 0,
            "joint_examples_in_training": 0,
            "test_rows_used": 0,
        },
        "active_set_protocol": {
            "subsets": 15,
            "ordered_configurations": 64,
            "forward_inputs": ["token_ids", "lengths"],
            "active_roles_model_input": False,
            "active_roles_evaluator_only": True,
            "four_queries_always_run": True,
            "absence": {"present": False, "pointer": None, "value_state": None, "decoded_value": None, "RAW": None, "CANON": None},
            "tie_policy": {"NULL_vs_real": "NULL wins ties", "real_vs_real": "minimum valid position wins"},
        },
        "evaluation_plan": {
            "per_seed_counts_by_subset_size": {"1": 128, "2": 11904, "3": 23808, "4": 23808},
            "per_seed_total": 59648,
            "per_seed_multi_clause": 59520,
            "all_seed_total": 298240,
            "all_seed_multi_clause": 297600,
            "derived_value_formulas": {
                "v2": "(seed+v0+v1+1)%32; increment modulo 32 until distinct",
                "v3": "(seed+v0+v1+v2+2)%32; increment modulo 32 until distinct",
            },
        },
        "midpoint_calibration": {
            "status": "prepared_not_executed",
            "formula": "b_r=N_r^max+(P_r^min-N_r^max)/2",
            "weights_must_be_frozen": True,
            "allowed_sources": ["128 atomic catalog rows", "40 background contexts"],
            "forbidden_sources": ["multi-clause evaluation rows", "joint examples", "active-set labels", "learned thresholds"],
            "interval_requirement": "N_r^max < b_r < P_r^min",
            "future_output": "one frozen b_r per role and seed, with extremal source evidence",
            "reference_status": "7301 design artifact contains engineering-only interval evidence; no fresh 750x calibration executed",
        },
        "files": files,
        "checker": checker_result,
        "training_performed": False,
        "model_constructed": False,
        "new_checkpoints": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    digest = write_self_hashed(OUTPUT, artifact)
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "checker": checker_result, "training_performed": False, "new_checkpoints": False}, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
