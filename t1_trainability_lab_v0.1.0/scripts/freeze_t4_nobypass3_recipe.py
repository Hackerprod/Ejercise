"""Materialize the frozen T4-NOBYPASS-3 recipe without importing model code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "t4_nobypass3_fresh" / "recipe_freeze.json"


def record(relative: str, symbols: list[str]) -> dict[str, object]:
    path = ROOT / relative
    return {"path": relative.replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size, "symbols": symbols}


def main() -> int:
    artifact = {
        "status": "frozen",
        "task": "T4-NOBYPASS-3-FRESH",
        "recipe_version": 1,
        "training_performed": False,
        "model": "T4RelKeyEncoder with shared W_v",
        "components": {
            "model": record("scripts/execute_t4_nobypass2_rcsep3_bg.py", ["T4RelKeyEncoder"]),
            "trainer": record("scripts/execute_t4_nobypass2_rcsep3_bg.py", ["train_seed", "main"]),
            "rcsep3_bg": record("scripts/execute_t4_nobypass2_rcsep3_bg.py", ["separation_with_background", "background_contexts", "objective_with_background"]),
            "hardptr_evaluator": record("scripts/execute_t4_nobypass1_three_active_roles.py", ["atomic_gate", "evaluate_test", "decode_state"]),
            "manifest_generator": record("scripts/generate_t4_nobypass3_fresh.py", ["assignments", "build_manifest", "match_value"]),
            "manifest_checker": record("scripts/check_t4_nobypass3_fresh.py", ["expected_mappings", "check_manifest", "main"]),
            "decoder_runtime": record("scripts/ctrl2_common.py", ["load_executor", "BASE_CHECKPOINT"]),
            "supervisor_runtime": record("scripts/train_t2_i0_baseline_b.py", ["LatentConditionedSupervisor", "CTRL7_CHECKPOINT"]),
            "source_runtime": record("scripts/train_t2_i2_r2.py", ["load_source"]),
            "value_decoder_contract": record("scripts/evaluate_u0c_c1_e_r_alu.py", ["VALUE_BASE", "VALUE_COUNT"]),
        },
        "flags": {
            "queries": 3,
            "shared_w_v": True,
            "rcsep_role_terms": 6,
            "rcsep_background_terms": 3,
            "background_contexts": 38,
            "multi_clause_train": 0,
            "local_background_negative_contexts": True,
            "hardptr_mask": False,
            "stage_b": False,
            "gate": False,
            "soft_mixture": False,
        },
        "literal_flags": [
            "queries=3",
            "shared_w_v=true",
            "rcsep_role_terms=6",
            "rcsep_background_terms=3",
            "background_contexts=38",
            "multi_clause_train=0",
            "local_background_negative_contexts=true",
            "hardptr_mask=false",
            "stage_b=false",
            "gate=false",
            "soft_mixture=false",
        ],
        "loss": {
            "total": "L=L_behavior^{F/A}+L_ref^{F/A/M}+L_sep^{3+BG}",
            "separation": "L_sep^{3+BG}=(1/9)[sum_r sum_{r'!=r} L_{r<-r'} + sum_r L_{r<-BG}]",
            "role_terms": ["F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A"],
            "background_terms": ["F<-BG", "A<-BG", "M<-BG"],
            "background_term": "mean_{i,z} softplus[BG_r(z)-SELF_r(i)] over [32,38]",
            "coefficient": 1.0,
        },
        "hyperparameters": {
            "updates": 5000,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": "1e-3->1e-5",
            "atomic_examples": 96,
            "atomic_rows": {"FLOOR": 32, "AVOID": 32, "MATCH": 32},
            "multi_clause_train": 0,
            "joint_examples_in_training": 0,
            "background_negatives": "local, active during training",
        },
        "hardptr": {"evaluator": "pure argmax over valid positions", "mask": False, "new_mask": False},
    }
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "training_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
