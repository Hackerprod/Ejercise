"""Prepare deterministic T7 development manifests without model or training work."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
SEEDS = (7701, 7702, 7703, 7704, 7705)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR", "NOOP")
OPERATORS = ("OP_V", "OP_W", "OP_X", "OP_Y", "OP_Z")
ARGUMENTS = tuple(f"ARG_{index:02d}" for index in range(32))
VALUE_COUNT = 32
ID_BLOCKS = {
    7701: (47000, 47499),
    7702: (48000, 48499),
    7703: (49000, 49499),
    7704: (50000, 50499),
    7705: (51000, 51499),
}


def all_assignments() -> tuple[dict[str, str], ...]:
    permutations = itertools.permutations(ROLES)
    return tuple(dict(zip(OPERATORS, role_order)) for role_order in permutations)


ALL_ASSIGNMENTS = all_assignments()
SELECTED_ASSIGNMENTS = tuple(random.Random(7700).sample(list(ALL_ASSIGNMENTS), 5))


def domain_permutation(seed: int) -> list[int]:
    return random.Random(seed).sample(range(VALUE_COUNT), VALUE_COUNT)


def base_token_ids(seed: int, token_order: list[str]) -> dict[str, int]:
    start, end = ID_BLOCKS[seed]
    physical_ids = random.Random(seed + 100000).sample(range(start, end + 1), len(token_order))
    return dict(zip(token_order, physical_ids))


def distractor_formula() -> str:
    return "D=(seed+sum(v_r for r in active_roles)+1)%32; increment modulo 32 until D differs from every active value"


def evaluation_spec() -> dict[str, Any]:
    base = {"1": {"orders": 1, "rows": 128}, "2": {"orders": 2, "rows": 11904}, "3": {"orders": 6, "rows": 23808}, "4": {"orders": 24, "rows": 23808}}
    augmented = {"1": {"orders": 2, "rows": 256}, "2": {"orders": 6, "rows": 35712}, "3": {"orders": 24, "rows": 95232}, "4": {"orders": 120, "rows": 119040}}
    return {
        "base": base,
        "augmented": augmented,
        "base_total": 59648,
        "augmented_total": 250240,
        "formula": "For each non-empty active subset of R and distinct active values, enumerate every permutation of active clauses; augmented enumerates every permutation after inserting exactly one NOOP clause.",
        "augmented_order_formula": "itertools.permutations(active_roles + (NOOP,))",
        "base_order_formula": "itertools.permutations(active_roles)",
        "pairing_rule": "The augmented row is paired to its base row by removing the NOOP clause outside the model; D is fixed across all orders for that assignment.",
        "per_seed": {"base": 59648, "augmented": 250240, "combined": 309888},
        "five_seed": {"base": 298240, "augmented": 1251200, "combined": 1549440},
    }


def build_manifest(seed: int, mapping: dict[str, str]) -> dict[str, Any]:
    noop_operator = next(operator for operator, role in mapping.items() if role == "NOOP")
    active_mapping = {operator: role for operator, role in mapping.items() if role != "NOOP"}
    active_operators = [operator for operator in OPERATORS if operator != noop_operator]
    stage_a_token_order = [*ARGUMENTS, *active_operators, "LINK"]
    stage_c_token_order = [*stage_a_token_order, noop_operator]
    permutation = domain_permutation(seed)
    token_ids = base_token_ids(seed, stage_a_token_order)
    operator_for_role = {role: operator for operator, role in active_mapping.items()}
    selected_index = SEEDS.index(seed)
    selected = SELECTED_ASSIGNMENTS[selected_index]
    omitted = [assignment for assignment in ALL_ASSIGNMENTS if assignment not in SELECTED_ASSIGNMENTS]
    return {
        "schema": "T7-noop-none-lexical-development-preparation-manifest-v1",
        "task": "T7-NOOP-NONE-LEXICAL-DEVELOPMENT-PREPARATION",
        "seed": seed,
        "operator_role_assignment": mapping,
        "active_operator_role_assignment": active_mapping,
        "noop_operator": noop_operator,
        "permutation": permutation,
        "token_order": stage_a_token_order,
        "token_ids": token_ids,
        "id_block": list(ID_BLOCKS[seed]),
        "id_semantics": "physical IDs fresh and disjoint; model uses manifest-local vocabulary indices",
        "operator_for_role": operator_for_role,
        "role_order": list(ROLES),
        "operator_order": list(OPERATORS),
        "role_assignment": {
            "all_count": 120,
            "formula": "random.Random(7700).sample(list(itertools.permutations((FLOOR,AVOID,MATCH,ANCHOR,NOOP))), 5)",
            "all_permutation_order": list(ROLES),
            "selection_index": selected_index,
            "selected_five": list(SELECTED_ASSIGNMENTS),
            "selected": selected,
            "omitted_count": len(omitted),
            "omitted": omitted,
        },
        "vocabulary": {
            "argument_tokens": list(ARGUMENTS),
            "stage_a_token_order": stage_a_token_order,
            "stage_a_noop_operator_present": False,
            "stage_c_token_order": stage_c_token_order,
            "stage_c_noop_row_internal_id": len(stage_a_token_order),
            "ids_preserved_when_appending": True,
            "noop_added_after_stage_b": True,
        },
        "stage_a": {
            "init": "fresh from zero per T7 domain; no historical checkpoint",
            "training_rows": 128,
            "training_rows_by_role": {role: 32 for role in ROLES if role != "NOOP"},
            "operator_vocab_excludes_noop": True,
            "objective": "L_behavior^(F,A)+L_ref^(F,A,M,H)+L_sep^(4+BG)",
            "L_sep": {"role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "reduction": "/16"},
            "background_contexts": 40,
            "updates": 5000,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear"},
            "noop_in_batch": False,
            "noop_in_objective": False,
            "noop_in_background_catalog": False,
        },
        "stage_b": {
            "core_frozen": True,
            "roles_calibrated": ["FLOOR", "AVOID", "MATCH", "ANCHOR"],
            "sources": ["128 atomic active-role relations", "40 base BG contexts"],
            "noop_contexts_used": False,
            "formula": "b_r=N_r^max+(P_r^min-N_r^max)/2",
            "strict_checks": ["P_r^min-b_r>0 in inference dtype", "b_r-N_r^max>0 in inference dtype"],
            "freeze_after_calibration": True,
            "recalibrate_after_stage_c": False,
        },
        "stage_c": {
            "trainable_parameters": ["e_N=e_{noop_operator}"],
            "trainable_dimension": 16,
            "initialization": "normal standard",
            "initialization_generator": "independent generator",
            "initialization_seed": 100000 + seed,
            "frozen": ["core", "old embeddings", "four queries", "key network", "W_v", "decoder", "b_r"],
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "updates": 5000,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear"},
            "selection": "last update; no early stopping",
            "contexts_per_update": 34,
            "loss": "(1/136)*sum_r sum_z softplus(s_r(z;e_N)-b_r)",
            "extra_terms": False,
            "active_role_loss": False,
            "value_loss": False,
            "noop_atomic_controls": {"rows": 32, "target": [None, None, None, None], "reported_separately": True},
        },
        "noop_contexts": {
            "count": 34,
            "formula": "U={OP_NOOP->ARG_D:D=0..31} union {START->OP_NOOP,LINK->OP_NOOP}",
            "score_positions": {"OP_NOOP->ARG_D": 1, "START->OP_NOOP": 0, "LINK->OP_NOOP": 1},
            "start_forward": "zero previous state at first token; no BOS invention",
            "link_rule": "score only OP_NOOP position; do not turn LINK->OP_NOOP into a complete instruction",
        },
        "evaluation": evaluation_spec(),
        "certificates": {
            "noop_to_null": "G_r^{NOOP->NULL}=b_r-max_{z in U}s_r(z)>0 for every r",
            "status": "formula sealed; not computed during preparation",
            "coverage": "all 32 possible NOOP argument values plus two structural candidates",
        },
        "separation_contract": {
            "stage_a_reads": ["stage A atomic active-role rows only"],
            "stage_b_reads": ["stage A frozen core", "128 atomic active-role rows", "40 base BG contexts"],
            "stage_c_reads": ["stage B frozen core and b_r", "34 NOOP contexts only"],
            "evaluation_reads": ["paired base and augmented logical rows only after all training stages"],
            "no_stage_reads_augmented_evaluation_during_training": True,
        },
        "status": "prepared_not_executed",
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return {"seed": manifest["seed"], "path": path.relative_to(ROOT).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    records = [write_manifest(MANIFEST_ROOT / f"manifest_{seed}_v1.json", build_manifest(seed, SELECTED_ASSIGNMENTS[index])) for index, seed in enumerate(SEEDS)]
    print(json.dumps({"status": "prepared", "manifests": records, "selected_assignments": list(SELECTED_ASSIGNMENTS), "all_assignments": len(ALL_ASSIGNMENTS)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
