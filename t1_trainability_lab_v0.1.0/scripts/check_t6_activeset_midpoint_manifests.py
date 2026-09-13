"""Independent checker for T6 active-set manifests; no model/training."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_preparation" / "manifests"
SEEDS = (7501, 7502, 7503, 7504, 7505)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
VALUE_COUNT = 32
EXPECTED_BY_SIZE = {1: 128, 2: 11904, 3: 23808, 4: 23808}
EXPECTED_TOTAL = 59648
EXPECTED_MULTI = 59520


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def derived_value(seed: int, values: list[int]) -> int:
    offset = 1 if len(values) == 2 else 2
    value = (seed + sum(values) + offset) % VALUE_COUNT
    while value in values:
        value = (value + 1) % VALUE_COUNT
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_physical_ids(value: Any, found: set[int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"token_ids", "physical_ids", "physical_token_ids"}:
                if isinstance(child, dict):
                    found.update(item for item in child.values() if isinstance(item, int))
                elif isinstance(child, list):
                    found.update(item for item in child if isinstance(item, int))
            else:
                collect_physical_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            collect_physical_ids(child, found)


def check(path: Path, seed: int, selected: list[dict[str, str]], omitted: list[dict[str, str]]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if data.get("schema") != "T6-activeset-midpoint-manifest-v1":
        failures.append("schema")
    if data.get("seed") != seed or data.get("batch_seed") != 7500:
        failures.append("seed_metadata")
    if data.get("operator_roles") != selected[SEEDS.index(seed)]:
        failures.append("role_assignment")
    assignment = data.get("role_assignment", {})
    if assignment.get("all_assignment_count") != 24 or assignment.get("computed_mappings") != selected or assignment.get("omitted_mappings") != omitted:
        failures.append("assignment_registry")
    if data.get("model_forward_inputs") != ["token_ids", "lengths"] or data.get("active_set_model_input") is not False or data.get("active_roles_evaluator_only") is not True:
        failures.append("active_set_forward_contract")
    train = data.get("train", [])
    if len(train) != 128 or data.get("multi_clause_train") != 0 or data.get("joint_examples_in_training") != 0 or data.get("test_rows_used") != 0:
        failures.append("training_contract")
    if any(row.get("kind") != "atomic" or "clauses" in row for row in train):
        failures.append("non_atomic_training_row")
    train_counts = {role: sum(row.get("role") == role for row in train) for role in ROLES}
    if train_counts != {role: 32 for role in ROLES}:
        failures.append("training_role_counts")
    permutation = data.get("permutation", [])
    inverse = {int(value): index for index, value in enumerate(permutation)}
    role_operator = data.get("operator_for_role", {})
    if len(permutation) != 32 or len(inverse) != 32:
        failures.append("permutation")
    for row in train:
        argument = str(row.get("argument", ""))
        index = int(argument[4:]) if argument.startswith("ARG_") else -1
        if index < 0 or row.get("value") != permutation[index] or row.get("operator") != role_operator.get(row.get("role")):
            failures.append("training_value_mapping")
            break
    evaluation = data.get("evaluation", [])
    counts_by_size = {size: 0 for size in range(1, 5)}
    subset_counts: dict[str, int] = {}
    assignment_orders: dict[tuple[str, tuple[tuple[str, int], ...]], set[str]] = {}
    assignment_order_keys: set[tuple[tuple[str, tuple[tuple[str, int], ...]], str]] = set()
    instruction_keys: set[tuple[str, ...]] = set()
    semantic_keys: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    train_keys = {(row["role"], int(row["value"])) for row in train}
    atomic_control_keys: set[tuple[str, int]] = set()
    for row in evaluation:
        size = int(row.get("subset_size", 0))
        subset = tuple(row.get("active_roles", []))
        subset_name = row.get("subset")
        if size not in counts_by_size or len(subset) != size or tuple(sorted(subset, key=ROLES.index)) != subset or len(set(subset)) != size:
            failures.append("subset_shape")
            break
        counts_by_size[size] += 1
        subset_counts[subset_name] = subset_counts.get(subset_name, 0) + 1
        if tuple(row.get("tokens", [])) in instruction_keys:
            failures.append("duplicate_instruction")
            break
        instruction_keys.add(tuple(row.get("tokens", [])))
        assignment = row.get("assignment", {})
        semantic = tuple((role, int(assignment[role])) for role in subset)
        semantic_key = (subset_name, semantic)
        assignment_order_key = (semantic_key, row.get("order"))
        if assignment_order_key in assignment_order_keys:
            failures.append("duplicate_semantic_assignment")
            break
        assignment_order_keys.add(assignment_order_key)
        semantic_keys.add(semantic_key)
        assignment_orders.setdefault(semantic_key, set()).add(row.get("order"))
        if size == 1:
            role = subset[0]
            value = int(assignment[role])
            atomic_control_keys.add((role, value))
            if row.get("kind") != "atomic_control" or (role, value) not in train_keys:
                failures.append("atomic_control_contract")
                break
        else:
            if row.get("kind") != "active_set_composition":
                failures.append("composition_kind")
                break
        if len(row.get("clauses", [])) != size or len(row.get("tokens", [])) != 3 * size - 1:
            failures.append("evaluation_shape")
            break
        expected_values = [int(assignment[role]) for role in subset]
        if len(set(expected_values)) != size:
            failures.append("assignment_distinctness")
            break
        if size >= 3 and expected_values[2] != derived_value(seed, expected_values[:2]):
            failures.append("v2_formula")
            break
        if size == 4 and expected_values[3] != derived_value(seed, expected_values[:3]):
            failures.append("v3_formula")
            break
    if counts_by_size != EXPECTED_BY_SIZE:
        failures.append("size_counts")
    if len(evaluation) != EXPECTED_TOTAL:
        failures.append("total_evaluation_rows")
    if sum(1 for row in evaluation if row.get("kind") == "active_set_composition") != EXPECTED_MULTI:
        failures.append("multi_clause_evaluation_count")
    expected_subset_counts = {"FLOOR": 32, "AVOID": 32, "MATCH": 32, "ANCHOR": 32, "FLOOR_AVOID": 1984, "FLOOR_MATCH": 1984, "FLOOR_ANCHOR": 1984, "AVOID_MATCH": 1984, "AVOID_ANCHOR": 1984, "MATCH_ANCHOR": 1984, "FLOOR_AVOID_MATCH": 5952, "FLOOR_AVOID_ANCHOR": 5952, "FLOOR_MATCH_ANCHOR": 5952, "AVOID_MATCH_ANCHOR": 5952, "FLOOR_AVOID_MATCH_ANCHOR": 23808}
    if subset_counts != expected_subset_counts:
        failures.append("subset_counts")
    if len(atomic_control_keys) != 128 or any(len(orders) != 1 for key, orders in assignment_orders.items() if len(key[0].split("_")) == 1):
        failures.append("atomic_control_duplicates")
    expected_orders_by_size = {2: 2, 3: 6, 4: 24}
    if any(len(orders) != expected_orders_by_size[len(key[0].split("_"))] for key, orders in assignment_orders.items() if len(key[0].split("_")) >= 2):
        failures.append("orders_per_assignment")
    token_ids = data.get("token_ids", {})
    if len(token_ids) != 37 or len(set(token_ids.values())) != 37:
        failures.append("physical_ids")
    return {"seed": seed, "path": str(path), "sha256": sha256(path), "status": "passed" if not failures else "failed", "failure_count": len(failures), "failures": failures, "training_rows": len(train), "multi_clause_train": data.get("multi_clause_train"), "evaluation_rows": len(evaluation), "counts_by_size": counts_by_size, "subset_count": len(subset_counts), "unique_instructions": len(instruction_keys), "unique_semantic_assignments": len(semantic_keys), "active_set_model_input": data.get("active_set_model_input"), "model_constructed": False, "training_performed": False, "new_checkpoints": False}


def main() -> int:
    all_assignments = [dict(zip(OPERATORS, permutation)) for permutation in itertools.permutations(ROLES)]
    selected = random.Random(7500).sample(all_assignments, 5)
    omitted = [item for item in all_assignments if item not in selected]
    reports = [check(MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, selected, omitted) for seed in SEEDS]
    current_ids: set[int] = set()
    for report in reports:
        current_ids.update(json.loads(Path(report["path"]).read_text(encoding="utf-8")).get("token_ids", {}).values())
    prior_ids: set[int] = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t6_activeset_midpoint_preparation" in path.parts:
            continue
        try:
            collect_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    failures = [failure for report in reports for failure in report["failures"]]
    if len(current_ids) != 185:
        failures.append("cross_seed_physical_id_overlap")
    if current_ids.intersection(prior_ids):
        failures.append("prior_physical_id_overlap")
    total_evaluation = sum(report["evaluation_rows"] for report in reports)
    if total_evaluation != EXPECTED_TOTAL * len(SEEDS):
        failures.append("total_evaluation_rows")
    result = {"status": "passed" if not failures and all(report["status"] == "passed" for report in reports) else "failed", "failure_count": len(failures), "failures": failures, "selected_mappings": selected, "omitted_mappings": omitted, "manifests": reports, "per_seed_counts": EXPECTED_BY_SIZE, "per_seed_total": EXPECTED_TOTAL, "per_seed_multi_clause": EXPECTED_MULTI, "total_evaluation_rows": total_evaluation, "expected_total_evaluation_rows": EXPECTED_TOTAL * len(SEEDS), "prior_physical_id_overlap": sorted(current_ids.intersection(prior_ids)), "model_constructed": False, "training_performed": False, "new_checkpoints": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
