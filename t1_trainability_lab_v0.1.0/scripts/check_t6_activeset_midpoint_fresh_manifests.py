"""Independent T6 fresh manifest checker; no model, training, or calibration."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "manifests"
OUTPUT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "checker.json"
SEEDS = (7601, 7602, 7603, 7604, 7605)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
EXPECTED_BY_SIZE = {1: 128, 2: 11904, 3: 23808, 4: 23808}
EXPECTED_TOTAL = 59648
EXPECTED_MULTI = 59520


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    path.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return digest


def derived_value(seed: int, values: list[int]) -> int:
    value = (seed + sum(values) + (1 if len(values) == 2 else 2)) % 32
    while value in values:
        value = (value + 1) % 32
    return value


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


def expected_assignments() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_assignments = [dict(zip(OPERATORS, permutation)) for permutation in itertools.permutations(ROLES)]
    selected = random.Random(7600).sample(all_assignments, 5)
    return selected, [item for item in all_assignments if item not in selected]


def check_manifest(path: Path, seed: int, selected: list[dict[str, str]], omitted: list[dict[str, str]]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if data.get("schema") != "T6-activeset-midpoint-fresh-manifest-v1": failures.append("schema")
    if data.get("seed") != seed or data.get("batch_seed") != 7600: failures.append("seed_metadata")
    if data.get("operator_roles") != selected[SEEDS.index(seed)]: failures.append("role_assignment")
    registry = data.get("role_assignment", {})
    if registry.get("computed_mappings") != selected or registry.get("omitted_mappings") != omitted or registry.get("all_assignment_count") != 24: failures.append("assignment_registry")
    if data.get("model_forward_inputs") != ["token_ids", "lengths"] or data.get("active_set_model_input") is not False or data.get("active_roles_evaluator_only") is not True: failures.append("forward_contract")
    train = data.get("train", [])
    if len(train) != 128 or data.get("multi_clause_train") != 0 or data.get("joint_examples_in_training") != 0 or data.get("test_rows_used") != 0: failures.append("training_contract")
    if any(row.get("kind") != "atomic" or "clauses" in row for row in train): failures.append("non_atomic_training_row")
    if {role: sum(row.get("role") == role for row in train) for role in ROLES} != {role: 32 for role in ROLES}: failures.append("training_role_counts")
    permutation = [int(value) for value in data.get("permutation", [])]
    inverse = {value: index for index, value in enumerate(permutation)}
    if len(permutation) != 32 or len(inverse) != 32: failures.append("permutation")
    role_for_operator = {operator: role for operator, role in data.get("operator_roles", {}).items()}
    train_keys = {(row.get("role"), int(row.get("value"))) for row in train}
    for row in train:
        index = int(str(row.get("argument", "ARG_-1"))[4:])
        if index < 0 or index >= 32 or row.get("value") != permutation[index] or row.get("role") != role_for_operator.get(row.get("operator")): failures.append("training_mapping"); break
    evaluations = data.get("evaluation", [])
    counts_by_size = {size: 0 for size in range(1, 5)}
    subset_counts: dict[str, int] = {}
    instruction_keys: set[tuple[str, ...]] = set()
    assignment_orders: dict[tuple[str, tuple[tuple[str, int], ...]], set[str]] = {}
    semantic_keys: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    atomic_keys: set[tuple[str, int]] = set()
    for row in evaluations:
        tokens = tuple(row.get("tokens", []))
        size = int(row.get("subset_size", 0))
        if size not in counts_by_size or len(tokens) != 3 * size - 1: failures.append("shape"); break
        counts_by_size[size] += 1
        subset = tuple(row.get("active_roles", []))
        subset_name = str(row.get("subset"))
        subset_counts[subset_name] = subset_counts.get(subset_name, 0) + 1
        if tokens in instruction_keys: failures.append("duplicate_instruction"); break
        instruction_keys.add(tokens)
        if len(subset) != size or tuple(role for role in ROLES if role in subset) != subset or len(set(subset)) != size: failures.append("subset_shape"); break
        parsed: list[tuple[str, str, int]] = []
        for index in range(size):
            operator = tokens[3 * index]
            argument = tokens[3 * index + 1]
            if index and tokens[3 * index - 1] != "LINK": failures.append("link_layout"); break
            role = role_for_operator.get(operator)
            arg_index = int(argument[4:]) if argument.startswith("ARG_") else -1
            if role is None or arg_index not in range(32): failures.append("token_semantics"); break
            parsed.append((role, argument, permutation[arg_index]))
        if failures and failures[-1] in {"link_layout", "token_semantics"}: break
        parsed_roles = tuple(item[0] for item in parsed)
        if parsed_roles != tuple(row.get("order", "").split("_")) or set(parsed_roles) != set(subset): failures.append("order_semantics"); break
        parsed_assignment = {role: value for role, _, value in parsed}
        if parsed_assignment != {role: int(value) for role, value in row.get("assignment", {}).items()}: failures.append("assignment_from_tokens"); break
        semantic = tuple((role, parsed_assignment[role]) for role in subset)
        semantic_key = (subset_name, semantic)
        semantic_keys.add(semantic_key)
        if row.get("kind") == "atomic_control":
            if size != 1 or (subset[0], parsed_assignment[subset[0]]) not in train_keys: failures.append("atomic_control"); break
            atomic_keys.add((subset[0], parsed_assignment[subset[0]]))
        elif row.get("kind") != "active_set_composition": failures.append("kind"); break
        if len(set(parsed_assignment.values())) != size: failures.append("assignment_distinctness"); break
        values = [parsed_assignment[role] for role in subset]
        if size >= 3 and values[2] != derived_value(seed, values[:2]): failures.append("v2_formula"); break
        if size == 4 and values[3] != derived_value(seed, values[:3]): failures.append("v3_formula"); break
        assignment_orders.setdefault(semantic_key, set()).add(str(row.get("order")))
    expected_subset_counts = {"FLOOR": 32, "AVOID": 32, "MATCH": 32, "ANCHOR": 32, "FLOOR_AVOID": 1984, "FLOOR_MATCH": 1984, "FLOOR_ANCHOR": 1984, "AVOID_MATCH": 1984, "AVOID_ANCHOR": 1984, "MATCH_ANCHOR": 1984, "FLOOR_AVOID_MATCH": 5952, "FLOOR_AVOID_ANCHOR": 5952, "FLOOR_MATCH_ANCHOR": 5952, "AVOID_MATCH_ANCHOR": 5952, "FLOOR_AVOID_MATCH_ANCHOR": 23808}
    if counts_by_size != EXPECTED_BY_SIZE: failures.append("size_counts")
    if len(evaluations) != EXPECTED_TOTAL: failures.append("total_rows")
    if sum(row.get("kind") == "active_set_composition" for row in evaluations) != EXPECTED_MULTI: failures.append("multi_clause_count")
    if subset_counts != expected_subset_counts: failures.append("subset_counts")
    if len(atomic_keys) != 128: failures.append("atomic_controls")
    expected_orders = {1: 1, 2: 2, 3: 6, 4: 24}
    if any(len(orders) != expected_orders[len(key[0].split("_"))] for key, orders in assignment_orders.items()): failures.append("order_counts")
    if len(data.get("token_ids", {})) != 37 or len(set(data.get("token_ids", {}).values())) != 37: failures.append("physical_ids")
    return {"seed": seed, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "status": "passed" if not failures else "failed", "failures": failures, "training_rows": len(train), "counts_by_size": counts_by_size, "evaluation_rows": len(evaluations), "multi_clause_evaluation_rows": sum(row.get("kind") == "active_set_composition" for row in evaluations), "subset_count": len(subset_counts), "semantic_assignment_count": len(semantic_keys), "unique_instructions": len(instruction_keys), "atomic_controls": len(atomic_keys), "active_set_model_input": data.get("active_set_model_input"), "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False}


def main() -> int:
    selected, omitted = expected_assignments()
    reports = [check_manifest(MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, selected, omitted) for seed in SEEDS]
    current_ids: set[int] = set()
    for report in reports:
        current_ids.update(json.loads((ROOT / report["path"]).read_text(encoding="utf-8")).get("token_ids", {}).values())
    prior_ids: set[int] = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t6_activeset_midpoint_fresh_preparation" in path.parts: continue
        try: collect_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError): pass
    failures = [failure for report in reports for failure in report["failures"]]
    if len(current_ids) != 185: failures.append("cross_seed_physical_id_overlap")
    if current_ids.intersection(prior_ids): failures.append("prior_physical_id_overlap")
    result = {"status": "passed" if not failures and all(report["status"] == "passed" for report in reports) else "failed", "task": "T6-ACTIVESET-MIDPOINT-FRESH-PREPARATION", "selected_mappings": selected, "omitted_mappings": omitted, "manifests": reports, "per_seed_counts": EXPECTED_BY_SIZE, "per_seed_total": EXPECTED_TOTAL, "per_seed_multi_clause": EXPECTED_MULTI, "total_evaluation_rows": sum(report["evaluation_rows"] for report in reports), "expected_total_evaluation_rows": EXPECTED_TOTAL * len(SEEDS), "failures": failures, "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    digest = write_self_hashed(OUTPUT, result)
    print(json.dumps({"status": result["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "training_performed": False, "calibration_performed": False, "new_checkpoints": False, "counts": EXPECTED_BY_SIZE, "total": result["total_evaluation_rows"]}, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
