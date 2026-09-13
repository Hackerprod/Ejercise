"""Independent checker for sealed 7401-7405 manifests; no model/training."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t5_n4_fresh_preparation" / "manifests"
SEEDS = (7401, 7402, 7403, 7404, 7405)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
VALUE_COUNT = 32
EXPECTED_ROWS = 992 * 24
EXPECTED_TOTAL = EXPECTED_ROWS * len(SEEDS)


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def expected_match(seed: int, lower: int, forbidden: int) -> int:
    value = (seed + lower + forbidden + 1) % VALUE_COUNT
    while value in (lower, forbidden):
        value = (value + 1) % VALUE_COUNT
    return value


def expected_anchor(seed: int, lower: int, forbidden: int, match: int) -> int:
    value = (seed + lower + forbidden + match + 2) % VALUE_COUNT
    while value in (lower, forbidden, match):
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
    if data.get("schema") != "T5-n4-fresh-manifest-v2":
        failures.append("schema")
    if data.get("seed") != seed or data.get("batch_seed") != 7400:
        failures.append("seed_metadata")
    if data.get("operator_roles") != selected[SEEDS.index(seed)]:
        failures.append("role_sample")
    assignment = data.get("role_assignment", {})
    if assignment.get("formula") != "random.Random(7400).sample(all_24_role_permutations_of_FAMH, 5)" or assignment.get("all_assignment_count") != 24 or assignment.get("computed_mappings") != selected or assignment.get("omitted_mappings") != omitted:
        failures.append("assignment_registry")
    if len(data.get("train", [])) != 128 or data.get("multi_clause_train") != 0 or data.get("joint_examples_in_training") != 0 or data.get("test_rows_used") != 0:
        failures.append("training_cardinality_or_contamination")
    if any(row.get("kind") != "atomic" or "clauses" in row or len(row) != 6 for row in data.get("train", [])):
        failures.append("non_atomic_training_row")
    if {role: sum(row.get("role") == role for row in data.get("train", [])) for role in ROLES} != {role: 32 for role in ROLES}:
        failures.append("training_role_counts")
    permutation = data.get("permutation", [])
    inverse = {int(value): index for index, value in enumerate(permutation)}
    mapping = data.get("operator_for_role", {})
    for row in data.get("train", []):
        argument_index = int(row.get("argument", "ARG_x")[4:]) if str(row.get("argument", "")).startswith("ARG_") else -1
        if argument_index < 0 or len(permutation) != 32 or row.get("value") != permutation[argument_index] or row.get("operator") != mapping.get(row.get("role")):
            failures.append("training_permutation_or_operator")
            break
    seen: set[tuple[int, int, int, int]] = set()
    order_counts: dict[tuple[int, int, int, int], set[str]] = {}
    expected_orders = {"_".join(order) for order in itertools.permutations(ROLES)}
    for row in data.get("test", []):
        lower, forbidden, match, anchor = (row.get(key) for key in ("L", "F", "E", "H"))
        quad = (lower, forbidden, match, anchor)
        if lower == forbidden or match != expected_match(seed, lower, forbidden) or anchor != expected_anchor(seed, lower, forbidden, match) or len(set(quad)) != 4:
            failures.append("quad_formula_or_distinctness")
            break
        order = row.get("order_name")
        if order not in expected_orders or row.get("order") != order or len(row.get("clauses", [])) != 4 or len(row.get("tokens", [])) != 11:
            failures.append("order_shape")
            break
        values = {"FLOOR": lower, "AVOID": forbidden, "MATCH": match, "ANCHOR": anchor}
        roles = tuple(order.split("_"))
        expected_clauses = [{"argument": arg_name(inverse[values[role]]), "operator": mapping[role], "role": role, "value": values[role]} for role in roles]
        expected_tokens: list[str] = []
        for index, clause in enumerate(expected_clauses):
            if index:
                expected_tokens.append("LINK")
            expected_tokens.extend((clause["operator"], clause["argument"]))
        if row.get("clauses") != expected_clauses or row.get("tokens") != expected_tokens:
            failures.append("clause_token_semantics")
            break
        if row.get("token_ids") != [data["token_ids"][token] for token in expected_tokens]:
            failures.append("physical_token_sequence")
            break
        seen.add(quad)
        order_counts.setdefault(quad, set()).add(order)
    if len(data.get("test", [])) != EXPECTED_ROWS or len(seen) != 992 or len(order_counts) != 992 or any(len(orders) != 24 for orders in order_counts.values()):
        failures.append("test_cardinality_or_order_coverage")
    token_ids = data.get("token_ids", {})
    if len(token_ids) != 37 or len(set(token_ids.values())) != 37:
        failures.append("physical_token_ids")
    return {"seed": seed, "path": str(path), "sha256": sha256(path), "status": "passed" if not failures else "failed", "failure_count": len(failures), "failures": failures, "atomic_training_rows": len(data.get("train", [])), "multi_clause_train": data.get("multi_clause_train"), "quadruples": len(seen), "test_rows": len(data.get("test", [])), "orders_per_quadruple": sorted({len(orders) for orders in order_counts.values()}), "model_constructed": False, "training_performed": False, "new_checkpoints": False}


def main() -> int:
    all_assignments = [dict(zip(OPERATORS, permutation)) for permutation in itertools.permutations(ROLES)]
    selected = random.Random(7400).sample(all_assignments, 5)
    omitted = [item for item in all_assignments if item not in selected]
    reports = [check(MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, selected, omitted) for seed in SEEDS]
    current_ids: set[int] = set()
    for report in reports:
        current_ids.update(json.loads(Path(report["path"]).read_text(encoding="utf-8")).get("token_ids", {}).values())
    prior_ids: set[int] = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t5_n4_fresh_preparation" in path.parts:
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
    total_rows = sum(report["test_rows"] for report in reports)
    if total_rows != EXPECTED_TOTAL:
        failures.append("total_test_rows")
    result = {"status": "passed" if not failures and all(report["status"] == "passed" for report in reports) else "failed", "failure_count": len(failures), "failures": failures, "selected_mappings": selected, "omitted_mappings": omitted, "manifests": reports, "total_atomic_training_rows": 640, "total_test_rows": total_rows, "expected_total_test_rows": EXPECTED_TOTAL, "prior_physical_id_overlap": sorted(current_ids.intersection(prior_ids)), "cross_seed_physical_id_overlap": len(current_ids) != 185, "model_constructed": False, "training_performed": False, "new_checkpoints": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
