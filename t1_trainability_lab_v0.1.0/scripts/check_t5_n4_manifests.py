"""Independent checker for sealed T5 N=4 manifests; never builds or trains a model."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t5_n4_preparation" / "manifests"
SEEDS = (7301, 7302, 7303, 7304, 7305)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
VALUE_COUNT = 32
EXPECTED_TEST_ROWS = 992 * 24
EXPECTED_TOTAL_TEST_ROWS = EXPECTED_TEST_ROWS * len(SEEDS)
EXPECTED_BLOCKS = {7301: (27000, 27499), 7302: (28000, 28499), 7303: (29000, 29499), 7304: (30000, 30499), 7305: (31000, 31499)}


def match_value(seed: int, lower: int, forbidden: int) -> int:
    candidate = (seed + lower + forbidden + 1) % VALUE_COUNT
    while candidate in (lower, forbidden):
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def anchor_value(seed: int, lower: int, forbidden: int, match: int) -> int:
    candidate = (seed + lower + forbidden + match + 2) % VALUE_COUNT
    while candidate in (lower, forbidden, match):
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


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


def check_manifest(path: Path, expected_seed: int, all_assignments: list[dict[str, str]], selected: list[dict[str, str]]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if data.get("schema") != "T5-n4-fresh-manifest-v1":
        failures.append("schema")
    if data.get("seed") is not None:
        failures.append("unexpected_seed_field")
    if data.get("multi_clause_train") != 0 or data.get("joint_train_examples") != 0 or data.get("test_rows_used") != 0:
        failures.append("train_test_contamination_metadata")
    if data.get("training", {}).get("total_rows") != 128 or data.get("training", {}).get("multi_clause_train") != 0:
        failures.append("training_counts")
    if len(data.get("train", [])) != 128:
        failures.append("training_rows")
    if len(data.get("test", [])) != EXPECTED_TEST_ROWS:
        failures.append("test_rows")
    mapping = data.get("operator_roles", {})
    expected_mapping = selected[expected_seed - SEEDS[0]]
    if mapping != expected_mapping:
        failures.append("selected_role_assignment")
    assignment = data.get("role_assignment", {})
    if assignment.get("all_assignment_count") != 24 or assignment.get("computed_mappings") != selected or assignment.get("omitted_mappings") != [item for item in all_assignments if item not in selected]:
        failures.append("assignment_registry")
    if set(mapping) != set(OPERATORS) or set(mapping.values()) != set(ROLES):
        failures.append("role_operator_bijection")
    token_ids = data.get("token_ids", {})
    if len(token_ids) != 37 or len(set(token_ids.values())) != 37:
        failures.append("token_id_cardinality")
    block_start, block_end = EXPECTED_BLOCKS[expected_seed]
    if any(not isinstance(value, int) or not block_start <= value <= block_end for value in token_ids.values()):
        failures.append("fresh_id_block")
    if len({row.get("argument") for row in data.get("train", [])}) != 32:
        failures.append("train_argument_coverage")
    if any(row.get("kind") != "atomic" or "clauses" in row for row in data.get("train", [])):
        failures.append("non_atomic_training_row")
    seen_quads: set[tuple[int, int, int, int]] = set()
    orders: dict[tuple[int, int, int, int], set[str]] = {}
    for row in data.get("test", []):
        lower, forbidden, match, anchor = row.get("L"), row.get("F"), row.get("E"), row.get("H")
        quad = (lower, forbidden, match, anchor)
        if len({lower, forbidden, match, anchor}) != 4:
            failures.append("non_distinct_quad")
            break
        if lower == forbidden or match != match_value(expected_seed, lower, forbidden) or anchor != anchor_value(expected_seed, lower, forbidden, match):
            failures.append("quad_formula")
            break
        seen_quads.add(quad)
        orders.setdefault(quad, set()).add(row.get("order", ""))
        if len(row.get("clauses", [])) != 4 or len(row.get("tokens", [])) != 11:
            failures.append("test_shape")
            break
    if len(seen_quads) != 992 or len(orders) != 992:
        failures.append("quad_count")
    if any(len(order_set) != 24 for order_set in orders.values()):
        failures.append("order_coverage")
    return {"seed": expected_seed, "path": str(path), "sha256": sha256(path), "status": "passed" if not failures else "failed", "failure_count": len(failures), "failures": failures, "atomic_training_rows": len(data.get("train", [])), "multi_clause_train": data.get("multi_clause_train"), "quadruples": len(seen_quads), "test_rows": len(data.get("test", [])), "orders_per_quadruple": sorted({len(order_set) for order_set in orders.values()})}


def main() -> int:
    all_assignments = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    selected = __import__("random").Random(7300).sample(all_assignments, 5)
    reports = [check_manifest(MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, all_assignments, selected) for seed in SEEDS]
    current_ids: set[int] = set()
    for report in reports:
        current_ids.update(json.loads(Path(report["path"]).read_text(encoding="utf-8")).get("token_ids", {}).values())
    prior_ids: set[int] = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t5_n4_preparation" in path.parts:
            continue
        try:
            collect_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    prior_overlap = sorted(current_ids & prior_ids)
    cross_seed_overlap = len(current_ids) != sum(37 for _ in SEEDS)
    total_rows = sum(report["test_rows"] for report in reports)
    failures = [failure for report in reports for failure in report["failures"]]
    if prior_overlap:
        failures.append("prior_physical_id_overlap")
    if cross_seed_overlap:
        failures.append("cross_seed_physical_id_overlap")
    if total_rows != EXPECTED_TOTAL_TEST_ROWS:
        failures.append("total_test_rows")
    result = {"status": "passed" if not failures and all(report["status"] == "passed" for report in reports) else "failed", "failure_count": len(failures), "failures": failures, "selected_mappings": selected, "omitted_mappings": [item for item in all_assignments if item not in selected], "manifests": reports, "total_atomic_training_rows": 640, "total_test_rows": total_rows, "expected_total_test_rows": EXPECTED_TOTAL_TEST_ROWS, "prior_physical_id_overlap": prior_overlap, "cross_seed_physical_id_overlap": cross_seed_overlap, "training_performed": False, "new_checkpoints": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
