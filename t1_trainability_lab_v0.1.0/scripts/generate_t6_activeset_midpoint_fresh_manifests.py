"""Generate T6 fresh-preparation manifests only; never constructs or loads models."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "manifests"
SEEDS = (7601, 7602, 7603, 7604, 7605)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
VALUE_COUNT = 32
BATCH_SEED = 7600
ID_BLOCKS = {7601: (42000, 42499), 7602: (43000, 43499), 7603: (44000, 44499), 7604: (45000, 45499), 7605: (46000, 46499)}
MATCH_FORMULA = "v2=(seed+v0+v1+1)%32; increment by 1 modulo 32 until v2 not in {v0,v1}"
ANCHOR_FORMULA = "v3=(seed+v0+v1+v2+2)%32; increment by 1 modulo 32 until v3 not in {v0,v1,v2}"


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def make_assignments() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_assignments = [dict(zip(OPERATORS, permutation)) for permutation in itertools.permutations(ROLES)]
    selected = random.Random(BATCH_SEED).sample(all_assignments, 5)
    return selected, [item for item in all_assignments if item not in selected]


def derived_value(seed: int, values: list[int]) -> int:
    offset = 1 if len(values) == 2 else 2
    value = (seed + sum(values) + offset) % VALUE_COUNT
    while value in values:
        value = (value + 1) % VALUE_COUNT
    return value


def make_manifest(seed: int, operator_roles: dict[str, str], selected: list[dict[str, str]], omitted: list[dict[str, str]]) -> dict[str, Any]:
    permutation = random.Random(seed).sample(range(VALUE_COUNT), VALUE_COUNT)
    inverse = {value: index for index, value in enumerate(permutation)}
    start, end = ID_BLOCKS[seed]
    physical_ids = random.Random(seed + 100000).sample(range(start, end + 1), 37)
    token_order = [*(arg_name(index) for index in range(VALUE_COUNT)), *OPERATORS, "LINK"]
    token_ids = dict(zip(token_order, physical_ids))
    operator_for_role = {role: operator for operator, role in operator_roles.items()}
    train = [{"argument": arg_name(index), "constraints": role.lower(), "kind": "atomic", "operator": operator_for_role[role], "role": role, "value": permutation[index]} for role in ROLES for index in range(VALUE_COUNT)]
    evaluation: list[dict[str, Any]] = []
    subset_registry: list[dict[str, Any]] = []
    for size in range(1, len(ROLES) + 1):
        for subset in itertools.combinations(ROLES, size):
            subset_name = "_".join(subset)
            if size == 1:
                assignments = [(permutation[index],) for index in range(VALUE_COUNT)]
                orders = (subset,)
                kind = "atomic_control"
            else:
                assignments = []
                for value0 in range(VALUE_COUNT):
                    for value1 in range(VALUE_COUNT):
                        if value0 == value1:
                            continue
                        values = [value0, value1]
                        if size >= 3:
                            values.append(derived_value(seed, values))
                        if size == 4:
                            values.append(derived_value(seed, values))
                        assignments.append(tuple(values))
                orders = tuple(itertools.permutations(subset))
                kind = "active_set_composition"
            for assignment in assignments:
                role_values = dict(zip(subset, assignment))
                for order_index, order in enumerate(orders):
                    clauses = [{"argument": arg_name(inverse[role_values[role]]), "operator": operator_for_role[role], "role": role, "value": role_values[role]} for role in order]
                    tokens: list[str] = []
                    for clause_index, clause in enumerate(clauses):
                        if clause_index:
                            tokens.append("LINK")
                        tokens.extend((clause["operator"], clause["argument"]))
                    evaluation.append({"active_roles": list(subset), "assignment": {role: role_values[role] for role in subset}, "clauses": clauses, "kind": kind, "order": "_".join(order), "order_index": order_index, "subset": subset_name, "subset_size": size, "tokens": tokens, "token_ids": [token_ids[token] for token in tokens]})
            subset_registry.append({"subset": subset_name, "roles": list(subset), "size": size, "expected_rows": len(assignments) * len(orders), "orders": len(orders), "kind": kind})
    return {"schema": "T6-activeset-midpoint-fresh-manifest-v1", "task": "T6-ACTIVESET-MIDPOINT-FRESH-PREPARATION", "version": 1, "seed": seed, "batch_seed": BATCH_SEED, "argument_count": VALUE_COUNT, "value_count": VALUE_COUNT, "role_names": list(ROLES), "operator_names": list(OPERATORS), "operator_roles": operator_roles, "operator_for_role": operator_for_role, "permutation": permutation, "token_order": token_order, "token_ids": token_ids, "id_block": [start, end], "id_semantics": "physical IDs fresh and disjoint; model uses manifest-local vocabulary indices", "fresh_init_required": True, "match_formula": MATCH_FORMULA, "anchor_formula": ANCHOR_FORMULA, "role_assignment": {"formula": "random.Random(7600).sample(all_24_role_permutations_of_FAMH, 5)", "batch_seed": BATCH_SEED, "all_assignment_count": 24, "computed_mappings": selected, "omitted_mappings": omitted, "selected_index": SEEDS.index(seed)}, "subset_registry": subset_registry, "train": train, "evaluation": evaluation, "active_roles_evaluator_only": True, "model_forward_inputs": ["token_ids", "lengths"], "active_set_model_input": False, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "training": {"atomic_rows": {role: VALUE_COUNT for role in ROLES}, "total_rows": len(train), "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0}, "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False}


def main() -> int:
    selected, omitted = make_assignments()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed, mapping in zip(SEEDS, selected):
        path = OUTPUT_ROOT / f"manifest_{seed}_v1.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite sealed manifest {path}")
        path.write_bytes((json.dumps(make_manifest(seed, mapping, selected, omitted), indent=2, sort_keys=True) + "\n").encode("utf-8"))
        records.append({"seed": seed, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    print(json.dumps({"status": "generated", "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False, "selected_mappings": selected, "omitted_mappings": omitted, "manifests": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
