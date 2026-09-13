"""Generate T4 development manifests; never trains or loads model artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "manifests"
SEEDS = (7101, 7102, 7103, 7104, 7105)
OPERATORS = ("OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH")
ORDERS = tuple(itertools.permutations(ROLES))
VALUE_COUNT = 32
ID_BLOCKS = {
    7101: (17000, 17499),
    7102: (18000, 18499),
    7103: (19000, 19499),
    7104: (20000, 20499),
    7105: (21000, 21499),
}

ROLE_ASSIGNMENT_FORMULA = (
    "all_six_assignments = [dict(zip(['OP_X', 'OP_Y', 'OP_Z'], role_perm)) "
    "for role_perm in itertools.permutations(['FLOOR', 'AVOID', 'MATCH'])]; "
    "selected = random.Random(7100).sample(all_six_assignments, 5); "
    "mapping = selected[seed - 7101]"
)
PERMUTATION_FORMULA = "permutation = random.Random(seed).sample(range(32), 32)"
ID_FORMULA = "ids = random.Random(seed + 100000).sample(range(block_start, block_end + 1), 36)"
MATCH_FORMULA = (
    "candidate = (seed + lower + forbidden + 1) % 32; "
    "while candidate in (lower, forbidden): candidate = (candidate + 1) % 32; "
    "E = candidate"
)


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def assignments() -> list[dict[str, str]]:
    all_six = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    return random.Random(7100).sample(all_six, 5)


def match_value(seed: int, lower: int, forbidden: int) -> int:
    candidate = (seed + lower + forbidden + 1) % VALUE_COUNT
    while candidate in (lower, forbidden):
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def build_manifest(seed: int, mapping: dict[str, str], selected: list[dict[str, str]]) -> dict[str, Any]:
    permutation = random.Random(seed).sample(range(VALUE_COUNT), VALUE_COUNT)
    inverse = {value: index for index, value in enumerate(permutation)}
    block_start, block_end = ID_BLOCKS[seed]
    physical_ids = random.Random(seed + 100000).sample(range(block_start, block_end + 1), 36)
    token_names = [*(arg_name(index) for index in range(VALUE_COUNT)), *OPERATORS, "LINK"]
    token_ids = dict(zip(token_names, physical_ids))
    operator_for_role = {role: operator for operator, role in mapping.items()}

    train = [
        {
            "argument": arg_name(index),
            "constraints": role.lower(),
            "kind": "atomic",
            "operator": operator_for_role[role],
            "role": role,
            "value": permutation[index],
        }
        for role in ROLES
        for index in range(VALUE_COUNT)
    ]

    test: list[dict[str, Any]] = []
    order_index = {order: index for index, order in enumerate(ORDERS)}
    for lower in range(VALUE_COUNT):
        for forbidden in range(VALUE_COUNT):
            if lower == forbidden:
                continue
            match = match_value(seed, lower, forbidden)
            values = {"FLOOR": lower, "AVOID": forbidden, "MATCH": match}
            for order in ORDERS:
                clauses = [
                    {
                        "argument": arg_name(inverse[values[role]]),
                        "operator": operator_for_role[role],
                        "role": role,
                        "value": values[role],
                    }
                    for role in order
                ]
                tokens: list[str] = []
                for index, clause in enumerate(clauses):
                    if index:
                        tokens.append("LINK")
                    tokens.extend((clause["operator"], clause["argument"]))
                test.append(
                    {
                        "E": match,
                        "F": forbidden,
                        "L": lower,
                        "clauses": clauses,
                        "kind": "three_active_roles",
                        "match": match,
                        "match_arg": arg_name(inverse[match]),
                        "order": "_".join(order),
                        "order_index": order_index[order],
                        "order_name": "_".join(order),
                        "forbidden": forbidden,
                        "lower": lower,
                        "tokens": tokens,
                        "token_ids": [token_ids[token] for token in tokens],
                    }
                )

    return {
        "argument_count": VALUE_COUNT,
        "fresh_init_required": True,
        "id_formula": ID_FORMULA,
        "ids_block": [block_start, block_end],
        "joint_train_examples": 0,
        "match_formula": MATCH_FORMULA,
        "multi_clause_train": 0,
        "operator_for_role": operator_for_role,
        "operator_names": list(OPERATORS),
        "operator_roles": mapping,
        "permutation": permutation,
        "role_assignment": {
            "all_assignment_count": 6,
            "computed_mappings": selected,
            "formula": ROLE_ASSIGNMENT_FORMULA,
            "operators": list(OPERATORS),
            "roles": list(ROLES),
            "selected_index": seed - SEEDS[0],
            "selection_seed": 7100,
        },
        "role_names": list(ROLES),
        "schema": "T4-nobypass1-three-active-roles-manifest-v1",
        "task": "T4-NOBYPASS-1-THREE-ACTIVE-ROLES",
        "test": test,
        "test_case_count": len(test),
        "token_ids": token_ids,
        "token_order": token_names,
        "train": train,
        "training": {
            "atomic_rows": {role: VALUE_COUNT for role in ROLES},
            "total_rows": len(train),
            "multi_clause_train": 0,
        },
        "value_count": VALUE_COUNT,
        "version": 1,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    selected = assignments()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed, mapping in zip(SEEDS, selected):
        path = OUTPUT_ROOT / f"manifest_{seed}_v1.json"
        payload = json.dumps(build_manifest(seed, mapping, selected), indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")
        records.append({"seed": seed, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    print(json.dumps({"status": "generated", "manifests": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
