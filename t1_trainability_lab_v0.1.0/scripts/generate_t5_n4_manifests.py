"""Generate sealed T5 N=4 development manifests; no model/training code."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "t5_n4_preparation" / "manifests"
SEEDS = (7301, 7302, 7303, 7304, 7305)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
ORDERS = tuple(itertools.permutations(ROLES))
VALUE_COUNT = 32
BATCH_SEED = 7300
ID_BLOCKS = {7301: (27000, 27499), 7302: (28000, 28499), 7303: (29000, 29499), 7304: (30000, 30499), 7305: (31000, 31499)}
ROLE_ASSIGNMENT_FORMULA = "all_24_role_permutations_of_FAMH = [dict(zip(['OP_W', 'OP_X', 'OP_Y', 'OP_Z'], role_perm)) for role_perm in itertools.permutations(['FLOOR', 'AVOID', 'MATCH', 'ANCHOR'])]; selected = random.Random(7300).sample(all_24_role_permutations_of_FAMH, 5); mapping = selected[seed - 7301]"
MATCH_FORMULA = "candidate = (seed + L + F + 1) % 32; while candidate in (L, F): candidate = (candidate + 1) % 32; E = candidate"
ANCHOR_FORMULA = "candidate = (seed + L + F + E + 2) % 32; while candidate in (L, F, E): candidate = (candidate + 1) % 32; H = candidate"


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def assignments() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    all_assignments = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    selected = random.Random(BATCH_SEED).sample(all_assignments, 5)
    omitted = [mapping for mapping in all_assignments if mapping not in selected]
    return selected, omitted, {"all_count": len(all_assignments)}


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


def build_manifest(seed: int, mapping: dict[str, str], selected: list[dict[str, str]], omitted: list[dict[str, str]], all_count: int) -> dict[str, Any]:
    permutation = random.Random(seed).sample(range(VALUE_COUNT), VALUE_COUNT)
    inverse = {value: index for index, value in enumerate(permutation)}
    block_start, block_end = ID_BLOCKS[seed]
    physical_ids = random.Random(seed + 100000).sample(range(block_start, block_end + 1), VALUE_COUNT + len(OPERATORS) + 1)
    token_names = [*(arg_name(index) for index in range(VALUE_COUNT)), *OPERATORS, "LINK"]
    token_ids = dict(zip(token_names, physical_ids))
    operator_for_role = {role: operator for operator, role in mapping.items()}
    train = [{"argument": arg_name(index), "constraints": role.lower(), "kind": "atomic", "operator": operator_for_role[role], "role": role, "value": permutation[index]} for role in ROLES for index in range(VALUE_COUNT)]
    test: list[dict[str, Any]] = []
    order_index = {order: index for index, order in enumerate(ORDERS)}
    for lower in range(VALUE_COUNT):
        for forbidden in range(VALUE_COUNT):
            if lower == forbidden:
                continue
            match = match_value(seed, lower, forbidden)
            anchor = anchor_value(seed, lower, forbidden, match)
            values = {"FLOOR": lower, "AVOID": forbidden, "MATCH": match, "ANCHOR": anchor}
            for order in ORDERS:
                clauses = [{"argument": arg_name(inverse[values[role]]), "operator": operator_for_role[role], "role": role, "value": values[role]} for role in order]
                tokens: list[str] = []
                for index, clause in enumerate(clauses):
                    if index:
                        tokens.append("LINK")
                    tokens.extend((clause["operator"], clause["argument"]))
                test.append({"E": match, "F": forbidden, "H": anchor, "L": lower, "clauses": clauses, "kind": "four_active_roles", "match": match, "match_arg": arg_name(inverse[match]), "anchor": anchor, "anchor_arg": arg_name(inverse[anchor]), "order": "_".join(order), "order_index": order_index[order], "order_name": "_".join(order), "forbidden": forbidden, "lower": lower, "tokens": tokens, "token_ids": [token_ids[token] for token in tokens]})
    return {"argument_count": VALUE_COUNT, "anchor_formula": ANCHOR_FORMULA, "batch_seed": BATCH_SEED, "fresh_init_required": True, "id_formula": "ids = random.Random(seed + 100000).sample(range(block_start, block_end + 1), 37)", "ids_block": [block_start, block_end], "joint_train_examples": 0, "match_formula": MATCH_FORMULA, "multi_clause_train": 0, "operator_for_role": operator_for_role, "operator_names": list(OPERATORS), "operator_roles": mapping, "permutation": permutation, "role_assignment": {"all_assignment_count": all_count, "batch_seed": BATCH_SEED, "computed_mappings": selected, "omitted_mappings": omitted, "formula": ROLE_ASSIGNMENT_FORMULA, "operators": list(OPERATORS), "roles": list(ROLES), "selected_index": seed - SEEDS[0]}, "role_names": list(ROLES), "schema": "T5-n4-fresh-manifest-v1", "task": "T5-N4-PREPARATION", "test": test, "test_case_count": len(test), "test_rows_used": 0, "token_ids": token_ids, "token_order": token_names, "train": train, "training": {"atomic_rows": {role: VALUE_COUNT for role in ROLES}, "total_rows": len(train), "multi_clause_train": 0, "joint_train_examples": 0, "test_rows_used": 0}, "value_count": VALUE_COUNT, "version": 1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    selected, omitted, metadata = assignments()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed, mapping in zip(SEEDS, selected):
        path = OUTPUT_ROOT / f"manifest_{seed}_v1.json"
        path.write_bytes((json.dumps(build_manifest(seed, mapping, selected, omitted, metadata["all_count"]), indent=2, sort_keys=True) + "\n").encode("utf-8"))
        records.append({"seed": seed, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    print(json.dumps({"status": "generated", "training_performed": False, "batch_seed": BATCH_SEED, "selected_mappings": selected, "omitted_mappings": omitted, "manifests": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
