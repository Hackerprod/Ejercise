"""Generate sealed T5 N=4 fresh manifests; preparation only."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "t5_n4_fresh_preparation" / "manifests"
SEEDS = (7401, 7402, 7403, 7404, 7405)
OPERATORS = ("OP_W", "OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
VALUE_COUNT = 32
BATCH_SEED = 7400
ID_BLOCKS = {7401: (32000, 32499), 7402: (33000, 33499), 7403: (34000, 34499), 7404: (35000, 35499), 7405: (36000, 36499)}
MATCH_FORMULA = "E=(seed+L+F+1)%32; increment by 1 modulo 32 until E not in {L,F}"
ANCHOR_FORMULA = "H=(seed+L+F+E+2)%32; increment by 1 modulo 32 until H not in {L,F,E}"
ASSIGNMENT_FORMULA = "random.Random(7400).sample(all_24_role_permutations_of_FAMH, 5)"


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def make_assignments() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_assignments = [dict(zip(OPERATORS, permutation)) for permutation in itertools.permutations(ROLES)]
    selected = random.Random(BATCH_SEED).sample(all_assignments, 5)
    return selected, [item for item in all_assignments if item not in selected]


def match_value(seed: int, lower: int, forbidden: int) -> int:
    value = (seed + lower + forbidden + 1) % VALUE_COUNT
    while value in (lower, forbidden):
        value = (value + 1) % VALUE_COUNT
    return value


def anchor_value(seed: int, lower: int, forbidden: int, match: int) -> int:
    value = (seed + lower + forbidden + match + 2) % VALUE_COUNT
    while value in (lower, forbidden, match):
        value = (value + 1) % VALUE_COUNT
    return value


def make_manifest(seed: int, mapping: dict[str, str], selected: list[dict[str, str]], omitted: list[dict[str, str]]) -> dict[str, Any]:
    permutation = random.Random(seed).sample(range(VALUE_COUNT), VALUE_COUNT)
    inverse = {value: index for index, value in enumerate(permutation)}
    start, end = ID_BLOCKS[seed]
    ids = random.Random(seed + 100000).sample(range(start, end + 1), 37)
    token_order = [*(arg_name(index) for index in range(VALUE_COUNT)), *OPERATORS, "LINK"]
    token_ids = dict(zip(token_order, ids))
    operator_for_role = {role: operator for operator, role in mapping.items()}
    train = [{"argument": arg_name(index), "constraints": role.lower(), "kind": "atomic", "operator": operator_for_role[role], "role": role, "value": permutation[index]} for role in ROLES for index in range(VALUE_COUNT)]
    orders = tuple(itertools.permutations(ROLES))
    test: list[dict[str, Any]] = []
    for lower in range(VALUE_COUNT):
        for forbidden in range(VALUE_COUNT):
            if lower == forbidden:
                continue
            match = match_value(seed, lower, forbidden)
            anchor = anchor_value(seed, lower, forbidden, match)
            values = {"FLOOR": lower, "AVOID": forbidden, "MATCH": match, "ANCHOR": anchor}
            for order_index, order in enumerate(orders):
                clauses = [{"argument": arg_name(inverse[values[role]]), "operator": operator_for_role[role], "role": role, "value": values[role]} for role in order]
                tokens: list[str] = []
                for index, clause in enumerate(clauses):
                    if index:
                        tokens.append("LINK")
                    tokens.extend((clause["operator"], clause["argument"]))
                test.append({"L": lower, "F": forbidden, "E": match, "H": anchor, "lower": lower, "forbidden": forbidden, "match": match, "anchor": anchor, "kind": "four_active_roles", "order": "_".join(order), "order_name": "_".join(order), "order_index": order_index, "clauses": clauses, "tokens": tokens, "token_ids": [token_ids[token] for token in tokens]})
    return {"schema": "T5-n4-fresh-manifest-v2", "task": "T5-N4-FRESH-PREPARATION", "version": 1, "seed": seed, "batch_seed": BATCH_SEED, "argument_count": VALUE_COUNT, "value_count": VALUE_COUNT, "role_names": list(ROLES), "operator_names": list(OPERATORS), "operator_roles": mapping, "operator_for_role": operator_for_role, "permutation": permutation, "token_order": token_order, "token_ids": token_ids, "id_block": [start, end], "fresh_init_required": True, "match_formula": MATCH_FORMULA, "anchor_formula": ANCHOR_FORMULA, "role_assignment": {"formula": ASSIGNMENT_FORMULA, "batch_seed": BATCH_SEED, "all_assignment_count": 24, "computed_mappings": selected, "omitted_mappings": omitted, "selected_index": SEEDS.index(seed)}, "train": train, "test": test, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "training": {"atomic_rows": {role: VALUE_COUNT for role in ROLES}, "total_rows": len(train), "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0}, "test_case_count": len(test), "training_performed": False, "new_checkpoints": False, "model_constructed": False}


def main() -> int:
    selected, omitted = make_assignments()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed, mapping in zip(SEEDS, selected):
        path = OUTPUT_ROOT / f"manifest_{seed}_v1.json"
        path.write_bytes((json.dumps(make_manifest(seed, mapping, selected, omitted), indent=2, sort_keys=True) + "\n").encode("utf-8"))
        records.append({"seed": seed, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    print(json.dumps({"status": "generated", "training_performed": False, "new_checkpoints": False, "selected_mappings": selected, "omitted_mappings": omitted, "manifests": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
