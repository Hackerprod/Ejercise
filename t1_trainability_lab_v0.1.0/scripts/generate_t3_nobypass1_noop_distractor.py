"""Generate the T3 no-bypass NOOP-distractor manifest battery."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "campaign" / "nb5_manifests"
SEEDS = (7001, 7002, 7003, 7004, 7005)
OPERATORS = ("OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "NOOP")
ORDERS = tuple(itertools.permutations(ROLES))
ID_BLOCKS = {
    7001: (12000, 12499),
    7002: (13000, 13499),
    7003: (14000, 14499),
    7004: (15000, 15499),
    7005: (16000, 16499),
}
ROLE_ASSIGNMENT_FORMULA = (
    "all_six_assignments = [dict(zip(['OP_X', 'OP_Y', 'OP_Z'], role_perm)) "
    "for role_perm in itertools.permutations(['FLOOR', 'AVOID', 'NOOP'])]; "
    "r = random.Random(7000); selected = r.sample(all_six_assignments, 5); "
    "mapping = selected[seed - 7001]"
)
DISTRACTOR_FORMULA = (
    "candidate=(seed + lower + forbidden) % 32; while candidate in "
    "(lower, forbidden): candidate=(candidate+1)%32; D=candidate"
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_role_mappings() -> tuple[dict[str, str], ...]:
    all_six = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    selected = random.Random(7000).sample(all_six, 5)
    return tuple(selected)


SELECTED_ROLE_MAPPINGS = selected_role_mappings()


def distractor(seed: int, lower: int, forbidden: int) -> int:
    candidate = (seed + lower + forbidden) % 32
    while candidate in (lower, forbidden):
        candidate = (candidate + 1) % 32
    return candidate


def build(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unsupported T3 seed: {seed}")

    # Per-seed RNG controls only the value permutation and physical IDs.
    rng = random.Random(seed)
    permutation = list(range(32))
    rng.shuffle(permutation)
    lower_id, upper_id = ID_BLOCKS[seed]
    physical_ids = rng.sample(range(lower_id, upper_id + 1), 36)
    token_names = [*(f"ARG_{index:02d}" for index in range(32)), *OPERATORS, "LINK"]
    token_ids = dict(zip(token_names, physical_ids))
    operator_roles = dict(SELECTED_ROLE_MAPPINGS[seed - SEEDS[0]])
    role_operators = {role: operator for operator, role in operator_roles.items()}
    inverse = {value: index for index, value in enumerate(permutation)}

    train: list[dict[str, Any]] = []
    constraint_names = {"FLOOR": "floor", "AVOID": "avoid", "NOOP": "none"}
    for role in ROLES:
        for index, value in enumerate(permutation):
            train.append(
                {
                    "kind": "atomic",
                    "role": role,
                    "operator": role_operators[role],
                    "argument": f"ARG_{index:02d}",
                    "value": value,
                    "constraints": constraint_names[role],
                }
            )

    test: list[dict[str, Any]] = []
    for lower in range(32):
        for forbidden in range(32):
            if lower == forbidden:
                continue
            distractor_value = distractor(seed, lower, forbidden)
            values_by_role = {"FLOOR": lower, "AVOID": forbidden, "NOOP": distractor_value}
            for order_index, order in enumerate(ORDERS):
                clauses = [
                    {
                        "role": role,
                        "operator": role_operators[role],
                        "argument": f"ARG_{inverse[values_by_role[role]]:02d}",
                        "value": values_by_role[role],
                    }
                    for role in order
                ]
                tokens: list[str] = []
                for clause_index, clause in enumerate(clauses):
                    if clause_index:
                        tokens.append("LINK")
                    tokens.extend((clause["operator"], clause["argument"]))
                test.append(
                    {
                        "order_index": order_index,
                        "order_name": "_".join(order),
                        "order": "_".join(order),
                        "clauses": clauses,
                        "tokens": tokens,
                        "token_ids": [token_ids[token] for token in tokens],
                        "lower": lower,
                        "forbidden": forbidden,
                        "distractor": distractor_value,
                        "distractor_value": distractor_value,
                        "distractor_arg": f"ARG_{inverse[distractor_value]:02d}",
                    }
                )

    return {
        "schema": "T3-nobypass1-noop-distractor-manifest-v1",
        "version": 1,
        "task": "T3-NOBYPASS-1-NOOP-DISTRACTOR",
        "domain_seed": seed,
        "value_count": 32,
        "argument_count": 32,
        "operator_names": list(OPERATORS),
        "role_names": list(ROLES),
        "permutation": permutation,
        "token_ids": token_ids,
        "operator_roles": operator_roles,
        "role_operator_mapping": operator_roles,
        "operator_for_role": role_operators,
        "role_assignment": {
            "operators": list(OPERATORS),
            "roles": list(ROLES),
            "all_assignment_count": 6,
            "selection_seed": 7000,
            "selected_index": seed - SEEDS[0],
            "formula": ROLE_ASSIGNMENT_FORMULA,
            "computed_mappings": [dict(mapping) for mapping in SELECTED_ROLE_MAPPINGS],
        },
        "distractor_formula": DISTRACTOR_FORMULA,
        "fresh_init_required": True,
        "weights_fresh_init_required": True,
        "multi_clause_train": 0,
        "joint_train_examples": 0,
        "forbidden_in_training": "joint",
        "train": train,
        "test": test,
        "train_count": len(train),
        "test_count": len(test),
        "test_pair_count": 32 * 31,
        "test_order_count": len(ORDERS),
    }


def write(seed: int, output_dir: Path) -> tuple[Path, str]:
    path = output_dir / f"manifest_{seed}_v1.json"
    payload = (json.dumps(build(seed), indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return path, sha_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        path, digest = write(seed, args.output_dir)
        rows.append({"seed": seed, "path": str(path), "sha256": digest})
    print(json.dumps({"seeds": list(SEEDS), "manifests": rows, "status": "generated"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
