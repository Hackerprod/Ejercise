"""Independently validate the T3 no-bypass NOOP-distractor battery."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = ROOT / "campaign" / "nb5_manifests"
DEFAULT_OUTPUT = ROOT / "campaign" / "t3_nobypass1_noop_distractor_manifest_check.json"
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
TOKEN_NAMES = {*(f"ARG_{index:02d}" for index in range(32)), *OPERATORS, "LINK"}
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
LEGACY_RANGE = range(2000, 6500)
OLD_FRESH_RANGE = range(7000, 11500)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_mappings() -> tuple[dict[str, str], ...]:
    assignments = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    return tuple(random.Random(7000).sample(assignments, 5))


def expected_distractor(seed: int, lower: int, forbidden: int) -> int:
    candidate = (seed + lower + forbidden) % 32
    while candidate in (lower, forbidden):
        candidate = (candidate + 1) % 32
    return candidate


def add_error(errors: list[str], message: str) -> None:
    if len(errors) < 100:
        errors.append(message)


def check_manifest(path: Path, seed: int, expected_mapping: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"seed": seed, "path": str(path), "rows": 0, "errors": [f"load: {error}"]}

    def require(condition: bool, message: str) -> None:
        if not condition:
            add_error(errors, message)

    require(manifest.get("schema") == "T3-nobypass1-noop-distractor-manifest-v1", "schema mismatch")
    require(manifest.get("version") == 1, "version mismatch")
    require(manifest.get("task") == "T3-NOBYPASS-1-NOOP-DISTRACTOR", "task mismatch")
    require(manifest.get("domain_seed") == seed, "domain_seed mismatch")
    require(manifest.get("value_count") == 32 and manifest.get("argument_count") == 32, "domain count mismatch")
    require(manifest.get("fresh_init_required") is True, "fresh_init_required mismatch")
    require(manifest.get("weights_fresh_init_required") is True, "weights_fresh_init_required mismatch")
    require(manifest.get("multi_clause_train") == 0, "multi_clause_train is not zero")
    require(manifest.get("joint_train_examples") == 0, "joint_train_examples is not zero")
    require(manifest.get("distractor_formula") == DISTRACTOR_FORMULA, "distractor formula mismatch")

    role_assignment = manifest.get("role_assignment")
    require(isinstance(role_assignment, dict), "role_assignment is not an object")
    if isinstance(role_assignment, dict):
        require(role_assignment.get("formula") == ROLE_ASSIGNMENT_FORMULA, "role assignment formula mismatch")
        require(role_assignment.get("selection_seed") == 7000, "role assignment seed mismatch")
        require(role_assignment.get("all_assignment_count") == 6, "role assignment count mismatch")
        require(role_assignment.get("computed_mappings") == list(expected_mappings()), "computed mappings mismatch")

    require(manifest.get("operator_roles") == expected_mapping, "operator_roles mismatch")
    require(manifest.get("role_operator_mapping") == expected_mapping, "role_operator_mapping mismatch")
    require(
        manifest.get("operator_for_role") == {role: operator for operator, role in expected_mapping.items()},
        "operator_for_role mismatch",
    )
    require(set(expected_mapping) == set(OPERATORS) and set(expected_mapping.values()) == set(ROLES), "mapping is not a bijection")
    require(manifest.get("role_names") == list(ROLES), "role names mismatch")
    require(manifest.get("operator_names") == list(OPERATORS), "operator names mismatch")

    token_ids = manifest.get("token_ids")
    require(isinstance(token_ids, dict), "token_ids is not an object")
    if isinstance(token_ids, dict):
        require(set(token_ids) == TOKEN_NAMES, "token key domain mismatch")
        values = list(token_ids.values())
        require(len(values) == 36 and all(isinstance(value, int) and not isinstance(value, bool) for value in values), "token IDs must contain 36 integers")
        require(len(set(values)) == 36, "token IDs are not unique")
        start, end = ID_BLOCKS[seed]
        require(all(start <= value <= end for value in values), "token ID outside assigned fresh block")

    permutation = manifest.get("permutation")
    valid_permutation = isinstance(permutation, list) and len(permutation) == 32 and set(permutation) == set(range(32))
    require(valid_permutation, "permutation is not a bijection over 0..31")
    inverse = {value: index for index, value in enumerate(permutation)} if valid_permutation else {}

    train = manifest.get("train")
    require(isinstance(train, list) and len(train) == 96, "train count is not 96")
    train_seen: set[tuple[str, str, str, int, int]] = set()
    train_counts = {role: 0 for role in ROLES}
    if isinstance(train, list):
        for index, row in enumerate(train):
            if not isinstance(row, dict):
                add_error(errors, f"train[{index}] is not an object")
                continue
            role = row.get("role")
            argument = row.get("argument")
            value = row.get("value")
            operator = row.get("operator")
            require(row.get("kind") == "atomic", f"train[{index}] is not atomic")
            require(role in ROLES, f"train[{index}] invalid role")
            require(isinstance(argument, str) and argument in {f"ARG_{i:02d}" for i in range(32)}, f"train[{index}] invalid argument")
            argument_index = int(argument[4:]) if isinstance(argument, str) and argument.startswith("ARG_") and argument[4:].isdigit() else -1
            require(valid_permutation and 0 <= argument_index < 32 and value == permutation[argument_index], f"train[{index}] value/permutation mismatch")
            require(role in ROLES and operator == {role_name: op for op, role_name in expected_mapping.items()}.get(role), f"train[{index}] operator mismatch")
            require(row.get("constraints") == {"FLOOR": "floor", "AVOID": "avoid", "NOOP": "none"}.get(role), f"train[{index}] constraints mismatch")
            require("clauses" not in row and row.get("kind") != "joint", f"train[{index}] contains multi-clause/joint data")
            if role in ROLES:
                train_counts[role] += 1
            if role in ROLES and isinstance(argument, str) and isinstance(value, int) and 0 <= argument_index < 32:
                train_seen.add((role, operator, argument, argument_index, value))
        require(train_counts == {"FLOOR": 32, "AVOID": 32, "NOOP": 32}, f"train role counts mismatch: {train_counts}")
        require(len(train_seen) == 96, "train rows are duplicated or incomplete")

    test = manifest.get("test")
    require(isinstance(test, list) and len(test) == 5952, "test count is not 5952")
    pairs_by_order = {order: set() for order in ORDERS}
    pair_distractors: dict[tuple[int, int], set[int]] = {}
    row_fingerprints: set[str] = set()
    checked_rows = len(test) if isinstance(test, list) else 0
    for index, row in enumerate(test if isinstance(test, list) else []):
        if not isinstance(row, dict):
            add_error(errors, f"test[{index}] is not an object")
            continue
        try:
            lower = row["lower"]
            forbidden = row["forbidden"]
            distractor_value = row["distractor_value"]
        except (KeyError, TypeError):
            add_error(errors, f"test[{index}] missing semantic fields")
            continue
        require(all(isinstance(value, int) and not isinstance(value, bool) for value in (lower, forbidden, distractor_value)), f"test[{index}] semantic values are not integers")
        require(0 <= lower < 32 and 0 <= forbidden < 32 and 0 <= distractor_value < 32, f"test[{index}] value outside 0..31")
        require(lower != forbidden, f"test[{index}] is diagonal")
        expected_d = expected_distractor(seed, lower, forbidden) if isinstance(lower, int) and isinstance(forbidden, int) else -1
        require(distractor_value == expected_d and row.get("distractor") == expected_d, f"test[{index}] distractor formula mismatch")
        require(distractor_value not in (lower, forbidden), f"test[{index}] distractor collides with pair")
        require(valid_permutation and row.get("distractor_arg") == f"ARG_{inverse[distractor_value]:02d}", f"test[{index}] distractor argument mismatch")

        order_index = row.get("order_index")
        order_name = row.get("order_name")
        require(isinstance(order_index, int) and 0 <= order_index < 6, f"test[{index}] invalid order index")
        expected_order = ORDERS[order_index] if isinstance(order_index, int) and 0 <= order_index < 6 else ()
        expected_order_name = "_".join(expected_order)
        require(order_name == expected_order_name and row.get("order") == expected_order_name, f"test[{index}] order name mismatch")
        pair = (lower, forbidden)
        if isinstance(order_index, int) and 0 <= order_index < 6:
            pairs_by_order[expected_order].add(pair)
        pair_distractors.setdefault(pair, set()).add(distractor_value)

        clauses = row.get("clauses")
        require(isinstance(clauses, list) and len(clauses) == 3, f"test[{index}] clauses malformed")
        expected_values = {"FLOOR": lower, "AVOID": forbidden, "NOOP": distractor_value}
        expected_clauses: list[dict[str, Any]] = []
        if isinstance(clauses, list):
            for clause_position, clause in enumerate(clauses):
                expected_role = expected_order[clause_position] if clause_position < len(expected_order) else "INVALID"
                expected_value = expected_values.get(expected_role, -1)
                expected_clause = {
                    "role": expected_role,
                    "operator": {role: operator for operator, role in expected_mapping.items()}.get(expected_role),
                    "argument": f"ARG_{inverse[expected_value]:02d}" if valid_permutation and expected_value in inverse else None,
                    "value": expected_value,
                }
                expected_clauses.append(expected_clause)
                require(clause == expected_clause, f"test[{index}] clause {clause_position} semantics mismatch")
        require(clauses == expected_clauses, f"test[{index}] clause list mismatch")

        expected_tokens: list[str] = []
        for clause_position, clause in enumerate(expected_clauses):
            if clause_position:
                expected_tokens.append("LINK")
            expected_tokens.extend((clause["operator"], clause["argument"]))
        require(row.get("tokens") == expected_tokens, f"test[{index}] flat token sequence mismatch")
        require(len(expected_tokens) == 8 and expected_tokens[2] == "LINK" and expected_tokens[5] == "LINK", f"test[{index}] LINK separators mismatch")
        if isinstance(token_ids, dict):
            require(row.get("token_ids") == [token_ids.get(token) for token in expected_tokens], f"test[{index}] physical token IDs mismatch")
        row_fingerprints.add(json.dumps(row, sort_keys=True, separators=(",", ":")))

    require(len(row_fingerprints) == checked_rows, "test rows contain duplicates")
    expected_pairs = {(lower, forbidden) for lower in range(32) for forbidden in range(32) if lower != forbidden}
    require(len(expected_pairs) == 992, "internal pair catalog mismatch")
    require(all(pairs == expected_pairs for pairs in pairs_by_order.values()), "each order does not contain all 992 pairs exactly once")
    require(all(len(values) == 1 for values in pair_distractors.values()), "distractor differs across order variants")
    require(len(pair_distractors) == 992, "pair catalog is not 992 non-diagonal pairs")
    return {
        "seed": seed,
        "path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
        "sha256": sha(path),
        "rows": checked_rows,
        "pair_count": len(pair_distractors),
        "order_count": len(ORDERS),
        "semantic_errors": len(errors),
        "errors": errors,
    }


def nested_token_ids(value: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        token_values = value.get("token_ids")
        if isinstance(token_values, dict) and all(isinstance(item, int) and not isinstance(item, bool) for item in token_values.values()):
            found.update(token_values.values())
        elif isinstance(token_values, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in token_values):
            found.update(token_values)
        for child in value.values():
            found.update(nested_token_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(nested_token_ids(child))
    return found


def existing_ids(selected: set[Path]) -> tuple[set[int], list[str]]:
    ids: set[int] = set()
    files: list[str] = []
    for path in sorted(ROOT.rglob("*.json")):
        resolved = path.resolve()
        if resolved in selected:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        found = nested_token_ids(payload)
        if found:
            ids.update(found)
            files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return ids, files


def seal_artifact(artifact: dict[str, Any], output: Path) -> str:
    artifact["artifact_self_hash"] = "__SELF_HASH__"
    unsigned = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(unsigned).hexdigest()
    artifact["artifact_self_hash"] = digest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    written = output.read_bytes()
    current = json.loads(written.decode("utf-8"))["artifact_self_hash"]
    placeholder = written.replace(f'"artifact_self_hash": "{current}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if hashlib.sha256(placeholder).hexdigest() != current:
        raise RuntimeError("artifact self-hash verification failed")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    expected = expected_mappings()
    selected_paths = {(args.manifest_dir / f"manifest_{seed}_v1.json").resolve() for seed in SEEDS}
    per_manifest = []
    for seed, mapping in zip(SEEDS, expected):
        path = args.manifest_dir / f"manifest_{seed}_v1.json"
        per_manifest.append(check_manifest(path, seed, mapping) if path.exists() else {"seed": seed, "path": str(path), "rows": 0, "semantic_errors": 1, "errors": ["manifest missing"]})

    old_ids, old_files = existing_ids(selected_paths | {args.output.resolve()})
    fresh_ids: set[int] = set()
    id_errors: list[str] = []
    for seed in SEEDS:
        path = args.manifest_dir / f"manifest_{seed}_v1.json"
        try:
            values = set(json.loads(path.read_text(encoding="utf-8"))["token_ids"].values())
        except Exception:
            values = set()
        overlap = fresh_ids & values
        if overlap:
            id_errors.append(f"seed {seed}: overlap with another T3 seed: {sorted(overlap)}")
        fresh_ids.update(values)
        start, end = ID_BLOCKS[seed]
        outside = values - set(range(start, end + 1))
        if outside:
            id_errors.append(f"seed {seed}: IDs outside block {start}-{end}: {sorted(outside)}")
        legacy_overlap = values & set(LEGACY_RANGE)
        old_fresh_overlap = values & set(OLD_FRESH_RANGE)
        existing_overlap = values & old_ids
        if legacy_overlap:
            id_errors.append(f"seed {seed}: overlap with legacy range: {sorted(legacy_overlap)}")
        if old_fresh_overlap:
            id_errors.append(f"seed {seed}: overlap with prior fresh range: {sorted(old_fresh_overlap)}")
        if existing_overlap:
            id_errors.append(f"seed {seed}: overlap with existing artifact IDs: {sorted(existing_overlap)}")

    semantic_errors = sum(item.get("semantic_errors", len(item.get("errors", []))) for item in per_manifest)
    errors = semantic_errors + len(id_errors)
    artifact: dict[str, Any] = {
        "status": "passed" if errors == 0 and sum(item.get("rows", 0) for item in per_manifest) == 29760 else "failed",
        "task": "T3-NOBYPASS-1-NOOP-DISTRACTOR",
        "schema": "T3-nobypass1-noop-distractor-check-v1",
        "version": 1,
        "seeds": list(SEEDS),
        "role_assignment": {
            "operators": list(OPERATORS),
            "roles": list(ROLES),
            "all_six_assignments": [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)],
            "selection_seed": 7000,
            "formula": ROLE_ASSIGNMENT_FORMULA,
            "computed_mappings": list(expected),
        },
        "distractor_formula": DISTRACTOR_FORMULA,
        "manifests": per_manifest,
        "per_manifest_hashes": {str(item["seed"]): item.get("sha256") for item in per_manifest},
        "fresh_ids": {
            "count": len(fresh_ids),
            "blocks": {str(seed): list(ID_BLOCKS[seed]) for seed in SEEDS},
            "existing_artifact_files_checked": old_files,
            "existing_artifact_ids_checked": len(old_ids),
            "legacy_range": [2000, 6499],
            "prior_fresh_range": [7000, 11499],
            "errors": id_errors,
        },
        "checked_rows": sum(item.get("rows", 0) for item in per_manifest),
        "checked_expected": 29760,
        "errors": errors,
        "error_details": id_errors + [f"seed {item['seed']}: {error}" for item in per_manifest for error in item.get("errors", [])],
        "validation": [
            "schema/version/domain seed",
            "token key domain, uniqueness, fresh blocks, and cross-artifact disjointness",
            "permutation bijection",
            "six role assignments and distinct selected mappings",
            "fresh init, train counts, atomic rows, and zero joint/multi-clause training",
            "5952 test rows, 992 non-diagonal pairs, and six orders",
            "deterministic distractor and invariant six-variant distractor",
            "clause role/operator/argument/value semantics",
            "LINK separators and flat token sequence",
            "physical token_ids consistency",
            "duplicate row rejection",
        ],
    }
    digest = seal_artifact(artifact, args.output)
    print(json.dumps({**artifact, "artifact_self_hash": digest}, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
