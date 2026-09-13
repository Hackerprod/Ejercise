"""Independent checker for T4 development manifests; imports no generator code."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "manifests"
OUTPUT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "manifest_check.json"
SEEDS = (7101, 7102, 7103, 7104, 7105)
OPERATORS = ("OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH")
ORDERS = tuple(itertools.permutations(ROLES))
VALUE_COUNT = 32
ID_BLOCKS = {7101: (17000, 17499), 7102: (18000, 18499), 7103: (19000, 19499), 7104: (20000, 20499), 7105: (21000, 21499)}
TOKEN_NAMES = {*(f"ARG_{index:02d}" for index in range(VALUE_COUNT)), *OPERATORS, "LINK"}
ROLE_ASSIGNMENT_FORMULA = "all_six_assignments = [dict(zip(['OP_X', 'OP_Y', 'OP_Z'], role_perm)) for role_perm in itertools.permutations(['FLOOR', 'AVOID', 'MATCH'])]; selected = random.Random(7100).sample(all_six_assignments, 5); mapping = selected[seed - 7101]"
MATCH_FORMULA = "candidate = (seed + lower + forbidden + 1) % 32; while candidate in (lower, forbidden): candidate = (candidate + 1) % 32; E = candidate"


def expected_mappings() -> tuple[dict[str, str], ...]:
    all_six = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    return tuple(random.Random(7100).sample(all_six, 5))


def expected_match(seed: int, lower: int, forbidden: int) -> int:
    candidate = (seed + lower + forbidden + 1) % VALUE_COUNT
    while candidate in (lower, forbidden):
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(errors: list[str], message: str) -> None:
    if len(errors) < 100:
        errors.append(message)


def nested_token_ids(value: Any) -> set[int]:
    if isinstance(value, dict):
        found = set(value["token_ids"].values()) if isinstance(value.get("token_ids"), dict) else set()
        for child in value.values():
            found.update(nested_token_ids(child))
        return found
    if isinstance(value, list):
        found: set[int] = set()
        for child in value:
            found.update(nested_token_ids(child))
        return found
    return set()


def existing_ids(excluded: set[Path]) -> set[int]:
    result: set[int] = set()
    for path in ROOT.rglob("*.json"):
        if path.resolve() in excluded:
            continue
        try:
            result.update(nested_token_ids(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return result


def check_manifest(path: Path, seed: int, mapping: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            add(errors, message)

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"seed": seed, "rows": 0, "errors": [f"load: {error}"], "semantic_errors": 1}

    require(manifest.get("schema") == "T4-nobypass1-three-active-roles-manifest-v1", "schema mismatch")
    require(manifest.get("version") == 1, "version mismatch")
    require(manifest.get("task") == "T4-NOBYPASS-1-THREE-ACTIVE-ROLES", "task mismatch")
    require(manifest.get("argument_count") == 32 and manifest.get("value_count") == 32, "domain count mismatch")
    require(manifest.get("fresh_init_required") is True, "fresh init flag missing")
    require(manifest.get("multi_clause_train") == 0 and manifest.get("joint_train_examples") == 0, "multi-clause training admitted")
    require(manifest.get("match_formula") == MATCH_FORMULA, "match formula mismatch")
    require(manifest.get("operator_roles") == mapping, "operator role mapping mismatch")
    require(set(mapping) == set(OPERATORS) and set(mapping.values()) == set(ROLES), "role mapping not bijective")
    require(manifest.get("operator_for_role") == {role: operator for operator, role in mapping.items()}, "inverse mapping mismatch")

    token_ids = manifest.get("token_ids")
    require(isinstance(token_ids, dict) and set(token_ids) == TOKEN_NAMES, "token name domain mismatch")
    ids = set(token_ids.values()) if isinstance(token_ids, dict) else set()
    require(len(ids) == 36 and all(isinstance(value, int) and not isinstance(value, bool) for value in ids), "physical IDs malformed")
    block = ID_BLOCKS[seed]
    require(all(block[0] <= value <= block[1] for value in ids), "physical ID outside fresh block")

    permutation = manifest.get("permutation")
    valid_permutation = isinstance(permutation, list) and len(permutation) == 32 and set(permutation) == set(range(32))
    require(valid_permutation, "permutation is not bijective")
    inverse = {value: index for index, value in enumerate(permutation)} if valid_permutation else {}

    train = manifest.get("train")
    require(isinstance(train, list) and len(train) == 96, "training row count mismatch")
    counts = {role: 0 for role in ROLES}
    seen_train: set[tuple[str, str]] = set()
    for index, row in enumerate(train if isinstance(train, list) else []):
        role = row.get("role")
        argument = row.get("argument")
        argument_index = int(argument[4:]) if isinstance(argument, str) and argument.startswith("ARG_") and argument[4:].isdigit() else -1
        require(row.get("kind") == "atomic" and "clauses" not in row, f"train[{index}] not atomic")
        require(role in ROLES, f"train[{index}] invalid role")
        require(0 <= argument_index < 32, f"train[{index}] invalid argument")
        require(row.get("operator") == {role_name: operator for operator, role_name in mapping.items()}.get(role), f"train[{index}] operator mismatch")
        require(valid_permutation and row.get("value") == permutation[argument_index], f"train[{index}] value mismatch")
        require(row.get("constraints") == role.lower(), f"train[{index}] constraint mismatch")
        counts[role] = counts.get(role, 0) + 1
        seen_train.add((role, argument))
    require(counts == {role: 32 for role in ROLES}, f"training role counts mismatch: {counts}")
    require(len(seen_train) == 96, "training rows duplicated or incomplete")

    test = manifest.get("test")
    require(isinstance(test, list) and len(test) == 5952, "test row count mismatch")
    expected_pairs = {(lower, forbidden) for lower in range(32) for forbidden in range(32) if lower != forbidden}
    pairs_by_order = {order: set() for order in ORDERS}
    pair_matches: dict[tuple[int, int], set[int]] = {}
    row_fingerprints: set[str] = set()
    for index, row in enumerate(test if isinstance(test, list) else []):
        lower, forbidden, match = row.get("lower"), row.get("forbidden"), row.get("match")
        require(all(isinstance(value, int) and not isinstance(value, bool) for value in (lower, forbidden, match)), f"test[{index}] values malformed")
        if not all(isinstance(value, int) for value in (lower, forbidden, match)):
            continue
        require(lower != forbidden and 0 <= lower < 32 and 0 <= forbidden < 32 and 0 <= match < 32, f"test[{index}] pair domain invalid")
        require(match == expected_match(seed, lower, forbidden) and row.get("E") == match, f"test[{index}] match formula mismatch")
        require(match not in (lower, forbidden), f"test[{index}] match collides with pair")
        require(valid_permutation and row.get("match_arg") == f"ARG_{inverse[match]:02d}", f"test[{index}] match arg mismatch")
        order_index = row.get("order_index")
        expected_order = ORDERS[order_index] if isinstance(order_index, int) and 0 <= order_index < 6 else ()
        expected_name = "_".join(expected_order)
        require(row.get("order") == expected_name and row.get("order_name") == expected_name, f"test[{index}] order mismatch")
        pair = (lower, forbidden)
        if expected_order:
            pairs_by_order[expected_order].add(pair)
        pair_matches.setdefault(pair, set()).add(match)
        values = {"FLOOR": lower, "AVOID": forbidden, "MATCH": match}
        expected_clauses = [
            {"argument": f"ARG_{inverse[values[role]]:02d}", "operator": {role_name: operator for operator, role_name in mapping.items()}[role], "role": role, "value": values[role]}
            for role in expected_order
        ]
        require(row.get("clauses") == expected_clauses, f"test[{index}] clause semantics mismatch")
        expected_tokens: list[str] = []
        for clause_index, clause in enumerate(expected_clauses):
            if clause_index:
                expected_tokens.append("LINK")
            expected_tokens.extend((clause["operator"], clause["argument"]))
        require(row.get("tokens") == expected_tokens, f"test[{index}] token sequence mismatch")
        require(isinstance(token_ids, dict) and row.get("token_ids") == [token_ids[token] for token in expected_tokens], f"test[{index}] physical token sequence mismatch")
        row_fingerprints.add(json.dumps(row, sort_keys=True, separators=(",", ":")))

    require(len(row_fingerprints) == len(test) if isinstance(test, list) else False, "test rows duplicated")
    require(all(pairs == expected_pairs for pairs in pairs_by_order.values()), "each order does not contain all 992 pairs")
    require(len(pair_matches) == 992 and all(len(values) == 1 for values in pair_matches.values()), "match values inconsistent across order variants")
    return {"seed": seed, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "rows": len(test) if isinstance(test, list) else 0, "pair_count": len(pair_matches), "semantic_errors": len(errors), "errors": errors}


def seal(artifact: dict[str, Any], path: Path) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    current = json.loads(written.decode("utf-8"))["artifact_self_hash"]
    placeholder = written.replace(f'"artifact_self_hash": "{current}"'.encode(), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if hashlib.sha256(placeholder).hexdigest() != current:
        raise RuntimeError("checker artifact self-hash verification failed")
    return digest


def main() -> int:
    mappings = expected_mappings()
    selected_paths = {(MANIFEST_ROOT / f"manifest_{seed}_v1.json").resolve() for seed in SEEDS}
    per_manifest = [check_manifest(MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, mapping) for seed, mapping in zip(SEEDS, mappings)]
    existing = existing_ids(selected_paths | {OUTPUT.resolve()})
    fresh_ids: set[int] = set()
    id_errors: list[str] = []
    for item in per_manifest:
        seed = item["seed"]
        path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
        try:
            values = set(json.loads(path.read_text(encoding="utf-8"))["token_ids"].values())
        except Exception:
            values = set()
        overlap = fresh_ids & values
        if overlap:
            id_errors.append(f"seed {seed}: T4 overlap {sorted(overlap)}")
        fresh_ids.update(values)
        if values & existing:
            id_errors.append(f"seed {seed}: overlap with existing artifact IDs {sorted(values & existing)}")
        if values & set(range(2000, 6500)):
            id_errors.append(f"seed {seed}: overlap with 680x range")
        if values & set(range(7000, 11500)):
            id_errors.append(f"seed {seed}: overlap with 690x range")
        if values & set(range(12000, 16500)):
            id_errors.append(f"seed {seed}: overlap with 700x/T3 range")
    semantic_errors = sum(item["semantic_errors"] for item in per_manifest)
    total_rows = sum(item["rows"] for item in per_manifest)
    errors = semantic_errors + len(id_errors)
    artifact: dict[str, Any] = {
        "status": "passed" if errors == 0 and total_rows == 29760 else "failed",
        "task": "T4-NOBYPASS-1-THREE-ACTIVE-ROLES",
        "schema": "T4-nobypass1-three-active-roles-check-v1",
        "version": 1,
        "seeds": list(SEEDS),
        "role_assignment": {"operators": list(OPERATORS), "roles": list(ROLES), "selection_seed": 7100, "formula": ROLE_ASSIGNMENT_FORMULA, "computed_mappings": list(mappings)},
        "match_formula": MATCH_FORMULA,
        "manifests": per_manifest,
        "checked_rows": total_rows,
        "checked_expected": 29760,
        "errors": errors,
        "error_details": id_errors + [f"seed {item['seed']}: {error}" for item in per_manifest for error in item["errors"]],
        "fresh_ids": {"count": len(fresh_ids), "blocks": {str(seed): list(ID_BLOCKS[seed]) for seed in SEEDS}, "existing_artifact_ids_checked": len(existing), "errors": id_errors},
        "validation": ["six order coverage per (L,F)", "sealed E(L,F) consistency and collision exclusion", "valid distinct role mappings", "96 atomic rows with 32 per active role and zero multi-clause", "fresh IDs disjoint from 680x/690x/700x and existing JSON artifacts"],
    }
    digest = seal(artifact, OUTPUT)
    print(json.dumps({"status": artifact["status"], "checked_rows": total_rows, "errors": errors, "fresh_ids": len(fresh_ids), "artifact": str(OUTPUT), "artifact_self_hash": digest}, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
