"""Independently validate sealed T4-NOBYPASS-3 manifests before training."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t4_nobypass3_fresh" / "manifests"
OUTPUT = ROOT / "campaign" / "t4_nobypass3_fresh" / "manifest_check.json"
SEEDS = (7201, 7202, 7203, 7204, 7205)
OPERATORS = ("OP_X", "OP_Y", "OP_Z")
ROLES = ("FLOOR", "AVOID", "MATCH")
ORDERS = tuple(itertools.permutations(ROLES))
VALUE_COUNT = 32
BATCH_SEED = 7200
ID_BLOCKS = {7201: (22000, 22499), 7202: (23000, 23499), 7203: (24000, 24499), 7204: (25000, 25499), 7205: (26000, 26499)}
TOKEN_NAMES = {*(f"ARG_{index:02d}" for index in range(VALUE_COUNT)), *OPERATORS, "LINK"}
OLD_BLOCKS = ((2000, 6499), (7000, 11499), (12000, 16499), (17000, 21499))
MATCH_FORMULA = "candidate = (seed + L + F + 1) % 32; while candidate in (L, F): candidate = (candidate + 1) % 32; E = candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_mappings() -> tuple[list[dict[str, str]], dict[str, str], list[dict[str, str]]]:
    all_six = [dict(zip(OPERATORS, role_perm)) for role_perm in itertools.permutations(ROLES)]
    selected = random.Random(BATCH_SEED).sample(all_six, 5)
    omitted = next(mapping for mapping in all_six if mapping not in selected)
    return selected, omitted, all_six


def expected_match(seed: int, lower: int, forbidden: int) -> int:
    candidate = (seed + lower + forbidden + 1) % VALUE_COUNT
    while candidate in (lower, forbidden):
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


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


def check_manifest(path: Path, seed: int, mapping: dict[str, str], selected: list[dict[str, str]], omitted: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition and len(errors) < 100:
            errors.append(message)

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"seed": seed, "path": str(path), "rows": 0, "pair_count": 0, "semantic_errors": 1, "errors": [f"load: {error}"]}
    require(manifest.get("schema") == "T4-nobypass3-fresh-manifest-v1", "schema mismatch")
    require(manifest.get("task") == "T4-NOBYPASS-3-FRESH", "task mismatch")
    require(manifest.get("version") == 1, "version mismatch")
    require(manifest.get("batch_seed") == BATCH_SEED, "batch seed mismatch")
    require(manifest.get("argument_count") == VALUE_COUNT and manifest.get("value_count") == VALUE_COUNT, "domain count mismatch")
    require(manifest.get("fresh_init_required") is True, "fresh init missing")
    require(manifest.get("multi_clause_train") == 0 and manifest.get("joint_train_examples") == 0 and manifest.get("test_rows_used") == 0, "training contamination flags mismatch")
    require(manifest.get("match_formula") == MATCH_FORMULA, "match formula mismatch")
    require(manifest.get("operator_roles") == mapping, "operator role mapping mismatch")
    require(set(mapping) == set(OPERATORS) and set(mapping.values()) == set(ROLES), "role mapping not bijective")
    require(manifest.get("operator_for_role") == {role: operator for operator, role in mapping.items()}, "inverse role mapping mismatch")
    assignment = manifest.get("role_assignment", {})
    require(assignment.get("all_assignment_count") == 6 and assignment.get("batch_seed") == BATCH_SEED, "assignment metadata mismatch")
    require(assignment.get("computed_mappings") == selected and assignment.get("omitted_mapping") == omitted, "assignment selection/omitted mismatch")
    token_ids = manifest.get("token_ids")
    require(isinstance(token_ids, dict) and set(token_ids) == TOKEN_NAMES, "token name domain mismatch")
    ids = set(token_ids.values()) if isinstance(token_ids, dict) else set()
    require(len(ids) == 36 and all(isinstance(value, int) and not isinstance(value, bool) for value in ids), "physical IDs malformed")
    block = ID_BLOCKS[seed]
    require(all(block[0] <= value <= block[1] for value in ids), "physical ID outside fresh block")
    permutation = manifest.get("permutation")
    valid_permutation = isinstance(permutation, list) and len(permutation) == VALUE_COUNT and set(permutation) == set(range(VALUE_COUNT))
    require(valid_permutation, "argument permutation is not bijective")
    inverse = {value: index for index, value in enumerate(permutation)} if valid_permutation else {}

    train = manifest.get("train")
    require(isinstance(train, list) and len(train) == 96, "training row count mismatch")
    counts = {role: 0 for role in ROLES}
    seen_train: set[tuple[str, str]] = set()
    for index, row in enumerate(train if isinstance(train, list) else []):
        role = row.get("role")
        argument = row.get("argument")
        argument_index = int(argument[4:]) if isinstance(argument, str) and argument.startswith("ARG_") and argument[4:].isdigit() else -1
        require(row.get("kind") == "atomic" and "clauses" not in row and "tokens" not in row, f"train[{index}] not atomic-only")
        require(role in ROLES and 0 <= argument_index < VALUE_COUNT, f"train[{index}] domain invalid")
        require(row.get("operator") == {role_name: operator for operator, role_name in mapping.items()}.get(role), f"train[{index}] operator mismatch")
        require(valid_permutation and row.get("value") == permutation[argument_index], f"train[{index}] value mismatch")
        require(row.get("constraints") == role.lower(), f"train[{index}] constraints mismatch")
        counts[role] = counts.get(role, 0) + 1
        seen_train.add((role, argument))
    require(counts == {role: 32 for role in ROLES}, f"training role counts mismatch: {counts}")
    require(len(seen_train) == 96, "training rows duplicated/incomplete")

    test = manifest.get("test")
    require(isinstance(test, list) and len(test) == 5952, "test row count mismatch")
    expected_pairs = {(lower, forbidden) for lower in range(VALUE_COUNT) for forbidden in range(VALUE_COUNT) if lower != forbidden}
    pairs_by_order = {order: set() for order in ORDERS}
    pair_matches: dict[tuple[int, int], set[int]] = {}
    fingerprints: set[str] = set()
    for index, row in enumerate(test if isinstance(test, list) else []):
        lower, forbidden, match = row.get("lower"), row.get("forbidden"), row.get("match")
        require(all(isinstance(value, int) and not isinstance(value, bool) for value in (lower, forbidden, match)), f"test[{index}] values malformed")
        if not all(isinstance(value, int) for value in (lower, forbidden, match)):
            continue
        require(lower != forbidden and 0 <= lower < VALUE_COUNT and 0 <= forbidden < VALUE_COUNT and 0 <= match < VALUE_COUNT, f"test[{index}] domain invalid")
        require(match == expected_match(seed, lower, forbidden) and row.get("E") == match and match not in (lower, forbidden), f"test[{index}] E formula/collision mismatch")
        require(valid_permutation and row.get("match_arg") == f"ARG_{inverse[match]:02d}", f"test[{index}] match arg mismatch")
        order_index = row.get("order_index")
        expected_order = ORDERS[order_index] if isinstance(order_index, int) and 0 <= order_index < 6 else ()
        expected_name = "_".join(expected_order)
        require(row.get("order") == expected_name and row.get("order_name") == expected_name, f"test[{index}] order mismatch")
        if expected_order:
            pairs_by_order[expected_order].add((lower, forbidden))
        pair_matches.setdefault((lower, forbidden), set()).add(match)
        values = {"FLOOR": lower, "AVOID": forbidden, "MATCH": match}
        expected_clauses = [{"argument": f"ARG_{inverse[values[role]]:02d}", "operator": {role_name: operator for operator, role_name in mapping.items()}[role], "role": role, "value": values[role]} for role in expected_order]
        require(row.get("clauses") == expected_clauses, f"test[{index}] clause mismatch")
        expected_tokens: list[str] = []
        for clause_index, clause in enumerate(expected_clauses):
            if clause_index:
                expected_tokens.append("LINK")
            expected_tokens.extend((clause["operator"], clause["argument"]))
        require(row.get("tokens") == expected_tokens, f"test[{index}] token sequence mismatch")
        require(isinstance(token_ids, dict) and row.get("token_ids") == [token_ids[token] for token in expected_tokens], f"test[{index}] physical sequence mismatch")
        fingerprints.add(json.dumps(row, sort_keys=True, separators=(",", ":")))
    require(len(fingerprints) == len(test) if isinstance(test, list) else False, "test rows duplicated")
    require(all(pairs == expected_pairs for pairs in pairs_by_order.values()), "each order lacks all 992 pairs")
    require(len(pair_matches) == 992 and all(len(values) == 1 for values in pair_matches.values()), "match values inconsistent across orders")
    return {"seed": seed, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "rows": len(test) if isinstance(test, list) else 0, "pair_count": len(pair_matches), "train_rows": len(train) if isinstance(train, list) else 0, "semantic_errors": len(errors), "errors": errors}


def seal(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(written)
    return digest


def main() -> int:
    selected, omitted, all_six = expected_mappings()
    selected_paths = {(MANIFEST_ROOT / f"manifest_{seed}_v1.json").resolve() for seed in SEEDS}
    per_manifest = [check_manifest(MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, mapping, selected, omitted) for seed, mapping in zip(SEEDS, selected)]
    existing = existing_ids(selected_paths | {OUTPUT.resolve()})
    fresh_ids: set[int] = set()
    id_errors: list[str] = []
    for item in per_manifest:
        seed = item["seed"]
        try:
            values = set(json.loads((MANIFEST_ROOT / f"manifest_{seed}_v1.json").read_text(encoding="utf-8"))["token_ids"].values())
        except Exception:
            values = set()
        overlap = fresh_ids & values
        if overlap:
            id_errors.append(f"seed {seed}: overlap with T4-NOBYPASS-3 IDs {sorted(overlap)}")
        fresh_ids.update(values)
        if values & existing:
            id_errors.append(f"seed {seed}: overlap with existing artifact IDs {sorted(values & existing)}")
        if any(start <= value <= end for start, end in OLD_BLOCKS for value in values):
            id_errors.append(f"seed {seed}: overlap with blocked 680x/690x/700x/710x ranges")
    semantic_errors = sum(item["semantic_errors"] for item in per_manifest)
    total_rows = sum(item["rows"] for item in per_manifest)
    errors = semantic_errors + len(id_errors)
    artifact = {"status": "passed" if errors == 0 and total_rows == 29760 and len(fresh_ids) == 180 else "failed", "task": "T4-NOBYPASS-3-FRESH", "schema": "T4-nobypass3-fresh-check-v1", "version": 1, "seeds": list(SEEDS), "role_assignment": {"operators": list(OPERATORS), "roles": list(ROLES), "batch_seed": BATCH_SEED, "all_assignment_count": 6, "selected_mappings": selected, "omitted_mapping": omitted, "all_six_mappings": all_six}, "match_formula": MATCH_FORMULA, "manifests": per_manifest, "checked_rows": total_rows, "checked_expected": 29760, "errors": errors, "error_details": id_errors + [f"seed {item['seed']}: {error}" for item in per_manifest for error in item["errors"]], "fresh_ids": {"count": len(fresh_ids), "blocks": {str(seed): list(ID_BLOCKS[seed]) for seed in SEEDS}, "existing_artifact_ids_checked": len(existing), "errors": id_errors}, "validation": ["992 (L,F) pairs × six exact orders per seed", "E(L,F) formula and E collision exclusion", "five valid role assignments plus explicit omitted sixth", "96 atomic training rows with 32 per role and zero multi-clause/test rows", "fresh physical IDs unique across seeds and disjoint from 680x/690x/700x/710x and existing JSON artifacts"]}
    digest = seal(artifact)
    print(json.dumps({"status": artifact["status"], "checked_rows": total_rows, "errors": errors, "fresh_ids": len(fresh_ids), "artifact": str(OUTPUT), "artifact_self_hash": digest}, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
