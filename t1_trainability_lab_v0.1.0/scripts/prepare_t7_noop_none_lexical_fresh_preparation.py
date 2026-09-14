"""Prepare sealed T7 fresh 7801-7805 manifests without model work."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import evaluate_t7_noop_none_lexical_paired_development as corrected_evaluator


ROOT = corrected_evaluator.ROOT
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_fresh_preparation"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
SEEDS = (7801, 7802, 7803, 7804, 7805)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR", "NOOP")
ACTIVE_ROLES = ROLES[:-1]
OPERATORS = ("OP_V", "OP_W", "OP_X", "OP_Y", "OP_Z")
ARGUMENTS = tuple(f"ARG_{index:02d}" for index in range(32))
ID_BLOCKS = {seed: (52000 + index * 1000, 52499 + index * 1000) for index, seed in enumerate(SEEDS)}
PLACEHOLDER = "__SELF_HASH__"
EXPECTED_BASE = {1: 128, 2: 11904, 3: 23808, 4: 23808}
EXPECTED_AUGMENTED = {1: 256, 2: 35712, 3: 95232, 4: 119040}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def all_assignments() -> tuple[dict[str, str], ...]:
    return tuple(dict(zip(OPERATORS, roles)) for roles in itertools.permutations(ROLES))


ALL_ASSIGNMENTS = all_assignments()
SELECTED_ASSIGNMENTS = tuple(random.Random(7800).sample(list(ALL_ASSIGNMENTS), 5))


def independent_derived_value(seed: int, values: list[int]) -> int:
    offset = 1 if len(values) == 2 else 2
    candidate = (seed + sum(values) + offset) % 32
    while candidate in values:
        candidate = (candidate + 1) % 32
    return candidate


def independent_distractor(seed: int, values: list[int]) -> int:
    candidate = (seed + sum(values) + 1) % 32
    while candidate in values:
        candidate = (candidate + 1) % 32
    return candidate


def independent_assignments(seed: int, subset: tuple[str, ...]) -> list[tuple[int, ...]]:
    if len(subset) <= 2:
        return list(itertools.permutations(range(32), len(subset)))
    result: list[tuple[int, ...]] = []
    for first_two in itertools.permutations(range(32), 2):
        values = list(first_two)
        while len(values) < len(subset):
            values.append(independent_derived_value(seed, values))
        result.append(tuple(values))
    return result


def physical_ids(seed: int, count: int) -> list[int]:
    start, end = ID_BLOCKS[seed]
    return random.Random(seed + 100000).sample(range(start, end + 1), count)


def build_manifest(seed: int, mapping: dict[str, str]) -> dict[str, Any]:
    noop_operator = next(operator for operator, role in mapping.items() if role == "NOOP")
    active_mapping = {operator: role for operator, role in mapping.items() if role != "NOOP"}
    base_token_order = [*ARGUMENTS, *(operator for operator in OPERATORS if operator != noop_operator), "LINK"]
    augmented_token_order = [*base_token_order, noop_operator]
    ids = physical_ids(seed, len(base_token_order))
    token_ids = dict(zip(base_token_order, ids))
    start, end = ID_BLOCKS[seed]
    inverse = {value: index for index, value in enumerate(random.Random(seed).sample(range(32), 32))}
    permutation = [value for value in inverse]
    role_operator = {role: operator for operator, role in active_mapping.items()}
    subset_registry = []
    for size in range(1, 5):
        for subset in itertools.combinations(ACTIVE_ROLES, size):
            assignments = len(independent_assignments(seed, subset))
            orders = math_factorial(size)
            subset_registry.append({"subset": "_".join(subset), "roles": list(subset), "size": size, "assignments": assignments, "orders": orders, "base_rows": assignments * orders, "augmented_rows": assignments * math_factorial(size + 1), "kind": "atomic_control" if size == 1 else "active_set_composition"})
    return {
        "schema": "T7-noop-none-lexical-fresh-preparation-manifest-v1",
        "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION",
        "seed": seed,
        "operator_role_assignment": mapping,
        "active_operator_role_assignment": active_mapping,
        "noop_operator": noop_operator,
        "operator_order": list(OPERATORS),
        "role_order": list(ROLES),
        "permutation": permutation,
        "inverse_permutation": {str(value): index for index, value in enumerate(permutation)},
        "token_order": base_token_order,
        "token_ids": token_ids,
        "id_block": [start, end],
        "id_semantics": "physical IDs fresh and disjoint; model uses manifest-local vocabulary indices",
        "stage_c_noop_external_id": end + 1,
        "stage_c_noop_internal_id": len(base_token_order),
        "stage_c_token_order": augmented_token_order,
        "vocabulary": {"stage_a_rows": 37, "stage_c_rows": 38, "base_indices_preserved": True, "noop_added_after_stage_b": True, "noop_external_id_reserved": end + 1},
        "operator_for_role": role_operator,
        "formulas": {
            "v2": "v2 = (seed + v0 + v1 + 1) mod 32; increment by 1 modulo 32 until v2 differs from v0,v1",
            "v3": "v3 = (seed + v0 + v1 + v2 + 2) mod 32; increment by 1 modulo 32 until v3 differs from v0,v1,v2",
            "D": "D = (seed + sum(valores_activos) + 1) mod 32; increment by 1 modulo 32 until D differs from every active value",
        },
        "stages": {
            "A": {"fresh_initialization": True, "active_roles": list(ACTIVE_ROLES), "atomic_rows": 128, "role_role_terms": 12, "role_bg_terms": 4, "background_contexts": 40, "reduction": "/16", "updates": 5000, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear"}, "checkpoint_selection": "last update; no evaluation selection", "training_multi_clause": 0},
            "B": {"core_frozen": True, "sources": ["128 atomic active-role relations", "40 base BG contexts"], "formula": "b_r = N_r_max + (P_r_min - N_r_max)/2", "storage_dtype": "inference dtype", "noop_contexts": False, "recalibration_after_C": False},
            "C": {"trainable_parameter": "e_N only", "dimension": 16, "objective": "(1/136) * sum_{r=1}^{4} sum_{z in U} softplus(s_r(z; e_N) - b_r)", "contexts": 34, "updates": 5000, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear"}, "initialization": "normal standard", "generator_seed": 100000 + seed, "core_and_thresholds_frozen": True},
        },
        "evaluation_plan": {"base_by_size": {str(k): EXPECTED_BASE[k] for k in EXPECTED_BASE}, "augmented_by_size": {str(k): EXPECTED_AUGMENTED[k] for k in EXPECTED_AUGMENTED}, "base_total": 59648, "augmented_total": 250240, "base_singleton_controls": 128, "base_multi_clause": 59520, "augmented_exactly_one_noop": True, "pair_metric_reuses_base_output": True, "subset_registry": subset_registry},
        "assignment_registry": {"random_seed": 7800, "all_count": 120, "selected_count": 5, "selected": list(SELECTED_ASSIGNMENTS), "omitted_count": 115, "omitted": [assignment for assignment in ALL_ASSIGNMENTS if assignment not in SELECTED_ASSIGNMENTS]},
        "checker_contract": {"reference_implementation": "independent_derived_value/independent_distractor in fresh checker", "forward_inputs": ["token_ids", "lengths"], "active_roles_model_input": False, "fallback_identity_permutation": False, "regression": {"seed": 7701, "values": [0, 1], "v2": 23, "v3": 15}},
        "training_performed": False,
        "model_constructed": False,
        "calibration_performed": False,
        "new_checkpoints": False,
    }


def math_factorial(value: int) -> int:
    result = 1
    for item in range(2, value + 1):
        result *= item
    return result


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json_bytes(path: Path, value: Any) -> None:
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_self_hashed(path: Path, value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned["artifact_self_hash"] = PLACEHOLDER
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    value["artifact_self_hash"] = digest
    write_json_bytes(path, value)
    return digest


def verify_self_hash_raw(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    stored = str(parsed["artifact_self_hash"])
    needle = stored.encode("utf-8")
    if data.count(needle) != 1:
        raise RuntimeError(f"self-hash field occurrence mismatch: {path}")
    verified = sha256_bytes(data.replace(needle, PLACEHOLDER.encode("utf-8"), 1))
    if verified != stored:
        raise RuntimeError(f"self-hash mismatch: {path}: {stored} != {verified}")
    return stored, sha256_bytes(data)


def writer_probe() -> dict[str, Any]:
    path = OUTPUT_ROOT / "writer_byte_exact_probe.json"
    payload = {"schema": "byte-exact-writer-probe-v1", "newline": "LF", "artifact_self_hash": PLACEHOLDER}
    unsigned = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected = sha256_bytes(unsigned)
    payload["artifact_self_hash"] = expected
    final = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes(path, final)
    actual = path.read_bytes()
    stored = json.loads(actual.decode("utf-8"))["artifact_self_hash"]
    raw_unsigned = actual.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)
    verified = sha256_bytes(raw_unsigned)
    if stored != expected or stored != verified or actual != final:
        raise RuntimeError("byte-exact writer probe failed")
    return {"path": path.relative_to(ROOT).as_posix(), "artifact_self_hash": stored, "file_sha256": sha256_bytes(actual), "verified": True, "write_method": "write_bytes(UTF-8)", "verification_method": "raw byte replacement; no JSON reserialization", "line_endings": {"CRLF": actual.count(b"\r\n"), "LF": actual.count(b"\n")}}


def collect_prior_physical_ids(value: Any, found: set[int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"token_ids", "physical_ids", "physical_token_ids"}:
                if isinstance(child, dict):
                    found.update(item for item in child.values() if isinstance(item, int))
                elif isinstance(child, list):
                    found.update(item for item in child if isinstance(item, int))
            else:
                collect_prior_physical_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            collect_prior_physical_ids(child, found)


def prior_snapshot() -> dict[str, Any]:
    paths = sorted((ROOT / "campaign").glob("**/cases.jsonl"))
    selected = [path for path in paths if "t7_noop_none_lexical_paired_development" in path.as_posix()]
    records = []
    for path in selected:
        records.append({**source_record(path), "preserved": True})
    prior_ids: set[int] = set()
    for path in (ROOT / "campaign").glob("**/*.json"):
        if OUTPUT_ROOT in path.parents:
            continue
        try:
            collect_prior_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return {"cases_jsonl": records, "cases_count": len(records), "all_prior_physical_id_count": len(prior_ids), "prior_physical_ids": prior_ids}


def independent_checker(seed: int, manifest: dict[str, Any], prior_ids: set[int]) -> dict[str, Any]:
    failures: list[str] = []
    base_ids = {token: index for index, token in enumerate(manifest["token_order"])}
    augmented_ids = {**base_ids, manifest["noop_operator"]: 37}
    if len(manifest["permutation"]) != 32 or sorted(manifest["permutation"]) != list(range(32)):
        failures.append("permutation is not exact 0..31")
    inverse = {value: index for index, value in enumerate(manifest["permutation"])}
    if manifest["inverse_permutation"] != {str(value): index for value, index in inverse.items()}:
        failures.append("inverse permutation mismatch")
    expected_noop = manifest["id_block"][1] + 1
    if expected_noop in prior_ids or any(value in prior_ids for value in manifest["token_ids"].values()):
        failures.append("physical ID overlaps prior campaign")
    if len(set(manifest["token_ids"].values())) != 37 or expected_noop in manifest["token_ids"].values():
        failures.append("physical ID uniqueness/reservation")
    base_counts = {size: 0 for size in range(1, 5)}
    augmented_counts = {size: 0 for size in range(1, 5)}
    seen_base: set[tuple[str, ...]] = set()
    seen_augmented: set[tuple[str, ...]] = set()
    expected_pairs = 0
    for size in range(1, 5):
        for subset in itertools.combinations(ACTIVE_ROLES, size):
            base_cases, augmented_cases = corrected_evaluator.build_cases_for_subset(seed, subset, base_ids, augmented_ids, manifest["operator_for_role"], manifest["noop_operator"], manifest["permutation"])
            expected_assignments = independent_assignments(seed, subset)
            actual_assignments = [tuple(case.assignment_dict[role] for role in subset) for case in base_cases[::math_factorial(size)]]
            if actual_assignments != expected_assignments:
                failures.append(f"exact assignments mismatch:{'_'.join(subset)}")
            base_by_pair = {case.pair_id: case for case in base_cases}
            for case in base_cases:
                base_counts[size] += 1
                if case.tokens in seen_base:
                    failures.append(f"duplicate base instruction:{case.case_id}")
                seen_base.add(case.tokens)
                assignment = case.assignment_dict
                for index, role in enumerate(case.order):
                    offset = index * 3
                    if case.tokens[offset] != manifest["operator_for_role"][role] or case.tokens[offset + 1] != f"ARG_{inverse[assignment[role]]:02d}":
                        failures.append(f"base clause mapping:{case.case_id}")
                        break
            for case in augmented_cases:
                augmented_counts[size] += 1
                if case.tokens in seen_augmented:
                    failures.append(f"duplicate augmented instruction:{case.case_id}")
                seen_augmented.add(case.tokens)
                assignment = case.assignment_dict
                expected_d = independent_distractor(seed, [assignment[role] for role in subset])
                if case.distractor != expected_d or case.tokens.count(manifest["noop_operator"]) != 1 or case.distractor in assignment.values():
                    failures.append(f"D/NOOP mismatch:{case.case_id}")
                active_order = tuple(role for role in case.order if role != "NOOP")
                paired = base_by_pair.get(case.pair_id)
                if paired is None or paired.order != active_order or paired.assignment_dict != assignment:
                    failures.append(f"pair identity mismatch:{case.case_id}")
                clauses = [(case.order[index], case.tokens[index * 3 + 1]) for index in range(size + 1)]
                clauses = [(role, argument) for role, argument in clauses if role != "NOOP"]
                rebuilt: list[str] = []
                for index, (role, argument) in enumerate(clauses):
                    if index:
                        rebuilt.append("LINK")
                    rebuilt.extend((manifest["operator_for_role"][role], argument))
                if paired is None or tuple(rebuilt) != paired.tokens:
                    failures.append(f"pair reconstruction mismatch:{case.case_id}")
                expected_pairs += 1
    if base_counts != EXPECTED_BASE or augmented_counts != EXPECTED_AUGMENTED:
        failures.append(f"counts mismatch:{base_counts}/{augmented_counts}")
    if independent_derived_value(7701, [0, 1]) != 23 or independent_derived_value(7701, [0, 1, 23]) != 15:
        failures.append("regression v2/v3 failed")
    return {"status": "passed" if not failures else "failed", "seed": seed, "base_counts": base_counts, "augmented_counts": augmented_counts, "base_total": sum(base_counts.values()), "augmented_total": sum(augmented_counts.values()), "pairs": expected_pairs, "unique_base_instructions": len(seen_base), "unique_augmented_instructions": len(seen_augmented), "regression": {"seed": 7701, "values": [0, 1], "v2": independent_derived_value(7701, [0, 1]), "v3": independent_derived_value(7701, [0, 1, 23])}, "failures": failures}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty fresh preparation root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prior_full = prior_snapshot()
    prior_ids = prior_full["prior_physical_ids"]
    prior = {key: value for key, value in prior_full.items() if key != "prior_physical_ids"}
    manifests = {}
    manifest_records = []
    for seed, mapping in zip(SEEDS, SELECTED_ASSIGNMENTS):
        path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
        write_json_bytes(path, build_manifest(seed, mapping))
        manifests[seed] = (json.loads(path.read_text(encoding="utf-8")), path)
        manifest_records.append(source_record(path) | {"seed": seed})
    checker_results = {str(seed): independent_checker(seed, manifests[seed][0], prior_ids) for seed in SEEDS}
    checker_value = {"schema": "T7-noop-none-lexical-fresh-independent-checker-v1", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION", "reference": "independent formula implementation; does not import evaluator formula", "regression": {"seed": 7701, "values": [0, 1], "v2": 23, "v3": 15}, "manifests": checker_results, "status": "passed" if all(item["status"] == "passed" for item in checker_results.values()) else "failed", "training_performed": False, "model_constructed": False, "calibration_performed": False}
    checker_path = OUTPUT_ROOT / "independent_checker.json"
    checker_hash = write_self_hashed(checker_path, checker_value)
    probe = writer_probe()
    if checker_value["status"] != "passed":
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION", "reason": "independent checker failed before any model work", "checker": checker_results, "checker_self_hash": checker_hash, "writer_probe": probe}
        write_json_bytes(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2
    source_paths = [
        ROOT / "scripts" / "t5_nrole_design_audit.py",
        ROOT / "t1_trainability" / "t7_production_core_noop_integration.py",
        ROOT / "scripts" / "execute_t7_noop_none_stage_a_development.py",
        ROOT / "scripts" / "execute_t7_noop_none_stage_b_development.py",
        ROOT / "scripts" / "execute_t7_noop_none_stage_c_development.py",
        ROOT / "scripts" / "execute_t6_activeset_midpoint_development.py",
        ROOT / "scripts" / "ctrl2_common.py",
        ROOT / "scripts" / "evaluate_t7_noop_none_lexical_paired_development.py",
        ROOT / "scripts" / "evaluate_t7_noop_none_lexical_paired_development_conformance.py",
        Path(__file__).resolve(),
    ]
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    environment = {"python": sys.version, "python_executable": "redacted", "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "implementation": platform.python_implementation(), "git_head": git_commit}
    before_prior = prior
    after_full = prior_snapshot()
    after_prior = {key: value for key, value in after_full.items() if key != "prior_physical_ids"}
    preservation = {"before": before_prior, "after": after_prior, "unchanged": before_prior == after_prior, "not_uploaded": True}
    freeze = {"schema": "T7-noop-none-lexical-fresh-preparation-v1", "status": "prepared", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION", "reference_valid_development": {"commit": "2769776e", "path": "campaign/t7_noop_none_lexical_paired_development_conformance/", "not_prior_plus2_reference": True}, "seeds": list(SEEDS), "selected_assignments": list(SELECTED_ASSIGNMENTS), "omitted_assignments": [assignment for assignment in ALL_ASSIGNMENTS if assignment not in SELECTED_ASSIGNMENTS], "manifests": manifest_records, "independent_checker": {"path": checker_path.relative_to(ROOT).as_posix(), "self_hash": checker_hash, "status": checker_value["status"]}, "byte_exact_writer_probe": probe, "source_registry": [source_record(path) for path in source_paths], "environment": environment, "prior_attempt_preservation": preservation, "scope": {"training_performed": False, "model_constructed": False, "calibration_performed": False, "e_N_initialized": False, "new_checkpoints": False, "old_770x_artifacts_touched": False, "old_770x_cases_uploaded": False}, "formulas": {"v2": "v2 = (seed + v0 + v1 + 1) mod 32; increment by 1 modulo 32 until collision-free", "v3": "v3 = (seed + v0 + v1 + v2 + 2) mod 32; increment by 1 modulo 32 until collision-free", "D": "D = (seed + sum(valores_activos) + 1) mod 32; increment by 1 modulo 32 until collision-free"}, "evaluation_totals": {"base": 298240, "augmented": 1251200, "combined": 1549440, "base_singleton_controls": 640, "base_multi_clause": 297600, "augmented_pairs": 1251200}, "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False}
    result_path = OUTPUT_ROOT / "results.json"
    result_hash = write_self_hashed(result_path, freeze)
    stored_hash, file_hash = verify_self_hash_raw(result_path)
    if stored_hash != result_hash:
        raise RuntimeError("consolidated result self-hash verification failed")
    print(json.dumps({"status": freeze["status"], "artifact": result_path.relative_to(ROOT).as_posix(), "artifact_self_hash": stored_hash, "file_sha256": file_hash, "checker": checker_value["status"], "manifests": manifest_records, "training_performed": False, "model_constructed": False, "calibration_performed": False, "e_N_initialized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
