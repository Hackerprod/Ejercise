"""Independent reconstruction audit for the sealed T7 paired-fresh JSONL evidence.

This audit never reads evaluator-produced correctness, margin, or semantic-signature
fields as answers. It reconstructs expected tokens, pointers, presence, RAW/CANON,
and pair signatures from each manifest plus the recorded output fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRESH_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_paired_fresh"
MANIFEST_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_fresh_preparation" / "manifests"
CALIBRATION_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_fresh" / "calibrations"
OUTPUT = ROOT / "campaign" / "t7_final_closure" / "audit_t7_paired_fresh_reconstructed.json"
SEEDS = (7801, 7802, 7803, 7804, 7805)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
EXPECTED_BASE_BY_SIZE = {1: 128, 2: 11904, 3: 23808, 4: 23808}
EXPECTED_AUGMENTED_BY_SIZE = {1: 256, 2: 35712, 3: 95232, 4: 119040}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def manifest_for(seed: int) -> tuple[Path, dict[str, Any]]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    return path, load_json(path)


def calibration_for(seed: int) -> tuple[Path, dict[str, Any]]:
    path = CALIBRATION_ROOT / f"calibration_{seed}_v1.json"
    return path, load_json(path)


def expected_tokens(manifest: dict[str, Any], record: dict[str, Any]) -> list[str]:
    assignment = record["assignment"]
    inverse = manifest["inverse_permutation"]
    order = record["order"].split("_")
    clauses: list[list[str]] = []
    for name in order:
        if name == "NOOP":
            value = int(record["D"])
            operator = manifest["noop_operator"]
        else:
            value = int(assignment[name])
            operator = manifest["operator_for_role"][name]
        argument_index = int(inverse[str(value)])
        clauses.append([operator, f"ARG_{argument_index:02d}"])
    tokens: list[str] = []
    for index, clause in enumerate(clauses):
        if index:
            tokens.append("LINK")
        tokens.extend(clause)
    return tokens


def reconstruct_record(manifest: dict[str, Any], thresholds: list[float], record: dict[str, Any]) -> tuple[tuple[tuple[Any, ...], ...], dict[str, int], dict[str, dict[str, Any]]]:
    subset = tuple(record["subset"].split("_"))
    if tuple(sorted(subset, key=ROLES.index)) != subset or not subset or any(role not in ROLES for role in subset):
        raise ValueError(f"invalid subset: {record.get('subset')}")
    if int(record["subset_size"]) != len(subset):
        raise ValueError(f"subset size mismatch: {record.get('case_id')}")
    if bool(record["augmented"]) != (record["kind"] == "augmented"):
        raise ValueError(f"kind/augmented mismatch: {record.get('case_id')}")
    actual_tokens = list(record["tokens"])
    if actual_tokens != expected_tokens(manifest, record):
        raise ValueError(f"instruction token mismatch: {record.get('case_id')}")
    order = record["order"].split("_")
    if len(order) != len(set(order)) or any(name not in (*ROLES, "NOOP") for name in order):
        raise ValueError(f"invalid order: {record.get('case_id')}")
    if record["kind"] == "base" and "NOOP" in order:
        raise ValueError(f"base contains NOOP: {record.get('case_id')}")
    if record["kind"] == "augmented" and order.count("NOOP") != 1:
        raise ValueError(f"augmented NOOP count mismatch: {record.get('case_id')}")
    assignment = record["assignment"]
    inverse = manifest["inverse_permutation"]
    signature: list[tuple[Any, ...]] = []
    counters = Counter(
        present_roles=0,
        absent_roles=0,
        present_pointer_pass=0,
        present_raw_pass=0,
        present_canon_pass=0,
        present_margins_recomputed_from_scores=0,
        absent_margin_sign_checked_from_recorded_numeric_margin=0,
        reconstructed_role_pass=0,
    )
    witnesses: dict[str, dict[str, Any]] = {}
    for role_index, role in enumerate(ROLES):
        output = record["roles"][role_index]
        active = role in subset
        expected_value = int(assignment[role]) if active else -1
        expected_argument = f"ARG_{int(inverse[str(expected_value)]):02d}" if active else ""
        if active:
            counters["present_roles"] += 1
            operator = manifest["operator_for_role"][role]
            expected_position = actual_tokens.index(expected_argument)
            if expected_position == 0 or actual_tokens[expected_position - 1] != operator:
                raise ValueError(f"argument pointer context mismatch: {record.get('case_id')}:{role}")
            pointer_pass = int(output["pointer"]) == expected_position and int(output["target_position"]) == expected_position and output["selected_token"] == expected_argument
            raw_pass = int(output["RAW"]) == expected_value
            canon_pass = int(output["CANON"]) == expected_value
            if not (pointer_pass and raw_pass and canon_pass and output["presence"] == "present" and bool(output["expected_present"])):
                raise ValueError(f"reconstructed present output mismatch: {record.get('case_id')}:{role}")
            counters["present_pointer_pass"] += int(pointer_pass)
            counters["present_raw_pass"] += int(raw_pass)
            counters["present_canon_pass"] += int(canon_pass)
            if not (finite(output["target_score"]) and finite(output["max_other_score"])):
                raise ValueError(f"missing present scores: {record.get('case_id')}:{role}")
            margin_null = float(output["target_score"]) - float(thresholds[role_index])
            margin_real = float(output["target_score"]) - float(output["max_other_score"])
            if not (margin_null > 0.0 and margin_real > 0.0):
                raise ValueError(f"present margin sign failure: {record.get('case_id')}:{role}")
            counters["present_margins_recomputed_from_scores"] += 1
            witness = {"case_id": record["case_id"], "pair_id": record["pair_id"], "subset": record["subset"], "order": record["order"], "role": role, "D": record["D"], "assignment": assignment, "tokens": actual_tokens, "target_position": expected_position}
            witnesses.setdefault("present_to_NULL", {**witness, "margin": margin_null})
            if margin_null < witnesses["present_to_NULL"]["margin"]:
                witnesses["present_to_NULL"] = {**witness, "margin": margin_null}
            witnesses.setdefault("present_to_real", {**witness, "margin": margin_real})
            if margin_real < witnesses["present_to_real"]["margin"]:
                witnesses["present_to_real"] = {**witness, "margin": margin_real}
            if record["kind"] == "augmented" and finite(output["noop_rejection_margin"]):
                noop_margin = float(output["noop_rejection_margin"])
                witnesses.setdefault("noop_rejection", {**witness, "margin": noop_margin})
                if noop_margin < witnesses["noop_rejection"]["margin"]:
                    witnesses["noop_rejection"] = {**witness, "margin": noop_margin}
            signature.append(("present", expected_argument, expected_value, expected_value))
        else:
            counters["absent_roles"] += 1
            if not (output["presence"] == "none" and not bool(output["expected_present"]) and int(output["pointer"]) == -1 and int(output["target_position"]) == -1 and output["selected_token"] == "" and int(output["RAW"]) == -1 and int(output["CANON"]) == -1):
                raise ValueError(f"reconstructed absent output mismatch: {record.get('case_id')}:{role}")
            if not finite(output["margin_absent"]) or float(output["margin_absent"]) <= 0.0:
                raise ValueError(f"absent margin sign failure: {record.get('case_id')}:{role}")
            counters["absent_margin_sign_checked_from_recorded_numeric_margin"] += 1
            absent_margin = float(output["margin_absent"])
            witness = {"case_id": record["case_id"], "pair_id": record["pair_id"], "subset": record["subset"], "order": record["order"], "role": role, "D": record["D"], "assignment": assignment, "tokens": actual_tokens, "margin": absent_margin}
            witnesses.setdefault("absent", witness)
            if absent_margin < witnesses["absent"]["margin"]:
                witnesses["absent"] = witness
            if record["kind"] == "augmented" and finite(output["noop_rejection_margin"]):
                noop_margin = float(output["noop_rejection_margin"])
                noop_witness = {**witness, "margin": noop_margin}
                witnesses.setdefault("noop_rejection", noop_witness)
                if noop_margin < witnesses["noop_rejection"]["margin"]:
                    witnesses["noop_rejection"] = noop_witness
            signature.append(("none", "", -1, -1))
        counters["reconstructed_role_pass"] += 1
    return tuple(signature), dict(counters), witnesses


def audit_seed(seed: int, manifest: dict[str, Any], thresholds: list[float], cases_path: Path) -> dict[str, Any]:
    base_pairs: dict[str, tuple[tuple[Any, ...], ...]] = {}
    case_ids: set[str] = set()
    pair_ids: set[str] = set()
    augmented_pair_ids: set[str] = set()
    augmented_pair_instances: set[tuple[str, str]] = set()
    counts: Counter[tuple[str, str, int]] = Counter()
    totals = Counter()
    metrics = Counter()
    duplicate_case_ids = 0
    duplicate_base_pairs = 0
    duplicate_augmented_pair_instances = 0
    missing_base_pairs = 0
    pair_mismatches: list[str] = []
    minima: dict[str, dict[str, Any]] = {}
    with cases_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if int(record["seed"]) != seed:
                raise ValueError(f"seed mismatch at {cases_path}:{line_number}")
            case_id = str(record["case_id"])
            if case_id in case_ids:
                duplicate_case_ids += 1
            case_ids.add(case_id)
            kind = str(record["kind"])
            if kind not in {"base", "augmented"}:
                raise ValueError(f"unknown kind at {cases_path}:{line_number}")
            signature, local_metrics, local_witnesses = reconstruct_record(manifest, thresholds, record)
            for metric, witness in local_witnesses.items():
                if metric not in minima or witness["margin"] < minima[metric]["margin"]:
                    minima[metric] = witness
            subset = str(record["subset"])
            size = int(record["subset_size"])
            counts[(kind, subset, size)] += 1
            totals[kind] += 1
            metrics.update(local_metrics)
            pair_id = str(record["pair_id"])
            if kind == "base":
                if pair_id in pair_ids:
                    duplicate_base_pairs += 1
                pair_ids.add(pair_id)
                base_pairs[pair_id] = signature
            else:
                if pair_id not in base_pairs:
                    missing_base_pairs += 1
                    if len(pair_mismatches) < 20:
                        pair_mismatches.append(f"missing base: {pair_id}")
                else:
                    pair_instance = (pair_id, str(record["order"]))
                    if pair_instance in augmented_pair_instances:
                        duplicate_augmented_pair_instances += 1
                    augmented_pair_instances.add(pair_instance)
                    augmented_pair_ids.add(pair_id)
                    if signature != base_pairs[pair_id]:
                        if len(pair_mismatches) < 20:
                            pair_mismatches.append(f"reconstructed semantic mismatch: {pair_id}")
                    else:
                        metrics["paired_reconstructed_semantic_pass"] += 1
    expected_counts: dict[str, int] = {}
    count_pass = True
    for size in range(1, 5):
        subsets = [tuple(role for role in ROLES if role in subset.split("_")) for subset in ("_".join(combo) for count in [size] for combo in __import__("itertools").combinations(ROLES, count))]
        for subset_tuple in subsets:
            subset = "_".join(subset_tuple)
            expected_base = EXPECTED_BASE_BY_SIZE[size] // len(list(__import__("itertools").combinations(ROLES, size)))
            expected_augmented = EXPECTED_AUGMENTED_BY_SIZE[size] // len(list(__import__("itertools").combinations(ROLES, size)))
            expected_counts[f"base:{subset}"] = expected_base
            expected_counts[f"augmented:{subset}"] = expected_augmented
            count_pass = count_pass and counts[("base", subset, size)] == expected_base and counts[("augmented", subset, size)] == expected_augmented
    passed = (
        duplicate_case_ids == 0 and duplicate_base_pairs == 0 and missing_base_pairs == 0 and
        duplicate_augmented_pair_instances == 0 and augmented_pair_ids == pair_ids and
        totals["base"] == sum(EXPECTED_BASE_BY_SIZE.values()) and totals["augmented"] == sum(EXPECTED_AUGMENTED_BY_SIZE.values()) and
        metrics["paired_reconstructed_semantic_pass"] == totals["augmented"] and not pair_mismatches and count_pass
    )
    return {
        "seed": seed,
        "status": "passed" if passed else "failed",
        "cases": source_record(cases_path),
        "counts": {"base": totals["base"], "augmented": totals["augmented"], "pairs_reconstructed": metrics["paired_reconstructed_semantic_pass"]},
        "expected_counts": {"base": sum(EXPECTED_BASE_BY_SIZE.values()), "augmented": sum(EXPECTED_AUGMENTED_BY_SIZE.values()), "pairs": sum(EXPECTED_AUGMENTED_BY_SIZE.values())},
        "counts_by_subset": {f"{kind}:{subset}": counts[(kind, subset, size)] for kind, subset, size in sorted(counts)},
        "expected_counts_by_subset": expected_counts,
        "reconstruction_metrics": dict(metrics),
        "minimum_margin_witnesses": minima,
        "uniqueness": {"case_ids": len(case_ids), "base_pair_ids": len(pair_ids), "augmented_pair_ids": len(augmented_pair_ids), "augmented_pair_instances": len(augmented_pair_instances), "duplicate_case_ids": duplicate_case_ids, "duplicate_base_pairs": duplicate_base_pairs, "duplicate_augmented_pair_instances": duplicate_augmented_pair_instances, "missing_base_pairs": missing_base_pairs, "base_augmented_pair_sets_identical": augmented_pair_ids == pair_ids, "pair_id_reuse_is_expected_for_augmented_insertion_orders": True},
        "pair_mismatches_sample": pair_mismatches,
        "method_scope": {"structured_correct_used": False, "margins_strict_used": False, "semantic_signature_used": False, "present_margins": "present_margins_recomputed_from_scores", "absent_margins": "absent_margin_sign_checked_from_recorded_numeric_margin; weaker because best_score is not recorded and no score reconstruction was attempted"},
    }


def write_self_hashed(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    unsigned_bytes = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(unsigned_bytes).hexdigest()
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return digest, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    runner_hash = sha256_file(Path(__file__).resolve())
    per_seed: list[dict[str, Any]] = []
    all_pass = True
    for seed in SEEDS:
        manifest_path, manifest = manifest_for(seed)
        calibration_path, calibration = calibration_for(seed)
        thresholds = [float(calibration["calibration"]["intervals"][role]["b_r_stored"]) for role in ROLES]
        cases_path = FRESH_ROOT / f"seed_{seed}" / "cases.jsonl"
        result = audit_seed(seed, manifest, thresholds, cases_path)
        result["manifest"] = source_record(manifest_path)
        result["calibration"] = source_record(calibration_path)
        result["thresholds_role_order"] = thresholds
        per_seed.append(result)
        all_pass = all_pass and result["status"] == "passed"
    payload = {
        "schema": "t7-paired-fresh-reconstructed-audit-v1",
        "status": "completed",
        "classification": "PASS_RECONSTRUCTED" if all_pass else "REAL_DISCREPANCY_FOUND",
        "runner": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256_before_audit": runner_hash},
        "scope": "Read-only audit of existing paired-fresh JSONL; zero model construction, forwards, training, recalibration, manifests, or checkpoints.",
        "independently_reconstructed": {"presence": True, "argument_position_from_inverse_permutation": True, "pointer": True, "RAW": True, "CANON": True, "semantic_signature_from_reconstructed_outputs": True},
        "margin_scope": {"present": "present_margins_recomputed_from_scores", "absent": "absent_margin_sign_checked_from_recorded_numeric_margin; weaker and explicitly non-circular because absent best_score is unavailable"},
        "global_expected": {"base": sum(EXPECTED_BASE_BY_SIZE.values()) * len(SEEDS), "augmented": sum(EXPECTED_AUGMENTED_BY_SIZE.values()) * len(SEEDS), "pairs": sum(EXPECTED_AUGMENTED_BY_SIZE.values()) * len(SEEDS)},
        "global_observed": {"base": sum(item["counts"]["base"] for item in per_seed), "augmented": sum(item["counts"]["augmented"] for item in per_seed), "pairs_reconstructed": sum(item["counts"]["pairs_reconstructed"] for item in per_seed)},
        "real_discrepancy": False if all_pass else True,
        "per_seed": per_seed,
    }
    digest, file_sha = write_self_hashed(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "classification": payload["classification"], "artifact": OUTPUT.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": file_sha, "real_discrepancy": payload["real_discrepancy"]}, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
