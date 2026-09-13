"""Repair T7 pair-generation conformance, then rerun the sealed evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

import torch

import evaluate_t7_noop_none_lexical_paired_development as evaluator


ROOT = evaluator.ROOT
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_paired_development_conformance"
PRIOR_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_paired_development"
PRIOR_COMMIT = "60f8d863d13eafefc812755dfc086c3f28ec023d"
PLACEHOLDER = "__SELF_HASH__"


def independent_derived_value(seed: int, values: list[int]) -> int:
    offset = 1 if len(values) == 2 else 2
    candidate = (seed + sum(values) + offset) % evaluator.VALUE_COUNT
    while candidate in values:
        candidate = (candidate + 1) % evaluator.VALUE_COUNT
    return candidate


def independent_distractor(seed: int, values: list[int]) -> int:
    candidate = (seed + sum(values) + 1) % evaluator.VALUE_COUNT
    while candidate in values:
        candidate = (candidate + 1) % evaluator.VALUE_COUNT
    return candidate


def independent_assignments(seed: int, subset: tuple[str, ...]) -> list[tuple[int, ...]]:
    if len(subset) <= 2:
        return list(itertools.permutations(range(evaluator.VALUE_COUNT), len(subset)))
    expected: list[tuple[int, ...]] = []
    for first_two in itertools.permutations(range(evaluator.VALUE_COUNT), 2):
        values = list(first_two)
        while len(values) < len(subset):
            values.append(independent_derived_value(seed, values))
        expected.append(tuple(values))
    return expected


def self_hashed(path: Path, value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned["artifact_self_hash"] = PLACEHOLDER
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    value["artifact_self_hash"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return digest


def conformance_checker(seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    base_ids = {token: index for index, token in enumerate(manifest["token_order"])}
    augmented_ids = {**base_ids, manifest["noop_operator"]: 37}
    permutation = [int(value) for value in manifest["permutation"]]
    inverse = {value: index for index, value in enumerate(permutation)}
    failures: list[str] = []
    base_total = augmented_total = 0
    base_by_pair: dict[str, evaluator.Case] = {}
    seen_base: set[tuple[str, ...]] = set()
    seen_augmented: set[tuple[str, ...]] = set()
    for size in range(1, 5):
        for subset in itertools.combinations(evaluator.ROLES, size):
            base_cases, augmented_cases = evaluator.build_cases_for_subset(seed, subset, base_ids, augmented_ids, manifest["operator_for_role"], manifest["noop_operator"], permutation)
            expected_assignments = independent_assignments(seed, subset)
            actual_assignments = [tuple(case.assignment_dict[role] for role in subset) for case in base_cases[::len(tuple(itertools.permutations(subset)))] ]
            if actual_assignments != expected_assignments:
                failures.append(f"exact assignment mismatch {seed}:{'_'.join(subset)}")
            for case in base_cases:
                base_total += 1
                base_by_pair[case.pair_id] = case
                if case.tokens in seen_base:
                    failures.append(f"duplicate base instruction {case.case_id}")
                seen_base.add(case.tokens)
                assignment = case.assignment_dict
                for index, role in enumerate(case.order):
                    token_offset = index * 3
                    if case.tokens[token_offset] != manifest["operator_for_role"][role] or case.tokens[token_offset + 1] != f"ARG_{inverse[assignment[role]]:02d}":
                        failures.append(f"base clause mapping {case.case_id}")
                        break
                if len(case.tokens) != 3 * size - 1:
                    failures.append(f"base clause shape {case.case_id}")
            for case in augmented_cases:
                augmented_total += 1
                if case.tokens in seen_augmented:
                    failures.append(f"duplicate augmented instruction {case.case_id}")
                seen_augmented.add(case.tokens)
                assignment = case.assignment_dict
                expected_d = independent_distractor(seed, [assignment[role] for role in subset])
                if case.distractor != expected_d or case.tokens.count(manifest["noop_operator"]) != 1:
                    failures.append(f"D/noop mismatch {case.case_id}")
                if len(case.tokens) != 3 * (size + 1) - 1:
                    failures.append(f"augmented clause shape {case.case_id}")
                for index, role in enumerate(case.order):
                    token_offset = index * 3
                    expected_operator = manifest["noop_operator"] if role == "NOOP" else manifest["operator_for_role"][role]
                    expected_value = expected_d if role == "NOOP" else assignment[role]
                    if case.tokens[token_offset] != expected_operator or case.tokens[token_offset + 1] != f"ARG_{inverse[expected_value]:02d}":
                        failures.append(f"augmented clause mapping {case.case_id}")
                        break
                active_order = tuple(role for role in case.order if role != "NOOP")
                matching = [base_by_pair[case.pair_id]] if case.pair_id in base_by_pair else []
                if len(matching) != 1 or matching[0].order != active_order or matching[0].distractor != case.distractor:
                    failures.append(f"pair/order mismatch {case.case_id}")
                if case.distractor in assignment.values():
                    failures.append(f"D collision {case.case_id}")
    expected_base = {1: 128, 2: 11904, 3: 23808, 4: 23808}
    expected_augmented = {1: 256, 2: 35712, 3: 95232, 4: 119040}
    counts_base = {size: 0 for size in range(1, 5)}
    counts_augmented = {size: 0 for size in range(1, 5)}
    for case in base_by_pair.values():
        counts_base[len(case.subset)] += 1
    # Recount augmented independently because base_by_pair intentionally has one row per pair.
    for size in range(1, 5):
        for subset in itertools.combinations(evaluator.ROLES, size):
            _, augmented_cases = evaluator.build_cases_for_subset(seed, subset, base_ids, augmented_ids, manifest["operator_for_role"], manifest["noop_operator"], permutation)
            counts_augmented[size] += len(augmented_cases)
    if counts_base != expected_base or counts_augmented != expected_augmented:
        failures.append(f"count mismatch {seed}: {counts_base}/{counts_augmented}")
    regression = {"v2": independent_derived_value(7701, [0, 1]), "v3": independent_derived_value(7701, [0, 1, 23]), "expected": {"v2": 23, "v3": 15}}
    if regression["v2"] != 23 or regression["v3"] != 15:
        failures.append("independent formula regression")
    return {"status": "passed" if not failures else "failed", "seed": seed, "base_counts": counts_base, "augmented_counts": counts_augmented, "base_total": base_total, "augmented_total": augmented_total, "unique_base_instructions": len(seen_base), "unique_augmented_instructions": len(seen_augmented), "regression": regression, "failures": failures}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty conformance output root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifests = {seed: evaluator.load_t7_manifest(seed) for seed in evaluator.SEEDS}
    prior_result_path = PRIOR_ROOT / "results.json"
    prior_result = json.loads(prior_result_path.read_text(encoding="utf-8")) if prior_result_path.exists() else {}
    specification = {"schema": "T7-noop-none-lexical-paired-development-conformance-v1", "task": "T7-PAIRED-DEVELOPMENT-CONFORMANCE-REPAIR", "prior_attempt_commit": PRIOR_COMMIT, "prior_attempt_artifact": evaluator.source_record(prior_result_path), "prior_attempt_self_hash": prior_result.get("artifact_self_hash"), "formulas": {"v2": "(seed + v0 + v1 + 1) mod 32; increment by 1 modulo 32 until distinct", "v3": "(seed + v0 + v1 + v2 + 2) mod 32; increment by 1 modulo 32 until distinct", "D": "(seed + sum(active values) + 1) mod 32; increment by 1 modulo 32 until distinct"}, "sealed_inputs": {"seeds": list(evaluator.SEEDS), "roles": list(evaluator.ROLES), "preserve_manifests": True, "preserve_prior_results": True, "checkpoints": "Stage C final only", "training": False, "recalibration": False}, "regression": {"seed": 7701, "values": [0, 1], "v2": 23, "v3": 15}}
    spec_hash = self_hashed(OUTPUT_ROOT / "conformance_specification.json", specification)
    checker_results = {str(seed): conformance_checker(seed, manifests[seed][0]) for seed in evaluator.SEEDS}
    if not all(item["status"] == "passed" for item in checker_results.values()):
        invalid = {"status": "INVALID/HARNESS BUG", "task": specification["task"], "conformance_specification": evaluator.source_record(OUTPUT_ROOT / "conformance_specification.json"), "conformance_specification_self_hash": spec_hash, "logical_checker": checker_results}
        evaluator.write_json(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2
    executor = evaluator.load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    runtime_before = {"file_sha256": evaluator.sha256_file(evaluator.BASE_CHECKPOINT), "tensor_state_hash": evaluator.state_hash(executor.state_dict())}
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(evaluator.VALUE_BASE, evaluator.VALUE_BASE + evaluator.VALUE_COUNT, dtype=torch.long)).detach()
    runtime_before["codebook_hash"] = evaluator.tensor_digest({"codebook": codebook})
    seed_reports: list[dict[str, Any]] = []
    try:
        for seed in evaluator.SEEDS:
            manifest, manifest_path = manifests[seed]
            core, identity = evaluator.load_verified_core(seed, manifest, manifest_path)
            base_ids = {token: index for index, token in enumerate(manifest["token_order"])}
            augmented_ids = {**base_ids, manifest["noop_operator"]: 37}
            seed_output = OUTPUT_ROOT / f"seed_{seed}"
            seed_output.mkdir(parents=True, exist_ok=True)
            cases_path = seed_output / "cases.jsonl"
            base_aggregate = evaluator.new_aggregate()
            augmented_aggregate = evaluator.new_aggregate()
            pair_count = pair_pass = 0
            base_lookup: dict[str, tuple[tuple[tuple[Any, ...], ...], bool]] = {}
            with cases_path.open("w", encoding="utf-8", newline="\n") as stream:
                for size in range(1, 5):
                    for subset in itertools.combinations(evaluator.ROLES, size):
                        base_cases, augmented_cases = evaluator.build_cases_for_subset(seed, subset, base_ids, augmented_ids, manifest["operator_for_role"], manifest["noop_operator"], [int(value) for value in manifest["permutation"]])
                        def write_base(_case: evaluator.Case, record: dict[str, Any]) -> None:
                            stream.write(json.dumps({"kind": "base", **record}, separators=(",", ":")) + "\n")
                            evaluator.update_aggregate(base_aggregate, record)
                            base_lookup[record["pair_id"]] = (tuple(tuple(item) for item in record["semantic_signature"]), bool(record["structured_correct"]))
                        evaluator.evaluate_cases(core, base_cases, identity["thresholds"], executor, codebook, write_base)
                        def write_augmented(_case: evaluator.Case, record: dict[str, Any]) -> None:
                            nonlocal pair_count, pair_pass
                            base_signature, base_structured = base_lookup[record["pair_id"]]
                            record["base_structured_correct"] = base_structured
                            record["semantic_equivalent"] = tuple(tuple(item) for item in record["semantic_signature"]) == base_signature
                            record["paired_structured_pass"] = bool(base_structured and record["structured_correct"] and record["semantic_equivalent"])
                            stream.write(json.dumps({"kind": "augmented", **record}, separators=(",", ":")) + "\n")
                            evaluator.update_aggregate(augmented_aggregate, record)
                            pair_count += 1
                            pair_pass += int(record["paired_structured_pass"])
                        evaluator.evaluate_cases(core, augmented_cases, identity["thresholds"], executor, codebook, write_augmented)
                        base_lookup.clear()
            identity["core_state_hash_after"] = evaluator.tensor_digest(dict(core.state_dict()))
            identity["core_state_hash_identical"] = identity["core_state_hash_before"] == identity["core_state_hash_after"]
            identity["threshold_hash_after"] = evaluator.tensor_digest({"thresholds": torch.tensor(identity["thresholds"], dtype=torch.float32)})
            identity["threshold_hash_identical"] = identity["threshold_hash_before"] == identity["threshold_hash_after"]
            identity["source_hashes_after"] = {"stage_a": evaluator.sha256_file(evaluator.STAGE_A_ROOT / f"seed_{seed}" / "final.pt"), "calibration": evaluator.sha256_file(evaluator.STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"), "manifest": evaluator.sha256_file(manifest_path), "stage_c": evaluator.sha256_file(evaluator.STAGE_C_ROOT / "training" / f"seed_{seed}" / "final.pt")}
            identity["source_files_identical"] = identity["source_hashes_before"] == identity["source_hashes_after"]
            seed_pass = base_aggregate["structured_correct"] == sum(evaluator.EXPECTED_BASE.values()) and base_aggregate["margins_strict"] == sum(evaluator.EXPECTED_BASE.values()) and augmented_aggregate["structured_correct"] == sum(evaluator.EXPECTED_AUGMENTED.values()) and augmented_aggregate["margins_strict"] == sum(evaluator.EXPECTED_AUGMENTED.values()) and pair_pass == pair_count and base_aggregate["false_positives"] == 0 and base_aggregate["false_negatives"] == 0 and augmented_aggregate["false_positives"] == 0 and augmented_aggregate["false_negatives"] == 0 and augmented_aggregate["RAW_correct"] == augmented_aggregate["present_roles"] and augmented_aggregate["CANON_correct"] == augmented_aggregate["present_roles"] and identity["core_state_hash_identical"] and identity["threshold_hash_identical"] and identity["source_files_identical"]
            seed_report = {"seed": seed, "status": "passed" if seed_pass else "failed", "identity": {key: value for key, value in identity.items() if key != "checkpoint_payload"}, "base": evaluator.finite_aggregate(base_aggregate), "augmented": evaluator.finite_aggregate(augmented_aggregate), "pairs": {"count": pair_count, "structured_semantic_pass": pair_pass}, "cases": evaluator.source_record(cases_path)}
            evaluator.write_json(seed_output / "summary.json", seed_report)
            seed_reports.append(seed_report)
        runtime_after = {"file_sha256": evaluator.sha256_file(evaluator.BASE_CHECKPOINT), "tensor_state_hash": evaluator.state_hash(executor.state_dict()), "codebook_hash": evaluator.tensor_digest({"codebook": executor.token_embedding(torch.arange(evaluator.VALUE_BASE, evaluator.VALUE_BASE + evaluator.VALUE_COUNT, dtype=torch.long)).detach()})}
        runtime_integrity = {"before": runtime_before, "after": runtime_after, "file_sha256_identical": runtime_before["file_sha256"] == runtime_after["file_sha256"], "tensor_state_hash_identical": runtime_before["tensor_state_hash"] == runtime_after["tensor_state_hash"], "codebook_hash_identical": runtime_before["codebook_hash"] == runtime_after["codebook_hash"]}
        all_base = sum(report["base"]["structured_correct"] for report in seed_reports)
        all_augmented = sum(report["augmented"]["structured_correct"] for report in seed_reports)
        all_pairs = sum(report["pairs"]["structured_semantic_pass"] for report in seed_reports)
        total_pairs = sum(report["pairs"]["count"] for report in seed_reports)
        result = {"schema": "T7-noop-none-lexical-paired-development-conformance-v1", "status": "completed", "classification": "T7-NOOP-NONE-LEXICAL DEVELOPMENT: CLOSED/PASS" if all(report["status"] == "passed" for report in seed_reports) and runtime_integrity["file_sha256_identical"] and runtime_integrity["tensor_state_hash_identical"] and runtime_integrity["codebook_hash_identical"] else "T7-NOOP-NONE-LEXICAL DEVELOPMENT: VALID FAIL", "task": "T7-PAIRED-DEVELOPMENT-CONFORMANCE-REPAIR", "training": False, "recalibration": False, "architecture_changes": False, "multi_clause_evaluation": True, "fresh_pass_strong": False, "prior_attempt_commit": PRIOR_COMMIT, "prior_attempt_preserved": True, "conformance_specification": {"path": str((OUTPUT_ROOT / "conformance_specification.json").relative_to(ROOT)).replace("\\", "/"), "self_hash": spec_hash}, "logical_checker": checker_results, "expected_totals": {"base": evaluator.EXPECTED_ALL_BASE, "augmented": evaluator.EXPECTED_ALL_AUGMENTED, "combined": evaluator.EXPECTED_ALL_BASE + evaluator.EXPECTED_ALL_AUGMENTED}, "observed_totals": {"base": all_base, "augmented": all_augmented, "paired_structured_semantic_pass": all_pairs, "pairs": total_pairs}, "runtime_integrity": runtime_integrity, "seeds": seed_reports, "output_root": str(OUTPUT_ROOT.relative_to(ROOT)).replace("\\", "/"), "elapsed_seconds": time.perf_counter() - started}
        unsigned = dict(result)
        unsigned["artifact_self_hash"] = PLACEHOLDER
        digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
        result["artifact_self_hash"] = digest
        evaluator.write_json(OUTPUT_ROOT / "results.json", result)
        print(json.dumps({"status": result["status"], "classification": result["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": digest, "base": all_base, "augmented": all_augmented, "pairs": total_pairs, "pair_pass": all_pairs}, sort_keys=True))
        return 0 if result["classification"].endswith("CLOSED/PASS") else 1
    except Exception as error:
        invalid = {"status": "INVALID/HARNESS BUG", "task": result["task"] if "result" in locals() else specification["task"], "error_type": type(error).__name__, "error": str(error), "completed_seed_reports": seed_reports, "logical_checker": checker_results, "conformance_specification_self_hash": spec_hash}
        evaluator.write_json(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
