"""Evaluate the frozen T7 Stage-C Fresh models on the sealed paired domains."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from execute_t6_activeset_midpoint_development import decode_states, state_hash  # noqa: E402
from evaluate_t7_noop_none_lexical_paired_development import (  # noqa: E402
    Case,
    EVAL_BATCH_SIZE,
    EXPECTED_AUGMENTED,
    EXPECTED_BASE,
    EXPECTED_ALL_AUGMENTED,
    EXPECTED_ALL_BASE,
    build_cases_for_subset,
    finite_aggregate,
    new_aggregate,
    semantic_signature,
    update_aggregate,
)
from execute_t7_noop_none_stage_a_fresh import fresh_manifest_view  # noqa: E402
from repair_t7_noop_none_stage_b_atomic_sanity import atomic_sanity as repaired_atomic_sanity  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402


SEEDS = (7801, 7802, 7803, 7804, 7805)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
STAGE_A_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_fresh" / "training"
STAGE_B_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_fresh"
STAGE_C_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_c_fresh"
PREPARATION_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_fresh_preparation"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_paired_fresh"
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER = "__SELF_HASH__"


class HarnessError(RuntimeError):
    """Implementation or integrity failure; never classify as scientific failure."""


class FrozenBasePlusNoop(torch.nn.Module):
    """Evaluation-only append-only embedding matching the Stage-C checkpoint."""

    def __init__(self, old_weights: torch.Tensor, noop_weights: torch.Tensor) -> None:
        super().__init__()
        self.old_embeddings = torch.nn.Parameter(old_weights.detach().clone(), requires_grad=False)
        self.noop_embedding = torch.nn.Parameter(noop_weights.detach().clone(), requires_grad=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.old_embeddings, self.noop_embedding.unsqueeze(0)), dim=0)[token_ids]


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"non-finite evaluation value: {result}")
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json_bytes(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_self_hashed(path: Path, value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned["artifact_self_hash"] = PLACEHOLDER
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    value["artifact_self_hash"] = digest
    write_json_bytes(path, value)
    actual = path.read_bytes()
    stored = json.loads(actual.decode("utf-8"))["artifact_self_hash"]
    if sha256_bytes(actual.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != stored:
        raise HarnessError(f"self-hash verification failed: {path}")
    return digest


def verify_json_self_hash(path: Path) -> tuple[dict[str, Any], str, str]:
    data = path.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    stored = parsed.get("artifact_self_hash")
    if not isinstance(stored, str) or sha256_bytes(data.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != stored:
        raise HarnessError(f"invalid self-hash: {path}")
    return parsed, stored, sha256_bytes(data)


def independent_derived_value(seed: int, values: list[int]) -> int:
    offset = 1 if len(values) == 2 else 2
    candidate = (seed + sum(values) + offset) % VALUE_COUNT
    while candidate in values:
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def independent_distractor(seed: int, values: Iterable[int]) -> int:
    active = list(values)
    candidate = (seed + sum(active) + 1) % VALUE_COUNT
    while candidate in active:
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def independent_assignments(seed: int, subset: tuple[str, ...]) -> list[tuple[int, ...]]:
    if len(subset) <= 2:
        return list(itertools.permutations(range(VALUE_COUNT), len(subset)))
    expected: list[tuple[int, ...]] = []
    for first_two in itertools.permutations(range(VALUE_COUNT), 2):
        values = list(first_two)
        while len(values) < len(subset):
            values.append(independent_derived_value(seed, values))
        expected.append(tuple(values))
    return expected


def independent_checker(seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check generated cases against an independently implemented logical model."""
    base_ids = {token: index for index, token in enumerate(manifest["token_order"])}
    augmented_ids = {**base_ids, manifest["noop_operator"]: 37}
    permutation = [int(value) for value in manifest["permutation"]]
    inverse = {value: index for index, value in enumerate(permutation)}
    failures: list[str] = []
    counts_base = {size: 0 for size in range(1, 5)}
    counts_augmented = {size: 0 for size in range(1, 5)}
    base_by_pair: dict[str, Case] = {}
    seen_base: set[tuple[str, ...]] = set()
    seen_augmented: set[tuple[str, ...]] = set()
    for size in range(1, 5):
        for subset in itertools.combinations(ROLES, size):
            base_cases, augmented_cases = build_cases_for_subset(seed, subset, base_ids, augmented_ids, manifest["operator_for_role"], manifest["noop_operator"], permutation)
            expected_assignments = independent_assignments(seed, subset)
            order_count = math.factorial(size)
            actual_assignments = [tuple(case.assignment_dict[role] for role in subset) for case in base_cases[::order_count]]
            if actual_assignments != expected_assignments:
                failures.append(f"assignment mismatch:{size}:{subset}")
            counts_base[size] += len(base_cases)
            counts_augmented[size] += len(augmented_cases)
            for case in base_cases:
                base_by_pair[case.pair_id] = case
                if case.tokens in seen_base:
                    failures.append(f"duplicate base:{case.case_id}")
                seen_base.add(case.tokens)
                assignment = case.assignment_dict
                if len(case.tokens) != 3 * size - 1:
                    failures.append(f"base length:{case.case_id}")
                for index, role in enumerate(case.order):
                    offset = index * 3
                    expected_argument = f"ARG_{inverse[assignment[role]]:02d}"
                    if case.tokens[offset : offset + 2] != (manifest["operator_for_role"][role], expected_argument):
                        failures.append(f"base mapping:{case.case_id}")
                        break
            for case in augmented_cases:
                if case.tokens in seen_augmented:
                    failures.append(f"duplicate augmented:{case.case_id}")
                seen_augmented.add(case.tokens)
                assignment = case.assignment_dict
                expected_d = independent_distractor(seed, [assignment[role] for role in subset])
                if case.distractor != expected_d or case.tokens.count(manifest["noop_operator"]) != 1:
                    failures.append(f"d/noop:{case.case_id}")
                if len(case.tokens) != 3 * (size + 1) - 1:
                    failures.append(f"augmented length:{case.case_id}")
                for index, role in enumerate(case.order):
                    offset = index * 3
                    expected_operator = manifest["noop_operator"] if role == "NOOP" else manifest["operator_for_role"][role]
                    expected_value = expected_d if role == "NOOP" else assignment[role]
                    expected_argument = f"ARG_{inverse[expected_value]:02d}"
                    if case.tokens[offset : offset + 2] != (expected_operator, expected_argument):
                        failures.append(f"augmented mapping:{case.case_id}")
                        break
                active_order = tuple(role for role in case.order if role != "NOOP")
                paired = base_by_pair.get(case.pair_id)
                if paired is None or paired.order != active_order or paired.distractor != case.distractor:
                    failures.append(f"pair/order:{case.case_id}")
                if case.distractor in assignment.values():
                    failures.append(f"d collision:{case.case_id}")
    if counts_base != EXPECTED_BASE:
        failures.append(f"base counts:{counts_base}")
    if counts_augmented != EXPECTED_AUGMENTED:
        failures.append(f"augmented counts:{counts_augmented}")
    regression = {"seed": 7701, "values_v2": [0, 1], "v2": independent_derived_value(7701, [0, 1]), "values_v3": [0, 1, 23], "v3": independent_derived_value(7701, [0, 1, 23]), "expected": {"v2": 23, "v3": 15}}
    if regression["v2"] != 23 or regression["v3"] != 15:
        failures.append("formula regression")
    return {"status": "passed" if not failures else "failed", "seed": seed, "base_counts": counts_base, "augmented_counts": counts_augmented, "base_total": sum(counts_base.values()), "augmented_total": sum(counts_augmented.values()), "unique_base_instructions": len(seen_base), "unique_augmented_instructions": len(seen_augmented), "regression": regression, "failures": failures}


def load_manifest(seed: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = PREPARATION_ROOT / "manifests" / f"manifest_{seed}_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "T7-noop-none-lexical-fresh-preparation-manifest-v1" or raw.get("seed") != seed:
        raise HarnessError(f"fresh manifest mismatch: {seed}")
    permutation = raw.get("permutation")
    if not isinstance(permutation, list) or len(permutation) != VALUE_COUNT or sorted(permutation) != list(range(VALUE_COUNT)):
        raise HarnessError(f"fresh permutation mismatch: {seed}")
    if raw.get("stage_c_noop_internal_id") != 37 or raw.get("stage_c_token_order") != [*raw["token_order"], raw["noop_operator"]]:
        raise HarnessError(f"sealed Stage-C mapping mismatch: {seed}")
    manifest = dict(raw)
    manifest["train"] = [{"argument": f"ARG_{index:02d}", "constraints": role.lower(), "kind": "atomic", "operator": manifest["operator_for_role"][role], "role": role, "value": int(permutation[index])} for role in ROLES for index in range(VALUE_COUNT)]
    manifest["multi_clause_train"] = 0
    manifest["joint_examples_in_training"] = 0
    manifest["test_rows_used"] = 0
    return manifest, fresh_manifest_view(manifest), path


def load_verified_core(seed: int, manifest: dict[str, Any], view: dict[str, Any], manifest_path: Path) -> tuple[GenericNRoleBinder, dict[str, Any]]:
    stage_a = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
    stage_b = STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
    stage_c = STAGE_C_ROOT / "training" / f"seed_{seed}" / "final.pt"
    stage_c_result = STAGE_C_ROOT / "results" / f"result_{seed}_v1.json"
    stage_a_result = STAGE_A_ROOT / f"seed_{seed}" / "results.json"
    for path in (stage_a, stage_b, stage_c, stage_c_result, stage_a_result):
        if not path.exists():
            raise HarnessError(f"missing frozen input: {seed}:{path}")
    a_result, _, _ = verify_json_self_hash(stage_a_result)
    b_calibration, b_self_hash, b_file_sha = verify_json_self_hash(stage_b)
    c_result, _, _ = verify_json_self_hash(stage_c_result)
    a_sha = sha256_file(stage_a)
    b_sha = sha256_file(stage_b)
    c_sha = sha256_file(stage_c)
    manifest_sha = sha256_file(manifest_path)
    a_payload = torch.load(stage_a, map_location="cpu", weights_only=False)
    c_payload = torch.load(stage_c, map_location="cpu", weights_only=False)
    if a_payload.get("seed") != seed or a_payload.get("manifest_sha256") != manifest_sha:
        raise HarnessError(f"Stage-A input provenance mismatch: {seed}")
    if c_payload.get("stage") != "C" or c_payload.get("seed") != seed or c_payload.get("stage_a_checkpoint_sha256") != a_sha or c_payload.get("calibration_sha256") != b_sha or c_payload.get("manifest_sha256") != manifest_sha:
        raise HarnessError(f"Stage-C input provenance mismatch: {seed}")
    if c_result.get("seed") != seed or c_result.get("invariants", {}).get("stage_c_checkpoint", {}).get("sha256") != c_sha:
        raise HarnessError(f"Stage-C result/checkpoint mismatch: {seed}")
    if b_calibration.get("manifest", {}).get("sha256") != manifest_sha or b_calibration.get("core_checkpoint", {}).get("sha256") != a_sha:
        raise HarnessError(f"Stage-B input association mismatch: {seed}")
    thresholds = [float(b_calibration["calibration"]["intervals"][role]["b_r_stored"]) for role in ROLES]
    if thresholds != [float(value) for value in b_calibration["calibration"]["threshold_vector"]] or thresholds != [float(value) for value in c_payload["threshold_vector_role_order"]]:
        raise HarnessError(f"threshold identity mismatch: {seed}")
    if c_payload.get("base_token_order") != manifest["token_order"] or c_payload.get("augmented_token_order") != manifest["stage_c_token_order"] or c_payload.get("noop_operator") != manifest["noop_operator"] or c_payload.get("noop_internal_id") != 37 or c_payload.get("trainable_parameter_names") != ["embedding.noop_embedding"]:
        raise HarnessError(f"Stage-C mapping metadata mismatch: {seed}")
    c_state = c_payload["encoder"]
    if not torch.equal(c_state["embedding.old_embeddings"], a_payload["encoder"]["embedding.weight"]):
        raise HarnessError(f"Stage-C old embedding differs from Stage-A: {seed}")
    for name, value in a_payload["encoder"].items():
        if name == "embedding.weight":
            continue
        if name not in c_state or not torch.equal(value, c_state[name]):
            raise HarnessError(f"Stage-C base component differs from Stage-A: {seed}:{name}")
    if c_result.get("C1", {}).get("final_e_N_hash") != tensor_digest({"e_N": c_state["embedding.noop_embedding"]}):
        raise HarnessError(f"Stage-C e_N hash mismatch: {seed}")
    core = GenericNRoleBinder(view, seed, list(ROLES))
    non_embedding = {name: value for name, value in c_state.items() if not name.startswith("embedding.")}
    missing, unexpected = core.load_state_dict(non_embedding, strict=False)
    if missing != ["embedding.weight"] or unexpected:
        raise HarnessError(f"Stage-C core load mismatch: {seed}:{missing}/{unexpected}")
    core.embedding = FrozenBasePlusNoop(c_state["embedding.old_embeddings"], c_state["embedding.noop_embedding"])
    if tuple(core.embedding.old_embeddings.shape) != (37, 16) or tuple(core.embedding.noop_embedding.shape) != (16,):
        raise HarnessError(f"Stage-C embedding shape mismatch: {seed}")
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    identity = {"seed": seed, "stage_a_checkpoint": source_record(stage_a), "stage_b_calibration": {**source_record(stage_b), "self_hash": b_self_hash, "file_sha256": b_file_sha}, "stage_c_checkpoint": source_record(stage_c), "stage_c_result": source_record(stage_c_result), "stage_a_result": source_record(stage_a_result), "manifest": source_record(manifest_path), "thresholds": thresholds, "threshold_hash_before": tensor_digest({"thresholds": torch.tensor(thresholds, dtype=torch.float32)}), "core_state_hash_before": tensor_digest(dict(c_state)), "e_N_final_hash": tensor_digest({"e_N": c_state["embedding.noop_embedding"]}), "noop_operator": manifest["noop_operator"], "noop_internal_id": 37, "noop_external_id": int(manifest["stage_c_noop_external_id"]), "a3_decoder_sha256": a_result["A3"]["decoder_checkpoint"]["sha256"], "c_payload": c_payload}
    return core, identity


def evaluate_cases_fresh(core: GenericNRoleBinder, cases: list[Case], thresholds: list[float], executor: torch.nn.Module, codebook: torch.Tensor, on_result: Callable[[Case, dict[str, Any]], None]) -> None:
    by_length: dict[int, list[Case]] = {}
    for case in cases:
        by_length.setdefault(len(case.token_ids), []).append(case)
    for length, grouped in sorted(by_length.items()):
        for start in range(0, len(grouped), EVAL_BATCH_SIZE):
            chunk = grouped[start : start + EVAL_BATCH_SIZE]
            ids = torch.tensor([case.token_ids for case in chunk], dtype=torch.long)
            lengths = torch.full((len(chunk),), length, dtype=torch.long)
            with torch.no_grad():
                details = core(ids, lengths)
            pending_decode: list[tuple[int, int, int]] = []
            role_data: list[list[dict[str, Any]]] = [[{} for _ in ROLES] for _ in chunk]
            for row_index, case in enumerate(chunk):
                for query_index, role in enumerate(ROLES):
                    scores = details["scores"][query_index][row_index, :length]
                    threshold = thresholds[query_index]
                    best, pointer = scores.max(dim=0)
                    pointer_index = int(pointer.item())
                    best_value = finite(best.item())
                    target_position = case.target_positions[query_index]
                    expected_present = role in case.subset
                    if expected_present:
                        target_score = finite(scores[target_position].item())
                        other = torch.cat((scores[:target_position], scores[target_position + 1 :]))
                        real_margin = finite(target_score - float(other.max().item()))
                        null_margin = finite(target_score - threshold)
                        predicted_present = best_value > threshold
                        if predicted_present:
                            pending_decode.append((row_index, query_index, pointer_index))
                        role_data[row_index][query_index] = {"presence": "present" if predicted_present else "none", "pointer": pointer_index if predicted_present else -1, "selected_token": case.tokens[pointer_index] if predicted_present else "", "RAW": -1, "CANON": -1, "expected_present": True, "target_position": target_position, "target_score": target_score, "max_other_score": finite(other.max().item()), "margin_present_to_NULL": null_margin, "margin_present_to_real": real_margin, "margin_strict": null_margin > 0.0 and real_margin > 0.0, "correct": False, "noop_rejection_margin": None, "noop_worst_score": None}
                    else:
                        absent_margin = finite(threshold - best_value)
                        predicted_present = best_value > threshold
                        if predicted_present:
                            pending_decode.append((row_index, query_index, pointer_index))
                        role_data[row_index][query_index] = {"presence": "present" if predicted_present else "none", "pointer": pointer_index if predicted_present else -1, "selected_token": case.tokens[pointer_index] if predicted_present else "", "RAW": -1, "CANON": -1, "expected_present": False, "target_position": -1, "target_score": None, "max_other_score": None, "margin_absent": absent_margin, "margin_strict": absent_margin > 0.0, "correct": not predicted_present, "noop_rejection_margin": None, "noop_worst_score": None}
                    if case.noop_positions:
                        noop_worst = finite(scores[list(case.noop_positions)].max().item())
                        role_data[row_index][query_index]["noop_worst_score"] = noop_worst
                        role_data[row_index][query_index]["noop_rejection_margin"] = finite(threshold - noop_worst)
            if pending_decode:
                states = torch.stack(tuple(details["values"][row, pointer] for row, _query, pointer in pending_decode))
                with torch.no_grad():
                    raw, canonical = decode_states(executor, codebook, states)
                for decode_index, (row, query, _pointer) in enumerate(pending_decode):
                    item = role_data[row][query]
                    item["RAW"] = int(raw[decode_index].item())
                    item["CANON"] = int(canonical[decode_index].item())
                    expected_value = dict(chunk[row].assignment).get(ROLES[query])
                    item["correct"] = item["presence"] == "present" and item["pointer"] == chunk[row].target_positions[query] and item["RAW"] == expected_value and item["CANON"] == expected_value and item["RAW"] == item["CANON"]
            for row_index, case in enumerate(chunk):
                roles = role_data[row_index]
                record = {"case_id": case.case_id, "pair_id": case.pair_id, "seed": case.seed, "subset": "_".join(case.subset), "subset_size": len(case.subset), "order": "_".join(case.order), "assignment": dict(case.assignment), "D": case.distractor, "augmented": case.augmented, "tokens": list(case.tokens), "roles": roles, "structured_correct": all(item["correct"] for item in roles), "margins_strict": all(item["margin_strict"] for item in roles), "semantic_signature": semantic_signature(roles)}
                on_result(case, record)


def update_counts_for_case(aggregate: dict[str, Any], record: dict[str, Any]) -> None:
    update_aggregate(aggregate, record)


def write_cases_and_evaluate(seed: int, manifest: dict[str, Any], core: GenericNRoleBinder, thresholds: list[float], executor: torch.nn.Module, codebook: torch.Tensor, seed_output: Path) -> dict[str, Any]:
    cases_path = seed_output / "cases.jsonl"
    base_aggregate = new_aggregate()
    augmented_aggregate = new_aggregate()
    pair_count = 0
    pair_pass = 0
    generated_base = 0
    generated_augmented = 0
    base_lookup: dict[str, tuple[tuple[tuple[Any, ...], ...], bool]] = {}
    with cases_path.open("wb") as stream:
        def emit(kind: str, record: dict[str, Any]) -> None:
            stream.write((json.dumps({"kind": kind, **record}, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8"))

        for size in range(1, 5):
            for subset in itertools.combinations(ROLES, size):
                base_ids = {token: index for index, token in enumerate(manifest["token_order"])}
                augmented_ids = {**base_ids, manifest["noop_operator"]: 37}
                base_cases, augmented_cases = build_cases_for_subset(seed, subset, base_ids, augmented_ids, manifest["operator_for_role"], manifest["noop_operator"], [int(value) for value in manifest["permutation"]])

                base_start = generated_base
                def write_base(_case: Case, record: dict[str, Any]) -> None:
                    nonlocal generated_base
                    emit("base", record)
                    update_counts_for_case(base_aggregate, record)
                    base_lookup[record["pair_id"]] = (tuple(tuple(item) for item in record["semantic_signature"]), bool(record["structured_correct"]))
                    generated_base += 1

                evaluate_cases_fresh(core, base_cases, thresholds, executor, codebook, write_base)
                if generated_base - base_start != len(base_cases):
                    raise HarnessError(f"base case execution count mismatch: {seed}/{subset}")

                augmented_start = generated_augmented
                def write_augmented(_case: Case, record: dict[str, Any]) -> None:
                    nonlocal generated_augmented, pair_count, pair_pass
                    base_signature, base_structured = base_lookup[record["pair_id"]]
                    record["base_structured_correct"] = base_structured
                    record["semantic_equivalent"] = tuple(tuple(item) for item in record["semantic_signature"]) == base_signature
                    record["paired_structured_pass"] = bool(base_structured and record["structured_correct"] and record["semantic_equivalent"])
                    emit("augmented", record)
                    update_counts_for_case(augmented_aggregate, record)
                    generated_augmented += 1
                    pair_count += 1
                    pair_pass += int(record["paired_structured_pass"])

                evaluate_cases_fresh(core, augmented_cases, thresholds, executor, codebook, write_augmented)
                if generated_augmented - augmented_start != len(augmented_cases):
                    raise HarnessError(f"augmented case execution count mismatch: {seed}/{subset}")
                base_lookup.clear()
    cases_sha = sha256_file(cases_path)
    expected_base = sum(EXPECTED_BASE.values())
    expected_augmented = sum(EXPECTED_AUGMENTED.values())
    base_report = finite_aggregate(base_aggregate)
    augmented_report = finite_aggregate(augmented_aggregate)
    expected_pairs = expected_augmented
    passed = generated_base == expected_base and generated_augmented == expected_augmented and pair_count == expected_pairs and base_report["structured_correct"] == expected_base and augmented_report["structured_correct"] == expected_augmented and base_report["margins_strict"] == expected_base and augmented_report["margins_strict"] == expected_augmented and base_report["false_positives"] == 0 and base_report["false_negatives"] == 0 and augmented_report["false_positives"] == 0 and augmented_report["false_negatives"] == 0 and pair_pass == expected_pairs and augmented_report["RAW_correct"] == augmented_report["present_roles"] and augmented_report["CANON_correct"] == augmented_report["present_roles"]
    return {"seed": seed, "status": "passed" if passed else "failed", "base": base_report, "augmented": augmented_report, "pairs": {"count": pair_count, "expected": expected_pairs, "structured_semantic_pass": pair_pass}, "generated_rows": {"base": generated_base, "augmented": generated_augmented, "total": generated_base + generated_augmented}, "cases": {"path": cases_path.relative_to(ROOT).as_posix(), "bytes": cases_path.stat().st_size, "sha256": cases_sha, "encoding": "UTF-8", "line_endings": "LF", "rows": generated_base + generated_augmented}}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty paired fresh root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    try:
        manifests = {seed: load_manifest(seed) for seed in SEEDS}
        checker_results: dict[str, Any] = {}
        checker_paths: dict[str, Any] = {}
        for seed in SEEDS:
            manifest, _view, manifest_path = manifests[seed]
            checker = independent_checker(seed, manifest)
            checker_path = OUTPUT_ROOT / "checker" / f"checker_{seed}_v1.json"
            checker_payload = {"schema": "t7-noop-none-lexical-paired-fresh-independent-checker-v1", "seed": seed, "manifest": source_record(manifest_path), "independently_implemented_formulas": {"v2": "(seed+v0+v1+1) mod 32; increment until distinct", "v3": "(seed+v0+v1+v2+2) mod 32; increment until distinct", "D": "(seed+sum(active)+1) mod 32; increment until distinct"}, "checker": checker, "model_forward_started": False}
            checker_self_hash = write_self_hashed(checker_path, checker_payload)
            if checker["status"] != "passed":
                raise HarnessError(f"independent checker failed before model forward: {seed}")
            checker_results[str(seed)] = checker
            checker_paths[str(seed)] = {**source_record(checker_path), "self_hash": checker_self_hash}
        runner_hash = sha256_file(SCRIPT_PATH)
        executor = load_executor()
        executor.eval()
        for parameter in executor.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in executor.parameters()):
            raise HarnessError("decoder runtime is not evaluation-frozen")
        runtime_before = {"file_sha256": sha256_file(BASE_CHECKPOINT), "tensor_state_hash": state_hash(executor.state_dict())}
        with torch.no_grad():
            codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
        runtime_before["codebook_hash"] = tensor_digest({"codebook": codebook})
        seed_reports: list[dict[str, Any]] = []
        for seed in SEEDS:
            manifest, view, manifest_path = manifests[seed]
            core, identity = load_verified_core(seed, manifest, view, manifest_path)
            if identity["a3_decoder_sha256"] != runtime_before["file_sha256"]:
                raise HarnessError(f"decoder does not match Stage-A A3 binding: {seed}")
            seed_output = OUTPUT_ROOT / f"seed_{seed}"
            seed_output.mkdir(parents=True, exist_ok=True)
            base_controls = repaired_atomic_sanity(core, view, executor, codebook, identity["thresholds"])
            evaluation = write_cases_and_evaluate(seed, manifest, core, identity["thresholds"], executor, codebook, seed_output)
            state_after = tensor_digest(dict(core.state_dict()))
            threshold_after = tensor_digest({"thresholds": torch.tensor(identity["thresholds"], dtype=torch.float32)})
            source_after = {"stage_a": sha256_file(STAGE_A_ROOT / f"seed_{seed}" / "final.pt"), "stage_b": sha256_file(STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"), "stage_c": sha256_file(STAGE_C_ROOT / "training" / f"seed_{seed}" / "final.pt"), "stage_c_result": sha256_file(STAGE_C_ROOT / "results" / f"result_{seed}_v1.json"), "manifest": sha256_file(manifest_path)}
            source_before = {"stage_a": identity["stage_a_checkpoint"]["sha256"], "stage_b": identity["stage_b_calibration"]["sha256"], "stage_c": identity["stage_c_checkpoint"]["sha256"], "stage_c_result": identity["stage_c_result"]["sha256"], "manifest": identity["manifest"]["sha256"]}
            identity["core_state_hash_after"] = state_after
            identity["core_state_hash_identical"] = identity["core_state_hash_before"] == state_after
            identity["threshold_hash_after"] = threshold_after
            identity["threshold_hash_identical"] = identity["threshold_hash_before"] == threshold_after
            identity["source_hashes_before"] = source_before
            identity["source_hashes_after"] = source_after
            identity["source_files_identical"] = source_before == source_after
            seed_pass = evaluation["status"] == "passed" and base_controls["passed"] and identity["core_state_hash_identical"] and identity["threshold_hash_identical"] and identity["source_files_identical"]
            seed_report = {"seed": seed, "status": "passed" if seed_pass else "failed", "identity": {key: value for key, value in identity.items() if key != "c_payload"}, "base_atomic_controls": base_controls, "evaluation": evaluation, "paired_pass_strong_conditions": {"base_complete": evaluation["base"]["structured_correct"] == sum(EXPECTED_BASE.values()), "augmented_complete": evaluation["augmented"]["structured_correct"] == sum(EXPECTED_AUGMENTED.values()), "all_pairs_equivalent": evaluation["pairs"]["structured_semantic_pass"] == evaluation["pairs"]["count"], "no_false_positives": evaluation["base"]["false_positives"] == 0 and evaluation["augmented"]["false_positives"] == 0, "no_false_negatives": evaluation["base"]["false_negatives"] == 0 and evaluation["augmented"]["false_negatives"] == 0, "all_margins_strict": evaluation["base"]["margins_strict"] == sum(EXPECTED_BASE.values()) and evaluation["augmented"]["margins_strict"] == sum(EXPECTED_AUGMENTED.values())}, "stage_c_gate": {"pass": seed_pass}}
            summary_path = seed_output / "summary.json"
            write_self_hashed(summary_path, seed_report)
            seed_report["summary"] = source_record(summary_path)
            seed_reports.append(seed_report)
        runtime_after = {"file_sha256": sha256_file(BASE_CHECKPOINT), "tensor_state_hash": state_hash(executor.state_dict())}
        with torch.no_grad():
            runtime_after["codebook_hash"] = tensor_digest({"codebook": executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()})
        runtime_integrity = {"before": runtime_before, "after": runtime_after, "file_sha256_identical": runtime_before["file_sha256"] == runtime_after["file_sha256"], "tensor_state_hash_identical": runtime_before["tensor_state_hash"] == runtime_after["tensor_state_hash"], "codebook_hash_identical": runtime_before["codebook_hash"] == runtime_after["codebook_hash"]}
        base_total = sum(report["evaluation"]["base"]["structured_correct"] for report in seed_reports)
        augmented_total = sum(report["evaluation"]["augmented"]["structured_correct"] for report in seed_reports)
        pair_total = sum(report["evaluation"]["pairs"]["structured_semantic_pass"] for report in seed_reports)
        pair_count = sum(report["evaluation"]["pairs"]["count"] for report in seed_reports)
        all_pass = all(report["status"] == "passed" for report in seed_reports) and runtime_integrity["file_sha256_identical"] and runtime_integrity["tensor_state_hash_identical"] and runtime_integrity["codebook_hash_identical"]
        consolidated = {"schema": "t7-noop-none-lexical-paired-fresh-v1", "status": "completed", "classification": "T7-NOOP-NONE-LEXICAL-FRESH: PASS_STRONG" if all_pass else "T7-NOOP-NONE-LEXICAL-FRESH: VALID SCIENTIFIC FAIL", "task": "T7-NOOP-NONE-LEXICAL / PAIRED FRESH EVALUATION", "training": False, "recalibration": False, "architecture_changes": False, "multi_clause_evaluation": True, "fresh_pass_strong": all_pass, "runner": {**source_record(SCRIPT_PATH), "sha256_before_first_evaluation_forward": runner_hash}, "logical_checker": checker_results, "checker_artifacts": checker_paths, "expected_totals": {"base": EXPECTED_ALL_BASE, "augmented": EXPECTED_ALL_AUGMENTED, "pairs": EXPECTED_ALL_AUGMENTED, "combined": EXPECTED_ALL_BASE + EXPECTED_ALL_AUGMENTED}, "observed_totals": {"base_structured_correct": base_total, "augmented_structured_correct": augmented_total, "paired_structured_semantic_pass": pair_total, "pairs": pair_count}, "runtime_integrity": runtime_integrity, "seeds": seed_reports, "output_root": OUTPUT_ROOT.relative_to(ROOT).as_posix(), "evaluation_set": {"seeds": list(SEEDS), "base_by_size": EXPECTED_BASE, "augmented_by_size": EXPECTED_AUGMENTED, "base_per_seed": sum(EXPECTED_BASE.values()), "augmented_per_seed": sum(EXPECTED_AUGMENTED.values()), "base_all": EXPECTED_ALL_BASE, "augmented_all": EXPECTED_ALL_AUGMENTED, "pairs_all": EXPECTED_ALL_AUGMENTED, "all_augmented_forwards_executed": True}, "forbidden_operations_confirmed": ["training", "backward", "optimizer", "recalibration", "architecture change", "C reinitialization", "NOOP suppression", "label-based clause removal", "automatic next stage"], "next_stage": "No next stage; evaluation complete.", "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unavailable"}, "elapsed_seconds": time.perf_counter() - started}
        artifact_path = OUTPUT_ROOT / "results.json"
        write_self_hashed(artifact_path, consolidated)
        parsed, _, file_sha = verify_json_self_hash(artifact_path)
        print(json.dumps({"status": parsed["status"], "classification": parsed["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_self_hash": parsed["artifact_self_hash"], "file_sha256": file_sha, "base": base_total, "augmented": augmented_total, "pairs": pair_count, "pair_pass": pair_total}, sort_keys=True))
        return 0 if all_pass else 1
    except Exception as error:
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7-NOOP-NONE-LEXICAL / PAIRED FRESH EVALUATION", "error_type": type(error).__name__, "error": str(error)}
        write_json_bytes(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
