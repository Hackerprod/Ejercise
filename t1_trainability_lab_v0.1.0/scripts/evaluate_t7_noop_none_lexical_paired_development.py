"""Evaluate frozen T7 Stage-C checkpoints against sealed base/NOOP pairs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from execute_t6_activeset_midpoint_development import decode_states  # noqa: E402
from execute_t7_noop_none_stage_a_development import (  # noqa: E402
    ROLES,
    SEEDS,
    load_t7_manifest,
    source_record,
    state_hash,
)
from t1_trainability.t7_production_core_noop_integration import AppendedNoopEmbedding  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402


STAGE_A_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_development" / "training"
STAGE_B_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_development"
STAGE_C_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_c_development"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_paired_development"
PREPARATION_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation"
UPDATES = 5000
DMODEL = 16
PLACEHOLDER = "__SELF_HASH__"
EXPECTED_BASE = {1: 128, 2: 11904, 3: 23808, 4: 23808}
EXPECTED_AUGMENTED = {1: 256, 2: 35712, 3: 95232, 4: 119040}
EXPECTED_TOTAL_BASE = 59648
EXPECTED_TOTAL_AUGMENTED = 250240
EXPECTED_ALL_BASE = 298240
EXPECTED_ALL_AUGMENTED = 1251200
EXPECTED_NOOP_ARGUMENTS = 32
EVAL_BATCH_SIZE = 2048


@dataclass(frozen=True)
class Case:
    case_id: str
    pair_id: str
    seed: int
    subset: tuple[str, ...]
    order: tuple[str, ...]
    assignment: tuple[tuple[str, int], ...]
    distractor: int
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    target_positions: tuple[int, ...]
    noop_positions: tuple[int, ...]
    augmented: bool

    @property
    def assignment_dict(self) -> dict[str, int]:
        return dict(self.assignment)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite evaluation value: {result}")
    return result


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def derived_value(seed: int, values: list[int]) -> int:
    offset = 1 if len(values) == 2 else 2
    candidate = (seed + sum(values) + offset) % VALUE_COUNT
    while candidate in values:
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def distractor_value(seed: int, values: Iterable[int]) -> int:
    active = list(values)
    candidate = (seed + sum(active) + 1) % VALUE_COUNT
    while candidate in active:
        candidate = (candidate + 1) % VALUE_COUNT
    return candidate


def value_assignments(seed: int, subset: tuple[str, ...]) -> Iterable[tuple[int, ...]]:
    if len(subset) <= 2:
        yield from itertools.permutations(range(VALUE_COUNT), len(subset))
        return
    for first_two in itertools.permutations(range(VALUE_COUNT), 2):
        values = list(first_two)
        while len(values) < len(subset):
            values.append(derived_value(seed, values))
        yield tuple(values)


def token_layout(
    order: tuple[str, ...],
    assignment: dict[str, int],
    distractor: int,
    operator_for_role: dict[str, str],
    noop_operator: str,
    inverse_permutation: dict[int, int],
    token_ids: dict[str, int],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    tokens: list[str] = []
    positions: dict[str, int] = {}
    noop_positions: list[int] = []
    for index, role in enumerate(order):
        if index:
            tokens.append("LINK")
        operator = operator_for_role[role] if role != "NOOP" else noop_operator
        value = assignment[role] if role != "NOOP" else distractor
        argument = f"ARG_{inverse_permutation[value]:02d}"
        positions[role] = len(tokens) + 1
        if role == "NOOP":
            noop_positions.extend((len(tokens), len(tokens) + 1))
        tokens.extend((operator, argument))
    ids = tuple(token_ids[token] for token in tokens)
    target_positions = tuple(positions.get(role, -1) for role in ROLES)
    return tuple(tokens), ids, target_positions, tuple(noop_positions)


def build_cases_for_subset(
    seed: int,
    subset: tuple[str, ...],
    base_token_ids: dict[str, int],
    augmented_token_ids: dict[str, int],
    operator_for_role: dict[str, str],
    noop_operator: str,
    permutation: list[int],
) -> tuple[list[Case], list[Case]]:
    inverse = {value: index for index, value in enumerate(permutation)}
    base_cases: list[Case] = []
    augmented_cases: list[Case] = []
    subset_name = "_".join(subset)
    assignment_index = 0
    for values in value_assignments(seed, subset):
        assignment = dict(zip(subset, values))
        distractor = distractor_value(seed, values)
        assignment_key = ",".join(f"{role}={assignment[role]}" for role in subset)
        for base_order in itertools.permutations(subset):
            order_name = "_".join(base_order)
            pair_id = f"{seed}|{subset_name}|{assignment_index}|{assignment_key}|{order_name}"
            base_id = f"{pair_id}|base"
            base_tokens, base_ids, positions, noop_positions = token_layout(
                base_order, assignment, distractor, operator_for_role, "", inverse, base_token_ids
            )
            base_cases.append(Case(base_id, pair_id, seed, subset, base_order, tuple(assignment.items()), distractor, base_tokens, base_ids, positions, noop_positions, False))
        for augmented_order in itertools.permutations((*subset, "NOOP")):
            active_order = tuple(role for role in augmented_order if role != "NOOP")
            order_name = "_".join(augmented_order)
            pair_id = f"{seed}|{subset_name}|{assignment_index}|{assignment_key}|{'_'.join(active_order)}"
            augmented_id = f"{pair_id}|augmented|{order_name}"
            augmented_assignment = {**assignment, "NOOP": distractor}
            tokens, ids, positions, noop_positions = token_layout(
                augmented_order, augmented_assignment, distractor, operator_for_role, noop_operator, inverse, augmented_token_ids
            )
            augmented_cases.append(Case(augmented_id, pair_id, seed, subset, augmented_order, tuple(assignment.items()), distractor, tokens, ids, positions, noop_positions, True))
        assignment_index += 1
    return base_cases, augmented_cases


def logical_checker(
    seed: int,
    base_manifest: dict[str, Any],
    base_token_ids: dict[str, int],
    augmented_token_ids: dict[str, int],
) -> dict[str, Any]:
    permutation = [int(value) for value in base_manifest["permutation"]]
    active_roles = tuple(ROLES)
    operator_for_role = dict(base_manifest["operator_for_role"])
    base_counts = {size: 0 for size in range(1, 5)}
    augmented_counts = {size: 0 for size in range(1, 5)}
    pair_count = 0
    failures: list[str] = []
    for size in range(1, 5):
        for subset in itertools.combinations(active_roles, size):
            base_cases, augmented_cases = build_cases_for_subset(seed, subset, base_token_ids, augmented_token_ids, operator_for_role, base_manifest["noop_operator"], permutation)
            base_counts[size] += len(base_cases)
            augmented_counts[size] += len(augmented_cases)
            base_by_pair = {case.pair_id: case for case in base_cases}
            for case in base_cases:
                if len(case.tokens) != 3 * size - 1:
                    failures.append(f"base length {case.case_id}")
                    break
                if tuple(role for role in case.order) != tuple(role for role in case.order if role in subset):
                    failures.append(f"base order {case.case_id}")
                    break
            for case in augmented_cases:
                if len(case.tokens) != 3 * (size + 1) - 1:
                    failures.append(f"augmented length {case.case_id}")
                    break
                if case.tokens.count(base_manifest["noop_operator"]) != 1:
                    failures.append(f"NOOP count {case.case_id}")
                    break
                active_order = tuple(role for role in case.order if role != "NOOP")
                paired = base_by_pair.get(case.pair_id)
                if paired is None or active_order != paired.order or case.distractor in paired.assignment_dict.values():
                    failures.append(f"pair identity {case.case_id}")
                    break
                # Reconstruct from semantic clauses, not by deleting arbitrary tokens.
                clauses = [(case.order[index], case.tokens[offset + 1]) for index, offset in enumerate(range(0, len(case.tokens), 3))]
                clauses = [(role, argument) for role, argument in clauses if role != "NOOP"]
                rebuilt: list[str] = []
                for index, (role, argument) in enumerate(clauses):
                    if index:
                        rebuilt.append("LINK")
                    rebuilt.extend((operator_for_role[role], argument))
                if tuple(rebuilt) != paired.tokens:
                    failures.append(f"reconstruction {case.case_id}")
                    break
                pair_count += 1
    if base_counts != EXPECTED_BASE:
        failures.append(f"base counts {base_counts}")
    if augmented_counts != EXPECTED_AUGMENTED:
        failures.append(f"augmented counts {augmented_counts}")
    return {"status": "passed" if not failures else "failed", "seed": seed, "base_counts": base_counts, "augmented_counts": augmented_counts, "base_total": sum(base_counts.values()), "augmented_total": sum(augmented_counts.values()), "pairs": pair_count, "failures": failures}


def verify_self_hash(path: Path) -> tuple[dict[str, Any], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stored = str(data["artifact_self_hash"])
    data["artifact_self_hash"] = PLACEHOLDER
    actual = sha256_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if actual != stored:
        raise RuntimeError(f"invalid self-hash: {path}")
    return json.loads(path.read_text(encoding="utf-8")), stored


def load_verified_core(seed: int, base_manifest: dict[str, Any], manifest_path: Path) -> tuple[GenericNRoleBinder, dict[str, Any]]:
    stage_a_checkpoint = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
    stage_b_calibration = STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
    stage_c_checkpoint = STAGE_C_ROOT / "training" / f"seed_{seed}" / "final.pt"
    stage_c_result = STAGE_C_ROOT / "results" / f"result_{seed}_v1.json"
    if not all(path.exists() for path in (stage_a_checkpoint, stage_b_calibration, stage_c_checkpoint, stage_c_result)):
        raise RuntimeError(f"missing A/B/C artifact for seed {seed}")
    calibration, calibration_self_hash = verify_self_hash(stage_b_calibration)
    c_payload = torch.load(stage_c_checkpoint, map_location="cpu", weights_only=False)
    c_result = json.loads(stage_c_result.read_text(encoding="utf-8"))
    if c_payload.get("stage") != "C" or c_payload.get("seed") != seed:
        raise RuntimeError(f"invalid Stage-C checkpoint provenance for seed {seed}")
    hashes = {"stage_a": sha256_file(stage_a_checkpoint), "calibration": sha256_file(stage_b_calibration), "manifest": sha256_file(manifest_path), "stage_c": sha256_file(stage_c_checkpoint)}
    if c_payload.get("stage_a_checkpoint_sha256") != hashes["stage_a"] or c_payload.get("calibration_sha256") != hashes["calibration"] or c_payload.get("manifest_sha256") != hashes["manifest"]:
        raise RuntimeError(f"Stage-C references do not match files for seed {seed}")
    state = c_payload["encoder"]
    a_payload = torch.load(stage_a_checkpoint, map_location="cpu", weights_only=False)
    if a_payload.get("seed") != seed or a_payload.get("noop_in_stage_a_vocab") is not False or a_payload.get("noop_in_batch") is not False:
        raise RuntimeError(f"invalid Stage-A checkpoint provenance for seed {seed}")
    if not torch.equal(state["embedding.old_embeddings"], a_payload["encoder"]["embedding.weight"]):
        raise RuntimeError(f"Stage-C old embedding differs from Stage-A for seed {seed}")
    for name, value in a_payload["encoder"].items():
        if name == "embedding.weight":
            continue
        if name not in state or not torch.equal(state[name], value):
            raise RuntimeError(f"Stage-C base component differs from Stage-A: {seed}:{name}")
    thresholds = [float(calibration["calibration"]["intervals"][role]["b_r_stored"]) for role in ROLES]
    if thresholds != [float(value) for value in calibration["calibration"]["threshold_vector"]] or thresholds != [float(value) for value in c_payload["threshold_vector_role_order"]]:
        raise RuntimeError(f"threshold identity mismatch for seed {seed}")
    expected_augmented = [*base_manifest["token_order"], base_manifest["noop_operator"]]
    if c_payload.get("base_token_order") != base_manifest["token_order"] or c_payload.get("augmented_token_order") != expected_augmented or c_payload.get("noop_operator") != base_manifest["noop_operator"] or c_payload.get("trainable_parameter_names") != ["embedding.noop_embedding"]:
        raise RuntimeError(f"Stage-C mapping metadata mismatch for seed {seed}")
    e_n_hash = tensor_digest({"e_N": state["embedding.noop_embedding"]})
    if c_result["invariants"]["e_N_final_hash"] != e_n_hash:
        raise RuntimeError(f"Stage-C final e_N identity mismatch for seed {seed}")
    core = GenericNRoleBinder(base_manifest, seed, list(ROLES))
    non_embedding = {name: value for name, value in state.items() if not name.startswith("embedding.")}
    missing, unexpected = core.load_state_dict(non_embedding, strict=False)
    if missing != ["embedding.weight"] or unexpected:
        raise RuntimeError(f"unexpected Stage-C core load result for seed {seed}: {missing}/{unexpected}")
    core.embedding = AppendedNoopEmbedding(state["embedding.old_embeddings"], state["embedding.noop_embedding"])
    if tuple(core.embedding.old_embeddings.shape) != (37, DMODEL) or tuple(core.embedding.noop_embedding.shape) != (DMODEL,):
        raise RuntimeError(f"Stage-C embedding shape mismatch for seed {seed}")
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    core_state_hash = tensor_digest(dict(state))
    threshold_hash = tensor_digest({"thresholds": torch.tensor(thresholds, dtype=torch.float32)})
    return core, {"seed": seed, "checkpoint": source_record(stage_c_checkpoint), "stage_a_checkpoint": source_record(stage_a_checkpoint), "calibration": {**source_record(stage_b_calibration), "self_hash": calibration_self_hash}, "manifest": source_record(manifest_path), "thresholds": thresholds, "threshold_hash_before": threshold_hash, "source_hashes_before": dict(hashes), "core_state_hash_before": core_state_hash, "noop_operator": base_manifest["noop_operator"], "noop_internal_id": 37, "e_N_final_hash": e_n_hash, "checkpoint_payload": c_payload}


def semantic_signature(roles: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple((item["presence"], item["selected_token"], item["RAW"], item["CANON"]) for item in roles)


def evaluate_cases(
    core: GenericNRoleBinder,
    cases: list[Case],
    thresholds: list[float],
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    on_result: Callable[[Case, dict[str, Any]], None],
) -> None:
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
                    present_expected = role in case.subset
                    if present_expected:
                        target_score = finite(scores[target_position].item())
                        other = torch.cat((scores[:target_position], scores[target_position + 1 :]))
                        real_margin = finite(target_score - float(other.max().item()))
                        null_margin = finite(target_score - threshold)
                        predicted_present = best_value > threshold
                        selected_state = details["values"][row_index, pointer_index]
                        pending_decode.append((row_index, query_index, pointer_index))
                        role_data[row_index][query_index] = {"presence": "present" if predicted_present else "none", "pointer": pointer_index if predicted_present else -1, "selected_token": case.tokens[pointer_index] if predicted_present else "", "RAW": -1, "CANON": -1, "expected_present": True, "target_position": target_position, "target_score": target_score, "max_other_score": finite(other.max().item()), "margin_present_to_NULL": null_margin, "margin_present_to_real": real_margin, "margin_strict": null_margin > 0.0 and real_margin > 0.0, "correct": False, "noop_rejection_margin": None, "noop_worst_score": None}
                    else:
                        absent_margin = finite(threshold - best_value)
                        predicted_present = best_value > threshold
                        if predicted_present:
                            pending_decode.append((row_index, query_index, pointer_index))
                        role_data[row_index][query_index] = {"presence": "present" if predicted_present else "none", "pointer": pointer_index if predicted_present else -1, "selected_token": case.tokens[pointer_index] if predicted_present else "", "RAW": -1, "CANON": -1, "expected_present": False, "target_position": -1, "target_score": None, "max_other_score": None, "margin_absent": absent_margin, "margin_strict": absent_margin > 0.0, "correct": not predicted_present, "noop_rejection_margin": None, "noop_worst_score": None}
                    if case.noop_positions:
                        noop_scores = scores[list(case.noop_positions)]
                        noop_worst = finite(noop_scores.max().item())
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
                    item["correct"] = item["presence"] == "present" and item["pointer"] == item["target_position"] and item["RAW"] == expected_value and item["CANON"] == expected_value and item["RAW"] == item["CANON"]
            for row_index, case in enumerate(chunk):
                roles = role_data[row_index]
                record = {"case_id": case.case_id, "pair_id": case.pair_id, "seed": case.seed, "subset": "_".join(case.subset), "subset_size": len(case.subset), "order": "_".join(case.order), "assignment": dict(case.assignment), "D": case.distractor, "augmented": case.augmented, "tokens": list(case.tokens), "roles": roles, "structured_correct": all(item["correct"] for item in roles), "margins_strict": all(item["margin_strict"] for item in roles), "semantic_signature": semantic_signature(roles)}
                on_result(case, record)


def update_aggregate(aggregate: dict[str, Any], record: dict[str, Any]) -> None:
    size = int(record["subset_size"])
    aggregate["counts_by_size"][str(size)] += 1
    aggregate["structured_correct"] += int(record["structured_correct"])
    aggregate["margins_strict"] += int(record["margins_strict"])
    for index, role in enumerate(ROLES):
        item = record["roles"][index]
        aggregate["false_positives"] += int(not item["expected_present"] and item["presence"] == "present")
        aggregate["false_negatives"] += int(item["expected_present"] and item["presence"] != "present")
        aggregate["RAW_correct"] += int(item["expected_present"] and item["RAW"] == dict(record["assignment"])[role])
        aggregate["CANON_correct"] += int(item["expected_present"] and item["CANON"] == dict(record["assignment"])[role])
        if item["expected_present"]:
            aggregate["present_roles"] += 1
            aggregate["present_to_NULL_margin_min"] = min(aggregate["present_to_NULL_margin_min"], item["margin_present_to_NULL"])
            aggregate["present_to_real_margin_min"] = min(aggregate["present_to_real_margin_min"], item["margin_present_to_real"])
        else:
            aggregate["absent_roles"] += 1
            aggregate["absent_margin_min"] = min(aggregate["absent_margin_min"], item["margin_absent"])
        if item["noop_rejection_margin"] is not None:
            aggregate["noop_rejection_margin_min"] = min(aggregate["noop_rejection_margin_min"], item["noop_rejection_margin"])


def new_aggregate() -> dict[str, Any]:
    return {"counts_by_size": {str(size): 0 for size in range(1, 5)}, "structured_correct": 0, "margins_strict": 0, "false_positives": 0, "false_negatives": 0, "RAW_correct": 0, "CANON_correct": 0, "present_roles": 0, "absent_roles": 0, "present_to_NULL_margin_min": float("inf"), "present_to_real_margin_min": float("inf"), "absent_margin_min": float("inf"), "noop_rejection_margin_min": float("inf")}


def finite_aggregate(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("present_to_NULL_margin_min", "present_to_real_margin_min", "absent_margin_min", "noop_rejection_margin_min"):
        if math.isinf(result[key]):
            result[key] = None
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty paired output root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    manifests: dict[int, tuple[dict[str, Any], Path]] = {seed: load_t7_manifest(seed) for seed in SEEDS}
    checker_results = {}
    for seed, (manifest, manifest_path) in manifests.items():
        base_token_ids = {token: index for index, token in enumerate(manifest["token_order"])}
        augmented_token_ids = {**base_token_ids, manifest["noop_operator"]: 37}
        checker_results[str(seed)] = logical_checker(seed, manifest, base_token_ids, augmented_token_ids)
    checker_pass = all(result["status"] == "passed" for result in checker_results.values())
    if not checker_pass:
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7-NOOP-NONE-LEXICAL / PAIRED DEVELOPMENT EVALUATION", "reason": "sealed logical checker failed before any model forward", "logical_checker": checker_results}
        write_json(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    runtime_before = {"file_sha256": sha256_file(BASE_CHECKPOINT), "tensor_state_hash": state_hash(executor.state_dict())}
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    runtime_before["codebook_hash"] = tensor_digest({"codebook": codebook})
    seed_reports: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            manifest, manifest_path = manifests[seed]
            core, identity = load_verified_core(seed, manifest, manifest_path)
            base_token_ids = {token: index for index, token in enumerate(manifest["token_order"])}
            augmented_token_ids = {**base_token_ids, manifest["noop_operator"]: 37}
            seed_output = OUTPUT_ROOT / f"seed_{seed}"
            seed_output.mkdir(parents=True, exist_ok=True)
            cases_path = seed_output / "cases.jsonl"
            base_aggregate = new_aggregate()
            augmented_aggregate = new_aggregate()
            pair_count = pair_pass = 0
            base_lookup: dict[str, tuple[tuple[tuple[Any, ...], ...], bool]] = {}
            with cases_path.open("w", encoding="utf-8", newline="\n") as stream:
                for size in range(1, 5):
                    for subset in itertools.combinations(ROLES, size):
                        base_cases, augmented_cases = build_cases_for_subset(seed, subset, base_token_ids, augmented_token_ids, manifest["operator_for_role"], manifest["noop_operator"], [int(value) for value in manifest["permutation"]])
                        def write_base(_case: Case, record: dict[str, Any]) -> None:
                            stream.write(json.dumps({"kind": "base", **record}, separators=(",", ":")) + "\n")
                            update_aggregate(base_aggregate, record)
                            base_lookup[record["pair_id"]] = (tuple(tuple(item) for item in record["semantic_signature"]), bool(record["structured_correct"]))
                        evaluate_cases(core, base_cases, identity["thresholds"], executor, codebook, write_base)
                        def write_augmented(_case: Case, record: dict[str, Any]) -> None:
                            nonlocal pair_count, pair_pass
                            base_signature, base_structured = base_lookup[record["pair_id"]]
                            record["base_structured_correct"] = base_structured
                            record["semantic_equivalent"] = tuple(tuple(item) for item in record["semantic_signature"]) == base_signature
                            record["paired_structured_pass"] = bool(base_structured and record["structured_correct"] and record["semantic_equivalent"])
                            stream.write(json.dumps({"kind": "augmented", **record}, separators=(",", ":")) + "\n")
                            update_aggregate(augmented_aggregate, record)
                            pair_count += 1
                            pair_pass += int(record["paired_structured_pass"])
                        evaluate_cases(core, augmented_cases, identity["thresholds"], executor, codebook, write_augmented)
                        base_lookup.clear()
            identity["core_state_hash_after"] = tensor_digest(dict(core.state_dict()))
            identity["core_state_hash_identical"] = identity["core_state_hash_before"] == identity["core_state_hash_after"]
            identity["threshold_hash_after"] = tensor_digest({"thresholds": torch.tensor(identity["thresholds"], dtype=torch.float32)})
            identity["threshold_hash_identical"] = identity["threshold_hash_before"] == identity["threshold_hash_after"]
            identity["source_hashes_after"] = {"stage_a": sha256_file(STAGE_A_ROOT / f"seed_{seed}" / "final.pt"), "calibration": sha256_file(STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"), "manifest": sha256_file(manifest_path), "stage_c": sha256_file(STAGE_C_ROOT / "training" / f"seed_{seed}" / "final.pt")}
            identity["source_files_identical"] = identity["source_hashes_before"] == identity["source_hashes_after"]
            seed_pass = base_aggregate["structured_correct"] == sum(EXPECTED_BASE.values()) and base_aggregate["margins_strict"] == sum(EXPECTED_BASE.values()) and augmented_aggregate["structured_correct"] == sum(EXPECTED_AUGMENTED.values()) and augmented_aggregate["margins_strict"] == sum(EXPECTED_AUGMENTED.values()) and pair_pass == pair_count and base_aggregate["false_positives"] == 0 and base_aggregate["false_negatives"] == 0 and augmented_aggregate["false_positives"] == 0 and augmented_aggregate["false_negatives"] == 0 and augmented_aggregate["RAW_correct"] == augmented_aggregate["present_roles"] and augmented_aggregate["CANON_correct"] == augmented_aggregate["present_roles"]
            seed_pass = seed_pass and identity["core_state_hash_identical"] and identity["threshold_hash_identical"] and identity["source_files_identical"]
            seed_report = {"seed": seed, "status": "passed" if seed_pass else "failed", "identity": {key: value for key, value in identity.items() if key != "checkpoint_payload"}, "base": finite_aggregate(base_aggregate), "augmented": finite_aggregate(augmented_aggregate), "pairs": {"count": pair_count, "structured_semantic_pass": pair_pass}, "cases": source_record(cases_path)}
            write_json(seed_output / "summary.json", seed_report)
            seed_reports.append(seed_report)
        runtime_after = {"file_sha256": sha256_file(BASE_CHECKPOINT), "tensor_state_hash": state_hash(executor.state_dict()), "codebook_hash": tensor_digest({"codebook": executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()})}
        runtime_integrity = {"before": runtime_before, "after": runtime_after, "file_sha256_identical": runtime_before["file_sha256"] == runtime_after["file_sha256"], "tensor_state_hash_identical": runtime_before["tensor_state_hash"] == runtime_after["tensor_state_hash"], "codebook_hash_identical": runtime_before["codebook_hash"] == runtime_after["codebook_hash"]}
        all_base = sum(report["base"]["structured_correct"] for report in seed_reports)
        all_augmented = sum(report["augmented"]["structured_correct"] for report in seed_reports)
        all_pairs = sum(report["pairs"]["structured_semantic_pass"] for report in seed_reports)
        total_pairs = sum(report["pairs"]["count"] for report in seed_reports)
        result = {"schema": "T7-noop-none-lexical-paired-development-v1", "status": "completed", "classification": "T7-NOOP-NONE-LEXICAL DEVELOPMENT: CLOSED/PASS" if all(report["status"] == "passed" for report in seed_reports) and runtime_integrity["file_sha256_identical"] and runtime_integrity["tensor_state_hash_identical"] and runtime_integrity["codebook_hash_identical"] else "T7-NOOP-NONE-LEXICAL DEVELOPMENT: VALID FAIL", "task": "T7-NOOP-NONE-LEXICAL / PAIRED DEVELOPMENT EVALUATION", "training": False, "recalibration": False, "architecture_changes": False, "multi_clause_evaluation": True, "fresh_pass_strong": False, "logical_checker": checker_results, "expected_totals": {"base": EXPECTED_ALL_BASE, "augmented": EXPECTED_ALL_AUGMENTED, "combined": EXPECTED_ALL_BASE + EXPECTED_ALL_AUGMENTED}, "observed_totals": {"base": all_base, "augmented": all_augmented, "paired_structured_semantic_pass": all_pairs, "pairs": total_pairs}, "runtime_integrity": runtime_integrity, "seeds": seed_reports, "output_root": str(OUTPUT_ROOT.relative_to(ROOT)).replace("\\", "/"), "elapsed_seconds": time.perf_counter() - started}
        unsigned = dict(result)
        unsigned["artifact_self_hash"] = PLACEHOLDER
        digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        result["artifact_self_hash"] = digest
        write_json(OUTPUT_ROOT / "results.json", result)
        print(json.dumps({"status": result["status"], "classification": result["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": digest, "base": all_base, "augmented": all_augmented, "pairs": total_pairs, "pair_pass": all_pairs}, sort_keys=True))
        return 0 if result["classification"].endswith("CLOSED/PASS") else 1
    except Exception as error:
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7-NOOP-NONE-LEXICAL / PAIRED DEVELOPMENT EVALUATION", "error_type": type(error).__name__, "error": str(error), "completed_seed_reports": seed_reports, "logical_checker": checker_results}
        write_json(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
