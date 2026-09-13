"""Repair T7 Stage-B atomic sanity using the approved frozen decoder path only."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
    write_self_hashed,
)
from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402


PREVIOUS_ATTEMPT_COMMIT = "df371328a97ce271d84a790a14928e6e69d7ea07"
PREVIOUS_STAGE_B_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_development"
STAGE_A_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_development" / "training"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_sanity_repair"
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER = "__SELF_HASH__"


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite sanity value: {result}")
    return result


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_self_hash(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    parsed = json.loads(data.decode("utf-8"))
    stored = parsed.get("artifact_self_hash")
    if not isinstance(stored, str):
        raise RuntimeError(f"missing self-hash: {path}")
    parsed["artifact_self_hash"] = PLACEHOLDER
    actual = hashlib.sha256((json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    if actual != stored:
        raise RuntimeError(f"invalid self-hash: {path}")
    return json.loads(data.decode("utf-8")), stored


def load_calibration(seed: int, manifest_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = PREVIOUS_STAGE_B_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
    calibration, self_hash = verify_self_hash(path)
    if calibration.get("seed") != seed:
        raise RuntimeError(f"calibration seed mismatch: {seed}")
    if calibration.get("manifest", {}).get("sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise RuntimeError(f"calibration manifest binding mismatch: {seed}")
    if calibration.get("core_checkpoint", {}).get("sha256") != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest():
        raise RuntimeError(f"calibration checkpoint binding mismatch: {seed}")
    intervals = calibration.get("calibration", {}).get("intervals", {})
    if set(intervals) != set(ROLES):
        raise RuntimeError(f"calibration roles mismatch: {seed}")
    thresholds = [float(intervals[role]["b_r_stored"]) for role in ROLES]
    expected = calibration.get("calibration", {}).get("threshold_vector")
    if thresholds != [float(value) for value in expected]:
        raise RuntimeError(f"role-ordered threshold vector mismatch: {seed}")
    for role in ROLES:
        interval = intervals[role]
        if float(torch.tensor(interval["b_r_stored"], dtype=torch.float32).item()) != float(interval["b_r_stored"]):
            raise RuntimeError(f"calibration reread dtype mismatch: {seed}/{role}")
        if interval["stored_strictly_inside"] is not True or not float(interval["P_r_min_minus_b_r"]) > 0.0 or not float(interval["b_r_minus_N_r_max"]) > 0.0:
            raise RuntimeError(f"calibration interval no longer strict: {seed}/{role}")
    return calibration, path, {"self_hash": self_hash, "thresholds": thresholds, "holdues": {role: [float(intervals[role]["P_r_min_minus_b_r"]), float(intervals[role]["b_r_minus_N_r_max"])] for role in ROLES}}


def atomic_sanity(encoder: GenericNRoleBinder, manifest: dict[str, Any], executor: torch.nn.Module, codebook: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    rows = manifest["train"]
    ids = torch.tensor([[encoder.vocab.encode(row["operator"]), encoder.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
    lengths = torch.full((len(rows),), 2, dtype=torch.long)
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32)
    with torch.no_grad():
        details = encoder(ids, lengths)
    outputs: list[dict[str, Any]] = []
    present_decisions = 0
    absent_decisions = 0
    present_pointer_correct = 0
    present_raw_correct = 0
    present_canon_correct = 0
    absent_null_correct = 0
    complete = 0
    errors: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_outputs: dict[str, Any] = {}
        row_complete = True
        expected_value = int(row["value"])
        for query_index, query_role in enumerate(ROLES):
            with torch.no_grad():
                scores = details["scores"][query_index][row_index, :2]
                best_score, pointer = scores.max(dim=0)
            present = bool(best_score.item() > threshold_tensor[query_index].item())
            if present:
                present_decisions += 1
                pointer_value = int(pointer.item())
                selected = details["values"][row_index, pointer_value].unsqueeze(0)
                with torch.no_grad():
                    raw, canonical = decode_states(executor, codebook, selected)
                raw_value = int(raw[0].item())
                canonical_value = int(canonical[0].item())
                expected_present = query_role == row["role"]
                pointer_ok = expected_present and pointer_value == 1
                raw_ok = pointer_ok and raw_value == expected_value
                canon_ok = pointer_ok and canonical_value == expected_value
                present_pointer_correct += int(pointer_ok)
                present_raw_correct += int(raw_ok)
                present_canon_correct += int(canon_ok)
                correct = pointer_ok and raw_ok and canon_ok and raw_value == canonical_value
                row_complete = row_complete and correct
                row_outputs[query_role] = {"presence": True, "pointer": pointer_value, "score": finite(best_score.item()), "expected_present": expected_present, "pointer_correct": pointer_ok, "RAW": raw_value, "CANON": canonical_value, "RAW_correct": raw_ok, "CANON_correct": canon_ok, "RAW_CANON_equal": raw_value == canonical_value, "structured_correct": correct}
            else:
                absent_decisions += 1
                expected_absent = query_role != row["role"]
                absent_ok = expected_absent
                absent_null_correct += int(absent_ok)
                row_complete = row_complete and absent_ok
                row_outputs[query_role] = {"presence": False, "pointer": None, "score": finite(best_score.item()), "expected_absent": expected_absent, "NULL": True, "structured_correct": absent_ok}
        complete += int(row_complete)
        outputs.append({"row_index": row_index, "role": row["role"], "argument": row["argument"], "expected_value": expected_value, "queries": row_outputs, "structured_complete": row_complete})
        if not row_complete and len(errors) < 20:
            errors.append(outputs[-1])
    return {"rows": len(rows), "expected_rows": 128, "decisions": len(rows) * len(ROLES), "present_decisions": present_decisions, "absent_decisions": absent_decisions, "expected_present_decisions": 128, "expected_absent_decisions": 384, "present_pointer_correct": present_pointer_correct, "present_RAW_correct": present_raw_correct, "present_CANON_correct": present_canon_correct, "absent_NULL_correct": absent_null_correct, "structured_complete": complete, "structured_expected": 128, "errors": errors, "passed": len(rows) == 128 and present_decisions == 128 and absent_decisions == 384 and present_pointer_correct == 128 and present_raw_correct == 128 and present_canon_correct == 128 and absent_null_correct == 384 and complete == 128 and not errors, "outputs": outputs}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty repair root: {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    runtime_file_before = hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()
    runtime_state_before = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_before = tensor_hash({"codebook": codebook})
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        encoder, manifest, manifest_path, checkpoint_before, core_state_before = _load_core(seed)
        calibration, calibration_path, calibration_binding = load_calibration(seed, manifest_path, STAGE_A_ROOT / f"seed_{seed}" / "final.pt")
        thresholds = calibration_binding["thresholds"]
        sanity = atomic_sanity(encoder, manifest, executor, codebook, thresholds)
        core_state_after = state_hash(encoder.state_dict())
        checkpoint_path = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
        checkpoint_after = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        result = {"seed": seed, "previous_invalid_attempt_commit": PREVIOUS_ATTEMPT_COMMIT, "calibration": {**source_record(calibration_path), "self_hash": calibration_binding["self_hash"], "reused_without_recalibration": True, "threshold_vector_role_order": thresholds, "holdues_role_order": calibration_binding["holdues"]}, "core_checkpoint": {**source_record(checkpoint_path), "sha256_before": checkpoint_before, "sha256_after": checkpoint_after, "sha256_identical": checkpoint_before == checkpoint_after}, "core_tensor_state": {"hash_before": core_state_before, "hash_after": core_state_after, "hash_identical": core_state_before == core_state_after}, "atomic_sanity": sanity, "passed": sanity["passed"] and checkpoint_before == checkpoint_after and core_state_before == core_state_after}
        result_path = OUTPUT_ROOT / "results" / f"result_{seed}_v1.json"
        write_self_hashed(result_path, result)
        result["result"] = source_record(result_path)
        per_seed.append(result)
    runtime_file_after = hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()
    runtime_state_after = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook_after_tensor = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_after = tensor_hash({"codebook": codebook_after_tensor})
    runtime_integrity = {"file_sha256_before": runtime_file_before, "file_sha256_after": runtime_file_after, "file_sha256_identical": runtime_file_before == runtime_file_after, "tensor_state_hash_before": runtime_state_before, "tensor_state_hash_after": runtime_state_after, "tensor_state_hash_identical": runtime_state_before == runtime_state_after, "codebook_hash_before": codebook_before, "codebook_hash_after": codebook_after, "codebook_hash_identical": codebook_before == codebook_after, "runtime_checkpoint": source_record(BASE_CHECKPOINT), "codebook_construction": "executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE+VALUE_COUNT,dtype=torch.long)).detach()"}
    all_pass = all(item["passed"] for item in per_seed) and all(runtime_integrity[key] for key in ("file_sha256_identical", "tensor_state_hash_identical", "codebook_hash_identical"))
    consolidated = {"schema": "T7-noop-none-stage-b-atomic-sanity-repair-v1", "status": "completed", "classification": "T7 STAGE-B DEVELOPMENT: CLOSED/PASS" if all_pass else "T7 STAGE-B DEVELOPMENT: HALT/IMPLEMENTATION DEFECT", "task": "T7-STAGE-B-ATOMIC-SANITY-REPAIR", "previous_invalid_attempt_commit": PREVIOUS_ATTEMPT_COMMIT, "authorization": "Sol-authorized repair; Stage C and multi-clause evaluation held", "summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "calibrations_reused": 5, "recalibration": False, "training": False, "backward": False, "optimizer": False, "NOOP_initialized": False, "NOOP_scores_measured": False, "multi_clause_evaluation": False, "present_decisions": sum(item["atomic_sanity"]["present_decisions"] for item in per_seed), "absent_decisions": sum(item["atomic_sanity"]["absent_decisions"] for item in per_seed), "structured_complete": sum(item["atomic_sanity"]["structured_complete"] for item in per_seed), "present_pointer_correct": sum(item["atomic_sanity"]["present_pointer_correct"] for item in per_seed), "present_RAW_correct": sum(item["atomic_sanity"]["present_RAW_correct"] for item in per_seed), "present_CANON_correct": sum(item["atomic_sanity"]["present_CANON_correct"] for item in per_seed)}, "runner": source_record(SCRIPT_PATH), "runtime_integrity": runtime_integrity, "calibrations": [item["calibration"] for item in per_seed], "cores": [item["core_checkpoint"] for item in per_seed], "per_seed_results": [item["result"] for item in per_seed], "per_seed": per_seed, "elapsed_seconds": time.perf_counter() - started, "next_stage": "Stage C remains held; no automatic continuation."}
    artifact_path = OUTPUT_ROOT / "results.json"
    digest = write_self_hashed(artifact_path, consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(), "artifact_self_hash": digest, "seeds": len(per_seed)}, sort_keys=True))
    return 0


def _load_core(seed: int) -> tuple[GenericNRoleBinder, dict[str, Any], Path, str, str]:
    manifest, manifest_path = load_t7_manifest(seed)
    checkpoint_path = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
    checkpoint_before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise RuntimeError(f"core checkpoint provenance mismatch: {seed}")
    encoder = GenericNRoleBinder(manifest, seed, list(ROLES))
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, manifest, manifest_path, checkpoint_before, state_hash(encoder.state_dict())


if __name__ == "__main__":
    raise SystemExit(main())
