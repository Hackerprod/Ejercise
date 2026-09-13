"""Execute only T7 Stage-B base NULL calibration on frozen Stage-A cores."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from execute_t7_noop_none_stage_a_development import (  # noqa: E402
    HarnessError,
    ROLES,
    SEEDS,
    VALUE_COUNT,
    load_t7_manifest,
    source_record,
    state_hash,
    write_self_hashed,
)
from t5_nrole_design_audit import (  # noqa: E402
    GenericNRoleBinder,
    generic_background_scores,
    generic_catalog_scores,
)


OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_development"
CHECKPOINT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_development" / "training"
SCRIPT_PATH = Path(__file__).resolve()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"non-finite calibration value: {result}")
    return result


def load_frozen_core(seed: int) -> tuple[GenericNRoleBinder, dict[str, Any], Path, str, str]:
    manifest, manifest_path = load_t7_manifest(seed)
    checkpoint = CHECKPOINT_ROOT / f"seed_{seed}" / "final.pt"
    if not checkpoint.exists():
        raise HarnessError(f"missing Stage-A checkpoint for {seed}")
    checkpoint_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise HarnessError(f"checkpoint provenance mismatch for {seed}")
    encoder = GenericNRoleBinder(manifest, seed, list(ROLES))
    encoder.load_state_dict(payload["encoder"], strict=True)
    if tuple(encoder.embedding.weight.shape) != (37, 16):
        raise HarnessError(f"frozen Stage-A core is not base 37x16 for {seed}")
    if payload.get("noop_in_stage_a_vocab") is not False or payload.get("noop_in_batch") is not False or payload.get("noop_in_objective") is not False or payload.get("noop_in_background_catalog") is not False:
        raise HarnessError(f"checkpoint NOOP exclusion provenance failed for {seed}")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, manifest, manifest_path, checkpoint_before, state_hash(encoder.state_dict())


def calibrate(encoder: GenericNRoleBinder, manifest: dict[str, Any], real_dtype: torch.dtype) -> dict[str, Any]:
    with torch.no_grad():
        catalog = generic_catalog_scores(encoder, manifest)
        backgrounds, contexts = generic_background_scores(encoder, manifest)
    if len(contexts) != 40 or any(context["kind"] == "NOOP" for context in contexts):
        raise HarnessError("base BG catalog is not exactly 40 non-NOOP contexts")
    intervals: dict[str, Any] = {}
    certificates: dict[str, Any] = {}
    for query_index, role in enumerate(ROLES):
        self_scores = catalog[query_index][query_index]
        p_value, p_index = self_scores.min(dim=0)
        negatives: list[dict[str, Any]] = []
        for source_index, source_role in enumerate(ROLES):
            if source_index == query_index:
                continue
            cross = catalog[query_index][source_index]
            n_value, n_index = cross.max(dim=0)
            negatives.append({"kind": "role→role", "source_role": source_role, "argument_index": int(n_index.item()), "score": finite(n_value.item()), "context": f"{source_role}/ARG_{int(n_index.item()):02d}"})
        bg = backgrounds[query_index]
        bg_value, bg_index = bg.max(dim=0)
        bg_context = contexts[int(bg_index.item())]
        negatives.append({"kind": "BG", "source_role": None, "background_index": int(bg_index.item()), "score": finite(bg_value.item()), "context": bg_context["context"], "background_kind": bg_context["kind"]})
        negative = max(negatives, key=lambda item: item["score"])
        p_min = finite(p_value.item())
        n_max = finite(negative["score"])
        width = finite(p_min - n_max)
        if not p_min > n_max:
            raise HarnessError(f"calibration interval does not exist for {role}: {p_min} <= {n_max}")
        midpoint = n_max + (p_min - n_max) / 2.0
        stored_tensor = torch.tensor(midpoint, dtype=real_dtype)
        stored = finite(stored_tensor.item())
        left = finite(p_min - stored)
        right = finite(stored - n_max)
        intervals[role] = {"P_r_min": p_min, "P_r_min_arg_index": int(p_index.item()), "P_r_min_context": f"{role}/ARG_{int(p_index.item()):02d}", "P_r_min_score_source": {"role": role, "argument": f"ARG_{int(p_index.item()):02d}", "argument_index": int(p_index.item()), "score": p_min}, "N_r_max": n_max, "N_r_max_source": negative, "interval_width": width, "interval_exists": True, "b_r_midpoint": finite(midpoint), "b_r_stored": stored, "storage_dtype": str(real_dtype), "P_r_min_minus_b_r": left, "b_r_minus_N_r_max": right, "stored_strictly_inside": bool(left > 0.0 and right > 0.0), "negative_candidates": negatives}
        certificates[role] = {"P_r_min": p_min, "N_r_max": n_max, "interval_width": width, "b_r_stored": stored, "strictly_positive_after_storage": bool(left > 0.0 and right > 0.0)}
    return {"formula": "b_r=N_r^max+(P_r^min-N_r^max)/2", "source": "128 atomic relations + 40 base BG contexts; 96 other-role arguments + 40 BG negatives", "real_inference_dtype": str(real_dtype), "intervals": intervals, "certificates": certificates, "interval_count": len(intervals), "all_strictly_inside": len(intervals) == 4 and all(item["stored_strictly_inside"] for item in intervals.values()), "all_finite": all(math.isfinite(value) for role in intervals.values() for key, value in role.items() if key in {"P_r_min", "N_r_max", "interval_width", "b_r_midpoint", "b_r_stored", "P_r_min_minus_b_r", "b_r_minus_N_r_max"}), "threshold_vector": [intervals[role]["b_r_stored"] for role in ROLES]}


def atomic_sanity(encoder: GenericNRoleBinder, manifest: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    rows = manifest["train"]
    ids = torch.tensor([[encoder.vocab.encode(row["operator"]), encoder.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
    lengths = torch.full((len(rows),), 2, dtype=torch.long)
    thresholds = torch.tensor(calibration["threshold_vector"], dtype=torch.float32)
    with torch.no_grad():
        details = encoder(ids, lengths)
        values_logits = details["values"]
    output_rows = 0
    present_decisions = 0
    absent_decisions = 0
    present_correct = 0
    absent_correct = 0
    complete_rows = 0
    errors: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_complete = True
        expected_value = int(row["value"])
        role_outputs: dict[str, Any] = {}
        for query_index, query_role in enumerate(ROLES):
            scores = details["scores"][query_index][row_index, :2]
            best_score, pointer = scores.max(dim=0)
            present = bool(best_score.item() > thresholds[query_index].item())
            if present:
                present_decisions += 1
                decoded = int(values_logits[row_index, int(pointer.item())].argmax().item())
                canonical = decoded
                expected_present = query_role == row["role"]
                correct = expected_present and int(pointer.item()) == 1 and decoded == expected_value
                present_correct += int(correct)
                if not correct:
                    row_complete = False
                role_outputs[query_role] = {"present": True, "pointer": int(pointer.item()), "decoded_index": decoded, "canonical_index": canonical, "score": finite(best_score.item()), "expected_present": expected_present, "correct": correct}
            else:
                absent_decisions += 1
                expected_absent = query_role != row["role"]
                correct = expected_absent
                absent_correct += int(correct)
                if not correct:
                    row_complete = False
                role_outputs[query_role] = {"present": False, "pointer": None, "decoded_index": None, "canonical_index": None, "score": finite(best_score.item()), "expected_absent": expected_absent, "correct": correct}
        output_rows += 1
        complete_rows += int(row_complete)
        if not row_complete and len(errors) < 20:
            errors.append({"row_index": row_index, "role": row["role"], "argument": row["argument"], "expected_value": expected_value, "outputs": role_outputs})
    return {"rows": output_rows, "expected_rows": 128, "present_decisions": present_decisions, "absent_decisions": absent_decisions, "expected_present_decisions": 128, "expected_absent_decisions": 384, "present_correct": present_correct, "absent_correct": absent_correct, "structured_complete": complete_rows, "structured_expected": 128, "errors": errors, "passed": output_rows == 128 and present_decisions == 128 and absent_decisions == 384 and present_correct == 128 and absent_correct == 384 and complete_rows == 128 and not errors}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-B output root: {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        encoder, manifest, manifest_path, checkpoint_before, state_before = load_frozen_core(seed)
        checkpoint_path = CHECKPOINT_ROOT / f"seed_{seed}" / "final.pt"
        calibration = calibrate(encoder, manifest, next(encoder.parameters()).dtype)
        calibration_path = OUTPUT_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
        calibration_payload = {"schema": "T7-noop-none-lexical-stage-b-calibration-v1", "task": "T7 STAGE-B DEVELOPMENT", "seed": seed, "manifest": source_record(manifest_path), "core_checkpoint": source_record(checkpoint_path), "core_checkpoint_sha256": checkpoint_before, "core_state_hash": state_before, "noop_operator": manifest["noop_operator"], "base_vocab_rows": 37, "noop_in_model": False, "calibration": calibration}
        write_self_hashed(calibration_path, calibration_payload)
        reread = json.loads(calibration_path.read_text(encoding="utf-8"))
        stored_vector = [float(reread["calibration"]["intervals"][role]["b_r_stored"]) for role in ROLES]
        reread_vector = [float(torch.tensor(value, dtype=next(encoder.parameters()).dtype).item()) for value in stored_vector]
        if stored_vector != reread_vector or stored_vector != calibration["threshold_vector"]:
            raise HarnessError(f"stored threshold reread mismatch for {seed}")
        sanity = atomic_sanity(encoder, manifest, calibration)
        state_after = state_hash(encoder.state_dict())
        checkpoint_after = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        immutability = {"checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after, "checkpoint_hash_identical": checkpoint_before == checkpoint_after, "tensor_state_hash_before": state_before, "tensor_state_hash_after": state_after, "tensor_state_hash_identical": state_before == state_after, "checkpoint_and_state_hashes_are_distinct_schemes": True}
        result = {"seed": seed, "calibration": source_record(calibration_path), "calibration_self_hash": reread["artifact_self_hash"], "intervals": calibration["intervals"], "threshold_vector": calibration["threshold_vector"], "reread_verification": {"stored_vector": stored_vector, "reconstructed_vector": reread_vector, "matches": stored_vector == reread_vector, "strictly_inside_after_reread": all(item["stored_strictly_inside"] for item in calibration["intervals"].values())}, "atomic_sanity": sanity, "immutability": immutability, "stage_b_gate": {"intervals": calibration["all_strictly_inside"], "finite": calibration["all_finite"], "atomic_sanity": sanity["passed"], "immutability": immutability["checkpoint_hash_identical"] and immutability["tensor_state_hash_identical"], "pass": calibration["all_strictly_inside"] and calibration["all_finite"] and sanity["passed"] and immutability["checkpoint_hash_identical"] and immutability["tensor_state_hash_identical"]}, "forbidden_operations": {"training": False, "backward": False, "optimizer": False, "noop_initialized": False, "noop_scores_measured": False, "multi_clause_evaluation": False, "stage_c": False}}
        result_path = OUTPUT_ROOT / "results" / f"result_{seed}_v1.json"
        write_self_hashed(result_path, result)
        result["result"] = source_record(result_path)
        per_seed.append(result)
    consolidated = {"schema": "T7-noop-none-lexical-stage-b-development-v1", "status": "completed", "classification": "T7 STAGE-B DEVELOPMENT: PASS" if all(item["stage_b_gate"]["pass"] for item in per_seed) else "T7 STAGE-B DEVELOPMENT: VALID FAIL", "task": "T7-NOOP-NONE-LEXICAL / STAGE-B DEVELOPMENT", "authorization": "Sol-authorized Stage B only; Stage C and multi-clause evaluation held", "elapsed_seconds": time.perf_counter() - started, "summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "intervals_checked": 20, "intervals_strictly_inside": all(item["stage_b_gate"]["intervals"] for item in per_seed), "atomic_present_decisions": sum(item["atomic_sanity"]["present_decisions"] for item in per_seed), "atomic_absent_decisions": sum(item["atomic_sanity"]["absent_decisions"] for item in per_seed), "atomic_structured_complete": sum(item["atomic_sanity"]["structured_complete"] for item in per_seed), "training": False, "noop_initialized": False, "multi_clause_evaluation": False}, "runner": source_record(SCRIPT_PATH), "core_source": source_record(SCRIPT_DIR / "t5_nrole_design_audit.py"), "preparation_manifests": {str(seed): source_record(load_t7_manifest(seed)[1]) for seed in SEEDS}, "calibrations": [item["calibration"] for item in per_seed], "per_seed_results": [item["result"] for item in per_seed], "per_seed": per_seed, "next_stage": "Stage C remains held; no automatic continuation."}
    artifact_path = OUTPUT_ROOT / "results.json"
    digest = write_self_hashed(artifact_path, consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(), "artifact_self_hash": digest, "seeds": len(per_seed), "intervals": 20}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
