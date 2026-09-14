"""Execute authorized T7 Stage-B fresh NULL calibration on frozen Stage-A cores."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from execute_t6_activeset_midpoint_development import (  # noqa: E402
    ROLES,
    generic_background_contexts,
    generic_background_scores,
    generic_catalog_scores,
    sha256_file,
    state_hash,
)
from execute_t7_noop_none_stage_a_fresh import fresh_manifest_view, load_manifest, verify_raw_self_hash  # noqa: E402
from repair_t7_noop_none_stage_b_atomic_sanity import atomic_sanity as repaired_atomic_sanity  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402


SEEDS = (7801, 7802, 7803, 7804, 7805)
STAGE_A_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_fresh" / "training"
STAGE_A_RESULT_ROOT = STAGE_A_ROOT
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_fresh"
SCRIPT_PATH = Path(__file__).resolve()
REPAIR_PATH = SCRIPT_DIR / "repair_t7_noop_none_stage_b_atomic_sanity.py"
REPAIR_COMMIT = "fea9ea43f255893451230db4714360d409e0363f"
PLACEHOLDER = "__SELF_HASH__"


class HarnessError(RuntimeError):
    """Implementation/provenance failure; never classify as scientific failure."""


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"non-finite value: {result}")
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    written = path.read_bytes()
    stored = json.loads(written.decode("utf-8"))["artifact_self_hash"]
    if sha256_bytes(written.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != stored:
        raise HarnessError(f"self-hash verification failed: {path}")
    return digest


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git_commit_contains_repair() -> bool:
    repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    if not repo_root:
        return False
    commit_path = Path(repo_root).joinpath(REPAIR_PATH).relative_to(Path(repo_root)).as_posix()
    completed = subprocess.run(["git", "cat-file", "-e", f"{REPAIR_COMMIT}:{commit_path}"], cwd=ROOT, capture_output=True, check=False)
    return completed.returncode == 0


def load_a_provenance(seed: int, manifest: dict[str, Any], manifest_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    result_path = STAGE_A_RESULT_ROOT / f"seed_{seed}" / "results.json"
    if not result_path.exists() or not checkpoint_path.exists():
        raise HarnessError(f"missing Stage-A fresh artifact for seed {seed}")
    verify_raw_self_hash(result_path)
    a_result = json.loads(result_path.read_bytes().decode("utf-8"))
    if a_result.get("seed") != seed:
        raise HarnessError(f"Stage-A result seed mismatch: {seed}")
    checkpoint_sha = sha256_file(checkpoint_path)
    a3 = a_result.get("A3", {})
    if a3.get("checkpoint", {}).get("sha256") != checkpoint_sha or a3.get("checkpoint_sha256_after_audits") != checkpoint_sha:
        raise HarnessError(f"Stage-A checkpoint result binding mismatch: {seed}")
    if a3.get("decoder_checkpoint", {}).get("path") != BASE_CHECKPOINT.relative_to(ROOT).as_posix():
        raise HarnessError(f"Stage-A decoder path mismatch: {seed}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise HarnessError(f"Stage-A checkpoint provenance mismatch: {seed}")
    if payload.get("fresh_init") is not True or payload.get("prior_binder_checkpoint_loaded") is not False:
        raise HarnessError(f"Stage-A fresh-init provenance mismatch: {seed}")
    if a_result.get("fresh_initialization", {}).get("base_vocab_rows") != 37 or payload.get("noop_in_stage_a_vocab") is not False or payload.get("noop_in_batch") is not False or payload.get("noop_in_objective") is not False or payload.get("noop_in_background_catalog") is not False:
        raise HarnessError(f"Stage-A NOOP exclusion provenance mismatch: {seed}")
    if payload.get("base_token_order") != manifest["token_order"] or payload.get("base_token_ids_physical") != manifest["token_ids"] or payload.get("operator_for_role") != manifest["operator_for_role"] or payload.get("permutation") != manifest["permutation"]:
        raise HarnessError(f"Stage-A checkpoint mapping metadata mismatch: {seed}")
    return a_result, payload, checkpoint_sha


def load_frozen_core(seed: int) -> tuple[GenericNRoleBinder, dict[str, Any], Path, str, str, dict[str, Any]]:
    manifest, manifest_path = load_manifest(seed)
    view = fresh_manifest_view(manifest)
    checkpoint_path = STAGE_A_ROOT / f"seed_{seed}" / "final.pt"
    a_result, payload, checkpoint_before = load_a_provenance(seed, manifest, manifest_path, checkpoint_path)
    encoder = GenericNRoleBinder(view, seed, list(ROLES))
    encoder.load_state_dict(payload["encoder"], strict=True)
    if tuple(encoder.embedding.weight.shape) != (37, 16):
        raise HarnessError(f"frozen Stage-A core is not 37x16: {seed}")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise HarnessError(f"Stage-A core not fully frozen: {seed}")
    return encoder, manifest, manifest_path, checkpoint_before, state_hash(encoder.state_dict()), a_result


def calibrate(encoder: GenericNRoleBinder, manifest: dict[str, Any], real_dtype: torch.dtype) -> dict[str, Any]:
    view = fresh_manifest_view(manifest)
    with torch.no_grad():
        catalog = generic_catalog_scores(encoder, view)
        backgrounds, contexts = generic_background_scores(encoder, view)
    if len(contexts) != 40 or any(context["kind"] == "NOOP" or "NOOP" in context["context"] for context in contexts):
        raise HarnessError("base BG catalog is not exactly 40 non-NOOP contexts")
    intervals: dict[str, Any] = {}
    for query_index, role in enumerate(ROLES):
        self_scores = catalog[query_index][query_index]
        p_value, p_index = self_scores.min(dim=0)
        p_min = finite(p_value.item())
        p_arg_index = int(p_index.item())
        negatives: list[dict[str, Any]] = []
        for source_index, source_role in enumerate(ROLES):
            if source_index == query_index:
                continue
            cross = catalog[query_index][source_index]
            n_value, n_index = cross.max(dim=0)
            negatives.append({"kind": "role→role", "source_role": source_role, "argument_index": int(n_index.item()), "score": finite(n_value.item()), "context": f"{source_role}/ARG_{int(n_index.item()):02d}", "score_position": 1, "candidate_count": VALUE_COUNT})
        bg = backgrounds[query_index]
        bg_value, bg_index = bg.max(dim=0)
        bg_position = int(bg_index.item())
        bg_context = contexts[bg_position]
        negatives.append({"kind": "BG", "source_role": None, "background_index": bg_position, "score": finite(bg_value.item()), "context": bg_context["context"], "background_kind": bg_context["kind"], "score_position": bg_context["score_position"], "candidate_count": len(contexts)})
        negative = max(negatives, key=lambda item: item["score"])
        n_max = finite(negative["score"])
        width = finite(p_min - n_max)
        midpoint = finite(n_max + (p_min - n_max) / 2.0)
        stored_tensor = torch.tensor(midpoint, dtype=real_dtype)
        stored = finite(stored_tensor.item())
        left = finite(p_min - stored)
        right = finite(stored - n_max)
        intervals[role] = {"P_r_min": p_min, "P_r_min_arg_index": p_arg_index, "P_r_min_context": f"{role}/ARG_{p_arg_index:02d}", "P_r_min_score_position": 1, "N_r_max": n_max, "N_r_max_source": negative, "negative_candidate_count": 96 + len(contexts), "negative_partition": {"other_role_arguments": 96, "base_background_contexts": len(contexts)}, "interval_width": width, "interval_exists": bool(p_min > n_max), "b_r_midpoint": midpoint, "b_r_stored": stored, "storage_dtype": str(real_dtype), "P_r_min_minus_b_r": left, "b_r_minus_N_r_max": right, "stored_strictly_inside": bool(left > 0.0 and right > 0.0)}
    all_finite = all(math.isfinite(float(interval[key])) for interval in intervals.values() for key in ("P_r_min", "N_r_max", "interval_width", "b_r_midpoint", "b_r_stored", "P_r_min_minus_b_r", "b_r_minus_N_r_max"))
    return {"formula": "b_r = N_r^max + (P_r^min - N_r^max)/2", "source": "128 atomic OP→ARG relations + 40 base BG contexts; 96 other-role arguments + 40 BG negatives", "score_positions": {"OP→ARG": 1, "ARG→LINK": 1, "START→OP": 0, "LINK→OP": 1, "START_prefix": "zero prefix; no BOS"}, "real_inference_dtype": str(real_dtype), "intervals": intervals, "interval_count": len(intervals), "all_finite": all_finite, "all_intervals_exist": all(item["interval_exists"] for item in intervals.values()), "all_strictly_inside": len(intervals) == 4 and all(item["stored_strictly_inside"] for item in intervals.values()), "threshold_vector": [intervals[role]["b_r_stored"] for role in ROLES]}


def reread_calibration(path: Path, calibration: dict[str, Any], real_dtype: torch.dtype) -> dict[str, Any]:
    stored_hash, file_sha = verify_raw_self_hash(path)
    stored = json.loads(path.read_bytes().decode("utf-8"))
    intervals = stored["calibration"]["intervals"]
    vector = [float(intervals[role]["b_r_stored"]) for role in ROLES]
    reconstructed = [float(torch.tensor(value, dtype=real_dtype).item()) for value in vector]
    strict_after = {}
    for role in ROLES:
        interval = intervals[role]
        p_min = float(interval["P_r_min"])
        n_max = float(interval["N_r_max"])
        threshold = reconstructed[ROLES.index(role)]
        strict_after[role] = {"P_r_min_minus_b_r": finite(p_min - threshold), "b_r_minus_N_r_max": finite(threshold - n_max), "strictly_inside": bool(p_min - threshold > 0.0 and threshold - n_max > 0.0)}
    if vector != reconstructed or vector != [float(value) for value in calibration["threshold_vector"]]:
        raise HarnessError(f"calibration threshold reread mismatch: {path}")
    return {"stored_vector_role_order": vector, "reconstructed_vector_role_order": reconstructed, "matches": vector == reconstructed, "strict_after_reread": strict_after, "strictly_inside_after_reread": all(item["strictly_inside"] for item in strict_after.values()), "self_hash": stored_hash, "file_sha256": file_sha}


def writer_probe() -> dict[str, Any]:
    path = OUTPUT_ROOT / "writer_byte_exact_probe.json"
    value = {"schema": "t7-stage-b-fresh-byte-exact-writer-probe-v1", "artifact_self_hash": PLACEHOLDER}
    digest = sha256_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    value["artifact_self_hash"] = digest
    written = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    actual = path.read_bytes()
    if sha256_bytes(actual.replace(digest.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1)) != digest:
        raise HarnessError("Stage-B fresh writer probe failed")
    return {"path": path.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": sha256_bytes(actual), "verified": True, "write_method": "write_bytes(UTF-8)", "line_endings": {"CRLF": actual.count(b"\r\n"), "LF": actual.count(b"\n")}}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-B fresh root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    if not git_commit_contains_repair():
        raise HarnessError(f"approved repair commit does not contain {REPAIR_PATH.name}: {REPAIR_COMMIT}")
    runner_hash = sha256_file(SCRIPT_PATH)
    repair_hash = sha256_file(REPAIR_PATH)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in executor.parameters()):
        raise HarnessError("decoder runtime not fully frozen")
    runtime_file_before = sha256_file(BASE_CHECKPOINT)
    runtime_state_before = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_before = tensor_hash({"codebook": codebook})
    per_seed: list[dict[str, Any]] = []
    decoder_hashes: set[str] = set()
    try:
        for seed in SEEDS:
            encoder, manifest, manifest_path, checkpoint_before, core_state_before, a_result = load_frozen_core(seed)
            decoder_hashes.add(a_result["A3"]["decoder_checkpoint"]["sha256"])
            if a_result["A3"]["decoder_checkpoint"]["sha256"] != runtime_file_before:
                raise HarnessError(f"decoder hash differs from Stage-A A3 binding: {seed}")
            calibration = calibrate(encoder, manifest, next(encoder.parameters()).dtype)
            calibration_path = OUTPUT_ROOT / "calibrations" / f"calibration_{seed}_v1.json"
            calibration_payload = {"schema": "t7-noop-none-lexical-stage-b-fresh-calibration-v1", "task": "T7 STAGE-B FRESH", "seed": seed, "manifest": source_record(manifest_path), "core_checkpoint": source_record(STAGE_A_ROOT / f"seed_{seed}" / "final.pt"), "core_checkpoint_sha256": checkpoint_before, "core_state_hash": core_state_before, "base_vocab_rows": 37, "noop_operator": manifest["noop_operator"], "noop_in_model": False, "noop_scores_measured": False, "calibration": calibration, "sanity_implementation": {"path": REPAIR_PATH.relative_to(ROOT).as_posix(), "commit": REPAIR_COMMIT, "sha256": repair_hash, "entrypoint": "atomic_sanity", "decoder_path": "decode_states(executor, codebook, selected)", "buggy_stage_b_atomic_sanity_executed": False}}
            write_self_hashed(calibration_path, calibration_payload)
            reread = reread_calibration(calibration_path, calibration, next(encoder.parameters()).dtype)
            sanity = repaired_atomic_sanity(encoder, manifest, executor, codebook, reread["reconstructed_vector_role_order"])
            checkpoint_after = sha256_file(STAGE_A_ROOT / f"seed_{seed}" / "final.pt")
            core_state_after = state_hash(encoder.state_dict())
            immutability = {"checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after, "checkpoint_hash_identical": checkpoint_before == checkpoint_after, "tensor_state_hash_before": core_state_before, "tensor_state_hash_after": core_state_after, "tensor_state_hash_identical": core_state_before == core_state_after, "checkpoint_and_tensor_hashes_are_distinct_schemes": True}
            gate = {"intervals": calibration["all_strictly_inside"], "finite": calibration["all_finite"], "reread": reread["matches"] and reread["strictly_inside_after_reread"], "atomic_sanity": sanity["passed"], "immutability": immutability["checkpoint_hash_identical"] and immutability["tensor_state_hash_identical"], "pass": calibration["all_strictly_inside"] and calibration["all_finite"] and reread["matches"] and reread["strictly_inside_after_reread"] and sanity["passed"] and immutability["checkpoint_hash_identical"] and immutability["tensor_state_hash_identical"]}
            result = {"seed": seed, "status": "passed" if gate["pass"] else "VALID SCIENTIFIC FAIL", "calibration": {**source_record(calibration_path), "self_hash": reread["self_hash"], "reread": reread}, "intervals": calibration["intervals"], "threshold_vector_role_order": reread["reconstructed_vector_role_order"], "atomic_sanity": sanity, "core_checkpoint": {**source_record(STAGE_A_ROOT / f"seed_{seed}" / "final.pt"), "sha256_before": checkpoint_before, "sha256_after": checkpoint_after}, "core_tensor_state": {"hash_before": core_state_before, "hash_after": core_state_after}, "immutability": immutability, "sanity_implementation": calibration_payload["sanity_implementation"], "stage_b_gate": gate, "forbidden_operations": {"training": False, "backward": False, "optimizer": False, "noop_initialized": False, "noop_scores_measured": False, "multi_clause_evaluation": False, "stage_c": False}}
            result_path = OUTPUT_ROOT / "results" / f"result_{seed}_v1.json"
            write_self_hashed(result_path, result)
            result["result"] = source_record(result_path)
            per_seed.append(result)
    except HarnessError as error:
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7 STAGE-B FRESH", "error_type": type(error).__name__, "error": str(error), "runner_sha256": runner_hash, "repair": {"path": REPAIR_PATH.relative_to(ROOT).as_posix(), "commit": REPAIR_COMMIT, "sha256": repair_hash}, "completed_seed_results": per_seed}
        write_json_bytes(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2
    runtime_file_after = sha256_file(BASE_CHECKPOINT)
    runtime_state_after = state_hash(executor.state_dict())
    with torch.no_grad():
        codebook_after_tensor = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    codebook_after = tensor_hash({"codebook": codebook_after_tensor})
    runtime_integrity = {"file_sha256_before": runtime_file_before, "file_sha256_after": runtime_file_after, "file_sha256_identical": runtime_file_before == runtime_file_after, "tensor_state_hash_before": runtime_state_before, "tensor_state_hash_after": runtime_state_after, "tensor_state_hash_identical": runtime_state_before == runtime_state_after, "codebook_hash_before": codebook_before, "codebook_hash_after": codebook_after, "codebook_hash_identical": codebook_before == codebook_after}
    probe = writer_probe()
    all_pass = len(per_seed) == 5 and len(decoder_hashes) == 1 and all(item["stage_b_gate"]["pass"] for item in per_seed) and all(runtime_integrity[key] for key in ("file_sha256_identical", "tensor_state_hash_identical", "codebook_hash_identical"))
    consolidated = {"schema": "t7-noop-none-lexical-stage-b-fresh-v1", "status": "completed", "classification": "T7 STAGE-B FRESH: CLOSED/PASS" if all_pass else "T7 STAGE-B FRESH: VALID SCIENTIFIC FAIL", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION / STAGE-B FRESH", "authorization": "Sol-authorized Stage B fresh only; Stage C and paired evaluation held", "summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "intervals_checked": len(per_seed) * 4, "intervals_expected": 20, "intervals_strictly_inside": all(item["stage_b_gate"]["intervals"] for item in per_seed), "atomic_present_decisions": sum(item["atomic_sanity"]["present_decisions"] for item in per_seed), "atomic_absent_decisions": sum(item["atomic_sanity"]["absent_decisions"] for item in per_seed), "atomic_present_expected": 640, "atomic_absent_expected": 1920, "atomic_present_pointer_correct": sum(item["atomic_sanity"]["present_pointer_correct"] for item in per_seed), "atomic_present_RAW_correct": sum(item["atomic_sanity"]["present_RAW_correct"] for item in per_seed), "atomic_present_CANON_correct": sum(item["atomic_sanity"]["present_CANON_correct"] for item in per_seed), "atomic_absent_NULL_correct": sum(item["atomic_sanity"]["absent_NULL_correct"] for item in per_seed), "structured_complete": sum(item["atomic_sanity"]["structured_complete"] for item in per_seed), "structured_expected": 640, "training": False, "backward": False, "optimizer": False, "noop_initialized": False, "noop_scores_measured": False, "multi_clause_evaluation": False, "stage_c": False}, "runner": {**source_record(SCRIPT_PATH), "sha256_used_for_run": runner_hash}, "sanity_implementation": {"path": REPAIR_PATH.relative_to(ROOT).as_posix(), "commit": REPAIR_COMMIT, "sha256": repair_hash, "entrypoint": "atomic_sanity", "decoder_path": "decode_states(executor, codebook, selected)", "buggy_stage_b_atomic_sanity_executed": False, "repair_commit_contains_file": True}, "decoder_a3_bindings": sorted(decoder_hashes), "runtime_integrity": runtime_integrity, "byte_exact_writer_probe": probe, "source_registry": [source_record(path) for path in (SCRIPT_PATH, REPAIR_PATH, SCRIPT_DIR / "execute_t7_noop_none_stage_a_fresh.py", SCRIPT_DIR / "execute_t6_activeset_midpoint_development.py", SCRIPT_DIR / "t5_nrole_design_audit.py", SCRIPT_DIR / "ctrl2_common.py")], "calibrations": [item["calibration"] for item in per_seed], "per_seed_results": [item["result"] for item in per_seed], "per_seed": per_seed, "next_stage": "Stage C remains held; no automatic continuation.", "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unavailable"}, "elapsed_seconds": time.perf_counter() - started}
    artifact_path = OUTPUT_ROOT / "results.json"
    digest = write_self_hashed(artifact_path, consolidated)
    stored, file_sha = verify_raw_self_hash(artifact_path)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_self_hash": stored, "file_sha256": file_sha, "seeds": len(per_seed), "intervals": 20}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
