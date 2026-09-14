"""Execute T7 Stage-A fresh training for the sealed 7801-7805 domains."""

from __future__ import annotations

import copy
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
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from execute_t6_activeset_midpoint_development import (  # noqa: E402
    BACKGROUND_COUNT,
    ROLES,
    UPDATES,
    build_training_batch,
    d1_atomic,
    gradient_sanity,
    objective_n4,
    sha256_file,
    state_hash,
)
from execute_t5_n4_development import d2_certificates  # noqa: E402
from t5_nrole_design_audit import GenericNRoleBinder  # noqa: E402
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7801, 7802, 7803, 7804, 7805)
ACTIVE_ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
PREPARATION_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_fresh_preparation"
MANIFEST_ROOT = PREPARATION_ROOT / "manifests"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_fresh"
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER = "__SELF_HASH__"


class HarnessError(RuntimeError):
    """Implementation or provenance error; never classify as scientific failure."""


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
    return digest


def verify_raw_self_hash(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    stored = str(json.loads(data.decode("utf-8"))["artifact_self_hash"])
    needle = stored.encode("utf-8")
    if data.count(needle) != 1:
        raise HarnessError(f"expected one self-hash value in {path}")
    verified = sha256_bytes(data.replace(needle, PLACEHOLDER.encode("utf-8"), 1))
    if verified != stored:
        raise HarnessError(f"raw self-hash mismatch in {path}: {stored} != {verified}")
    return stored, sha256_bytes(data)


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    if not path.exists():
        raise HarnessError(f"missing sealed fresh manifest: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "T7-noop-none-lexical-fresh-preparation-manifest-v1" or raw.get("seed") != seed:
        raise HarnessError(f"unexpected fresh manifest schema/seed: {seed}")
    permutation = raw.get("permutation")
    if not isinstance(permutation, list) or len(permutation) != VALUE_COUNT or sorted(permutation) != list(range(VALUE_COUNT)):
        raise HarnessError(f"missing or invalid permutation for seed {seed}")
    if raw.get("token_order")[-1:] == [raw.get("noop_operator")]:
        raise HarnessError(f"fresh Stage-A manifest already includes NOOP in base token_order: {seed}")
    if len(raw.get("token_order", [])) != 37 or len(raw.get("token_ids", {})) != 37:
        raise HarnessError(f"fresh Stage-A base vocabulary is not 37 rows: {seed}")
    if set(raw["token_ids"]) != set(raw["token_order"]):
        raise HarnessError(f"physical token set differs from local token order: {seed}")
    if raw.get("stage_c_noop_internal_id") != 37 or raw.get("stage_c_noop_external_id") != raw["id_block"][1] + 1:
        raise HarnessError(f"invalid reserved NOOP mapping: {seed}")
    if set(raw.get("operator_for_role", {})) != set(ACTIVE_ROLES):
        raise HarnessError(f"active role mapping mismatch: {seed}")
    base = copy.deepcopy(raw)
    base["operator_for_role"] = dict(raw["operator_for_role"])
    base["train"] = [{"argument": f"ARG_{index:02d}", "constraints": role.lower(), "kind": "atomic", "operator": raw["operator_for_role"][role], "role": role, "value": int(permutation[index])} for role in ACTIVE_ROLES for index in range(VALUE_COUNT)]
    base["multi_clause_train"] = 0
    base["joint_examples_in_training"] = 0
    base["test_rows_used"] = 0
    return base, path


def target_mapping_check(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    permutation = manifest["permutation"]
    physical = manifest["token_ids"]
    internal = {token: index for index, token in enumerate(manifest["token_order"])}
    rows = manifest["train"]
    mapping: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        argument = str(row["argument"])
        arg_index = int(argument[4:])
        expected = int(permutation[arg_index])
        actual = int(row["value"])
        entry = {"seed": seed, "role": row["role"], "operator": row["operator"], "argument": argument, "arg_index": arg_index, "target_value": actual, "expected_permutation_value": expected, "physical_operator_id": int(physical[row["operator"]]), "physical_argument_id": int(physical[argument]), "internal_operator_index": internal[row["operator"]], "internal_argument_index": internal[argument]}
        mapping.append(entry)
        if expected != actual:
            mismatches.append(entry)
    counts = {role: sum(row["role"] == role for row in rows) for role in ACTIVE_ROLES}
    if len(rows) != 128 or counts != {role: 32 for role in ACTIVE_ROLES} or mismatches:
        raise HarnessError(f"fresh exhaustive target mapping failed for seed {seed}")
    return {"rows_checked": len(rows), "expected_rows": 128, "checks": len(mapping), "mismatch_count": 0, "role_counts": counts, "passed": True, "mapping": mapping}


def fresh_manifest_view(manifest: dict[str, Any]) -> dict[str, Any]:
    ordered_token_ids = {token: manifest["token_ids"][token] for token in manifest["token_order"]}
    view = {"permutation": list(manifest["permutation"]), "token_order": list(manifest["token_order"]), "token_ids": ordered_token_ids, "operator_for_role": dict(manifest["operator_for_role"]), "train": list(manifest["train"]), "role_names": list(ACTIVE_ROLES), "value_count": VALUE_COUNT}
    return view


def train_seed(seed: int, manifest: dict[str, Any], manifest_path: Path, observations: dict[str, Tensor], labels: dict[str, Tensor], supervisor: torch.nn.Module, executor: torch.nn.Module, codebook: Tensor, runner_hash: str) -> dict[str, Any]:
    destination = OUTPUT_ROOT / "training" / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    mapping = target_mapping_check(manifest, seed)
    view = fresh_manifest_view(manifest)
    encoder = GenericNRoleBinder(view, seed, list(ACTIVE_ROLES))
    if tuple(encoder.embedding.weight.shape) != (37, 16):
        raise HarnessError(f"fresh encoder shape mismatch: {seed}")
    batch = build_training_batch(encoder, view, labels)
    if batch["token_ids"].shape != (128, 2) or any(int(row["value"]) != int(manifest["permutation"][int(row["argument"][4:])]) for row in manifest["train"]):
        raise HarnessError(f"constructed fresh batch mapping mismatch: {seed}")
    initial_state_hash = state_hash(encoder.state_dict())
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last_losses: dict[str, float] = {}
    started = time.perf_counter()
    for step in range(1, UPDATES + 1):
        progress = (step - 1) / (UPDATES - 1)
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.zero_grad(set_to_none=True)
        losses = objective_n4(encoder, supervisor, executor, codebook, view, batch, observations, labels)
        last_losses = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        final_losses_tensor = objective_n4(encoder, supervisor, executor, codebook, view, batch, observations, labels)
    final_losses = {name: finite(final_losses_tensor[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    torch.save({"encoder": encoder.state_dict(), "seed": seed, "updates": UPDATES, "manifest_sha256": sha256_file(manifest_path), "fresh_init": True, "prior_binder_checkpoint_loaded": False, "base_vocab_only": True, "noop_operator": manifest["noop_operator"], "noop_in_stage_a_vocab": False, "noop_in_batch": False, "noop_in_objective": False, "noop_in_background_catalog": False, "operator_role_assignment": manifest["operator_role_assignment"], "operator_for_role": manifest["operator_for_role"], "base_token_order": manifest["token_order"], "base_token_ids_physical": manifest["token_ids"], "permutation": manifest["permutation"], "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_sep_coefficient": 1.0, "L_sep_terms": 16, "background_contexts": BACKGROUND_COUNT, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "fresh_runner_sha256_before_first_step": runner_hash}, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    state_before_audits = state_hash(encoder.state_dict())
    atomic = d1_atomic(encoder, executor, codebook, view)
    certificates = d2_certificates(encoder, view)
    gradient = gradient_sanity(encoder, supervisor, executor, codebook, view, batch, observations, labels)
    state_after_audits = state_hash(encoder.state_dict())
    checkpoint_hash_after = sha256_file(checkpoint)
    a1_pass = bool(atomic["pass"])
    a2_pass = bool(certificates["all_formal_positive"] and gradient["strictly_positive"])
    result = {"status": "trained", "seed": seed, "elapsed_seconds": time.perf_counter() - started, "updates": UPDATES, "runner_sha256_before_first_optimizer_step": runner_hash, "manifest": {**source_record(manifest_path), "original_manifest_untouched": True}, "fresh_initialization": {"constructor": "GenericNRoleBinder(fresh base manifest view, seed)", "seed": seed, "prior_binder_checkpoint_loaded": False, "base_vocab_rows": 37, "noop_operator": manifest["noop_operator"], "noop_row_created": False, "e_N_initialized": False}, "base_vocab_mapping": {"token_order": manifest["token_order"], "physical_to_internal": [{"token": token, "physical_id": int(manifest["token_ids"][token]), "internal_index": index} for index, token in enumerate(manifest["token_order"])], "operator_for_role": manifest["operator_for_role"]}, "exhaustive_target_check": mapping, "training": {"rows": {role: 32 for role in ACTIVE_ROLES} | {"total": 128}, "objective": "L_behavior^(F,A)+L_ref^(F,A,M,H)+L_sep^(4+BG)", "L_behavior_roles": ["FLOOR", "AVOID"], "L_ref_roles": list(ACTIVE_ROLES), "L_sep": {"role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "coefficient": 1.0, "reduction": "mean over 16 terms"}, "background_contexts": 40, "multi_clause_train": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "last_update_losses": last_losses, "final_losses": final_losses}, "architecture": {"name": "RELKEY", "d_model": 16, "queries": list(ACTIVE_ROLES), "query_shape": [4, 16], "address_key": "k_t=LN(W_k[e_prev,e_t]+a_k)", "activation_between_linear_and_LN": False, "shared_w_v": True, "gate": False, "stage_b": False, "soft_mixture": False, "hard_pointer": "pure argmax over valid positions"}, "A1": {"atomic": atomic, "pass": a1_pass}, "A2": {"separation": certificates, "gradient_LINK": gradient, "pass": a2_pass}, "A3": {"checkpoint": source_record(checkpoint), "checkpoint_sha256_before_audits": checkpoint_hash, "checkpoint_sha256_after_audits": checkpoint_hash_after, "checkpoint_unchanged_after_audits": checkpoint_hash == checkpoint_hash_after, "core_state_hash_before_audits": state_before_audits, "core_state_hash_after_audits": state_after_audits, "core_state_unchanged_after_audits": state_before_audits == state_after_audits, "frozen_for_future_stage_b": True, "decoder_checkpoint": source_record(BASE_CHECKPOINT), "supervisor_checkpoint": source_record(CTRL7_CHECKPOINT)}, "noop_exclusion": {"operator": manifest["noop_operator"], "in_base_vocab": False, "in_train_batch": False, "in_objective": False, "in_background_catalog": False, "e_N_initialized": False, "stage_b": False, "evaluation": False}, "stage_a_gate": {"A1": a1_pass, "A2": a2_pass, "pass": a1_pass and a2_pass}, "initial_state_hash": initial_state_hash}
    result_path = destination / "results.json"
    result_hash = write_self_hashed(result_path, result)
    result["result"] = {"path": result_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(result_path), "artifact_self_hash": result_hash}
    return result


def writer_probe() -> dict[str, Any]:
    path = OUTPUT_ROOT / "writer_byte_exact_probe.json"
    payload = {"schema": "t7-stage-a-fresh-byte-exact-writer-probe-v1", "artifact_self_hash": PLACEHOLDER}
    unsigned = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected = sha256_bytes(unsigned)
    payload["artifact_self_hash"] = expected
    final = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes = path
    write_bytes.parent.mkdir(parents=True, exist_ok=True)
    write_bytes.write_bytes(final)
    actual = write_bytes.read_bytes()
    stored = json.loads(actual.decode("utf-8"))["artifact_self_hash"]
    verified = sha256_bytes(actual.replace(stored.encode("utf-8"), PLACEHOLDER.encode("utf-8"), 1))
    if actual != final or stored != expected or stored != verified:
        raise HarnessError("Stage-A fresh byte-exact writer probe failed")
    return {"path": path.relative_to(ROOT).as_posix(), "artifact_self_hash": stored, "file_sha256": sha256_bytes(actual), "verified": True, "write_method": "write_bytes(UTF-8)", "verification_method": "raw byte replacement without JSON reserialization", "line_endings": {"CRLF": actual.count(b"\r\n"), "LF": actual.count(b"\n")}}


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-A fresh root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    runner_hash = sha256_file(SCRIPT_PATH)
    manifests = {seed: load_manifest(seed) for seed in SEEDS}
    observations, labels = load_source()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    for parameter in supervisor.parameters():
        parameter.requires_grad_(False)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            manifest, manifest_path = manifests[seed]
            results.append(train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, runner_hash))
        runtime_file = sha256_file(BASE_CHECKPOINT)
        runtime_state = state_hash(executor.state_dict())
        with torch.no_grad():
            runtime_codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()
        runtime_codebook_hash = sha256_bytes(runtime_codebook.cpu().contiguous().numpy().tobytes())
        probe = writer_probe()
        source_paths = [SCRIPT_PATH, SCRIPT_DIR / "execute_t7_noop_none_stage_a_development.py", SCRIPT_DIR / "execute_t7_noop_none_stage_b_development.py", SCRIPT_DIR / "execute_t7_noop_none_stage_c_development.py", SCRIPT_DIR / "evaluate_t7_noop_none_lexical_paired_development_conformance.py", SCRIPT_DIR / "prepare_t7_noop_none_lexical_fresh_preparation.py", SCRIPT_DIR / "t5_nrole_design_audit.py", SCRIPT_DIR / "execute_t6_activeset_midpoint_development.py", SCRIPT_DIR / "ctrl2_common.py", SCRIPT_DIR / "train_t2_i0_baseline_b.py", SCRIPT_DIR / "train_t2_i2_r2.py", ROOT / "t1_trainability" / "t7_production_core_noop_integration.py"]
        environment = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "python_implementation": platform.python_implementation(), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unavailable"}
        all_pass = all(item["stage_a_gate"]["pass"] for item in results)
        consolidated = {"schema": "t7-noop-none-lexical-stage-a-fresh-v1", "status": "completed", "classification": "T7 STAGE-A FRESH: PASS" if all_pass else "T7 STAGE-A FRESH: VALID SCIENTIFIC FAIL", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION / STAGE-A FRESH", "authorization": "Sol-authorized Stage A fresh only; B/C/paired evaluation held", "seeds": list(SEEDS), "recipe": {"fresh_init_per_seed": True, "active_roles": list(ACTIVE_ROLES), "queries": list(ACTIVE_ROLES), "base_vocab_rows": 37, "noop_excluded": True, "batch_rows": {role: 32 for role in ACTIVE_ROLES} | {"total": 128}, "updates": UPDATES, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A)+L_ref^(F,A,M,H)+L_sep^(4+BG)", "L_sep_terms": {"role_role": 12, "role_bg": 4, "total": 16, "coefficient": 1.0, "background_contexts": 40}, "multi_clause_train": 0, "null_calibration": False}, "executive_summary": {"seeds_completed": len(results), "seeds_expected": 5, "A1_all_seeds": all(item["stage_a_gate"]["A1"] for item in results), "A2_all_seeds": all(item["stage_a_gate"]["A2"] for item in results), "atomic_mapping_controls": 640, "certificates_per_seed": 16, "training_performed": True, "stage_b_started": False, "stage_c_started": False, "evaluation_opened": False, "new_checkpoints": True}, "runner": {"path": SCRIPT_PATH.relative_to(ROOT).as_posix(), "sha256_before_first_optimizer_step": runner_hash}, "source_registry": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_paths], "decoder_runtime": {"checkpoint": source_record(BASE_CHECKPOINT), "supervisor": source_record(CTRL7_CHECKPOINT), "runtime_file_sha256": runtime_file, "runtime_tensor_state_hash": runtime_state, "codebook_hash": runtime_codebook_hash}, "byte_exact_writer_probe": probe, "environment": environment, "manifests": {str(item["seed"]): item["manifest"] for item in results}, "checkpoints": [item["A3"]["checkpoint"] for item in results], "per_seed_results": results, "forbidden_operations_confirmed": ["stage B", "stage C", "NULL calibration", "NOOP/e_N initialization", "multi-clause evaluation", "770x artifact load"], "elapsed_seconds": time.perf_counter() - started}
        artifact = OUTPUT_ROOT / "results.json"
        digest = write_self_hashed(artifact, consolidated)
        stored, file_hash = verify_raw_self_hash(artifact)
        print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact.relative_to(ROOT).as_posix(), "artifact_self_hash": stored, "file_sha256": file_hash, "seeds": len(results)}, sort_keys=True))
        return 0
    except HarnessError as error:
        invalid = {"status": "INVALID/HARNESS BUG", "task": "T7-NOOP-NONE-LEXICAL-FRESH-PREPARATION / STAGE-A FRESH", "error_type": type(error).__name__, "error": str(error), "runner_sha256": runner_hash, "completed_seed_results": results}
        write_json_bytes(OUTPUT_ROOT / "INVALID_HARNESS_BUG.json", invalid)
        print(json.dumps(invalid, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
