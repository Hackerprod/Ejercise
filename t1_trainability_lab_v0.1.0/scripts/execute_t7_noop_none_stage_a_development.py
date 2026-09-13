"""Execute only T7 Stage-A base-binder development for seeds 7701-7705."""

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
from t5_nrole_design_audit import (  # noqa: E402
    GenericNRoleBinder,
    generic_background_scores,
    generic_catalog_scores,
)
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7701, 7702, 7703, 7704, 7705)
MANIFEST_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation" / "manifests"
OUTPUT_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_development"
TRAINING_ROOT = OUTPUT_ROOT / "training"
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER = "__SELF_HASH__"


class HarnessError(RuntimeError):
    """Implementation or provenance error; never classify as scientific failure."""


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise HarnessError(f"non-finite value: {result}")
    return result


def source_record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = PLACEHOLDER
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    parsed = json.loads(written.decode("utf-8"))
    parsed["artifact_self_hash"] = PLACEHOLDER
    verified = hashlib.sha256((json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    if verified != digest:
        raise HarnessError(f"self-hash verification failed: {path}")
    return digest


def load_t7_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    if not path.exists():
        raise HarnessError(f"missing sealed T7 manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T7-noop-none-lexical-development-preparation-manifest-v1":
        raise HarnessError(f"unexpected T7 manifest schema for {seed}")
    if int(manifest.get("seed")) != seed:
        raise HarnessError(f"manifest seed mismatch for {seed}")
    permutation = manifest.get("permutation")
    if not isinstance(permutation, list) or len(permutation) != VALUE_COUNT or sorted(permutation) != list(range(VALUE_COUNT)):
        raise HarnessError(f"invalid value permutation for {seed}")
    stage_a_tokens = manifest.get("vocabulary", {}).get("stage_a_token_order")
    token_ids = manifest.get("token_ids")
    if not isinstance(stage_a_tokens, list) or len(stage_a_tokens) != 37 or not isinstance(token_ids, dict) or set(token_ids) != set(stage_a_tokens) or len(set(token_ids.values())) != 37:
        raise HarnessError(f"invalid base vocabulary metadata for {seed}")
    noop_operator = manifest.get("noop_operator")
    if noop_operator in stage_a_tokens or manifest.get("vocabulary", {}).get("stage_a_noop_operator_present") is not False:
        raise HarnessError(f"NOOP present in Stage-A vocabulary for {seed}")
    operator_roles = manifest.get("operator_role_assignment", {})
    operator_for_role = {role: operator for operator, role in operator_roles.items() if role != "NOOP"}
    if set(operator_for_role) != set(ROLES) or manifest.get("operator_for_role") != operator_for_role:
        raise HarnessError(f"active operator inverse mismatch for {seed}")

    base = copy.deepcopy(manifest)
    base["token_order"] = list(stage_a_tokens)
    base["token_ids"] = {token: int(token_ids[token]) for token in stage_a_tokens}
    base["operator_for_role"] = operator_for_role
    train: list[dict[str, Any]] = []
    for role in ROLES:
        for index in range(VALUE_COUNT):
            train.append({
                "argument": f"ARG_{index:02d}",
                "constraints": role.lower(),
                "kind": "atomic",
                "operator": operator_for_role[role],
                "role": role,
                "value": int(permutation[index]),
            })
    base["train"] = train
    base["multi_clause_train"] = 0
    base["joint_examples_in_training"] = 0
    base["test_rows_used"] = 0
    return base, path


def exhaustive_target_check(manifest: dict[str, Any]) -> dict[str, Any]:
    permutation = manifest["permutation"]
    rows = manifest["train"]
    mismatches: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for row in rows:
        argument = str(row["argument"])
        if not argument.startswith("ARG_"):
            raise HarnessError(f"invalid argument token: {argument}")
        index = int(argument[4:])
        expected = int(permutation[index])
        actual = int(row["value"])
        entry = {"role": row["role"], "operator": row["operator"], "argument": argument, "arg_index": index, "target_value": actual, "expected_permutation_value": expected}
        mapping.append(entry)
        if actual != expected:
            mismatches.append(entry)
    counts = {role: sum(row["role"] == role for row in rows) for role in ROLES}
    if len(rows) != 128 or counts != {role: 32 for role in ROLES}:
        raise HarnessError(f"invalid Stage-A row counts: {counts}")
    result = {"rows_checked": len(rows), "expected_rows": 128, "checks": len(rows), "expected_checks": 128, "role_counts": counts, "mismatch_count": len(mismatches), "passed": not mismatches, "mismatches": mismatches, "mapping": mapping}
    if mismatches:
        raise HarnessError(f"exhaustive target permutation mismatch: {mismatches[:2]}")
    return result


def atomic_source_mapping(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    physical_ids = manifest["token_ids"]
    token_order = manifest["token_order"]
    internal = {token: index for index, token in enumerate(token_order)}
    return [
        {
            "role": row["role"],
            "operator": row["operator"],
            "argument": row["argument"],
            "arg_index": int(row["argument"][4:]),
            "target_value": int(row["value"]),
            "physical_operator_id": int(physical_ids[row["operator"]]),
            "physical_argument_id": int(physical_ids[row["argument"]]),
            "internal_operator_index": internal[row["operator"]],
            "internal_argument_index": internal[row["argument"]],
        }
        for row in manifest["train"]
    ]


def separation_certificates(encoder: GenericNRoleBinder, manifest: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        catalog = generic_catalog_scores(encoder, manifest)
        backgrounds, contexts = generic_background_scores(encoder, manifest)
    certificates: dict[str, Any] = {}
    arg_link_certificates: dict[str, Any] = {}
    for query_index, role in enumerate(ROLES):
        self_scores = catalog[query_index][query_index]
        self_min, self_index = self_scores.min(dim=0)
        for source_index, source_role in enumerate(ROLES):
            if source_index == query_index:
                continue
            cross = catalog[query_index][source_index]
            cross_max, cross_index = cross.max(dim=0)
            margin_matrix = self_scores[:, None] - cross[None, :]
            margin, flat = margin_matrix.reshape(-1).min(dim=0)
            self_arg, cross_arg = divmod(int(flat.item()), VALUE_COUNT)
            certificates[f"{role}<-{source_role}"] = {
                "query_role": role,
                "source_role": source_role,
                "G": finite(margin.item()),
                "positive": bool(margin.item() > 0.0),
                "self_min": finite(self_min.item()),
                "self_min_arg_index": int(self_index.item()),
                "self_min_context": f"{role}/{role}/ARG_{int(self_index.item()):02d}",
                "cross_max": finite(cross_max.item()),
                "cross_max_arg_index": int(cross_index.item()),
                "cross_max_context": f"{source_role}/ARG_{int(cross_index.item()):02d}",
                "extreme_pair": {"self_arg_index": self_arg, "cross_arg_index": cross_arg, "self_score": finite(self_scores[self_arg].item()), "cross_score": finite(cross[cross_arg].item())},
            }
        bg = backgrounds[query_index]
        bg_max, bg_index = bg.max(dim=0)
        bg_margin, bg_flat = (self_scores[:, None] - bg[None, :]).reshape(-1).min(dim=0)
        bg_self_index, bg_context_index = divmod(int(bg_flat.item()), len(contexts))
        bg_context = contexts[bg_context_index]
        certificates[f"{role}<-BG"] = {
            "query_role": role,
            "G": finite(bg_margin.item()),
            "positive": bool(bg_margin.item() > 0.0),
            "self_arg_index": bg_self_index,
            "self_score": finite(self_scores[bg_self_index].item()),
            "background_index": bg_context_index,
            "background_score": finite(bg[bg_context_index].item()),
            "background_context": bg_context["context"],
            "background_kind": bg_context["kind"],
            "global_background_max": finite(bg_max.item()),
            "global_background_max_context": contexts[int(bg_index.item())]["context"],
        }
        arg_link_indices = [index for index, context in enumerate(contexts) if context["kind"] == "ARG→LINK"]
        arg_link = bg[arg_link_indices]
        arg_link_max, arg_link_local_index = arg_link.max(dim=0)
        arg_link_margin, arg_link_flat = (self_scores[:, None] - arg_link[None, :]).reshape(-1).min(dim=0)
        arg_index, local_context = divmod(int(arg_link_flat.item()), len(arg_link_indices))
        arg_link_context_index = arg_link_indices[local_context]
        arg_link_certificates[role] = {
            "query_role": role,
            "G_r_ARG_to_LINK": finite(arg_link_margin.item()),
            "positive": bool(arg_link_margin.item() > 0.0),
            "self_arg_index": arg_index,
            "self_score": finite(self_scores[arg_index].item()),
            "arg_to_link_index": arg_link_context_index,
            "arg_to_link_score": finite(arg_link[arg_link_local_index].item()),
            "arg_to_link_context": contexts[arg_link_context_index]["context"],
            "arg_to_link_kind": contexts[arg_link_context_index]["kind"],
            "candidate_count": len(arg_link_indices),
        }
    role_role = {key: value for key, value in certificates.items() if key.endswith(tuple(f"<-{role}" for role in ROLES))}
    role_bg = {key: value for key, value in certificates.items() if key.endswith("<-BG")}
    return {
        "certificates": certificates,
        "role_role_count": len(role_role),
        "role_bg_count": len(role_bg),
        "certificate_count": len(certificates),
        "all_16_strictly_positive": len(certificates) == 16 and all(item["positive"] for item in certificates.values()),
        "arg_to_link": arg_link_certificates,
        "background_context_count": len(contexts),
        "background_context_kinds": {kind: sum(context["kind"] == kind for context in contexts) for kind in ("START→OP", "ARG→LINK", "LINK→OP")},
    }


def train_seed(seed: int, manifest: dict[str, Any], manifest_path: Path, observations: dict[str, Tensor], labels: dict[str, Tensor], supervisor: torch.nn.Module, executor: torch.nn.Module, codebook: Tensor) -> dict[str, Any]:
    destination = TRAINING_ROOT / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    target_check = exhaustive_target_check(manifest)
    mapping = atomic_source_mapping(manifest)
    encoder = GenericNRoleBinder(manifest, seed, list(ROLES))
    if encoder.embedding.weight.shape[0] != 37:
        raise HarnessError(f"Stage-A binder vocabulary is not 37 rows for {seed}")
    batch = build_training_batch(encoder, manifest, labels)
    if batch["token_ids"].shape != (128, 2) or any(int(row["value"]) != int(manifest["permutation"][int(row["argument"][4:])]) for row in manifest["train"]):
        raise HarnessError(f"constructed Stage-A batch failed exhaustive target verification for {seed}")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last_losses: dict[str, float] = {}
    started = time.perf_counter()
    for step in range(1, UPDATES + 1):
        progress = (step - 1) / (UPDATES - 1)
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.zero_grad(set_to_none=True)
        losses = objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
        last_losses = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        final_losses_tensor = objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    final_losses = {name: finite(final_losses_tensor[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    checkpoint_payload = {
        "encoder": encoder.state_dict(),
        "seed": seed,
        "updates": UPDATES,
        "manifest_sha256": sha256_file(manifest_path),
        "fresh_init": True,
        "prior_binder_checkpoint_loaded": False,
        "base_vocab_only": True,
        "noop_operator": manifest["noop_operator"],
        "noop_in_stage_a_vocab": False,
        "noop_in_batch": False,
        "noop_in_objective": False,
        "noop_in_background_catalog": False,
        "operator_role_assignment": manifest["operator_role_assignment"],
        "operator_for_role": manifest["operator_for_role"],
        "base_token_order": manifest["token_order"],
        "base_token_ids_physical": manifest["token_ids"],
        "permutation": manifest["permutation"],
        "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)",
        "L_sep_coefficient": 1.0,
        "L_sep_terms": 16,
        "background_contexts": BACKGROUND_COUNT,
        "multi_clause_train": 0,
        "joint_examples_in_training": 0,
        "test_rows_used": 0,
    }
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    state_before_audits = state_hash(encoder.state_dict())
    atomic = d1_atomic(encoder, executor, codebook, manifest)
    separation = separation_certificates(encoder, manifest)
    gradient = gradient_sanity(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    state_after_audits = state_hash(encoder.state_dict())
    checkpoint_hash_after = sha256_file(checkpoint)
    a1_pass = bool(atomic["pass"])
    a2_pass = bool(separation["all_16_strictly_positive"] and gradient["strictly_positive"])
    result = {
        "status": "trained",
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "updates": UPDATES,
        "manifest": {**source_record(manifest_path), "original_manifest_untouched": True},
        "fresh_initialization": {"constructor": "GenericNRoleBinder(base manifest view, seed)", "seed": seed, "prior_binder_checkpoint_loaded": False, "base_vocab_rows": 37, "noop_operator": manifest["noop_operator"], "noop_row_created": False},
        "base_vocab_mapping": {"token_order": manifest["token_order"], "physical_to_internal": [{"token": token, "physical_id": int(manifest["token_ids"][token]), "internal_index": index} for index, token in enumerate(manifest["token_order"])], "operator_for_role": manifest["operator_for_role"]},
        "exhaustive_target_check": target_check,
        "effective_atomic_mapping": mapping,
        "training": {"rows": {role: 32 for role in ROLES} | {"total": 128}, "objective": "L_behavior^(F,A)+L_ref^(F,A/M/H)+L_sep^(4+BG)", "L_behavior_roles": ["FLOOR", "AVOID"], "L_ref_roles": list(ROLES), "L_sep": {"role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "coefficient": 1.0, "reduction": "mean over 16 terms"}, "background_contexts": 40, "multi_clause_train": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "last_update_losses": last_losses, "final_losses": final_losses},
        "architecture": {"name": "RELKEY", "address_key": "k_t=LN(W_k[e_prev,e_t]+a_k)", "activation_between_linear_and_LN": False, "queries": list(ROLES), "query_shape": [4, 16], "shared_w_v": True, "hard_pointer": "argmax over valid positions"},
        "A1": {"atomic": atomic, "pass": a1_pass},
        "A2": {"separation": separation, "gradient_LINK": gradient, "pass": a2_pass},
        "A3": {"checkpoint": source_record(checkpoint), "checkpoint_sha256_before_audits": checkpoint_hash, "checkpoint_sha256_after_audits": checkpoint_hash_after, "checkpoint_unchanged_after_audits": checkpoint_hash == checkpoint_hash_after, "core_state_hash_before_audits": state_before_audits, "core_state_hash_after_audits": state_after_audits, "core_state_unchanged_after_audits": state_before_audits == state_after_audits, "frozen_for_future_stage_b": True, "decoder_checkpoint": source_record(BASE_CHECKPOINT), "supervisor_checkpoint": source_record(CTRL7_CHECKPOINT)},
        "noop_exclusion": {"operator": manifest["noop_operator"], "in_base_vocab": False, "in_train_batch": False, "in_objective": False, "in_background_catalog": False, "in_multi_clause_evaluation": "not opened", "e_N_initialized": False},
        "stage_a_gate": {"A1": a1_pass, "A2": a2_pass, "pass": a1_pass and a2_pass},
    }
    write_self_hashed(destination / "results.json", result)
    return result


def main() -> int:
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-A output root: {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifests = {seed: load_t7_manifest(seed) for seed in SEEDS}
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
    for seed in SEEDS:
        manifest, manifest_path = manifests[seed]
        results.append(train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook))
    all_pass = all(item["stage_a_gate"]["pass"] for item in results)
    consolidated = {
        "schema": "T7-noop-none-lexical-stage-a-development-v1",
        "status": "completed",
        "classification": "T7 STAGE-A DEVELOPMENT: PASS" if all_pass else "T7 STAGE-A DEVELOPMENT: VALID FAIL",
        "task": "T7-NOOP-NONE-LEXICAL / STAGE-A DEVELOPMENT",
        "authorization": "Sol-authorized Stage A only; B/C/evaluation held",
        "seeds": list(SEEDS),
        "executive_summary": {"seeds_completed": len(results), "seeds_expected": 5, "A1_all_seeds": all(item["stage_a_gate"]["A1"] for item in results), "A2_all_seeds": all(item["stage_a_gate"]["A2"] for item in results), "atomic_checks": 640, "certificates_per_seed": 16, "training_performed": True, "stage_b_started": False, "stage_c_started": False, "evaluation_opened": False, "new_checkpoints": True},
        "recipe": {"fresh_init_per_seed": True, "base_vocab_rows": 37, "noop_excluded": True, "batch_rows": {role: 32 for role in ROLES} | {"total": 128}, "updates": UPDATES, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A)+L_ref^(F,A/M/H)+L_sep^(4+BG)", "L_sep_terms": {"role_role": 12, "role_bg": 4, "total": 16, "coefficient": 1.0, "background_contexts": 40}, "multi_clause_train": 0, "null_calibration": False},
        "runner": source_record(SCRIPT_PATH),
        "core_source": source_record(SCRIPT_DIR / "t5_nrole_design_audit.py"),
        "manifests": {str(seed): results[index]["manifest"] for index, seed in enumerate(SEEDS)},
        "checkpoints": [item["A3"]["checkpoint"] for item in results],
        "per_seed_results": [{"seed": item["seed"], "path": str(TRAINING_ROOT / f"seed_{item['seed']}" / "results.json").replace("\\", "/"), "sha256": sha256_file(TRAINING_ROOT / f"seed_{item['seed']}" / "results.json")} for item in results],
        "per_seed": results,
        "failure_policy": "Scientific A1/A2 failures complete all five seeds; implementation/provenance errors halt as INVALID/HARNESS BUG.",
        "forbidden_operations_confirmed": ["stage_b", "stage_c", "NULL calibration", "multi_clause evaluation", "780x generation", "NOOP/e_N initialization"],
    }
    artifact_path = OUTPUT_ROOT / "results.json"
    digest = write_self_hashed(artifact_path, consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": artifact_path.relative_to(ROOT).as_posix(), "artifact_sha256": sha256_file(artifact_path), "artifact_self_hash": digest, "seeds": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
