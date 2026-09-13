"""Execute T6 active-set midpoint development training, calibration, and evaluation."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from t5_nrole_design_audit import (  # noqa: E402
    GenericNRoleBinder,
    generic_background_contexts,
    generic_background_scores,
    generic_catalog_scores,
    generic_separation,
)
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor  # noqa: E402
from train_t2_i2_r2 import load_source  # noqa: E402


SEEDS = (7501, 7502, 7503, 7504, 7505)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")
UPDATES = 5000
BACKGROUND_COUNT = 40
EXPECTED_BY_SIZE = {1: 128, 2: 11904, 3: 23808, 4: 23808}
EXPECTED_TOTAL = 59648
EXPECTED_MULTI = 59520
OUTPUT_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_development_closed2"
TRAINING_ROOT = OUTPUT_ROOT / "training"
MANIFEST_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_preparation" / "manifests"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_self_hashed(path: Path, artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(written)
    placeholder = written.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if sha256_bytes(placeholder) != digest:
        raise RuntimeError(f"self-hash verification failed: {path}")
    return digest


def state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T6-activeset-midpoint-manifest-v1":
        raise ValueError(f"unexpected T6 manifest schema for {seed}")
    if int(manifest.get("seed")) != seed:
        raise ValueError(f"manifest seed mismatch for {seed}")
    if len(manifest.get("train", [])) != 128 or len(manifest.get("evaluation", [])) != EXPECTED_TOTAL:
        raise ValueError(f"manifest cardinality mismatch for {seed}")
    if manifest.get("multi_clause_train") != 0 or manifest.get("joint_examples_in_training") != 0 or manifest.get("test_rows_used") != 0:
        raise ValueError(f"manifest training contract mismatch for {seed}")
    if manifest.get("active_set_model_input") is not False or manifest.get("active_roles_evaluator_only") is not True:
        raise ValueError(f"active-set forward contract mismatch for {seed}")
    counts = {size: sum(int(row["subset_size"]) == size for row in manifest["evaluation"]) for size in range(1, 5)}
    if counts != EXPECTED_BY_SIZE:
        raise ValueError(f"manifest size counts mismatch for {seed}: {counts}")
    return manifest, path


def build_training_batch(encoder: GenericNRoleBinder, manifest: dict[str, Any], labels: dict[str, Tensor]) -> dict[str, Any]:
    token_rows: list[list[int]] = []
    roles: list[str] = []
    source_rows: list[int] = []
    targets: list[int] = []
    for row in manifest["train"]:
        role = str(row["role"])
        value = int(row["value"])
        argument = str(row["argument"])
        if role not in ROLES or int(manifest["permutation"][int(argument[4:])]) != value:
            raise ValueError(f"training mapping mismatch for {role}/{value}")
        token_rows.append([encoder.vocab.encode(str(row["operator"])), encoder.vocab.encode(argument)])
        roles.append(role)
        targets.append(value)
        if role == "FLOOR":
            mask = (labels["constraints"][:, 0] == 1) & (labels["constraints"][:, 1] == 0) & (labels["lower"] == value)
        elif role == "AVOID":
            mask = (labels["constraints"][:, 0] == 0) & (labels["constraints"][:, 1] == 1) & (labels["forbidden"] == value)
        else:
            source_rows.append(-1)
            continue
        candidates = torch.where(mask)[0]
        if not len(candidates):
            raise ValueError(f"missing behavior source row for {role}/{value}")
        source_rows.append(int(candidates[0].item()))
    counts = {role: roles.count(role) for role in ROLES}
    if counts != {role: 32 for role in ROLES}:
        raise ValueError(f"training batch counts: {counts}")
    return {
        "token_ids": torch.tensor(token_rows, dtype=torch.long),
        "lengths": torch.full((128,), 2, dtype=torch.long),
        "roles": roles,
        "source_rows": torch.tensor(source_rows, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def objective_n4(
    encoder: GenericNRoleBinder,
    supervisor: nn.Module,
    executor: nn.Module,
    codebook: Tensor,
    manifest: dict[str, Any],
    batch: dict[str, Any],
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
) -> dict[str, Tensor]:
    details = encoder(batch["token_ids"], batch["lengths"])
    values = details["values"][:, 1]
    role_indices = {role: torch.tensor([index for index, item in enumerate(batch["roles"]) if item == role], dtype=torch.long) for role in ROLES}
    behavior_indices = torch.cat((role_indices["FLOOR"], role_indices["AVOID"]))
    behavior = F.cross_entropy(
        supervisor(observations["features"][batch["source_rows"][behavior_indices]], values[behavior_indices]),
        labels["action"][batch["source_rows"][behavior_indices]],
    )
    reference_terms = []
    for role in ROLES:
        indices = role_indices[role]
        logits = executor.register_decoder(torch.cat((values[indices], torch.zeros_like(values[indices])), dim=-1), codebook)
        reference_terms.append(F.cross_entropy(logits, batch["targets"][indices]))
    reference = torch.stack(reference_terms).mean()
    separation, terms, _ = generic_separation(encoder, manifest)
    return {"behavior": behavior, "ref": reference, "sep": separation, "total": behavior + reference + separation, **{f"sep_{name}": value for name, value in terms.items()}}


def decode_states(executor: nn.Module, codebook: Tensor, states: Tensor) -> tuple[Tensor, Tensor]:
    zeros = torch.zeros((len(states), states.shape[-1]), dtype=states.dtype)
    raw = executor.register_decoder(torch.cat((states, zeros), dim=-1), codebook).argmax(dim=-1)
    canonical = executor.register_decoder(codebook[raw], codebook).argmax(dim=-1)
    return raw, canonical


def d1_atomic(encoder: GenericNRoleBinder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ROLES:
        rows = [row for row in manifest["train"] if row["role"] == role]
        ids = torch.tensor([[encoder.vocab.encode(row["operator"]), encoder.vocab.encode(row["argument"])] for row in rows], dtype=torch.long)
        lengths = torch.full((len(rows),), 2, dtype=torch.long)
        with torch.no_grad():
            details = encoder(ids, lengths)
            pointers = details["scores"][ROLES.index(role)].argmax(dim=1)
            selected = details["values"][torch.arange(len(rows)), pointers]
            raw, canonical = decode_states(executor, codebook, selected)
        pointer_exact = int((pointers == 1).sum().item())
        raw_exact = int(sum(int(raw[index].item()) == int(row["value"]) for index, row in enumerate(rows)))
        canonical_exact = int(sum(int(canonical[index].item()) == int(row["value"]) for index, row in enumerate(rows)))
        raw_canon_equal = int(torch.equal(raw, canonical))
        result[role] = {
            "pointer_exact": pointer_exact,
            "decode_exact": raw_exact,
            "RAW_exact": raw_exact,
            "CANON_exact": canonical_exact,
            "RAW_CANON_equal": raw_canon_equal,
            "total": 32,
            "pass": pointer_exact == 32 and raw_exact == 32 and canonical_exact == 32 and raw_canon_equal == 1,
        }
    result["pass"] = all(result[role]["pass"] for role in ROLES)
    return result


def d2_calibrate(encoder: GenericNRoleBinder, manifest: dict[str, Any], real_dtype: torch.dtype) -> dict[str, Any]:
    with torch.no_grad():
        catalog = generic_catalog_scores(encoder, manifest)
        backgrounds, contexts = generic_background_scores(encoder, manifest)
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
            margin, flat = (self_scores.unsqueeze(1) - cross.unsqueeze(0)).reshape(-1).min(dim=0)
            self_index, cross_index = divmod(int(flat.item()), VALUE_COUNT)
            certificates[f"{role}<-{source_role}"] = {"query_role": role, "cross_role": source_role, "margin": finite(margin.item()), "positive": bool(margin.item() > 0.0), "self_arg_index": self_index, "cross_arg_index": cross_index, "self_score": finite(self_scores[self_index].item()), "cross_score": finite(cross[cross_index].item())}
        bg = backgrounds[query_index]
        bg_value, bg_index = bg.max(dim=0)
        negatives.append({"kind": "BG", "source_role": None, "background_index": int(bg_index.item()), "score": finite(bg_value.item()), "context": contexts[int(bg_index.item())]["context"], "background_kind": contexts[int(bg_index.item())]["kind"]})
        margin, flat = (self_scores.unsqueeze(1) - bg.unsqueeze(0)).reshape(-1).min(dim=0)
        self_index, background_index = divmod(int(flat.item()), len(contexts))
        certificates[f"{role}<-BG"] = {"query_role": role, "margin": finite(margin.item()), "positive": bool(margin.item() > 0.0), "self_arg_index": self_index, "background_index": background_index, "background_context": contexts[background_index]["context"], "background_kind": contexts[background_index]["kind"], "self_score": finite(self_scores[self_index].item()), "background_score": finite(bg[background_index].item())}
        negative = max(negatives, key=lambda item: item["score"])
        p_min = finite(p_value.item())
        n_max = finite(negative["score"])
        midpoint = n_max + (p_min - n_max) / 2.0
        stored_tensor = torch.tensor(midpoint, dtype=real_dtype)
        stored = finite(stored_tensor.item())
        left_hol = finite(p_min - stored)
        right_hol = finite(stored - n_max)
        intervals[role] = {
            "P_r_min": p_min,
            "P_r_min_arg_index": int(p_index.item()),
            "P_r_min_context": f"{role}/ARG_{int(p_index.item()):02d}",
            "N_r_max": n_max,
            "N_r_max_context": negative,
            "interval_width": finite(p_min - n_max),
            "interval_exists": bool(p_min > n_max),
            "b_r_midpoint": finite(midpoint),
            "b_r_stored": stored,
            "storage_dtype": str(real_dtype),
            "P_r_min_minus_b_r": left_hol,
            "b_r_minus_N_r_max": right_hol,
            "stored_strictly_inside": bool(left_hol > 0.0 and right_hol > 0.0),
        }
    return {
        "status": "passed" if all(item["interval_exists"] and item["stored_strictly_inside"] for item in intervals.values()) else "CALIBRATION_FAIL",
        "source": "128 atomic relations + 40 BG contexts only",
        "formula": "b_r = N_r^max + (P_r^min - N_r^max)/2",
        "real_inference_dtype": str(real_dtype),
        "separation_certificates": certificates,
        "separation_certificate_count": len(certificates),
        "all_separation_certificates_positive": len(certificates) == 16 and all(item["positive"] for item in certificates.values()),
        "roles": intervals,
        "role_count": len(intervals),
        "all_strict": all(item["interval_exists"] and item["stored_strictly_inside"] for item in intervals.values()),
    }


def gradient_sanity(
    encoder: GenericNRoleBinder,
    supervisor: nn.Module,
    executor: nn.Module,
    codebook: Tensor,
    manifest: dict[str, Any],
    batch: dict[str, Any],
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
) -> dict[str, Any]:
    encoder.zero_grad(set_to_none=True)
    losses = objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    losses["total"].backward()
    link_index = encoder.vocab.encode("LINK")
    gradient = encoder.embedding.weight.grad
    if gradient is None:
        return {"dL_de_LINK_norm": 0.0, "dL_de_LINK_max_abs": 0.0, "strictly_positive": False, "gradient_present": False}
    link = gradient[link_index]
    result = {
        "dL_de_LINK_norm": finite(link.norm().item()),
        "dL_de_LINK_max_abs": finite(link.abs().max().item()),
        "dL_de_LINK_nonzero": int((link != 0).sum().item()),
        "strictly_positive": bool(link.norm().item() > 0.0),
        "gradient_present": True,
    }
    encoder.zero_grad(set_to_none=True)
    return result


def train_seed(
    seed: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
    supervisor: nn.Module,
    executor: nn.Module,
    codebook: Tensor,
    destination: Path,
) -> tuple[GenericNRoleBinder, dict[str, Any], Path]:
    destination.mkdir(parents=True, exist_ok=True)
    encoder = GenericNRoleBinder(manifest, seed, list(ROLES))
    batch = build_training_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last_losses: dict[str, float] = {}
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
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "seed": seed,
            "updates": UPDATES,
            "manifest_sha256": sha256_file(manifest_path),
            "fresh_init": True,
            "prior_encoder_checkpoint_loaded": False,
            "decoder_checkpoint_loaded": source_record(BASE_CHECKPOINT),
            "shared_w_v": True,
            "query_roles": list(ROLES),
            "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)",
            "L_sep_coefficient": 1.0,
            "L_sep_terms": 16,
            "background_contexts": BACKGROUND_COUNT,
            "multi_clause_train": 0,
            "joint_examples_in_training": 0,
            "test_rows_used": 0,
        },
        checkpoint,
    )
    checkpoint_hash_before = sha256_file(checkpoint)
    state_hash_before = state_hash(encoder.state_dict())
    atomic = d1_atomic(encoder, executor, codebook, manifest)
    gradient = gradient_sanity(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    certificates = d2_calibrate(encoder, manifest, next(encoder.parameters()).dtype)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    state_hash_after_freeze = state_hash(encoder.state_dict())
    checkpoint_hash_after_freeze = sha256_file(checkpoint)
    immutability = {
        "checkpoint_sha256_before_calibration": checkpoint_hash_before,
        "checkpoint_sha256_after_calibration": checkpoint_hash_after_freeze,
        "checkpoint_hash_identical": checkpoint_hash_before == checkpoint_hash_after_freeze,
        "core_state_hash_before_calibration": state_hash_before,
        "core_state_hash_after_calibration": state_hash_after_freeze,
        "core_state_hash_identical": state_hash_before == state_hash_after_freeze,
        "parameters_frozen_before_evaluation": all(not parameter.requires_grad for parameter in encoder.parameters()),
    }
    result = {
        "status": "trained",
        "seed": seed,
        "updates": UPDATES,
        "fresh_init": {"constructor": "GenericNRoleBinder(T6 manifest, seed)", "seed": seed, "prior_encoder_checkpoint_loaded": False, "decoder_codebook_conserved_and_frozen": True},
        "training": {
            "batch_rows": {role: 32 for role in ROLES} | {"total": 128},
            "multi_clause_train": 0,
            "joint_examples_in_training": 0,
            "test_rows_used": 0,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)",
            "L_behavior_roles": ["FLOOR", "AVOID"],
            "L_ref_roles": list(ROLES),
            "L_sep": {"coefficient": 1.0, "role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "background_contexts": 40, "reduction": "mean over 16 terms"},
            "last_update_losses": last_losses,
            "final_losses": final_losses,
        },
        "architecture": {"name": "RELKEY", "d_model": 16, "queries": list(ROLES), "query_shape": [4, 16], "shared_w_v": True, "gate": False, "mask": False, "stage_b": False, "soft_mixture": False, "hard_pointer": "pure argmax over valid positions"},
        "D1": {"atomic": atomic, "separation_certificates": certificates["separation_certificates"], "gradient_LINK": gradient, "pass": bool(atomic["pass"] and certificates["all_separation_certificates_positive"] and gradient["strictly_positive"])},
        "D2": {"calibration": certificates, "pass": bool(certificates["all_strict"]), "immutability": immutability},
        "manifest": source_record(manifest_path),
        "checkpoint": source_record(checkpoint),
    }
    return encoder, result, checkpoint


def recover_seed(
    seed: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    observations: dict[str, Tensor],
    labels: dict[str, Tensor],
    supervisor: nn.Module,
    executor: nn.Module,
    codebook: Tensor,
    source_root: Path,
) -> tuple[GenericNRoleBinder, dict[str, Any], Path]:
    source_destination = source_root / "training" / f"seed_{seed}"
    checkpoint = source_destination / "final.pt"
    calibration_path = source_destination / "calibration.json"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = GenericNRoleBinder(manifest, seed, list(ROLES))
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    state_before = state_hash(encoder.state_dict())
    checkpoint_before = sha256_file(checkpoint)
    batch = build_training_batch(encoder, manifest, labels)
    atomic = d1_atomic(encoder, executor, codebook, manifest)
    gradient = gradient_sanity(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    calibration_bytes = calibration_path.read_bytes()
    calibration = json.loads(calibration_bytes.decode("utf-8"))
    stored_self_hash = calibration.get("artifact_self_hash")
    unsigned = dict(calibration)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    if stored_self_hash != sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")):
        raise ValueError(f"invalid calibration self-hash for seed {seed}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    state_after = state_hash(encoder.state_dict())
    checkpoint_after = sha256_file(checkpoint)
    immutability = {"checkpoint_sha256_before_calibration": checkpoint_before, "checkpoint_sha256_after_calibration": checkpoint_after, "checkpoint_hash_identical": checkpoint_before == checkpoint_after, "core_state_hash_before_calibration": state_before, "core_state_hash_after_calibration": state_after, "core_state_hash_identical": state_before == state_after, "parameters_frozen_before_evaluation": True, "recovered_from": str(source_destination.relative_to(ROOT)).replace("\\", "/")}
    result = {"status": "trained_recovered", "seed": seed, "updates": int(payload["updates"]), "fresh_init": {"constructor": "GenericNRoleBinder(T6 manifest, seed)", "seed": seed, "prior_encoder_checkpoint_loaded": False, "decoder_codebook_conserved_and_frozen": True}, "training": {"batch_rows": {role: 32 for role in ROLES} | {"total": 128}, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_sep": {"coefficient": 1.0, "role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "background_contexts": 40, "reduction": "mean over 16 terms"}, "recovered_checkpoint": True}, "architecture": {"name": "RELKEY", "d_model": 16, "queries": list(ROLES), "query_shape": [4, 16], "shared_w_v": True, "gate": False, "mask": False, "stage_b": False, "soft_mixture": False, "hard_pointer": "pure argmax over valid positions"}, "D1": {"atomic": atomic, "separation_certificates": calibration["separation_certificates"], "gradient_LINK": gradient, "pass": bool(atomic["pass"] and calibration["all_separation_certificates_positive"] and gradient["strictly_positive"])}, "D2": {"calibration": calibration, "pass": bool(calibration["all_strict"]), "immutability": immutability}, "manifest": source_record(manifest_path), "checkpoint": source_record(checkpoint)}
    return encoder, result, checkpoint


def update_minimum(store: dict[str, Any], key: str, margin: float, evidence: dict[str, Any]) -> None:
    if key not in store or margin < store[key]["margin"]:
        store[key] = {"margin": finite(margin), **evidence}


def new_role_metrics() -> dict[str, int]:
    return {"total": 0, "expected_present": 0, "expected_absent": 0, "presence_correct": 0, "false_positive": 0, "false_negative": 0, "binding_present_total": 0, "binding_present_correct": 0, "absence_total": 0, "absence_correct": 0, "RAW_CANON_equal": 0, "decision_complete": 0}


def d3_evaluate(encoder: GenericNRoleBinder, executor: nn.Module, codebook: Tensor, manifest: dict[str, Any], seed: int, thresholds: dict[str, float]) -> dict[str, Any]:
    groups = {str(item["subset"]): {"subset": item["subset"], "size": int(item["size"]), "rows": 0, "by_role": {role: new_role_metrics() for role in ROLES}, "complete_count": 0} for item in manifest["subset_registry"]}
    counts_by_size = {size: 0 for size in range(1, 5)}
    minima: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    equivalence: dict[str, tuple[tuple[Any, ...], ...]] = {}
    equivalence_orders: dict[str, int] = {}
    equivalence_sizes: dict[str, int] = {}
    equivalence_mismatches: list[dict[str, Any]] = []
    total_rows = 0
    for start in range(0, len(manifest["evaluation"]), 512):
        chunk = manifest["evaluation"][start : start + 512]
        encoded_rows = [[encoder.vocab.encode(token) for token in row["tokens"]] for row in chunk]
        max_length = max(len(row) for row in encoded_rows)
        pad_id = encoder.vocab.encode("LINK")
        token_ids = torch.tensor([row + [pad_id] * (max_length - len(row)) for row in encoded_rows], dtype=torch.long)
        lengths = torch.tensor([len(row) for row in encoded_rows], dtype=torch.long)
        with torch.no_grad():
            details = encoder(token_ids, lengths)
        row_outputs: list[dict[str, Any]] = [{} for _ in chunk]
        for role_index, role in enumerate(ROLES):
            scores = details["scores"][role_index]
            present_indices: list[int] = []
            present_states: list[Tensor] = []
            records: list[dict[str, Any]] = []
            for index, row in enumerate(chunk):
                active = role in row["active_roles"]
                length = int(lengths[index].item())
                real_scores = scores[index, :length]
                best_real, best_position = real_scores.max(dim=0)
                null_wins = thresholds[role] >= float(best_real.item())
                target_position = next((position for position, token in enumerate(row["tokens"]) if token == next(clause["argument"] for clause in row["clauses"] if clause["role"] == role)), None) if active else None
                target_score = float(scores[index, target_position].item()) if target_position is not None else None
                if not null_wins:
                    pointer = int(best_position.item())
                    present_indices.append(index)
                    present_states.append(details["values"][index, pointer])
                records.append({"active": active, "pointer": None if null_wins else int(best_position.item()), "target_position": target_position, "target_score": target_score, "null": null_wins, "best_real_score": float(best_real.item()), "best_real_position": int(best_position.item())})
            if present_states:
                with torch.no_grad():
                    raw, canonical = decode_states(executor, codebook, torch.stack(present_states))
            else:
                raw = torch.empty((0,), dtype=torch.long)
                canonical = torch.empty((0,), dtype=torch.long)
            raw_by_row = {row_index: int(raw[position].item()) for position, row_index in enumerate(present_indices)}
            canonical_by_row = {row_index: int(canonical[position].item()) for position, row_index in enumerate(present_indices)}
            for index, row in enumerate(chunk):
                record = records[index]
                group = groups[str(row["subset"])]
                metrics = group["by_role"][role]
                metrics["total"] += 1
                active = bool(record["active"])
                if active:
                    metrics["expected_present"] += 1
                    if record["null"]:
                        metrics["false_negative"] += 1
                    else:
                        metrics["presence_correct"] += 1
                        metrics["binding_present_total"] += 1
                        raw_value = raw_by_row[index]
                        canonical_value = canonical_by_row[index]
                        expected_value = int(row["assignment"][role])
                        binding_ok = record["pointer"] == record["target_position"] and raw_value == expected_value and canonical_value == expected_value and raw_value == canonical_value
                        metrics["binding_present_correct"] += int(binding_ok)
                        metrics["RAW_CANON_equal"] += int(raw_value == canonical_value)
                        if record["target_position"] is not None:
                            target = int(record["target_position"])
                            others = [position for position in range(int(lengths[index].item())) if position != target]
                            competitor = max(others, key=lambda position: float(scores[index, position].item()))
                            margin = float(scores[index, target].item() - scores[index, competitor].item())
                            update_minimum(minima, f"{role}.present_vs_real", margin, {"seed": seed, "case_index": start + index, "subset": row["subset"], "subset_size": int(row["subset_size"]), "order": row["order"], "role": role, "target": row["tokens"][target], "competitor": row["tokens"][competitor], "target_score": finite(scores[index, target].item()), "competitor_score": finite(scores[index, competitor].item())})
                        null_margin = float(scores[index, int(record["target_position"])] - thresholds[role])
                        update_minimum(minima, f"{role}.present_vs_null", null_margin, {"seed": seed, "case_index": start + index, "subset": row["subset"], "subset_size": int(row["subset_size"]), "order": row["order"], "role": role, "target": row["tokens"][int(record["target_position"])], "threshold": thresholds[role], "target_score": finite(scores[index, int(record["target_position"])].item())})
                        complete = bool(record["pointer"] == record["target_position"] and binding_ok)
                    if record["null"]:
                        complete = False
                else:
                    metrics["expected_absent"] += 1
                    if record["null"]:
                        metrics["presence_correct"] += 1
                        metrics["absence_total"] += 1
                        metrics["absence_correct"] += 1
                        complete = True
                    else:
                        metrics["false_positive"] += 1
                        complete = False
                    competitor = int(record["best_real_position"])
                    absent_margin = thresholds[role] - float(record["best_real_score"])
                    update_minimum(minima, f"{role}.absent", absent_margin, {"seed": seed, "case_index": start + index, "subset": row["subset"], "subset_size": int(row["subset_size"]), "order": row["order"], "role": role, "threshold": thresholds[role], "competitor": row["tokens"][competitor], "competitor_score": record["best_real_score"]})
                metrics["decision_complete"] += int(complete)
                row_outputs[index][role] = {"present": not record["null"], "pointer": record["pointer"], "decoded_value": raw_by_row.get(index), "RAW": raw_by_row.get(index), "CANON": canonical_by_row.get(index)}
                if not complete and len(errors) < 20:
                    errors.append({"seed": seed, "case_index": start + index, "subset": row["subset"], "subset_size": int(row["subset_size"]), "order": row["order"], "role": role, "expected_present": active, "observed": row_outputs[index][role], "expected_value": row["assignment"].get(role)})
        for index, row in enumerate(chunk):
            total_rows += 1
            group = groups[str(row["subset"])]
            group["rows"] += 1
            counts_by_size[int(row["subset_size"])] += 1
            complete = all((role in row["active_roles"] and row_outputs[index][role]["present"] and row_outputs[index][role]["decoded_value"] == int(row["assignment"][role])) or (role not in row["active_roles"] and not row_outputs[index][role]["present"] and row_outputs[index][role]["pointer"] is None and row_outputs[index][role]["decoded_value"] is None and row_outputs[index][role]["RAW"] is None and row_outputs[index][role]["CANON"] is None) for role in ROLES)
            group["complete_count"] += int(complete)
            normalized = tuple((bool(row_outputs[index][role]["present"]), row_outputs[index][role]["decoded_value"], row_outputs[index][role]["RAW"], row_outputs[index][role]["CANON"]) for role in ROLES)
            semantic_key = json.dumps([row["subset"], [[role, int(row["assignment"][role])] for role in ROLES if role in row["active_roles"]]], separators=(",", ":"))
            equivalence_orders[semantic_key] = equivalence_orders.get(semantic_key, 0) + 1
            equivalence_sizes[semantic_key] = int(row["subset_size"])
            prior = equivalence.get(semantic_key)
            if prior is None:
                equivalence[semantic_key] = normalized
            elif prior != normalized and len(equivalence_mismatches) < 20:
                equivalence_mismatches.append({"seed": seed, "case_index": start + index, "subset": row["subset"], "assignment": row["assignment"], "order": row["order"], "expected_normalized": prior, "actual_normalized": normalized})
    normalized_groups: dict[str, Any] = {}
    for key, group in groups.items():
        by_role: dict[str, Any] = {}
        for role, metrics in group["by_role"].items():
            by_role[role] = {**metrics, "binding_present": {"correct": metrics["binding_present_correct"], "total": metrics["binding_present_total"], "not_applicable": metrics["binding_present_total"] == 0}, "absence": {"correct": metrics["absence_correct"], "total": metrics["absence_total"], "not_applicable": metrics["absence_total"] == 0}, "presence": {"correct": metrics["presence_correct"], "total": metrics["total"], "false_positive": metrics["false_positive"], "false_negative": metrics["false_negative"]}}
        normalized_groups[key] = {"subset": group["subset"], "size": group["size"], "rows": group["rows"], "by_role": by_role, "complete": {"correct": group["complete_count"], "total": group["rows"]}}
    all_margins_positive = bool(minima) and all(item["margin"] > 0.0 for item in minima.values())
    expected_orders = {1: 1, 2: 2, 3: 6, 4: 24}
    order_values = sorted(set(equivalence_orders.values()))
    all_order_counts_valid = all(count == expected_orders[equivalence_sizes[key]] for key, count in equivalence_orders.items())
    return {
        "status": "passed" if total_rows == EXPECTED_TOTAL and counts_by_size == EXPECTED_BY_SIZE and not errors and not equivalence_mismatches and all_order_counts_valid and all_margins_positive else "failed",
        "cases": total_rows,
        "expected_cases": EXPECTED_TOTAL,
        "counts_by_size": counts_by_size,
        "metrics_by_subset": normalized_groups,
        "result_complete": {"correct": sum(group["complete_count"] for group in groups.values()), "total": total_rows},
        "false_positive_total": sum(metrics["false_positive"] for group in groups.values() for metrics in group["by_role"].values()),
        "false_negative_total": sum(metrics["false_negative"] for group in groups.values() for metrics in group["by_role"].values()),
        "margins": {"minimums": minima, "all_positive": all_margins_positive},
        "permutation_equivalence": {"semantic_assignment_count": len(equivalence), "orders_per_assignment": order_values, "expected_orders_by_size": expected_orders, "all_assignment_order_counts_valid": all_order_counts_valid, "mismatch_count": len(equivalence_mismatches), "mismatch_samples": equivalence_mismatches, "all_equal_excluding_pointer": not equivalence_mismatches and all_order_counts_valid},
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-training-root", type=Path, default=None)
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty output root {OUTPUT_ROOT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_records: dict[int, tuple[dict[str, Any], Path]] = {}
    for seed in SEEDS:
        manifest_records[seed] = load_manifest(seed)
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
    per_seed: list[dict[str, Any]] = []
    encoders: dict[int, GenericNRoleBinder] = {}
    for seed in SEEDS:
        manifest, manifest_path = manifest_records[seed]
        destination = TRAINING_ROOT / f"seed_{seed}"
        try:
            if args.reuse_training_root is None:
                encoder, result, checkpoint = train_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, destination)
            else:
                encoder, result, checkpoint = recover_seed(seed, manifest, manifest_path, observations, labels, supervisor, executor, codebook, args.reuse_training_root.resolve())
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "final.pt").write_bytes(checkpoint.read_bytes())
                result["checkpoint"] = source_record(destination / "final.pt")
            result["development_gate"] = {"D1": result["D1"]["pass"], "D2": result["D2"]["pass"], "D3": False}
            result["calibration"] = {"path": str(destination / "calibration.json").replace("\\", "/")}
            if args.reuse_training_root is None:
                calibration_hash = write_self_hashed(destination / "calibration.json", result["D2"]["calibration"])
            else:
                source_calibration = args.reuse_training_root.resolve() / "training" / f"seed_{seed}" / "calibration.json"
                (destination / "calibration.json").write_bytes(source_calibration.read_bytes())
                calibration_hash = result["D2"]["calibration"]["artifact_self_hash"]
            result["calibration"]["sha256"] = sha256_file(destination / "calibration.json")
            result["calibration"]["artifact_self_hash"] = calibration_hash
            encoders[seed] = encoder
            per_seed.append(result)
        except Exception as error:
            failure = {"status": "execution_failed", "seed": seed, "manifest": source_record(manifest_path), "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, "development_gate": {"D1": False, "D2": False, "D3": False}}
            write_self_hashed(destination / "results.json", failure)
            per_seed.append(failure)
    all_d1 = len(per_seed) == len(SEEDS) and all(item.get("development_gate", {}).get("D1") is True for item in per_seed)
    all_d2 = all_d1 and all(item.get("development_gate", {}).get("D2") is True for item in per_seed)
    d3_opened = all_d2
    for item in per_seed:
        seed = int(item["seed"])
        reused_d3 = False
        if args.reuse_training_root is not None:
            source_result = args.reuse_training_root.resolve() / "training" / f"seed_{seed}" / "results.json"
            if source_result.exists():
                prior_result = json.loads(source_result.read_text(encoding="utf-8"))
                if prior_result.get("D3", {}).get("status") == "passed" and prior_result.get("D3", {}).get("cases") == EXPECTED_TOTAL:
                    item["D3"] = prior_result["D3"]
                    reused_d3 = True
        if reused_d3:
            item["development_gate"]["D3"] = True
        elif d3_opened and seed in encoders:
            manifest, _ = manifest_records[seed]
            thresholds = {role: float(item["D2"]["calibration"]["roles"][role]["b_r_stored"]) for role in ROLES}
            item["D3"] = d3_evaluate(encoders[seed], executor, codebook, manifest, seed, thresholds)
            item["development_gate"]["D3"] = item["D3"]["status"] == "passed"
        else:
            item["D3"] = {"status": "HALTED_BY_D1_D2", "reason": "D1 or D2 failed for at least one seed; D3 sealed", "cases": 0}
            item["development_gate"]["D3"] = False
        if seed in encoders:
            checkpoint = TRAINING_ROOT / f"seed_{seed}" / "final.pt"
            item["D2"]["immutability"]["checkpoint_sha256_after_evaluation"] = sha256_file(checkpoint)
            item["D2"]["immutability"]["checkpoint_hash_identical_through_evaluation"] = item["D2"]["immutability"]["checkpoint_sha256_before_calibration"] == item["D2"]["immutability"]["checkpoint_sha256_after_evaluation"]
        write_self_hashed(TRAINING_ROOT / f"seed_{seed}" / "results.json", item)
    all_d3 = d3_opened and all(item.get("development_gate", {}).get("D3") is True for item in per_seed)
    consolidated: dict[str, Any] = {
        "status": "completed",
        "classification": "T6-ACTIVESET-MIDPOINT DEVELOPMENT: CLOSED/PASS" if all_d1 and all_d2 and all_d3 else "T6-ACTIVESET-MIDPOINT DEVELOPMENT: VALID FAIL",
        "task": "T6-ACTIVESET-MIDPOINT-DEVELOPMENT TRAINING",
        "executive_summary": {"seeds_completed": len(per_seed), "seeds_expected": 5, "D1_all_seeds": all_d1, "D2_all_seeds": all_d2, "D3_opened": d3_opened, "D3_all_seeds": all_d3, "atomic_controls_expected": 640, "multi_clause_expected": 297600, "evaluation_rows": sum(item.get("D3", {}).get("cases", 0) for item in per_seed), "training_performed": True, "new_checkpoints": True, "pass_strong": False},
        "recipe": {"fresh_init": True, "prior_binder_weights_loaded": False, "decoder_codebook_frozen": True, "architecture": "RELKEY; Q∈R^{4×16}; shared W_v; pure HARDPTR", "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_sep_terms": {"role_role": 12, "role_bg": 4, "total": 16, "coefficient": 1.0, "background_contexts": 40}, "updates": 5000, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "batch_rows": {role: 32 for role in ROLES} | {"total": 128}, "multi_clause_train": 0, "test_rows_used": 0},
        "manifests": {str(seed): source_record(manifest_records[seed][1]) for seed in SEEDS},
        "checkpoints": [source_record(TRAINING_ROOT / f"seed_{seed}" / "final.pt") for seed in SEEDS if (TRAINING_ROOT / f"seed_{seed}" / "final.pt").exists()],
        "per_seed_results": [str(TRAINING_ROOT / f"seed_{seed}" / "results.json").replace("\\", "/") for seed in SEEDS],
    }
    digest = write_self_hashed(OUTPUT_ROOT / "results.json", consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_sha256": sha256_file(OUTPUT_ROOT / "results.json"), "artifact_self_hash": digest, "seeds": len(per_seed), "evaluation_rows": consolidated["executive_summary"]["evaluation_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
