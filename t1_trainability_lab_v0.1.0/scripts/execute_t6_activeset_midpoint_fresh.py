"""Execute authorized T6 fresh training through the frozen development recipe."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import execute_t6_activeset_midpoint_development as development


ROOT = development.ROOT
SEEDS = (7601, 7602, 7603, 7604, 7605)
MANIFEST_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "manifests"
OUTPUT_ROOT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_development"


def load_manifest(seed: int) -> tuple[dict[str, object], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T6-activeset-midpoint-fresh-manifest-v1":
        raise ValueError(f"unexpected fresh manifest schema for {seed}")
    if int(manifest.get("seed")) != seed or len(manifest.get("train", [])) != 128 or len(manifest.get("evaluation", [])) != development.EXPECTED_TOTAL:
        raise ValueError(f"fresh manifest contract mismatch for {seed}")
    if manifest.get("active_set_model_input") is not False or manifest.get("active_roles_evaluator_only") is not True:
        raise ValueError(f"fresh active-set contract mismatch for {seed}")
    return manifest, path


def train_seed(
    seed: int,
    manifest: dict[str, object],
    manifest_path: Path,
    observations: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    supervisor: torch.nn.Module,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    destination: Path,
) -> tuple[development.GenericNRoleBinder, dict[str, object], Path]:
    destination.mkdir(parents=True, exist_ok=True)
    encoder = development.GenericNRoleBinder(manifest, seed, list(development.ROLES))
    batch = development.build_training_batch(encoder, manifest, labels)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last_losses: dict[str, float] = {}
    for step in range(1, development.UPDATES + 1):
        progress = (step - 1) / (development.UPDATES - 1)
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.zero_grad(set_to_none=True)
        losses = development.objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
        last_losses = {name: development.finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        final_losses_tensor = development.objective_n4(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    final_losses = {name: development.finite(final_losses_tensor[name].item()) for name in ("behavior", "ref", "sep", "total")}
    checkpoint = destination / "final.pt"
    torch.save({"encoder": encoder.state_dict(), "seed": seed, "updates": development.UPDATES, "manifest_sha256": development.sha256_file(manifest_path), "fresh_init": True, "prior_encoder_checkpoint_loaded": False, "decoder_checkpoint_loaded": development.source_record(development.BASE_CHECKPOINT), "shared_w_v": True, "query_roles": list(development.ROLES), "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_sep_coefficient": 1.0, "L_sep_terms": 16, "background_contexts": development.BACKGROUND_COUNT, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0}, checkpoint)
    checkpoint_hash_before = development.sha256_file(checkpoint)
    state_hash_before = development.state_hash(encoder.state_dict())
    atomic = development.d1_atomic(encoder, executor, codebook, manifest)
    gradient = development.gradient_sanity(encoder, supervisor, executor, codebook, manifest, batch, observations, labels)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    certificates = development.d2_calibrate(encoder, manifest, next(encoder.parameters()).dtype)
    state_hash_after = development.state_hash(encoder.state_dict())
    checkpoint_hash_after = development.sha256_file(checkpoint)
    immutability = {"checkpoint_sha256_before_calibration": checkpoint_hash_before, "checkpoint_sha256_after_calibration": checkpoint_hash_after, "checkpoint_hash_identical": checkpoint_hash_before == checkpoint_hash_after, "core_state_hash_before_calibration": state_hash_before, "core_state_hash_after_calibration": state_hash_after, "core_state_hash_identical": state_hash_before == state_hash_after, "parameters_frozen_before_evaluation": all(not parameter.requires_grad for parameter in encoder.parameters())}
    result: dict[str, object] = {"status": "trained", "seed": seed, "updates": development.UPDATES, "fresh_init": {"constructor": "GenericNRoleBinder(T6 fresh manifest, seed)", "seed": seed, "prior_encoder_checkpoint_loaded": False, "decoder_codebook_conserved_and_frozen": True}, "training": {"batch_rows": {role: 32 for role in development.ROLES} | {"total": 128}, "multi_clause_train": 0, "joint_examples_in_training": 0, "test_rows_used": 0, "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"}, "objective": "L_behavior^(F,A) + L_ref^(F,A,M,H) + L_sep^(4+BG)", "L_behavior_roles": ["FLOOR", "AVOID"], "L_ref_roles": list(development.ROLES), "L_sep": {"coefficient": 1.0, "role_role_terms": 12, "role_bg_terms": 4, "total_terms": 16, "background_contexts": 40, "reduction": "mean over 16 terms"}, "last_update_losses": last_losses, "final_losses": final_losses}, "architecture": {"name": "RELKEY", "d_model": 16, "queries": list(development.ROLES), "query_shape": [4, 16], "shared_w_v": True, "gate": False, "mask": False, "stage_b": False, "soft_mixture": False, "hard_pointer": "pure argmax over valid positions"}, "D1": {"atomic": atomic, "separation_certificates": certificates["separation_certificates"], "gradient_LINK": gradient, "pass": bool(atomic["pass"] and certificates["all_separation_certificates_positive"] and gradient["strictly_positive"])}, "D2": {"calibration": certificates, "pass": bool(certificates["all_strict"]), "immutability": immutability}, "manifest": development.source_record(manifest_path), "checkpoint": development.source_record(checkpoint)}
    return encoder, result, checkpoint


def main() -> int:
    development.SEEDS = SEEDS
    development.MANIFEST_ROOT = MANIFEST_ROOT
    development.OUTPUT_ROOT = OUTPUT_ROOT
    development.TRAINING_ROOT = OUTPUT_ROOT / "training"
    development.load_manifest = load_manifest
    development.train_seed = train_seed
    return development.main()


if __name__ == "__main__":
    raise SystemExit(main())
