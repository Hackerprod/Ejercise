"""OMEGA ER64 Quality Scoping A runner.

Phase 1 provides a synthetic CPU-only smoke path. Real execution requires the
explicit ``--full --confirm-quality-scoping-a`` flags and is intentionally not
run by this module's tests.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
R1_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
ER64_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er64_scaling_gate"
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(ER64_DIR))

from run_scientific_scoping_a import (  # noqa: E402
    ADAMW_BETAS,
    ADAMW_EPS,
    BASE_LR,
    CLIP_NORM,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    EFFECTIVE_BATCH,
    FULL_PAIRS,
    MODEL_ID,
    MODEL_REVISION,
    MIN_AVAILABLE_BYTES,
    PHYSICAL_BATCH,
    TEMPERATURE,
    TOKENIZER_VOCAB,
    VALIDATION_DOCUMENTS,
    WINDOW_TOKENS,
    _distillation_loss_from_trace,
    _finite_gradients,
    _load_real_documents,
    build_manifest,
    canonical_hash,
    collect_all_eligible_documents,
    evaluate_validation,
    file_hash,
    implementation_identity as r1_implementation_identity,
    make_f_model,
    memory_guard,
    public_document,
    set_seed,
    synthetic_documents,
    teacher_window_logits,
    validate_policy,
)
from omega_fast_er64 import OmegaCoreLMFastER64  # noqa: E402


CAMPAIGN_ID = "OMEGA-CORE-LM-0-ER64-QUALITY-SCOPING-A"
SEEDS = (20260913, 20260914)
VARIANTS = ("ER64-K1", "ER64-K4")
ROUNDS = {"ER64-K1": 1, "ER64-K4": 4}
TOTAL_UPDATES = 2000
CHECKPOINT_INTERVAL = 500
BOUNDARIES = (0, 500, 1000, 1500, 2000)
SMOKE_UPDATES = 4
SMOKE_BOUNDARIES = (0, 2, 4)
DIMENSIONS = (128, 8)
EXPERIMENTAL_SEED_FORBIDDEN = 20260917
QUALITY_THRESHOLD = 0.10
SECONDARY_VALIDATION_DOCUMENTS = 60


class QualityScopingError(RuntimeError):
    """Raised when frozen-baseline or resume identity is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    return {
        "runner": file_hash(Path(__file__).resolve()),
        "r1_runner": file_hash(R1_DIR / "run_scientific_scoping_a.py"),
        "er64_adapter": file_hash(ER64_DIR / "omega_fast_er64.py"),
    }


def checkpoint_boundaries(total_updates: int = TOTAL_UPDATES, interval: int = CHECKPOINT_INTERVAL) -> tuple[int, ...]:
    if total_updates < 0 or interval <= 0 or total_updates % interval:
        raise ValueError("updates must be a non-negative multiple of checkpoint interval")
    return tuple(range(0, total_updates + 1, interval))


def schedule(total_updates: int) -> list[dict[str, int]]:
    if total_updates < 0:
        raise ValueError("updates must be non-negative")
    return [{"update": update, "window": update % 2, "pair": update // 2} for update in range(total_updates)]


def validate_resume_boundary(update: int, interval: int = CHECKPOINT_INTERVAL) -> None:
    if update < 0 or update % interval:
        raise ValueError(f"resume checkpoint must be on a {interval}-update boundary")


def _variant_rounds(variant: str) -> int:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    return ROUNDS[variant]


def load_frozen_r1_baselines(results_root: Path | None = None, *, updates: tuple[int, ...] = BOUNDARIES) -> dict[int, dict[str, Any]]:
    root = results_root or R1_DIR / "results" / "full_campaign" / "runs"
    baselines: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        per_seed: dict[str, Any] = {}
        for variant, key in (("shared_K1", "K1"), ("shared_K4", "K4")):
            path = root / f"{variant}_seed_{seed}" / "validation_curve.json"
            if not path.is_file():
                raise FileNotFoundError(f"frozen R1 validation curve missing: {path}")
            curve = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(curve, list):
                raise QualityScopingError(f"R1 curve is not a list: {path}")
            points: dict[int, dict[str, Any]] = {}
            for point in curve:
                update = int(point["update"])
                if update in points:
                    raise QualityScopingError(f"duplicate R1 update {update}: {path}")
                if not math.isfinite(float(point["nll"])):
                    raise QualityScopingError(f"non-finite R1 NLL: {path}")
                points[update] = {"update": update, "nll": float(point["nll"]), "tokens": int(point.get("tokens", 0))}
            missing = [update for update in updates if update not in points]
            if missing:
                raise QualityScopingError(f"R1 curve is missing required updates {missing}: {path}")
            per_seed[key] = {
                "path": path.as_posix(),
                "sha256": file_hash(path),
                "curve": [points[update] for update in updates],
                "selected_updates": list(updates),
                "ignored_extra_updates": [update for update in sorted(points) if update not in updates],
            }
        per_seed["delta_R1"] = [
            {"update": update, "delta": per_seed["K1"]["curve"][index]["nll"] - per_seed["K4"]["curve"][index]["nll"]}
            for index, update in enumerate(updates)
        ]
        baselines[seed] = per_seed
    return baselines


def build_checkpoint_metrics(seed: int, er64_curves: dict[str, list[dict[str, Any]]], baselines: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = baselines[seed]
    curves = {key: {int(point["update"]): point for point in curve} for key, curve in er64_curves.items()}
    expected = tuple(int(item["update"]) for item in baseline["K1"]["curve"])
    if any(tuple(sorted(curve)) != expected for curve in curves.values()):
        raise QualityScopingError(f"ER64 curves do not match baseline boundaries for seed {seed}")
    result: list[dict[str, Any]] = []
    for index, update in enumerate(expected):
        r1_k1 = float(baseline["K1"]["curve"][index]["nll"])
        r1_k4 = float(baseline["K4"]["curve"][index]["nll"])
        er_k1 = float(curves["K1"][update]["nll"])
        er_k4 = float(curves["K4"][update]["nll"])
        result.append({
            "update": update,
            "NLL_R1_K1": r1_k1,
            "NLL_ER64_K1": er_k1,
            "delta_K1": er_k1 - r1_k1,
            "NLL_R1_K4": r1_k4,
            "NLL_ER64_K4": er_k4,
            "delta_K4": er_k4 - r1_k4,
            "Delta_R1": float(baseline["delta_R1"][index]["delta"]),
            "Delta_ER64": er_k1 - er_k4,
        })
    return result


def classify_quality(metrics_by_seed: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    final = {seed: points[-1] for seed, points in metrics_by_seed.items()}
    promising = all(point["delta_K1"] <= QUALITY_THRESHOLD and point["delta_K4"] <= QUALITY_THRESHOLD and point["Delta_ER64"] > 0 for point in final.values())
    mean_delta_k1 = sum(point["delta_K1"] for point in final.values()) / len(final)
    mean_delta_k4 = sum(point["delta_K4"] for point in final.values()) / len(final)
    mean_delta_er64 = sum(point["Delta_ER64"] for point in final.values()) / len(final)
    no_go = mean_delta_k1 > QUALITY_THRESHOLD and mean_delta_k4 > QUALITY_THRESHOLD and mean_delta_er64 <= 0
    classification = "QUALITY-PROMISING-A" if promising else "QUALITY-NO-GO-A" if no_go else "QUALITY-MIXED-A"
    return {
        "update": int(next(iter(final.values()))["update"]),
        "classification": classification,
        "per_seed": {str(seed): point for seed, point in sorted(final.items())},
        "means": {"delta_K1": mean_delta_k1, "delta_K4": mean_delta_k4, "Delta_ER64": mean_delta_er64},
        "quality_pass": False,
    }


def build_fresh_er64(seed: int, variant: str, *, vocab_size: int = TOKENIZER_VOCAB, dimensions: tuple[int, int] = DIMENSIONS) -> OmegaCoreLMFastER64:
    if EXPERIMENTAL_SEED_FORBIDDEN == seed:
        raise QualityScopingError("technical cost-gate seed cannot be used as a quality seed")
    set_seed(seed)
    f_reference = make_f_model(
        vocab_size=vocab_size,
        dimensions=dimensions,
        variant="shared_K1" if variant == "ER64-K1" else "shared_K4",
    )
    model = OmegaCoreLMFastER64.from_f_reference(f_reference, experimental_seed=seed, implementation="efficient")
    del f_reference
    gc.collect()
    return model


def verify_shared_factor_initialization(models: dict[str, OmegaCoreLMFastER64]) -> dict[str, bool]:
    if set(models) != set(VARIANTS):
        raise QualityScopingError("both ER64 K variants are required for factor-sharing verification")
    same_c = torch.equal(models["ER64-K1"].embedding.C, models["ER64-K4"].embedding.C)
    same_u = torch.equal(models["ER64-K1"].embedding.U, models["ER64-K4"].embedding.U)
    if not same_c or not same_u:
        raise QualityScopingError("ER64 K1/K4 lexical factors are not bit-identical")
    return {"C_equal": same_c, "U_equal": same_u, "experimental_seed": True}


def _batch_loss(model: OmegaCoreLMFastER64, teacher: torch.nn.Module, documents: list[dict[str, Any]], window: int, previous_states: list[torch.Tensor] | None) -> tuple[torch.Tensor, list[torch.Tensor], int]:
    source = torch.tensor([document["tokens"] for document in documents], dtype=torch.long)
    start = window * WINDOW_TOKENS
    inputs = source[:, start : start + WINDOW_TOKENS]
    targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
    state = model.initial_state(len(documents), device=torch.device("cpu")) if window == 0 else torch.cat(previous_states or [], dim=0)
    result = model.forward_window(inputs, state)
    next_state, logits = result[0], result[1]
    trace = result[2] if len(result) >= 3 else None
    teacher_logits = teacher_window_logits(teacher, source, window)[:, :WINDOW_TOKENS]
    mask = torch.ones_like(targets, dtype=torch.bool)
    if trace is not None:
        loss = _distillation_loss_from_trace(model, trace, teacher_logits, targets, mask)
    else:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_teacher = teacher_logits.reshape_as(flat_logits)
        flat_targets = targets.reshape(-1)
        ce = F.cross_entropy(flat_logits, flat_targets)
        kl = F.kl_div(F.log_softmax(flat_logits / TEMPERATURE, dim=-1), F.softmax(flat_teacher / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2
        loss = 0.5 * ce + 0.5 * kl
    return loss, [next_state[index : index + 1].detach() for index in range(len(documents))], int(mask.sum().item())


def _checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, run_id: str, seed: int, variant: str, update: int, config: dict[str, Any], train_manifest_hash: str, validation_manifest_hash: str, baseline_hash: str) -> dict[str, Any]:
    return {
        "schema": "omega-core-lm-0-er64-quality-scoping-a-checkpoint-v1",
        "run_id": run_id,
        "seed": seed,
        "variant": variant,
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_states": {"torch": torch.get_rng_state(), "python": random.getstate()},
        "config": config,
        "config_hash": canonical_hash(config),
        "train_manifest_sha256": train_manifest_hash,
        "validation_manifest_sha256": validation_manifest_hash,
        "baseline_sha256": baseline_hash,
        "factor_initialization": "experimental_seed=campaign seed; efficient ER64 only",
        "implementation_identity": {"er64": "omega_fast_er64.OmegaCoreLMFastER64", "f": r1_implementation_identity()},
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_hash(path)


def evaluate_secondary_final(model: torch.nn.Module, documents: list[dict[str, Any]], update: int) -> dict[str, Any]:
    if update != TOTAL_UPDATES:
        raise ValueError("secondary validation is final-update-only")
    if len(documents) != SECONDARY_VALIDATION_DOCUMENTS:
        raise ValueError(f"secondary validation requires exactly {SECONDARY_VALIDATION_DOCUMENTS} documents")
    return {"update": update, "document_count": len(documents), **evaluate_validation(model, documents)}


def run_single(*, run_dir: Path, run_id: str, seed: int, variant: str, train_documents: list[dict[str, Any]], validation_documents: list[dict[str, Any]], train_manifest: dict[str, Any], validation_manifest: dict[str, Any], teacher: torch.nn.Module, baselines: dict[int, dict[str, Any]], total_updates: int = TOTAL_UPDATES, checkpoint_interval: int = CHECKPOINT_INTERVAL, dimensions: tuple[int, int] = DIMENSIONS, resume_checkpoint: Path | None = None, smoke: bool = False, max_updates: int | None = None, secondary_validation_documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    validate_policy()
    rounds = _variant_rounds(variant)
    boundaries = checkpoint_boundaries(total_updates, checkpoint_interval)
    if max_updates is not None and not 0 <= max_updates <= total_updates:
        raise ValueError("max_updates must be within total_updates")
    baseline_hash = canonical_hash(baselines[seed])
    config = {"campaign_id": CAMPAIGN_ID, "seed": seed, "variant": variant, "rounds": rounds, "rank": 64, "dimensions": list(dimensions), "updates": total_updates, "checkpoint_boundaries": list(boundaries), "device": "cpu", "dtype": "float32", "execution": "eager", "experimental_seed": seed, "teacher": "synthetic-tiny-teacher" if smoke else "distilgpt2-pinned", "test_split": False, "secondary_validation": {"document_count": SECONDARY_VALIDATION_DOCUMENTS, "final_update_only": True, "gates_primary_quality": False}}
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "train_manifest.json", train_manifest)
    write_json(run_dir / "validation_manifest.json", validation_manifest)
    curve_path = run_dir / "validation_curve.json"
    ledger_path = run_dir / "ledger.jsonl"
    model = build_fresh_er64(seed, variant, vocab_size=17 if smoke else TOKENIZER_VOCAB, dimensions=(4, 1) if smoke else dimensions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=0.0)
    start_update = 0
    curve: list[dict[str, Any]] = []
    secondary_result: dict[str, Any] | None = None
    if resume_checkpoint is None:
        initial = _checkpoint_payload(model, optimizer, run_id=run_id, seed=seed, variant=variant, update=0, config=config, train_manifest_hash=train_manifest["manifest_sha256"], validation_manifest_hash=validation_manifest["manifest_sha256"], baseline_hash=baseline_hash)
        checkpoint_hash = save_checkpoint(run_dir / "checkpoint_00000.pt", initial)
        curve.append({"update": 0, **evaluate_validation(model, validation_documents)})
        ledger_path.write_text(json.dumps({"record_type": "checkpoint", "update": 0, "checkpoint_hash": checkpoint_hash}) + "\n", encoding="utf-8")
    else:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        start_update = int(payload["update"])
        validate_resume_boundary(start_update, checkpoint_interval)
        if payload.get("config_hash") != canonical_hash(config) or payload.get("baseline_sha256") != baseline_hash:
            raise QualityScopingError("resume identity mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng_states"]["torch"])
        random.setstate(payload["rng_states"]["python"])
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        curve = [point for point in curve if int(point["update"]) <= start_update]
        with ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"record_type": "resume", "update": start_update, "parent_checkpoint": file_hash(resume_checkpoint)}) + "\n")
    previous_states: list[torch.Tensor] | None = None
    model.train()
    execution_end = total_updates if max_updates is None else max_updates
    for item in schedule(total_updates)[start_update:execution_end]:
        memory_guard(f"update_{item['update']}")
        pair_start = item["pair"] * PHYSICAL_BATCH
        documents = [train_documents[(pair_start + offset) % len(train_documents)] for offset in range(PHYSICAL_BATCH)]
        optimizer.zero_grad(set_to_none=True)
        loss, next_states, valid_tokens = _batch_loss(model, teacher, documents, item["window"], previous_states)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        _finite_gradients(model, f"update_{item['update']}")
        clip_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        if not math.isfinite(float(clip_grad_norm)):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        previous_states = next_states if item["window"] == 0 else None
        completed = item["update"] + 1
        with ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"record_type": "update", "update": completed, "window": item["window"], "pair": item["pair"], "valid_tokens": valid_tokens}) + "\n")
        if completed in boundaries:
            curve.append({"update": completed, **evaluate_validation(model, validation_documents)})
            if completed == TOTAL_UPDATES == total_updates and secondary_validation_documents is not None:
                secondary_result = evaluate_secondary_final(model, secondary_validation_documents, completed)
            write_json(curve_path, curve)
            payload = _checkpoint_payload(model, optimizer, run_id=run_id, seed=seed, variant=variant, update=completed, config=config, train_manifest_hash=train_manifest["manifest_sha256"], validation_manifest_hash=validation_manifest["manifest_sha256"], baseline_hash=baseline_hash)
            checkpoint_hash = save_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", payload)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"record_type": "checkpoint", "update": completed, "checkpoint_hash": checkpoint_hash}) + "\n")
    curve = sorted({int(point["update"]): point for point in curve}.values(), key=lambda point: int(point["update"]))
    write_json(curve_path, curve)
    return {
        "run_id": run_id,
        "seed": seed,
        "variant": variant,
        "rounds": rounds,
        "validation_curve": curve,
        "secondary_validation": secondary_result,
        "run_dir": run_dir.as_posix(),
        "factor_initialization": {
            "experimental_seed": seed,
            "technical_seed_rejected": seed != EXPERIMENTAL_SEED_FORBIDDEN,
        },
    }


def build_quality_report(results: list[dict[str, Any]], baselines: dict[int, dict[str, Any]], *, campaign_root: Path = HERE) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for result in results:
        grouped.setdefault(int(result["seed"]), {})["K1" if result["variant"] == "ER64-K1" else "K4"] = result["validation_curve"]
    metrics = {seed: build_checkpoint_metrics(seed, variants, baselines) for seed, variants in grouped.items()}
    classification = classify_quality(metrics)
    return {"schema": "omega-core-lm-0-er64-quality-scoping-a-report-v1", "campaign_id": CAMPAIGN_ID, "mode": "smoke_only" if any(result.get("mode") == "smoke" for result in results) else "campaign", "results": results, "metrics_by_seed": {str(seed): points for seed, points in sorted(metrics.items())}, "classification_at_2000": classification, "r1_baseline": baselines, "source_hashes": source_hashes(), "quality_pass": False, "secondary_validation_is_observational": True}


def synthetic_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], torch.nn.Module, dict[int, dict[str, Any]]]:
    train, train_manifest = build_manifest(synthetic_documents(16, 17), split_name="synthetic_train", pair_count=2)
    validation, validation_manifest = build_manifest(synthetic_documents(2, 17), split_name="synthetic_validation")
    baselines = {seed: {"K1": {"path": "synthetic", "sha256": "synthetic", "curve": [{"update": update, "nll": 4.0 - update * 0.01, "tokens": 1} for update in SMOKE_BOUNDARIES]}, "K4": {"path": "synthetic", "sha256": "synthetic", "curve": [{"update": update, "nll": 3.9 - update * 0.01, "tokens": 1} for update in SMOKE_BOUNDARIES]}, "delta_R1": [{"update": update, "delta": 0.1} for update in SMOKE_BOUNDARIES]} for seed in SEEDS}
    return train, validation, train_manifest, validation_manifest, __import__("run_scientific_scoping_a").TinyTeacher(17), baselines


def run_smoke(output_dir: Path) -> dict[str, Any]:
    train, validation, train_manifest, validation_manifest, teacher, baselines = synthetic_inputs()
    results = []
    for seed in SEEDS:
        for variant in VARIANTS:
            result = run_single(run_dir=output_dir / "runs" / f"{variant}_seed_{seed}", run_id=f"smoke_{variant}_{seed}", seed=seed, variant=variant, train_documents=train, validation_documents=validation, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=teacher, baselines=baselines, total_updates=SMOKE_UPDATES, checkpoint_interval=2, dimensions=(4, 1), smoke=True)
            result["mode"] = "smoke"
            results.append(result)
    report = build_quality_report(results, baselines)
    write_json(output_dir / "quality_scoping_report.json", report)
    return report


def load_secondary_validation_documents(train_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load disjoint observational validation documents; never used for gates."""
    validate_policy()
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise RuntimeError(f"tokenizer vocab mismatch: expected {TOKENIZER_VOCAB}, got {len(tokenizer)}")
    validation_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION, download_config=DownloadConfig(local_files_only=True))
    train_keys = {(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in train_documents}
    candidates = [item for item in collect_all_eligible_documents(validation_dataset, tokenizer) if (item["full_text_sha256"], item["retained_513_token_sha256"]) not in train_keys]
    if len(candidates) < SECONDARY_VALIDATION_DOCUMENTS:
        raise RuntimeError(f"only found {len(candidates)} eligible secondary validation documents")
    return candidates[:SECONDARY_VALIDATION_DOCUMENTS]


def run_full(output_dir: Path) -> dict[str, Any]:
    baselines = load_frozen_r1_baselines()
    train, validation, train_manifest, validation_manifest, teacher = _load_real_documents(pair_count=FULL_PAIRS)
    secondary_validation = load_secondary_validation_documents(train)
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        factor_models = {variant: build_fresh_er64(seed, variant) for variant in VARIANTS}
        factor_evidence = verify_shared_factor_initialization(factor_models)
        del factor_models
        gc.collect()
        for variant in VARIANTS:
            result = run_single(
                run_dir=output_dir / "runs" / f"{variant}_seed_{seed}",
                run_id=f"{CAMPAIGN_ID}_{variant}_{seed}",
                seed=seed,
                variant=variant,
                train_documents=train,
                validation_documents=validation,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                teacher=teacher,
                baselines=baselines,
                secondary_validation_documents=secondary_validation,
            )
            result["factor_initialization"] = factor_evidence
            results.append(result)
    report = build_quality_report(results, baselines)
    write_json(output_dir / "quality_scoping_report.json", report)
    return report


def run_resume(output_dir: Path, checkpoint: Path, seed: int, variant: str) -> dict[str, Any]:
    """Resume one authorized seed/K combination from its valid checkpoint."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError("resume requires an authorized Quality-Scoping-A seed and variant")
    baselines = load_frozen_r1_baselines()
    train, validation, train_manifest, validation_manifest, teacher = _load_real_documents(pair_count=FULL_PAIRS)
    secondary_validation = load_secondary_validation_documents(train)
    result = run_single(
        run_dir=checkpoint.parent,
        run_id=f"{CAMPAIGN_ID}_{variant}_{seed}",
        seed=seed,
        variant=variant,
        train_documents=train,
        validation_documents=validation,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        baselines=baselines,
        resume_checkpoint=checkpoint,
        secondary_validation_documents=secondary_validation,
    )
    report = {
        "schema": "omega-core-lm-0-er64-quality-scoping-a-resume-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": "authorized_resume",
        "seed": seed,
        "variant": variant,
        "checkpoint": checkpoint.as_posix(),
        "run": result,
        "source_hashes": source_hashes(),
    }
    write_json(output_dir / f"resume_report_{variant}_seed_{seed}.json", report)
    return report


def run_fresh_single(output_dir: Path, seed: int, variant: str) -> dict[str, Any]:
    """Run one authorized seed/K combination from update zero."""
    if seed not in SEEDS or variant not in VARIANTS:
        raise ValueError("fresh run requires an authorized Quality-Scoping-A seed and variant")
    baselines = load_frozen_r1_baselines()
    train, validation, train_manifest, validation_manifest, teacher = _load_real_documents(pair_count=FULL_PAIRS)
    secondary_validation = load_secondary_validation_documents(train)
    result = run_single(
        run_dir=output_dir / "runs" / f"{variant}_seed_{seed}",
        run_id=f"{CAMPAIGN_ID}_{variant}_{seed}",
        seed=seed,
        variant=variant,
        train_documents=train,
        validation_documents=validation,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        baselines=baselines,
        resume_checkpoint=None,
        secondary_validation_documents=secondary_validation,
    )
    report = {
        "schema": "omega-core-lm-0-er64-quality-scoping-a-fresh-run-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "mode": "authorized_fresh_single",
        "seed": seed,
        "variant": variant,
        "run": result,
        "source_hashes": source_hashes(),
    }
    write_json(output_dir / f"fresh_report_{variant}_seed_{seed}.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-quality-scoping-a", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "quality_scoping_a")
    args = parser.parse_args(argv)
    if args.smoke:
        if args.resume_checkpoint or args.seed is not None or args.variant is not None:
            parser.error("resume/seed/variant selectors cannot be combined with --smoke")
        print(json.dumps(run_smoke(args.output_dir), indent=2, sort_keys=True))
        return 0
    if not (args.full and args.confirm_quality_scoping_a):
        parser.error("real Quality-Scoping-A execution requires --full --confirm-quality-scoping-a")
    if args.resume_checkpoint:
        if args.seed is None or args.variant is None:
            parser.error("--resume-checkpoint requires --seed and --variant")
        print(json.dumps(run_resume(args.output_dir, args.resume_checkpoint, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    if args.seed is not None or args.variant is not None:
        if args.seed is None or args.variant is None:
            parser.error("fresh single run requires both --seed and --variant")
        print(json.dumps(run_fresh_single(args.output_dir, args.seed, args.variant), indent=2, sort_keys=True))
        return 0
    print(json.dumps(run_full(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
