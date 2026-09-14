"""Bounded real-teacher CPU timing probe for the Scientific Scoping A batch fix."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_scientific_scoping_a import (  # noqa: E402
    ADAMW_BETAS,
    ADAMW_EPS,
    BASE_LR,
    CLIP_NORM,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PHYSICAL_BATCH,
    TOKENIZER_VOCAB,
    VARIANTS,
    _batch_loss,
    build_manifest,
    collect_all_eligible_documents,
    file_hash,
    parameter_hash,
    set_seed,
    source_hashes,
)
from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical  # noqa: E402


UPDATES = 6


def load_approved_train_inputs() -> tuple[list[dict[str, object]], torch.nn.Module, dict[str, object]]:
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    download_config = DownloadConfig(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise RuntimeError(f"tokenizer vocab mismatch: expected {TOKENIZER_VOCAB}, got {len(tokenizer)}")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True).to(dtype=torch.float32)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=download_config)
    all_train = collect_all_eligible_documents(train_dataset, tokenizer)
    train_documents, train_manifest = build_manifest(all_train, split_name="train", pair_count=1)
    approved = train_documents[:PHYSICAL_BATCH]
    if len(approved) != PHYSICAL_BATCH:
        raise RuntimeError(f"expected {PHYSICAL_BATCH} approved documents, got {len(approved)}")
    return approved, teacher, {
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "split": "train", "test_split_loaded": False},
        "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "cache_only": True},
        "train_manifest_sha256": train_manifest["manifest_sha256"],
        "approved_document_indices": [document["document_index"] for document in approved],
        "approved_document_keys": [[document["full_text_sha256"], document["retained_513_token_sha256"]] for document in approved],
    }


def measure_variant(documents: list[dict[str, object]], teacher: torch.nn.Module, variant: str) -> dict[str, object]:
    set_seed(20260913)
    model = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, dimension=128, slots=8, rounds=1 if variant == "shared_K1" else 4, variant="shared").to(dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=0.0)
    pair_states: list[torch.Tensor] | None = None
    updates: list[dict[str, object]] = []
    for update in range(UPDATES):
        window = update % 2
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss, next_states, valid_tokens = _batch_loss(model, teacher, documents, window, pair_states)
        loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
        optimizer.step()
        elapsed = time.perf_counter() - started
        pair_states = next_states if window == 0 else None
        updates.append({"update": update, "window": window, "elapsed_seconds": elapsed, "loss": float(loss.detach()), "valid_tokens": valid_tokens, "pre_clip_grad_norm": pre_clip})
    measured = updates[2:]
    total_seconds = sum(float(item["elapsed_seconds"]) for item in measured)
    return {"variant": variant, "updates": updates, "measured_updates": len(measured), "measured_total_seconds": total_seconds, "measured_seconds_per_update": total_seconds / len(measured), "all_finite": all(torch.isfinite(torch.tensor(float(item["loss"]))) for item in updates), "parameter_hash": parameter_hash(model)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "batch_fix_timing_probe" / "real_teacher_timing_report.json")
    args = parser.parse_args(argv)
    documents, teacher, provenance = load_approved_train_inputs()
    results = [measure_variant(documents, teacher, variant) for variant in VARIANTS]
    report = {
        "schema": "omega-core-lm-0-r1-scientific-scoping-a-batch-fix-timing-v1",
        "status": "completed",
        "campaign_started": False,
        "full_campaign_launched": False,
        "purpose": "bounded post-fix timing only; not scientific campaign evidence",
        "policy": {"device": "cpu", "dtype": "float32", "execution": "eager", "physical_batch": PHYSICAL_BATCH, "effective_batch": 8, "updates_per_variant": UPDATES, "warmup_updates": 2, "measured_updates_per_variant": 4, "optimizer": "AdamW", "clip_norm": CLIP_NORM, "spend": False},
        "provenance": {**provenance, "runner_source_sha256": file_hash(Path(__file__)), "scoping_runner_hashes": source_hashes(), "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
