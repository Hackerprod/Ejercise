"""OMEGA expanded frozen validation.

Synthetic smoke is safe by default. Real WikiText-2/checkpoint evaluation is
reachable only with ``--full --confirm-secondary-frozen-validation``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parents[1]
ROOT = LAB_ROOT.parent
CAMPAIGN_ROOT = LAB_ROOT / "campaign"
R1_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
ER32_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er32_quality_scoping_a"
ER32_ADAPTER_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_er32_integration_and_cost_gate"
R1_RUNNER = R1_DIR / "run_scientific_scoping_a.py"
ER32_RUNNER = ER32_DIR / "run_er32_quality_scoping_a.py"
ER32_ADAPTER = ER32_ADAPTER_DIR / "omega_fast_er32.py"
SCRIPTS_DIR = LAB_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(R1_DIR))
sys.path.insert(0, str(ER32_DIR))
sys.path.insert(0, str(ER32_ADAPTER_DIR))

from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    reconstruct_documents,
)


CAMPAIGN_ID = "OMEGA-EXPANDED-FROZEN-VALIDATION"
VALIDATION_ROLE = "SECONDARY_FROZEN_VALIDATION"
SEEDS = (20260913, 20260914)
VARIANTS = ("K1", "K4")
FAMILIES = ("R1", "ER32")
UPDATE = 2000
RETAINED_TOKENS = 513
ORIGINAL_VALIDATION_DOCUMENTS = 8


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def token_hash(tokens: Iterable[int]) -> str:
    return sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in tokens))


def artifact_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_hashes() -> dict[str, str]:
    return {
        "runner": file_hash(Path(__file__).resolve()),
        "r1_runner": file_hash(R1_RUNNER),
        "er32_runner": file_hash(ER32_RUNNER),
        "er32_adapter": file_hash(ER32_ADAPTER),
    }


def is_eligible_document(document: dict[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    text = str(document["text"])
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    if len(tokens) < RETAINED_TOKENS:
        return None
    retained = tokens[:RETAINED_TOKENS]
    return {
        "document_index": int(document["document_index"]),
        "row_range": list(document["row_range"]),
        "header": str(document["header"]),
        "token_count": len(tokens),
        "selected_token_count": len(retained),
        "full_text_sha256": sha256_text(text),
        "retained_513_token_sha256": token_hash(retained),
        "tokens": retained,
    }


def collect_all_eligible_documents(
    dataset: Any,
    tokenizer: Any,
    *,
    excluded_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return every first-occurrence eligible document in reconstructed order."""
    excluded = excluded_keys or set()
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for document_index, document in enumerate(reconstruct_documents(dataset)):
        eligible = is_eligible_document({**document, "document_index": document_index}, tokenizer)
        if eligible is None:
            continue
        key = (eligible["full_text_sha256"], eligible["retained_513_token_sha256"])
        if key in seen or key in excluded:
            continue
        seen.add(key)
        selected.append(eligible)
    return selected


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "tokens"}


def build_validation_manifest(documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        raise ValueError("expanded validation requires at least one eligible document")
    keys: set[tuple[str, str]] = set()
    previous_index = -1
    for document in documents:
        if int(document["document_index"]) <= previous_index:
            raise ValueError("expanded validation document order is not strictly increasing")
        previous_index = int(document["document_index"])
        if int(document["token_count"]) < RETAINED_TOKENS or int(document["selected_token_count"]) != RETAINED_TOKENS:
            raise ValueError("expanded validation document violates 513-token eligibility")
        key = (str(document["full_text_sha256"]), str(document["retained_513_token_sha256"]))
        if key in keys:
            raise ValueError("expanded validation document set is not deduplicated")
        keys.add(key)
    manifest = {
        "schema": "omega-expanded-frozen-validation-document-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "validation_role": VALIDATION_ROLE,
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "split": "validation"},
        "reconstruction": {
            "header_rule": "level-one header is '= Title ='; deeper headings do not start documents",
            "implementation": "current R1 reconstruct_documents",
        },
        "eligibility": "all reconstructed documents with >=513 GPT-2 tokens; retain first 513",
        "dedupe_key": ["full_text_sha256", "retained_513_token_sha256"],
        "stable_order": "validation row order after reconstruction and first-occurrence dedupe",
        "selection_scope": "all eligible validation documents; no eight-document cap",
        "original_validation_document_count": ORIGINAL_VALIDATION_DOCUMENTS,
        "document_count": len(documents),
        "documents": [public_document(document) for document in documents],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def frozen_checkpoint_path(family: str, variant: str, seed: int) -> Path:
    if family not in FAMILIES or variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("unsupported frozen target")
    if family == "R1":
        return R1_DIR / "results" / "full_campaign" / "runs" / f"shared_{variant}_seed_{seed}" / f"checkpoint_{UPDATE:05d}.pt"
    return ER32_DIR / "results" / "quality_scoping_a" / "runs" / f"ER32-{variant}_seed_{seed}" / f"checkpoint_{UPDATE:05d}.pt"


def frozen_target_matrix() -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "variant": variant,
            "seed": seed,
            "update": UPDATE,
            "checkpoint_path": artifact_path(frozen_checkpoint_path(family, variant, seed)),
        }
        for family in FAMILIES
        for variant in VARIANTS
        for seed in SEEDS
    ]


def write_self_hashed(path: Path, payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = sha256_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    placeholder = encoded.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if sha256_bytes(placeholder) != digest:
        raise RuntimeError("secondary artifact self-hash verification failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return digest


def verify_self_hashed(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.get("artifact_self_hash")
    if not isinstance(digest, str):
        return False
    payload["artifact_self_hash"] = "__SELF_HASH__"
    expected = sha256_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return digest == expected


def _synthetic_documents(count: int = 10, vocab_size: int = 17) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for index in range(count):
        tokens = [((index + 3) * 5 + position) % vocab_size for position in range(RETAINED_TOKENS)]
        text = f"synthetic-{index}"
        documents.append({
            "document_index": index,
            "row_range": [index, index + 1],
            "header": f"= Synthetic {index} =",
            "token_count": RETAINED_TOKENS,
            "selected_token_count": RETAINED_TOKENS,
            "full_text_sha256": sha256_text(text),
            "retained_513_token_sha256": token_hash(tokens),
            "tokens": tokens,
        })
    return documents


class _SyntheticFrozenModel(torch.nn.Module):
    vocab_size = 17

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, 1, 1, device=device)

    def forward_window(self, input_ids: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size, dtype=torch.float32)
        return state, logits


def _synthetic_metric(documents: list[dict[str, Any]]) -> dict[str, Any]:
    model = _SyntheticFrozenModel()
    from run_scientific_scoping_a import evaluate_validation

    return evaluate_validation(model, documents)


def build_report(
    *,
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    mode: str,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "omega-expanded-frozen-validation-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "validation_role": VALIDATION_ROLE,
        "mode": mode,
        "primary_gate": {
            "status": "UNTOUCHED",
            "reclassification": False,
            "statement": "Secondary evidence never replaces or reclassifies primary R1/ER32 gate results.",
        },
        "evaluation": {"dataset": "WikiText-2", "split": "validation", "update": UPDATE, "all_eligible_documents": True, "document_count": manifest["document_count"]},
        "manifest": {"sha256": manifest["manifest_sha256"], "document_count": manifest["document_count"]},
        "frozen_targets": results,
        "target_matrix": frozen_target_matrix(),
        "source_hashes": source_hashes(),
        "output_path": artifact_path(output_path),
    }


def run_smoke(output_dir: Path) -> dict[str, Any]:
    documents = _synthetic_documents()
    manifest = build_validation_manifest(documents)
    metric = _synthetic_metric(documents)
    results = [
        {
            "family": family,
            "variant": variant,
            "seed": seed,
            "update": UPDATE,
            "checkpoint": {"path": frozen_checkpoint_path(family, variant, seed).as_posix(), "loaded": False, "reason": "synthetic smoke"},
            "validation": metric,
        }
        for family in FAMILIES
        for variant in VARIANTS
        for seed in SEEDS
    ]
    output_path = output_dir / "secondary_frozen_validation_smoke.json"
    report = build_report(manifest=manifest, results=results, mode="smoke_only", output_path=output_path)
    report["real_data_loaded"] = False
    report["real_checkpoints_loaded"] = False
    write_self_hashed(output_path, report)
    return report


def _load_real_validation() -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    download_config = DownloadConfig(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    train_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=download_config)
    validation_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation", revision=DATASET_REVISION, download_config=download_config)
    train_documents = collect_all_eligible_documents(train_dataset, tokenizer)
    train_keys = {(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in train_documents}
    documents = collect_all_eligible_documents(validation_dataset, tokenizer, excluded_keys=train_keys)
    return documents, build_validation_manifest(documents), tokenizer


def _load_frozen_model(target: dict[str, Any], checkpoint_path: Path) -> torch.nn.Module:
    import run_scientific_scoping_a as r1

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("update", -1)) != UPDATE or int(payload.get("seed", -1)) != target["seed"]:
        raise ValueError(f"frozen checkpoint identity mismatch: {checkpoint_path}")
    if target["family"] == "R1":
        model = r1.make_f_model(vocab_size=50257, dimensions=(128, 8), variant=f"shared_{target['variant']}")
    else:
        import run_er32_quality_scoping_a as er32

        model = er32.build_fresh_er32(target["seed"], f"ER32-{target['variant']}")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def run_real(output_dir: Path) -> dict[str, Any]:
    documents, manifest, _tokenizer = _load_real_validation()
    results: list[dict[str, Any]] = []
    from run_scientific_scoping_a import evaluate_validation

    for target in frozen_target_matrix():
        checkpoint = ROOT / target["checkpoint_path"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
        model = _load_frozen_model(target, checkpoint)
        with torch.no_grad():
            validation = evaluate_validation(model, documents)
        results.append({**target, "checkpoint_sha256": file_hash(checkpoint), "validation": validation})
    output_path = output_dir / "secondary_frozen_validation.json"
    report = build_report(manifest=manifest, results=results, mode="real_frozen_evaluation", output_path=output_path)
    report["real_data_loaded"] = True
    report["real_checkpoints_loaded"] = True
    write_self_hashed(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CAMPAIGN_ID)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-secondary-frozen-validation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.smoke:
        if args.full or args.confirm_secondary_frozen_validation:
            parser.error("--smoke cannot be combined with real-evaluation flags")
        print(json.dumps(run_smoke(args.output_dir.resolve()), indent=2, sort_keys=True))
        return 0
    if not (args.full and args.confirm_secondary_frozen_validation):
        parser.error("real SECONDARY_FROZEN_VALIDATION requires --full --confirm-secondary-frozen-validation")
    print(json.dumps(run_real(args.output_dir.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
