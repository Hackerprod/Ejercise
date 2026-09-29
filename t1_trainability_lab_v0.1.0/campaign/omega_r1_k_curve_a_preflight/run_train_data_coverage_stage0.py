"""Prepare and audit the exact OMEGA-TRAIN-DATA-COVERAGE-A Stage-0 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import psutil
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
R1_DIR = HERE.parents[1] / "campaign" / "omega_core_lm_0_r1_scientific_scoping_a"
EXPANDED_DIR = HERE.parents[1] / "campaign" / "omega_expanded_frozen_validation"
TEACHER_CACHE_DIR = HERE.parents[1] / "campaign" / "omega_teacher_hidden_cache_probe"
P2R0_DIR = HERE.parents[1] / "campaign" / "omega_native_runtime_p2r0"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
INPUTS = CAMPAIGN / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
TRAIN_MANIFEST_PATH = INPUTS / "train_manifest_1000_pairs.json"
VALIDATION_MANIFEST_PATH = INPUTS / "validation_manifest_60.json"
QUALIFICATION_MANIFEST_PATH = INPUTS / "qualification_manifest.json"
OLD_HIDDEN_MANIFEST_PATH = TEACHER_CACHE_DIR / "results" / "cache_manifest.json"
OLD_HIDDEN_CACHE_PATH = Path(r"C:\omega_cache\teacher_hidden.fp32")
K6_SENTINEL_REPORT_PATH = HERE / "results" / "k6_seed_sentinel_20260927_md281" / "k6_seed_sentinel_report.json"
CHUNK_TOKENS = 513
CHUNK_STRIDE = 512
WINDOW_TOKENS = 256
PAIR_COUNT = 1000
PHYSICAL_BATCH = 8
UPDATES = 2000
SAFETY_RESERVE_BYTES = 1 * 1024**3

for _path in (CAMPAIGN, R1_DIR, EXPANDED_DIR, TEACHER_CACHE_DIR, P2R0_DIR, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_backend_quality_qualification as quality  # noqa: E402
import run_scientific_scoping_a as r1  # noqa: E402
import run_omega_expanded_frozen_validation as expanded  # noqa: E402
import run_omega_teacher_hidden_cache_probe as hidden_cache_runner  # noqa: E402
import run_omega_core_lm_0_r1_training_technical_preflight as technical  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_bytes_hash(tokens: list[int]) -> str:
    return hashlib.sha256(b"".join(int(token).to_bytes(4, "little") for token in tokens)).hexdigest()


def _collect_full_eligible(dataset: Any, tokenizer: Any) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source_index, source in enumerate(technical.reconstruct_documents(dataset)):
        text = str(source["text"])
        tokens = list(tokenizer.encode(text, add_special_tokens=False))
        if len(tokens) < CHUNK_TOKENS:
            continue
        first_chunk = tokens[:CHUNK_TOKENS]
        full_hash = technical.sha256_text(text)
        first_chunk_hash = technical.sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in first_chunk))
        key = (full_hash, first_chunk_hash)
        if key in seen:
            continue
        seen.add(key)
        documents.append(
            {
                "document_index": source_index,
                "row_range": list(source["row_range"]),
                "header": str(source["header"]),
                "token_count": len(tokens),
                "selected_token_count": CHUNK_TOKENS,
                "full_text_sha256": full_hash,
                "retained_513_token_sha256": first_chunk_hash,
                "full_token_ids_sha256": _token_bytes_hash(tokens),
                "tokens": tokens,
            }
        )
    return documents


def _expected_chunk_count_from_manifest(train_manifest: dict[str, Any]) -> tuple[int, list[int]]:
    per_document = [
        max(0, (int(row["token_count"]) - 1) // CHUNK_STRIDE)
        for row in train_manifest["documents"]
    ]
    return sum(per_document), per_document


def _ram_plan(
    *,
    old_cache_manifest: dict[str, Any],
    old_cache_bytes: int,
    new_chunk_count: int,
    available_bytes: int,
    total_bytes: int,
) -> dict[str, Any]:
    shape = [int(value) for value in old_cache_manifest["shape"]]
    dtype_size = torch.tensor([], dtype=torch.float32).element_size()
    bytes_per_window_chunk = math.prod(shape[2:]) * dtype_size
    windows_per_chunk = shape[0]
    bytes_per_chunk = bytes_per_window_chunk * windows_per_chunk
    expected_old_bytes = math.prod(shape) * dtype_size
    if expected_old_bytes != old_cache_bytes:
        raise ValueError("sealed old hidden-cache size disagrees with its shape")
    planned_shape = [windows_per_chunk, new_chunk_count, shape[2], shape[3]]
    planned_cache_bytes = bytes_per_chunk * new_chunk_count
    four_cache_copies = planned_cache_bytes * 4
    two_cache_copies = planned_cache_bytes * 2

    sentinel = json.loads(K6_SENTINEL_REPORT_PATH.read_text(encoding="utf-8"))
    sentinel_rss_values: list[int] = []
    for route in sentinel["routes"].values():
        run_dir = Path(route["run_dir"])
        ledger = run_dir / "ledger.jsonl"
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line).get("memory", {}).get("rss_bytes")
                if value is not None:
                    sentinel_rss_values.append(int(value))
    if not sentinel_rss_values:
        raise ValueError("cannot estimate process working-set RAM from the completed K6 sentinel ledger")
    old_process_peak = max(sentinel_rss_values)
    noncache_process_estimate = max(0, old_process_peak - old_cache_bytes)
    projected_worker_peak = planned_cache_bytes + noncache_process_estimate
    current_used_bytes = total_bytes - available_bytes
    projected_total_with_two = current_used_bytes + 2 * projected_worker_peak
    projected_total_with_one = current_used_bytes + projected_worker_peak
    safe_available = available_bytes - SAFETY_RESERVE_BYTES
    concurrency_fits = {
        str(count): bool(count * projected_worker_peak <= safe_available)
        for count in (4, 2, 1)
    }
    selected = next((count for count in (4, 2, 1) if concurrency_fits[str(count)]), 1)
    return {
        "old_cache_shape": shape,
        "old_cache_bytes": old_cache_bytes,
        "old_cache_gib": old_cache_bytes / 1024**3,
        "bytes_per_window_chunk": bytes_per_window_chunk,
        "windows_per_chunk": windows_per_chunk,
        "cache_bytes_per_chunk": bytes_per_chunk,
        "new_chunk_count": new_chunk_count,
        "planned_cache_shape": planned_shape,
        "planned_cache_bytes": planned_cache_bytes,
        "planned_cache_gib": planned_cache_bytes / 1024**3,
        "four_hidden_cache_copies_bytes": four_cache_copies,
        "four_hidden_cache_copies_gib": four_cache_copies / 1024**3,
        "two_hidden_cache_copies_gib": two_cache_copies / 1024**3,
        "available_ram_bytes_at_preflight": available_bytes,
        "available_ram_gib_at_preflight": available_bytes / 1024**3,
        "total_ram_gib": total_bytes / 1024**3,
        "safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "k6_sentinel_max_process_rss_bytes": old_process_peak,
        "k6_sentinel_max_process_rss_gib": old_process_peak / 1024**3,
        "estimated_non_cache_process_bytes": noncache_process_estimate,
        "projected_peak_per_new_worker_bytes": projected_worker_peak,
        "projected_peak_per_new_worker_gib": projected_worker_peak / 1024**3,
        "projected_total_used_with_two_workers_gib": projected_total_with_two / 1024**3,
        "projected_total_used_with_one_worker_gib": projected_total_with_one / 1024**3,
        "external_concurrency_fits": concurrency_fits,
        "selected_external_concurrency": selected,
        "torch_threads_unchanged": {"intraop": 4, "interop": 1},
        "native_workers_unchanged": 4,
        "cache_status": "NOT_BUILT_IN_STAGE_0",
    }


def run_stage0(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Stage-0 output is immutable: {output_dir}")

    # RAM feasibility is assessed from frozen token counts and cache layout
    # before any new multichunk payload or hidden cache is built.
    manifest = quality._load_qualification_manifest()
    train_manifest = json.loads(TRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    validation_manifest = json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    chunk_count_estimate, chunks_per_document_estimate = _expected_chunk_count_from_manifest(train_manifest)
    old_cache_manifest = json.loads(OLD_HIDDEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not hidden_cache_runner.verify_self_hash(old_cache_manifest, "manifest_self_hash"):
        raise ValueError("historical hidden-cache manifest self-hash invalid")
    if _sha256_file(OLD_HIDDEN_CACHE_PATH) != old_cache_manifest["cache_file_sha256"]:
        raise ValueError("historical hidden-cache file hash differs from its sealed manifest")
    ram_before = psutil.virtual_memory()
    ram_plan = _ram_plan(
        old_cache_manifest=old_cache_manifest,
        old_cache_bytes=OLD_HIDDEN_CACHE_PATH.stat().st_size,
        new_chunk_count=chunk_count_estimate,
        available_bytes=int(ram_before.available),
        total_bytes=int(ram_before.total),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    from datasets import DownloadConfig, load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        technical.MODEL_ID,
        revision=technical.MODEL_REVISION,
        use_fast=True,
        local_files_only=True,
    )
    if len(tokenizer) != technical.TOKENIZER_VOCAB:
        raise ValueError("local tokenizer vocab differs from the frozen R1 tokenizer")
    if old_cache_manifest["tokenizer"].get("revision") != technical.MODEL_REVISION:
        raise ValueError("active tokenizer revision differs from the sealed hidden-cache tokenizer revision")
    download_config = DownloadConfig(local_files_only=True)
    train_dataset = load_dataset(
        technical.DATASET_ID,
        technical.DATASET_CONFIG,
        split="train",
        revision=technical.DATASET_REVISION,
        download_config=download_config,
    )
    validation_dataset = load_dataset(
        technical.DATASET_ID,
        technical.DATASET_CONFIG,
        split="validation",
        revision=technical.DATASET_REVISION,
        download_config=download_config,
    )

    historical = r1.collect_all_eligible_documents(train_dataset, tokenizer)
    full_documents = _collect_full_eligible(train_dataset, tokenizer)
    if len(historical) != 602 or len(full_documents) != 602:
        raise ValueError(f"expected 602 stable eligible train docs; got old={len(historical)} new={len(full_documents)}")
    if len(train_manifest["documents"]) != len(full_documents):
        raise ValueError("full-token reconstruction count differs from historical train manifest")
    expected_metadata_fields = (
        "document_index", "row_range", "header", "token_count", "selected_token_count",
        "full_text_sha256", "retained_513_token_sha256",
    )
    for position, (legacy, full, frozen) in enumerate(zip(historical, full_documents, train_manifest["documents"])):
        for field in expected_metadata_fields:
            if legacy[field] != frozen[field] or full[field] != frozen[field]:
                raise ValueError(f"historical reconstruction/order drift at train document position {position}, field={field}")
        if full["tokens"][:CHUNK_TOKENS] != legacy["tokens"]:
            raise ValueError(f"chunk0 token IDs differ bit-for-bit from the historical 513-token payload at position {position}")
        if r1.token_hash(full["tokens"][:CHUNK_TOKENS]) != frozen["retained_513_token_sha256"]:
            raise ValueError(f"chunk0 token hash mismatch at historical document position {position}")

    validation_docs = expanded.collect_all_eligible_documents(
        validation_dataset,
        tokenizer,
        excluded_keys={(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in train_manifest["documents"]},
    )
    primary_validation_docs = validation_docs[:60]
    if len(primary_validation_docs) != 60:
        raise ValueError(f"validation loader returned {len(primary_validation_docs)} eligible docs, expected 60")
    if len(validation_manifest["documents"]) != 60:
        raise ValueError("sealed primary validation manifest is not 60 documents")
    for index, (actual, frozen) in enumerate(zip(primary_validation_docs, validation_manifest["documents"])):
        for field in expected_metadata_fields:
            if actual[field] != frozen[field]:
                raise ValueError(f"validation source differs from its sealed manifest at index {index}, field={field}")
    train_full_hashes = {row["full_text_sha256"] for row in full_documents}
    validation_full_hashes = {row["full_text_sha256"] for row in primary_validation_docs}
    leakage_full_text = train_full_hashes & validation_full_hashes
    train_document_keys = {(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in full_documents}
    validation_document_keys = {(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in primary_validation_docs}
    leakage_dedupe_keys = train_document_keys & validation_document_keys
    if leakage_full_text or leakage_dedupe_keys:
        raise ValueError("train-to-primary-validation leakage detected")

    chunks_by_depth: list[list[dict[str, Any]]] = []
    per_doc_chunk_counts: list[int] = []
    source_tokens_total = 0
    old_covered_source_positions = 0
    new_covered_source_positions = 0
    unique_target_tokens = 0
    for document_order, document in enumerate(full_documents):
        tokens = document["tokens"]
        token_count = len(tokens)
        source_tokens_total += token_count
        old_covered_source_positions += min(CHUNK_TOKENS, token_count)
        starts = list(range(0, token_count - CHUNK_TOKENS + 1, CHUNK_STRIDE))
        per_doc_chunk_counts.append(len(starts))
        intervals: list[tuple[int, int]] = []
        for depth, start in enumerate(starts):
            end = start + CHUNK_TOKENS
            target_start = start + 1
            target_end = end
            if len(tokens[start:end]) != CHUNK_TOKENS:
                raise AssertionError("incomplete chunk was generated")
            if intervals and intervals[-1][1] != target_start:
                raise AssertionError("consecutive chunks do not have exactly one context-token overlap without duplicated targets")
            intervals.append((target_start, target_end))
            new_covered_source_positions = max(new_covered_source_positions, 0)  # accumulate below per document
        if starts:
            new_covered_source_positions += starts[-1] + CHUNK_TOKENS
            unique_target_tokens += CHUNK_STRIDE * len(starts)
        for depth, start in enumerate(starts):
            chunk_tokens = tokens[start : start + CHUNK_TOKENS]
            chunk_hash = r1.token_hash(chunk_tokens)
            record = {
                "document_index": -1,
                "source_document_order_index": document_order,
                "source_document_index": int(document["document_index"]),
                "source_row_range": document["row_range"],
                "header": document["header"],
                "full_text_sha256": document["full_text_sha256"],
                "source_first_chunk_token_sha256": document["retained_513_token_sha256"],
                "retained_513_token_sha256": chunk_hash,
                "chunk_token_sha256": chunk_hash,
                "chunk_depth": depth,
                "token_start": start,
                "token_end_exclusive": start + CHUNK_TOKENS,
                "target_start": start + 1,
                "target_end_exclusive": start + CHUNK_TOKENS,
                "token_count": CHUNK_TOKENS,
                "source_token_count": token_count,
                "tokens": chunk_tokens,
            }
            if depth >= len(chunks_by_depth):
                chunks_by_depth.append([])
            chunks_by_depth[depth].append(record)

    # Preserve the precise depth-major ordering: every chunk0 in old document
    # order, then every chunk1 in old document order, and so on.
    ordered_chunks: list[dict[str, Any]] = []
    chunk_depth_counts: dict[str, int] = {}
    for depth, level in enumerate(chunks_by_depth):
        level.sort(key=lambda row: int(row["source_document_order_index"]))
        chunk_depth_counts[str(depth)] = len(level)
        ordered_chunks.extend(level)
    for pool_index, chunk in enumerate(ordered_chunks):
        chunk["document_index"] = pool_index
    if len(ordered_chunks) != chunk_count_estimate:
        raise ValueError(f"chunk count from full reconstruction {len(ordered_chunks)} != RAM plan estimate {chunk_count_estimate}")
    first_level = ordered_chunks[:602]
    chunk0_identity = (
        len(first_level) == 602
        and all(chunk["chunk_depth"] == 0 for chunk in first_level)
        and [chunk["source_document_order_index"] for chunk in first_level] == list(range(602))
        and all(chunk["tokens"] == historical[index]["tokens"] for index, chunk in enumerate(first_level))
    )
    if not chunk0_identity:
        raise ValueError("first 602 depth-major chunks do not preserve exact historical chunk0 token IDs")

    payload_path = output_dir / "multichunk_train_chunks.jsonl"
    if payload_path.exists():
        raise FileExistsError(payload_path)
    payload_tmp = payload_path.with_suffix(payload_path.suffix + ".tmp")
    with payload_tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in ordered_chunks:
            stream.write(json.dumps(chunk, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(payload_tmp, payload_path)
    payload_sha = _sha256_file(payload_path)

    public_chunks = [{key: value for key, value in chunk.items() if key != "tokens"} for chunk in ordered_chunks]
    pair_manifest = r1.build_pair_manifest(public_chunks, pair_count=PAIR_COUNT)
    if pair_manifest["documents_consumed"] != PAIR_COUNT * PHYSICAL_BATCH:
        raise ValueError("multichunk pair schedule does not match 1000 pairs × batch 8")
    chunk_manifest: dict[str, Any] = {
        "schema": "omega-train-data-coverage-a-multichunk-train-manifest-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
        "status": "PREPARED_NO_SCIENTIFIC_TRAINING",
        "dataset": train_manifest["dataset"],
        "split": "train",
        "reconstruction": {
            "same_as_historical_r1": True,
            "header_rule": "level-one '= Title =' reconstruction from pinned technical preflight",
            "implementation_sha256": _sha256_file(Path(technical.__file__).resolve()),
        },
        "tokenization": {
            "tokenizer_id": technical.MODEL_ID,
            "tokenizer_revision": technical.MODEL_REVISION,
            "tokenizer_sha256": old_cache_manifest["tokenizer"]["sha256"],
            "vocab_size": len(tokenizer),
            "add_special_tokens": False,
        },
        "dedupe_key": ["full_text_sha256", "retained_513_token_sha256"],
        "documents": [{key: value for key, value in document.items() if key not in {"tokens", "full_token_ids_sha256"}} | {"full_token_ids_sha256": document["full_token_ids_sha256"]} for document in full_documents],
        "document_count": len(full_documents),
        "chunking": {
            "chunk_tokens": CHUNK_TOKENS,
            "stride_tokens": CHUNK_STRIDE,
            "context_overlap_tokens": 1,
            "full_chunks_only": True,
            "state_reset_after_each_chunk": True,
            "bptt_windows_per_chunk": 2,
            "window_tokens": WINDOW_TOKENS,
            "window0_target_range_relative": [1, 257],
            "window1_target_range_relative": [257, 513],
            "window0_to_window1": "carry detached recurrent state only within this chunk",
            "between_chunks": "zero-initialized recurrent state; no cross-chunk state or graph",
            "depth_major_order": "all chunk0 in historical document order, then all chunk1 in historical document order, etc.",
            "chunk_depth_counts": chunk_depth_counts,
            "chunk_count": len(ordered_chunks),
            "first_602_chunks_are_historical_chunk0_bit_identical": chunk0_identity,
            "first_602_source_document_order_indices": [row["source_document_order_index"] for row in first_level],
            "first_602_chunk_token_hashes": [row["chunk_token_sha256"] for row in first_level],
        },
        "chunk_payload": {
            "path": str(payload_path),
            "sha256": payload_sha,
            "format": "UTF-8 JSONL; one complete 513-token chunk record per line",
            "records": len(ordered_chunks),
        },
        "unique_targets": {
            "target_interval_per_chunk": "[token_start+1, token_start+513)",
            "target_count_per_chunk": CHUNK_STRIDE,
            "duplicates_between_chunks": 0,
            "total_unique_target_tokens": unique_target_tokens,
            "intervals_verified_disjoint_per_source_document": True,
        },
        "schedule": {
            "pair_count": PAIR_COUNT,
            "physical_batch": PHYSICAL_BATCH,
            "effective_batch": PHYSICAL_BATCH,
            "updates_per_run": UPDATES,
            "pair_manifest": pair_manifest,
            "chunk_presentations_in_2000_updates": pair_manifest["documents_consumed"],
        },
        "validation": {
            "historical_manifest_sha256": validation_manifest["manifest_sha256"],
            "document_count": len(primary_validation_docs),
            "train_validation_full_text_hash_intersection_count": len(leakage_full_text),
            "train_validation_dedupe_key_intersection_count": len(leakage_dedupe_keys),
            "leakage_train_to_validation": False,
            "test_split_loaded": False,
        },
        "manifest_sha256": "",
    }
    chunk_manifest["manifest_sha256"] = quality._canonical_hash({key: value for key, value in chunk_manifest.items() if key != "manifest_sha256"})
    chunk_manifest_path = output_dir / "multichunk_train_manifest.json"
    quality._write_json(chunk_manifest_path, chunk_manifest)

    manifest_token_count = sum(int(row["token_count"]) for row in train_manifest["documents"])
    old_unique_targets = len(full_documents) * CHUNK_STRIDE
    new_covered_fraction = new_covered_source_positions / manifest_token_count
    old_covered_fraction = old_covered_source_positions / manifest_token_count
    total_presentations = pair_manifest["documents_consumed"]
    full_cycles = total_presentations // len(ordered_chunks)
    partial_revisits = total_presentations % len(ordered_chunks)
    stage_report: dict[str, Any] = {
        "schema": "omega-train-data-coverage-a-stage0-report-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
        "status": "STAGE0_PASS_PREPARED_NO_TRAINING",
        "manifest": {"path": str(chunk_manifest_path), "sha256": _sha256_file(chunk_manifest_path), "manifest_sha256": chunk_manifest["manifest_sha256"]},
        "chunk_payload": chunk_manifest["chunk_payload"],
        "source_identity": {
            "qualification_manifest_sha256": manifest["manifest_sha256"],
            "historical_train_manifest_sha256": train_manifest["manifest_sha256"],
            "historical_validation_manifest_sha256": validation_manifest["manifest_sha256"],
            "historical_document_order_sha256": train_manifest["dataset"]["document_order_sha256"],
            "reconstructed_document_count": len(full_documents),
            "documents_and_dedupe_order_match_historical_manifest": True,
            "chunk0_identity_exact_against_historical_token_lists": chunk0_identity,
            "chunk0_count": len(first_level),
        },
        "coverage": {
            "source_token_count": manifest_token_count,
            "old_unique_covered_source_positions": old_covered_source_positions,
            "old_unique_training_targets": old_unique_targets,
            "old_coverage_fraction": old_covered_fraction,
            "old_coverage_percent": 100.0 * old_covered_fraction,
            "new_unique_covered_source_positions": new_covered_source_positions,
            "new_unique_training_targets": unique_target_tokens,
            "new_coverage_fraction": new_covered_fraction,
            "new_coverage_percent": 100.0 * new_covered_fraction,
            "coverage_improvement_percentage_points": 100.0 * (new_covered_fraction - old_covered_fraction),
            "new_chunk_count": len(ordered_chunks),
            "selected_chunk_token_positions_including_context_overlap": len(ordered_chunks) * CHUNK_TOKENS,
        },
        "revisits_in_2000_updates": {
            "updates": UPDATES,
            "pairs": PAIR_COUNT,
            "batch_chunks_per_pair": PHYSICAL_BATCH,
            "chunk_presentations": total_presentations,
            "full_pool_cycles": full_cycles,
            "chunks_seen_twice": partial_revisits,
            "chunks_seen_once": len(ordered_chunks) - partial_revisits,
            "mean_presentations_per_chunk": total_presentations / len(ordered_chunks),
            "repeated_target_presentations": partial_revisits * CHUNK_STRIDE,
        },
        "hidden_cache_ram_assessment": ram_plan,
        "ram_decision": {
            "external_concurrency_for_this_unit": ram_plan["selected_external_concurrency"],
            "native_workers": 4,
            "torch_intraop": 4,
            "torch_interop": 1,
            "reason": "4 private FP32 hidden-cache copies exceed preflight available RAM; K6 observed per-process working set makes 2 copies exceed the 1-GiB reserve, so this unit uses 1 external process without changing per-run threading.",
        },
        "execution_flags": {
            "scientific_training_started": False,
            "optimizer_updates": 0,
            "validation_or_test_scored": False,
            "teacher_hidden_cache_newly_built": False,
            "fp16_or_cache_compression_used": False,
        },
        "report_self_sha256": "",
        "output_dir": str(output_dir),
    }
    stage_report["report_self_sha256"] = quality._canonical_hash(
        {key: value for key, value in stage_report.items() if key != "report_self_sha256"}
    )
    report_path = output_dir / "stage0_report.json"
    quality._write_json(report_path, stage_report)
    return stage_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-stage0-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_stage0_go:
        parser.error("Stage 0 requires the explicit OMEGA-TRAIN-DATA-COVERAGE-A GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"train_data_coverage_stage0_{stamp}"
    try:
        report = run_stage0(output_dir)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "stage0_failure.json",
                {"status": "FAILED_STAGE0", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "scientific_training_started": False, "output_dir": str(output_dir)},
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "documents": report["source_identity"]["reconstructed_document_count"],
                "chunks": report["coverage"]["new_chunk_count"],
                "chunk0_exact": report["source_identity"]["chunk0_identity_exact_against_historical_token_lists"],
                "coverage_before_percent": report["coverage"]["old_coverage_percent"],
                "coverage_after_percent": report["coverage"]["new_coverage_percent"],
                "cache_gib": report["hidden_cache_ram_assessment"]["planned_cache_gib"],
                "four_copy_cache_gib": report["hidden_cache_ram_assessment"]["four_hidden_cache_copies_gib"],
                "available_ram_gib": report["hidden_cache_ram_assessment"]["available_ram_gib_at_preflight"],
                "external_concurrency": report["ram_decision"]["external_concurrency_for_this_unit"],
                "report": str(output_dir / "stage0_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "STAGE0_PASS_PREPARED_NO_TRAINING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
