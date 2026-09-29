"""Freeze the Coverage-C training stream and fail closed on calibration OOVs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parents[2]
CAMPAIGN = LAB_ROOT / "t1_trainability_lab_v0.1.0" / "campaign"
R1_CAMPAIGN = CAMPAIGN / "omega_r1_k_curve_a_preflight"
QUALITY_CAMPAIGN = CAMPAIGN / "omega_backend_quality_qualification"
INPUTS = QUALITY_CAMPAIGN / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
STAGE_A_ROOT = R1_CAMPAIGN / "results" / "train_data_coverage_stage0_20260927_md288_retry2"
STAGE_A_REPORT_PATH = STAGE_A_ROOT / "stage0_report.json"
STAGE_A_TRAIN_MANIFEST_PATH = STAGE_A_ROOT / "multichunk_train_manifest.json"
STAGE_A_CHUNK_PAYLOAD_PATH = STAGE_A_ROOT / "multichunk_train_chunks.jsonl"
STAGE_C_ROOT = R1_CAMPAIGN / "results" / "coverage_c_document_balanced_20260928_md291"
STAGE_C_REPORT_PATH = STAGE_C_ROOT / "coverage_c_training_report.json"
STAGE_C_ADAPTER_PATH = STAGE_C_ROOT / "input_manifests" / "coverage_c_document_balanced_execution_manifest.json"
STAGE_C0_REPORT_PATH = R1_CAMPAIGN / "results" / "coverage_c_stage0_document_balanced_20260928_md291" / "coverage_c_stage0_report.json"
STAGE_C_EVAL_PATH = R1_CAMPAIGN / "results" / "coverage_c_distribution_evaluation_20260928_md291" / "coverage_c_distribution_evaluation_report.json"
B_REPORT_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_distribution_report.json"
VALL_MANIFEST_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_validation_manifest.json"
VALL_PAYLOAD_PATH = R1_CAMPAIGN / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_validation_chunks.jsonl"
HISTORICAL_TRAIN_MANIFEST_PATH = INPUTS / "train_manifest_1000_pairs.json"
DEFAULT_OUTPUT_DIR = HERE / "results" / "phase_b_20260929_md298"

EXPECTED_MAIN_HEAD = "e4d4322dc082b2c7de6e06fb0d3076234afbeffa"
EXPECTED_PARENT = "6b61c04cb40642e8dcbf18ba1a0a5524e54346f3"
EXPECTED_PROTOCOL_BLOB = "fc750a2ae9fb7d3933c54fb91f09ce5568d035ec"
TOKENIZER_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DOCUMENT_COUNT = 602
PAIR_COUNT = 1000
BATCH_SIZE = 8
CHUNK_TOKENS = 513
TARGETS_PER_CHUNK = 512
VALL_CHUNKS = 452
VALL_TARGETS = 231424
FIXED_GENERATOR_SEED = 20260929


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")
    return str(signature)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=LAB_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _verify_provenance() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    parent = _git("show", "-s", "--format=%P", "HEAD")
    protocol_blob = _git("rev-parse", "HEAD:Conversacion LN.md")
    protocol_status = _git("status", "--porcelain", "--", "Conversacion LN.md")
    values = {
        "repository_root": str(LAB_ROOT),
        "repository": "Hackerprod/Ejercise",
        "branch": branch,
        "main_head": head,
        "head_parent": parent,
        "protocol_authority": "Conversacion LN.md",
        "protocol_blob_sha256": protocol_blob,
        "protocol_worktree_status": protocol_status,
    }
    if (
        branch != "main"
        or head != EXPECTED_MAIN_HEAD
        or parent != EXPECTED_PARENT
        or protocol_blob != EXPECTED_PROTOCOL_BLOB
        or protocol_status
    ):
        raise RuntimeError(f"FAIL_PROVENANCE: expected main/head/parent/blob/clean authority, observed {values}")
    return values


def _load_and_verify_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    historical_train = json.loads(HISTORICAL_TRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    historical_train_sig = _check_self_hash(historical_train, "manifest_sha256", "historical train manifest")
    stage_a = json.loads(STAGE_A_REPORT_PATH.read_text(encoding="utf-8"))
    stage_a_sig = _check_self_hash(stage_a, "report_self_sha256", "Coverage-A Stage-0 report")
    stage_a_manifest = json.loads(STAGE_A_TRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    stage_a_manifest_sig = _check_self_hash(stage_a_manifest, "manifest_sha256", "Coverage-A train manifest")
    stage_c = json.loads(STAGE_C_REPORT_PATH.read_text(encoding="utf-8"))
    stage_c_sig = _check_self_hash(stage_c, "report_self_sha256", "Coverage-C training report")
    stage_c0 = json.loads(STAGE_C0_REPORT_PATH.read_text(encoding="utf-8"))
    stage_c0_sig = _check_self_hash(stage_c0, "report_self_sha256", "Coverage-C Stage-0 report")
    stage_c_adapter = json.loads(STAGE_C_ADAPTER_PATH.read_text(encoding="utf-8"))
    stage_c_adapter_sig = _check_self_hash(stage_c_adapter, "manifest_sha256", "Coverage-C execution manifest")
    stage_c_eval = json.loads(STAGE_C_EVAL_PATH.read_text(encoding="utf-8"))
    stage_c_eval_sig = _check_self_hash(stage_c_eval, "report_self_sha256", "Coverage-C evaluation report")
    b_report = json.loads(B_REPORT_PATH.read_text(encoding="utf-8"))
    b_sig = _check_self_hash(b_report, "report_self_sha256", "Coverage-B report")
    vall_manifest = json.loads(VALL_MANIFEST_PATH.read_text(encoding="utf-8"))
    vall_sig = _check_self_hash(vall_manifest, "manifest_sha256", "frozen VALL chunk manifest")

    if historical_train_sig != stage_a["source_identity"]["historical_train_manifest_sha256"]:
        raise ValueError("historical train manifest self-hash differs from Coverage-A provenance")
    if _sha256_file(STAGE_A_TRAIN_MANIFEST_PATH) != stage_a["manifest"]["sha256"]:
        raise ValueError("Coverage-A train manifest file hash mismatch")
    if _sha256_file(STAGE_A_CHUNK_PAYLOAD_PATH) != stage_a_manifest["chunk_payload"]["sha256"]:
        raise ValueError("Coverage-A chunk payload SHA mismatch")
    if _sha256_file(VALL_PAYLOAD_PATH) != vall_manifest["chunk_payload"]["sha256"]:
        raise ValueError("VALL payload SHA mismatch")
    if stage_c.get("status") != "TRAINING_COMPLETE_AWAITING_C_DISTRIBUTION_EVALUATION" or int(stage_c.get("scientific_updates_completed", -1)) != 10000:
        raise ValueError("Coverage-C training report not complete at 5×2000")
    if stage_c_eval.get("status") != "COVERAGE_C_COMPLETE" or stage_c_eval.get("primary_metric") != "VALL_ALL_FULL_CHUNKS token-weighted NLL":
        raise ValueError("Coverage-C VALL primary-selection report is not final")
    if stage_c_adapter_sig != stage_c["source_identity"]["execution_manifest_sha256"]:
        raise ValueError("Coverage-C training did not consume the frozen C adapter")
    if stage_c0_sig != stage_c_adapter["coverage_c_stage0_self_sha256"]:
        raise ValueError("Coverage-C adapter and Stage-0 identities differ")
    if stage_a_manifest_sig != stage_c_adapter["coverage_a_stage0_manifest_sha256"]:
        raise ValueError("Coverage-C adapter and Coverage-A chunk-pool identities differ")
    if vall_sig != b_report["validation_chunk_manifest_sha256"] or b_report.get("status") != "DISTRIBUTION_DIAGNOSTIC_COMPLETE":
        raise ValueError("frozen VALL does not match the completed Coverage-B source")
    if vall_manifest["strata"]["VALL_ALL_FULL_CHUNKS"]["chunk_count"] != VALL_CHUNKS or vall_manifest["strata"]["VALL_ALL_FULL_CHUNKS"]["target_tokens"] != VALL_TARGETS:
        raise ValueError("VALL identity/target count changed")
    if any(row.get("route_pass") is not True for row in stage_c["runs"].values()) or len(stage_c["runs"]) != 5:
        raise ValueError("Coverage-C training routes are not all PASS")

    input_hashes = {
        "historical_train_manifest_path": str(HISTORICAL_TRAIN_MANIFEST_PATH),
        "historical_train_manifest_sha256": historical_train_sig,
        "historical_train_manifest_file_sha256": _sha256_file(HISTORICAL_TRAIN_MANIFEST_PATH),
        "coverage_a_stage0_report_path": str(STAGE_A_REPORT_PATH),
        "coverage_a_stage0_report_self_sha256": stage_a_sig,
        "coverage_a_train_manifest_path": str(STAGE_A_TRAIN_MANIFEST_PATH),
        "coverage_a_train_manifest_self_sha256": stage_a_manifest_sig,
        "coverage_a_chunk_payload_path": str(STAGE_A_CHUNK_PAYLOAD_PATH),
        "coverage_a_chunk_payload_sha256": stage_a_manifest["chunk_payload"]["sha256"],
        "coverage_c_stage0_report_path": str(STAGE_C0_REPORT_PATH),
        "coverage_c_stage0_report_self_sha256": stage_c0_sig,
        "coverage_c_execution_manifest_path": str(STAGE_C_ADAPTER_PATH),
        "coverage_c_execution_manifest_self_sha256": stage_c_adapter_sig,
        "coverage_c_training_report_path": str(STAGE_C_REPORT_PATH),
        "coverage_c_training_report_self_sha256": stage_c_sig,
        "coverage_c_training_report_file_sha256": _sha256_file(STAGE_C_REPORT_PATH),
        "coverage_c_evaluation_report_path": str(STAGE_C_EVAL_PATH),
        "coverage_c_evaluation_report_self_sha256": stage_c_eval_sig,
        "coverage_c_evaluation_report_file_sha256": _sha256_file(STAGE_C_EVAL_PATH),
        "vall_manifest_path": str(VALL_MANIFEST_PATH),
        "vall_manifest_self_sha256": vall_sig,
        "vall_payload_path": str(VALL_PAYLOAD_PATH),
        "vall_payload_sha256": vall_manifest["chunk_payload"]["sha256"],
        "coverage_b_report_path": str(B_REPORT_PATH),
        "coverage_b_report_self_sha256": b_sig,
    }
    return historical_train, stage_a, stage_a_manifest, stage_c, stage_c0, stage_c_adapter, vall_manifest, input_hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _freeze_stream_and_oov(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Phase-B output is immutable: {output_dir}")
    provenance = _verify_provenance()
    historical_train, stage_a, stage_a_manifest, stage_c, stage_c0, stage_c_adapter, vall_manifest, input_hashes = _load_and_verify_sources()
    stage_a_chunks = _read_jsonl(STAGE_A_CHUNK_PAYLOAD_PATH)
    vall_chunks = _read_jsonl(VALL_PAYLOAD_PATH)
    if len(stage_a_chunks) != 4378 or len(vall_chunks) != VALL_CHUNKS:
        raise ValueError("chunk payload row count differs from the frozen Coverage-A/VALL manifests")
    for index, row in enumerate(stage_a_chunks):
        if int(row["document_index"]) != index or len(row["tokens"]) != CHUNK_TOKENS:
            raise ValueError(f"Coverage-A chunk payload identity mismatch at pool index {index}")
    if any(len(row["tokens"]) != CHUNK_TOKENS or int(row["token_count"]) != CHUNK_TOKENS for row in vall_chunks):
        raise ValueError("VALL contains an incomplete chunk")

    adapter_pairs = stage_c_adapter.get("pairs", [])
    if len(adapter_pairs) != PAIR_COUNT or int(stage_c_adapter.get("documents_consumed", -1)) != PAIR_COUNT * BATCH_SIZE:
        raise ValueError("Coverage-C presentation manifest is not exactly 1000 pairs × batch 8")
    train_target_counts: Counter[int] = Counter()
    train_doc_target_counts: dict[int, Counter[int]] = {}
    train_visits: Counter[int] = Counter()
    unique_chunk_indices: set[int] = set()
    presentation_rows: list[dict[str, Any]] = []
    stream_digest = hashlib.sha256()
    target_presentations = 0
    for pair_index, pair in enumerate(adapter_pairs):
        if int(pair.get("pair", -1)) != pair_index:
            raise ValueError(f"Coverage-C pair index order mismatch at pair {pair_index}")
        selected = [int(value) for value in pair["document_indices"]]
        docs = [int(value) for value in pair["source_document_order_indices"]]
        ordinals = [int(value) for value in pair["document_visit_ordinals"]]
        depths = [int(value) for value in pair["selected_chunk_depths"]]
        if len(selected) != BATCH_SIZE or len(docs) != BATCH_SIZE or len(ordinals) != BATCH_SIZE or len(depths) != BATCH_SIZE:
            raise ValueError(f"Coverage-C pair {pair_index} is not batch size 8")
        for batch_slot, (doc_order, pool_index, ordinal, depth) in enumerate(zip(docs, selected, ordinals, depths)):
            expected_doc = (BATCH_SIZE * pair_index + batch_slot) % DOCUMENT_COUNT
            if doc_order != expected_doc or ordinal != train_visits[doc_order]:
                raise ValueError(f"Coverage-C historical document/visit schedule mismatch at pair={pair_index}, slot={batch_slot}")
            chunk = stage_a_chunks[pool_index]
            if int(chunk["source_document_order_index"]) != doc_order or int(chunk["chunk_depth"]) != depth:
                raise ValueError(f"Coverage-C selected chunk identity mismatch at pair={pair_index}, slot={batch_slot}")
            n_doc_chunks = max(0, (int(stage_a_manifest["documents"][doc_order]["token_count"]) - 1) // 512)
            if not n_doc_chunks or depth != ordinal % n_doc_chunks:
                raise ValueError(f"Coverage-C visit rotation mismatch at pair={pair_index}, slot={batch_slot}")
            if len(chunk["tokens"]) != CHUNK_TOKENS:
                raise ValueError(f"selected Coverage-C chunk has invalid token length at pool index {pool_index}")
            target_ids = [int(value) for value in chunk["tokens"][1:]]
            if len(target_ids) != TARGETS_PER_CHUNK:
                raise ValueError("Coverage-C chunk does not contain exactly 512 target presentations")
            train_target_counts.update(target_ids)
            train_doc_target_counts.setdefault(doc_order, Counter()).update(target_ids)
            train_visits[doc_order] += 1
            unique_chunk_indices.add(pool_index)
            target_presentations += len(target_ids)
            event = {
                "presentation_index": pair_index * BATCH_SIZE + batch_slot,
                "pair_index": pair_index,
                "batch_slot": batch_slot,
                "document_order_index": doc_order,
                "source_document_index": int(chunk["source_document_index"]),
                "full_text_sha256": chunk["full_text_sha256"],
                "chunk_pool_index": pool_index,
                "chunk_depth": depth,
                "chunk_token_sha256": chunk["chunk_token_sha256"],
                "target_token_presentations": TARGETS_PER_CHUNK,
            }
            presentation_rows.append(event)
            stream_digest.update(_canonical_bytes(event))
    if len(presentation_rows) != 8000 or target_presentations != 4_096_000:
        raise AssertionError("Coverage-C stream presentation/target count mismatch")
    if len(unique_chunk_indices) != int(stage_c0["coverage"]["unique_chunks_reached"]):
        raise ValueError("reconstructed unique chunk reach differs from Coverage-C Stage 0")
    if len(unique_chunk_indices) != 3869 or len(unique_chunk_indices) * TARGETS_PER_CHUNK != int(stage_c0["coverage"]["unique_targets_reached"]):
        raise ValueError("Coverage-C unique target coverage differs from its sealed Stage 0")
    if train_visits != Counter({int(row["document_order_index"]): int(row["historical_visits"]) for row in stage_c0["per_document_reachability"]}):
        raise ValueError("Coverage-C document visit counts differ from Stage 0")

    train_vocab = set(train_target_counts)
    vall_target_counts: Counter[int] = Counter()
    vall_doc_counts: dict[int, Counter[int]] = {}
    for row in vall_chunks:
        doc_order = int(row["validation_document_order_index"])
        targets = [int(value) for value in row["tokens"][1:]]
        vall_target_counts.update(targets)
        vall_doc_counts.setdefault(doc_order, Counter()).update(targets)
    if sum(vall_target_counts.values()) != VALL_TARGETS:
        raise AssertionError("VALL target-event count differs from its sealed manifest")

    oov_ids = sorted(set(vall_target_counts) - train_vocab)
    oov_rows = [
        {
            "token_id": token_id,
            "vall_target_occurrences": int(vall_target_counts[token_id]),
            "vall_document_count": sum(token_id in counts for counts in vall_doc_counts.values()),
        }
        for token_id in oov_ids
    ]
    oov_target_occurrences = sum(row["vall_target_occurrences"] for row in oov_rows)
    train_unigram_counts = {str(token_id): int(count) for token_id, count in sorted(train_target_counts.items())}
    vall_target_count_rows = {str(token_id): int(count) for token_id, count in sorted(vall_target_counts.items())}

    output_dir.mkdir(parents=True, exist_ok=False)
    stream_manifest: dict[str, Any] = {
        "schema": "omega-absolute-lm-calibration-document-balanced-stream-v1",
        "unit": "OMEGA-ABSOLUTE-LM-CALIBRATION",
        "policy": {
            "candidate_data_recipe": "DOCUMENT_BALANCED_MULTICHUNK",
            "optimizer_updates": 2000,
            "pair_positions": PAIR_COUNT,
            "document_presentations": PAIR_COUNT * BATCH_SIZE,
            "physical_effective_batch": BATCH_SIZE,
            "document_formula": "doc_position=((8*p+i) mod 602), p=0..999, i=0..7",
            "chunk_formula": "on zero-based document visit j, chunk_index=j mod n_d",
            "ngram_boundary_policy": "reset at every selected full chunk and document; never form cross-chunk/document n-grams",
            "target_positions_per_chunk": TARGETS_PER_CHUNK,
            "target_slice": "chunk token ids [1:513], matching the 512 supervised target positions over W0/W1",
            "bos_context": "four explicit out-of-vocabulary context sentinels per full chunk when extracting shorter start-of-chunk histories; sentinels are never targets",
        },
        "dataset": {
            "id": stage_a_manifest["dataset"]["id"],
            "config": stage_a_manifest["dataset"]["config"],
            "split": "train",
            "revision": stage_a_manifest["dataset"]["revision"],
            "document_order_sha256": stage_a_manifest["dataset"]["document_order_sha256"],
        },
        "tokenizer": {
            "id": stage_a_manifest["tokenization"]["tokenizer_id"],
            "revision": stage_a_manifest["tokenization"]["tokenizer_revision"],
            "sha256": stage_a_manifest["tokenization"]["tokenizer_sha256"],
            "vocab_size": int(stage_a_manifest["tokenization"]["vocab_size"]),
        },
        "source_hashes": input_hashes,
        "presentation_count": len(presentation_rows),
        "target_token_presentations": target_presentations,
        "unique_chunks_reached": len(unique_chunk_indices),
        "unique_chunk_ids": [
            {
                "chunk_pool_index": pool_index,
                "source_document_order_index": int(stage_a_chunks[pool_index]["source_document_order_index"]),
                "source_document_index": int(stage_a_chunks[pool_index]["source_document_index"]),
                "chunk_depth": int(stage_a_chunks[pool_index]["chunk_depth"]),
                "full_text_sha256": stage_a_chunks[pool_index]["full_text_sha256"],
                "chunk_token_sha256": stage_a_chunks[pool_index]["chunk_token_sha256"],
            }
            for pool_index in sorted(unique_chunk_indices)
        ],
        "document_order_ids": [
            {
                "document_order_index": int(row["document_order_index"]),
                "source_document_index": int(row["document_index"]),
                "full_text_sha256": row["full_text_sha256"],
                "historical_visits": int(row["historical_visits"]),
            }
            for row in stage_c0["per_document_reachability"]
        ],
        "presentation_stream_sha256": stream_digest.hexdigest(),
        "presentations": presentation_rows,
        "unigram_train_target_counts": train_unigram_counts,
        "vall_target_counts": vall_target_count_rows,
        "vocab_observed_by_unigram_target_stream": len(train_vocab),
        "vall_observed_target_types": len(vall_target_counts),
        "vall_oov_type_count": len(oov_ids),
        "vall_oov_target_occurrences": oov_target_occurrences,
        "vall_oov_types": oov_rows,
        "status": "ABSOLUTE_CALIBRATION_OOV_POLICY_REQUIRED" if oov_ids else "DATA_FREEZE_PASS_OOV_ZERO",
        "manifest_sha256": "",
    }
    stream_manifest["manifest_sha256"] = _canonical_hash({key: value for key, value in stream_manifest.items() if key != "manifest_sha256"})
    stream_path = output_dir / "training_stream_manifest.json"
    _write_json(stream_path, stream_manifest)

    train_counts_path = output_dir / "unigram_train_target_counts.json"
    _write_json(train_counts_path, {"schema": "omega-unigram-target-counts-v1", "counts": train_unigram_counts, "count_sha256": _canonical_hash(train_unigram_counts)})
    vall_counts_path = output_dir / "vall_target_counts.json"
    _write_json(vall_counts_path, {"schema": "omega-vall-target-counts-v1", "counts": vall_target_count_rows, "count_sha256": _canonical_hash(vall_target_count_rows)})

    phase_manifest: dict[str, Any] = {
        "schema": "omega-minimum-language-competence-phase-b-manifest-v1",
        "unit": "OMEGA-ABSOLUTE-LM-CALIBRATION+MINIMUM-LANGUAGE-COMPETENCE-PHASE-B",
        "status": stream_manifest["status"],
        "provenance": provenance,
        "fixed_generator_seed": FIXED_GENERATOR_SEED,
        "stream_manifest_path": str(stream_path),
        "stream_manifest_self_sha256": stream_manifest["manifest_sha256"],
        "stream_manifest_file_sha256": _sha256_file(stream_path),
        "train_unigram_counts_path": str(train_counts_path),
        "train_unigram_counts_file_sha256": _sha256_file(train_counts_path),
        "vall_target_counts_path": str(vall_counts_path),
        "vall_target_counts_file_sha256": _sha256_file(vall_counts_path),
        "vall_oov_type_count": len(oov_ids),
        "vall_oov_target_occurrences": oov_target_occurrences,
        "vall_oov_types": oov_rows,
        "NLLs_computed": False,
        "training_updates": 0,
        "test_split_loaded": False,
        "omega_checkpoint_loaded": False,
        "phase_b_manifest_sha256": "",
    }

    if oov_ids:
        phase_manifest["hold_reason"] = "VALL contains target token ids unseen by the allowed training presentation stream; unigram/MKN5 would require an unspecified OOV mapping or prior."
        phase_manifest["status"] = "ABSOLUTE_CALIBRATION_HOLD"
        phase_manifest["phase_b_manifest_sha256"] = _canonical_hash({key: value for key, value in phase_manifest.items() if key != "phase_b_manifest_sha256"})
        phase_manifest_path = output_dir / "phase_b_manifest.json"
        _write_json(phase_manifest_path, phase_manifest)
        oov_report = {
            "schema": "omega-absolute-calibration-oov-hold-v1",
            "status": "ABSOLUTE_CALIBRATION_OOV_POLICY_REQUIRED",
            "reason": phase_manifest["hold_reason"],
            "train_target_token_presentations": target_presentations,
            "train_observed_vocabulary_types": len(train_vocab),
            "vall_target_tokens": VALL_TARGETS,
            "vall_observed_types": len(vall_target_counts),
            "vall_oov_type_count": len(oov_ids),
            "vall_oov_target_occurrences": oov_target_occurrences,
            "vall_oov_types": oov_rows,
            "nlls_computed": False,
            "phase_b_manifest_sha256": phase_manifest["phase_b_manifest_sha256"],
            "report_self_sha256": "",
        }
        oov_report["report_self_sha256"] = _canonical_hash({key: value for key, value in oov_report.items() if key != "report_self_sha256"})
        oov_report_path = output_dir / "absolute_calibration_oov_hold.json"
        _write_json(oov_report_path, oov_report)
        absolute_report = {
            "schema": "omega-absolute-lm-calibration-v1",
            "status": "ABSOLUTE_CALIBRATION_HOLD",
            "reason": "OOV policy is unspecified; no non-comparable NLL was produced.",
            "unigram": None,
            "modified_kneser_ney_5gram": None,
            "distilgpt2": None,
            "oov_report_path": str(oov_report_path),
            "report_self_sha256": "",
        }
        absolute_report["report_self_sha256"] = _canonical_hash({key: value for key, value in absolute_report.items() if key != "report_self_sha256"})
        _write_json(output_dir / "absolute_lm_calibration.json", absolute_report)
        phase_report = (
            "# PHASE B — OMEGA-ABSOLUTE-LM-CALIBRATION + MINIMUM-LANGUAGE-COMPETENCE instrument preflight\n\n"
            "## Provenance\n\n"
            f"- Main HEAD: `{provenance['main_head']}`\n"
            f"- Parent: `{provenance['head_parent']}`\n"
            f"- Conversacion LN.md blob: `{provenance['protocol_blob_sha256']}`\n\n"
            "## Data freeze\n\n"
            f"- Candidate stream: {target_presentations:,} target presentations; {len(unique_chunk_indices):,} unique chunks; {int(stage_c0['coverage']['unique_targets_reached']):,} unique target positions.\n"
            f"- VALL: {VALL_CHUNKS} chunks; {VALL_TARGETS:,} targets.\n"
            f"- Training-observed token types: {len(train_vocab):,}.\n"
            f"- VALL unseen token types: {len(oov_ids):,}; target occurrences: {oov_target_occurrences:,}.\n\n"
            "No OOV mapping, `<UNK>` substitution, or probability prior was specified. NLL scoring was stopped before any baseline NLL was produced.\n\n"
            f"- Phase-B manifest self-hash: `{phase_manifest['phase_b_manifest_sha256']}`\n"
            f"- Stream manifest self-hash: `{stream_manifest['manifest_sha256']}`\n"
            f"- OOV report self-hash: `{oov_report['report_self_sha256']}`\n\n"
            "ABSOLUTE_CALIBRATION_HOLD\n"
        )
        phase_report_path = output_dir / "PHASE_B_REPORT.md"
        phase_report_path.write_text(phase_report, encoding="utf-8", newline="\n")
        phase_report_sha = _sha256_file(phase_report_path)
        (output_dir / "PHASE_B_REPORT.md.sha256").write_text(phase_report_sha + "\n", encoding="ascii", newline="\n")
        return {
            "status": "ABSOLUTE_CALIBRATION_HOLD",
            "phase_b_manifest_path": str(phase_manifest_path),
            "phase_b_manifest_sha256": phase_manifest["phase_b_manifest_sha256"],
            "oov_report_path": str(oov_report_path),
            "oov_type_count": len(oov_ids),
            "oov_target_occurrences": oov_target_occurrences,
            "nlls_computed": False,
            "phase_b_report_path": str(phase_report_path),
            "phase_b_report_sha256": phase_report_sha,
        }

    phase_manifest["phase_b_manifest_sha256"] = _canonical_hash({key: value for key, value in phase_manifest.items() if key != "phase_b_manifest_sha256"})
    phase_manifest_path = output_dir / "phase_b_manifest.json"
    _write_json(phase_manifest_path, phase_manifest)
    return {
        "status": "DATA_FREEZE_PASS_OOV_ZERO",
        "phase_b_manifest_path": str(phase_manifest_path),
        "phase_b_manifest_sha256": phase_manifest["phase_b_manifest_sha256"],
        "oov_type_count": 0,
        "oov_target_occurrences": 0,
        "nlls_computed": False,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-phase-b-data-freeze-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.confirm_phase_b_data_freeze_go:
        parser.error("OMEGA Phase-B data freeze requires the explicit MD/298 authorization")
    try:
        report = _freeze_stream_and_oov(args.output_dir.resolve())
    except Exception as error:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            failure = {
                "status": "FAIL_PROVENANCE" if "provenance" in str(error).lower() else "FAILED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "training_updates": 0,
            }
            failure["report_self_sha256"] = _canonical_hash(failure)
            _write_json(output_dir / "phase_b_data_freeze_failure.json", failure)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in ("ABSOLUTE_CALIBRATION_HOLD", "DATA_FREEZE_PASS_OOV_ZERO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
