"""Compute exact document-balanced Coverage-C reachability without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
INPUTS = CAMPAIGN / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
HISTORICAL_TRAIN_MANIFEST = INPUTS / "train_manifest_1000_pairs.json"
COVERAGE_A_ROOT = HERE / "results" / "train_data_coverage_stage0_20260927_md288_retry2"
COVERAGE_A_REPORT = COVERAGE_A_ROOT / "stage0_report.json"
COVERAGE_A_MANIFEST = COVERAGE_A_ROOT / "multichunk_train_manifest.json"
PAIR_COUNT = 1000
PHYSICAL_BATCH = 8
UPDATES = 2000
DOCUMENT_COUNT = 602
CHUNK_STRIDE = 512
CHUNK_TOKENS = 513


def _canonical_hash(value: Any) -> str:
    body = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")
    return str(signature)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def run_stage0(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Coverage-C Stage-0 output is immutable: {output_dir}")

    coverage_a = json.loads(COVERAGE_A_REPORT.read_text(encoding="utf-8"))
    coverage_a_self_hash = _verify_self_hash(coverage_a, "report_self_sha256", "Coverage-A Stage-0 report")
    multichunk = json.loads(COVERAGE_A_MANIFEST.read_text(encoding="utf-8"))
    multichunk_self_hash = _verify_self_hash(multichunk, "manifest_sha256", "multichunk train manifest")
    historical = json.loads(HISTORICAL_TRAIN_MANIFEST.read_text(encoding="utf-8"))
    historical_self_hash = _verify_self_hash(historical, "manifest_sha256", "historical train manifest")

    if coverage_a.get("status") != "STAGE0_PASS_PREPARED_NO_TRAINING":
        raise ValueError("Coverage-A Stage-0 source report is not sealed/prepared")
    if _sha256_file(COVERAGE_A_MANIFEST) != coverage_a["manifest"]["sha256"]:
        raise ValueError("Coverage-A manifest file hash differs from its Stage-0 report")
    if _sha256_file(COVERAGE_A_ROOT / "multichunk_train_chunks.jsonl") != multichunk["chunk_payload"]["sha256"]:
        raise ValueError("Coverage-A chunk payload hash differs from its manifest")
    if historical_self_hash != coverage_a["source_identity"]["historical_train_manifest_sha256"]:
        raise ValueError("historical train manifest differs from the sealed Coverage-A source identity")
    if _canonical_hash({key: value for key, value in coverage_a.items() if key != "report_self_sha256"}) != coverage_a_self_hash:
        raise AssertionError("Coverage-A report canonical identity changed during verification")

    documents = multichunk.get("documents", [])
    historical_documents = historical.get("documents", [])
    if len(documents) != DOCUMENT_COUNT or len(historical_documents) != DOCUMENT_COUNT:
        raise ValueError("Coverage-C requires the exact frozen 602-document historical order")
    historical_schedule = historical.get("schedule", {})
    if int(historical_schedule.get("pair_count", -1)) != PAIR_COUNT:
        raise ValueError("historical schedule is not frozen at 1000 pair positions")
    if int(historical.get("dataset", {}).get("document_count", -1)) != DOCUMENT_COUNT:
        raise ValueError("historical pair manifest has an unexpected document count")
    if int(historical_schedule.get("documents_consumed", -1)) != PAIR_COUNT * PHYSICAL_BATCH:
        raise ValueError("historical pair manifest has an unexpected presentation count")
    if historical_schedule.get("formula") != "doc_position=((8*p+i) mod 602), p=0..999, i=0..7" or historical_schedule.get("prefix_formula_verified_for_all_pairs") is not True:
        raise ValueError("historical schedule formula is not the frozen cyclic 8×1000 schedule")

    chunks_per_document: list[int] = []
    token_counts: list[int] = []
    for index, (row, frozen) in enumerate(zip(documents, historical_documents)):
        for field in ("document_index", "full_text_sha256", "retained_513_token_sha256", "token_count"):
            if row.get(field) != frozen.get(field):
                raise ValueError(f"historical document identity/order mismatch at position {index}, field={field}")
        token_count = int(row["token_count"])
        if token_count < CHUNK_TOKENS:
            raise ValueError(f"ineligible document in frozen train manifest at position {index}")
        token_counts.append(token_count)
        chunks_per_document.append(max(0, (token_count - 1) // CHUNK_STRIDE))

    total_chunks = sum(chunks_per_document)
    depth_counts = {
        str(depth): sum(chunk_count > depth for chunk_count in chunks_per_document)
        for depth in range(max(chunks_per_document))
    }
    if total_chunks != int(multichunk["chunking"]["chunk_count"]):
        raise ValueError("token-count-derived complete chunk count differs from Coverage-A")
    if depth_counts != multichunk["chunking"]["chunk_depth_counts"]:
        raise ValueError("token-count-derived per-depth counts differ from Coverage-A")
    all_complete_chunk_targets = CHUNK_STRIDE * total_chunks
    if all_complete_chunk_targets != int(multichunk["unique_targets"]["total_unique_target_tokens"]):
        raise ValueError("full-chunk target denominator differs from Coverage-A")

    pair_rows = historical["pairs"]
    if len(pair_rows) != PAIR_COUNT:
        raise ValueError("historical schedule does not contain exactly 1000 pairs")
    visits_per_document = [0] * DOCUMENT_COUNT
    for pair in range(PAIR_COUNT):
        expected_indices = [((PHYSICAL_BATCH * pair + offset) % DOCUMENT_COUNT) for offset in range(PHYSICAL_BATCH)]
        actual_indices = [int(index) for index in pair_rows[pair]["document_indices"]]
        if actual_indices != expected_indices:
            raise ValueError(f"historical cyclic schedule differs from d=((8p+i) mod 602) at pair {pair}")
        for document_index in expected_indices:
            visits_per_document[document_index] += 1

    if sum(visits_per_document) != PAIR_COUNT * PHYSICAL_BATCH:
        raise AssertionError("document presentation schedule count mismatch")

    seen_chunk_indices: list[set[int]] = [set() for _ in range(DOCUMENT_COUNT)]
    visit_ordinals = [0] * DOCUMENT_COUNT
    for pair in range(PAIR_COUNT):
        for batch_index in range(PHYSICAL_BATCH):
            document_index = (PHYSICAL_BATCH * pair + batch_index) % DOCUMENT_COUNT
            chunk_count = chunks_per_document[document_index]
            if chunk_count <= 0:
                raise ValueError(f"document {document_index} has no full chunk")
            j = visit_ordinals[document_index]
            seen_chunk_indices[document_index].add(j % chunk_count)
            visit_ordinals[document_index] += 1

    covered_chunks_per_document = [len(indices) for indices in seen_chunk_indices]
    covered_chunks = sum(covered_chunks_per_document)
    if covered_chunks != sum(min(visits_per_document[i], chunks_per_document[i]) for i in range(DOCUMENT_COUNT)):
        raise AssertionError("rotation enumeration disagrees with per-document reachability")
    reached_unique_targets = covered_chunks * CHUNK_STRIDE
    source_token_count = sum(token_counts)
    all_corpus_next_token_targets = source_token_count - DOCUMENT_COUNT
    if source_token_count != int(coverage_a["coverage"]["source_token_count"]):
        raise ValueError("source token denominator differs from Coverage-A")

    presentation_count = PAIR_COUNT * PHYSICAL_BATCH
    target_presentations = presentation_count * CHUNK_STRIDE
    full_pool_coverage = reached_unique_targets / all_complete_chunk_targets
    all_corpus_target_coverage = reached_unique_targets / all_corpus_next_token_targets
    source_positions_reached = reached_unique_targets + DOCUMENT_COUNT
    source_position_coverage = source_positions_reached / source_token_count

    per_document = []
    for index, row in enumerate(documents):
        per_document.append(
            {
                "document_order_index": index,
                "document_index": int(row["document_index"]),
                "full_text_sha256": row["full_text_sha256"],
                "retained_513_token_sha256": row["retained_513_token_sha256"],
                "source_token_count": token_counts[index],
                "full_chunk_count": chunks_per_document[index],
                "historical_visits": visits_per_document[index],
                "reached_chunk_count": covered_chunks_per_document[index],
                "unreached_full_chunk_count": chunks_per_document[index] - covered_chunks_per_document[index],
                "reached_unique_targets": covered_chunks_per_document[index] * CHUNK_STRIDE,
            }
        )

    report: dict[str, Any] = {
        "schema": "omega-train-data-coverage-c-document-balanced-stage0-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-C-DOCUMENT-BALANCED-MULTICHUNK",
        "status": "STAGE0_COMPLETE_NO_TRAINING",
        "policy": {
            "document_schedule": "historical cyclic order d=((8p+i) mod 602), p=0..999, i=0..7",
            "chunk_schedule": "on zero-based document visit j, choose chunk_index=j mod n_d",
            "updates": UPDATES,
            "pair_positions": PAIR_COUNT,
            "physical_batch": PHYSICAL_BATCH,
            "documents_per_update": PHYSICAL_BATCH * PAIR_COUNT // UPDATES,
            "documents": DOCUMENT_COUNT,
            "chunk_tokens": CHUNK_TOKENS,
            "stride_tokens": CHUNK_STRIDE,
            "unique_target_tokens_per_full_chunk": CHUNK_STRIDE,
        },
        "source_identity": {
            "coverage_a_stage0_report_path": str(COVERAGE_A_REPORT),
            "coverage_a_stage0_report_self_sha256": coverage_a_self_hash,
            "coverage_a_multichunk_manifest_path": str(COVERAGE_A_MANIFEST),
            "coverage_a_multichunk_manifest_file_sha256": _sha256_file(COVERAGE_A_MANIFEST),
            "coverage_a_multichunk_manifest_self_sha256": multichunk_self_hash,
            "coverage_a_chunk_payload_sha256": multichunk["chunk_payload"]["sha256"],
            "historical_train_manifest_path": str(HISTORICAL_TRAIN_MANIFEST),
            "historical_train_manifest_sha256": historical_self_hash,
            "historical_document_order_sha256": coverage_a["source_identity"]["historical_document_order_sha256"],
            "documents_and_token_counts_match_historical_order": True,
            "historical_pair_schedule_matches_requested_formula": True,
        },
        "coverage": {
            "document_visit_count_min": min(visits_per_document),
            "document_visit_count_max": max(visits_per_document),
            "documents_with_13_visits": visits_per_document.count(13),
            "documents_with_14_visits": visits_per_document.count(14),
            "document_visit_counts_match_8000_positions": sum(visits_per_document) == presentation_count,
            "full_chunks_in_corpus": total_chunks,
            "all_complete_chunk_targets": all_complete_chunk_targets,
            "all_corpus_next_token_targets": all_corpus_next_token_targets,
            "document_chunk_presentations": presentation_count,
            "target_presentations_in_2000_updates": target_presentations,
            "unique_chunks_reached": covered_chunks,
            "unique_targets_reached": reached_unique_targets,
            "unreached_full_chunks": total_chunks - covered_chunks,
            "documents_with_all_full_chunks_reached": sum(covered_chunks_per_document[i] == chunks_per_document[i] for i in range(DOCUMENT_COUNT)),
            "unique_target_coverage_of_all_full_chunks_fraction": full_pool_coverage,
            "unique_target_coverage_of_all_full_chunks_percent": 100.0 * full_pool_coverage,
            "unique_target_coverage_of_all_corpus_next_token_targets_fraction": all_corpus_target_coverage,
            "unique_target_coverage_of_all_corpus_next_token_targets_percent": 100.0 * all_corpus_target_coverage,
            "source_position_coverage_percent_comparable_to_coverage_a": 100.0 * source_position_coverage,
            "coverage_a_reported_coverage_percent_for_comparison": coverage_a["coverage"]["new_coverage_percent"],
            "coverage_a_coverage_basis_note": "Coverage-A reports source positions including one context token per document; this is close to, but not identical to, the strict next-token-target ratio, which excludes one initial context token from numerator and denominator.",
        },
        "execution_flags": {
            "optimizer_updates": 0,
            "scientific_training_started": False,
            "checkpoint_loaded": False,
            "teacher_forward_used": False,
            "hidden_cache_used": False,
            "validation_or_test_scored": False,
            "train_split_reconstructed_or_tokenized": False,
        },
        "per_document_reachability": per_document,
        "report_self_sha256": "",
    }
    report["report_self_sha256"] = _canonical_hash({key: value for key, value in report.items() if key != "report_self_sha256"})
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "coverage_c_stage0_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-coverage-c-stage0-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_coverage_c_stage0_go:
        parser.error("OMEGA-TRAIN-DATA-COVERAGE-C Stage-0 requires the explicit MD/291 GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir else HERE / "results" / f"coverage_c_stage0_document_balanced_{stamp}"
    report = run_stage0(output_dir)
    summary = {key: report["coverage"][key] for key in (
        "documents_with_13_visits",
        "documents_with_14_visits",
        "full_chunks_in_corpus",
        "all_complete_chunk_targets",
        "all_corpus_next_token_targets",
        "unique_chunks_reached",
        "unique_targets_reached",
        "unreached_full_chunks",
        "unique_target_coverage_of_all_full_chunks_percent",
        "unique_target_coverage_of_all_corpus_next_token_targets_percent",
        "source_position_coverage_percent_comparable_to_coverage_a",
    )}
    summary.update({"status": report["status"], "report": str(output_dir / "coverage_c_stage0_report.json"), "report_self_sha256": report["report_self_sha256"]})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
