"""Read-only document-macro rescoring for the completed Coverage-B checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch


HERE = Path(__file__).resolve().parent
B_REPORT_PATH = HERE / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_distribution_report.json"
DEFAULT_OUTPUT_DIR = HERE / "results" / "coverage_b2_document_macro_20260928_md291"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_coverage_b_distribution_diagnostic as coverage_b  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != coverage_b.quality._canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")
    return str(signature)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    coverage_b.quality._write_json(path, value)


def _load_validation_chunks(report: dict[str, Any]) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], str]:
    manifest_path = Path(report["validation_chunk_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sig = _check_self_hash(manifest, "manifest_sha256", "Coverage-B validation-chunk manifest")
    if manifest_sig != report["validation_chunk_manifest_sha256"]:
        raise ValueError("Coverage-B report and validation manifest identities disagree")
    payload_path = Path(manifest["chunk_payload"]["path"])
    payload_hash = _sha256_file(payload_path)
    if payload_hash != manifest["chunk_payload"]["sha256"] or payload_hash != report["validation_chunk_payload_sha256"]:
        raise ValueError("Coverage-B validation chunk payload SHA mismatch")
    if int(manifest.get("document_count", -1)) != 60 or int(manifest["chunk_policy"].get("chunk_count_all", -1)) != 452:
        raise ValueError("B2 requires the sealed 60-document, 452-chunk Vall manifest")
    if manifest.get("leakage", {}).get("train_validation_full_text_sha_intersection") != 0:
        raise ValueError("Coverage-B validation manifest reports train leakage")
    if manifest.get("leakage", {}).get("train_validation_dedupe_key_intersection") != 0:
        raise ValueError("Coverage-B validation manifest reports train dedupe-key leakage")
    if manifest.get("leakage", {}).get("test_split_loaded") is not False:
        raise ValueError("Coverage-B validation manifest unexpectedly loaded test data")

    document_metadata = manifest["documents"]
    if len(document_metadata) != 60:
        raise ValueError("validation document identity list is incomplete")
    grouped: dict[int, list[dict[str, Any]]] = {index: [] for index in range(60)}
    with payload_path.open("r", encoding="utf-8") as stream:
        record_count = 0
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            record_count += 1
            doc_order = int(row["validation_document_order_index"])
            if doc_order not in grouped:
                raise ValueError(f"validation chunk has out-of-range document order {doc_order}")
            if int(row["token_count"]) != 513 or len(row["tokens"]) != 513 or int(row["target_end_exclusive"]) - int(row["target_start"]) != 512:
                raise ValueError("validation chunk does not contain exactly 512 targets")
            if row["full_text_sha256"] != document_metadata[doc_order]["full_text_sha256"]:
                raise ValueError(f"validation chunk/document SHA identity mismatch at document {doc_order}")
            grouped[doc_order].append(row)
    if record_count != 452:
        raise ValueError(f"validation payload has {record_count} rows, expected 452")
    for doc_order, rows in grouped.items():
        rows.sort(key=lambda row: int(row["chunk_index"]))
        if not rows or [int(row["chunk_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"validation chunk indices are incomplete for document {doc_order}")
    return manifest, grouped, payload_hash


def _input_identity(report: dict[str, Any], manifest: dict[str, Any], payload_hash: str) -> dict[str, Any]:
    return {
        "coverage_b_report_path": str(B_REPORT_PATH),
        "coverage_b_report_sha256": _sha256_file(B_REPORT_PATH),
        "coverage_b_report_self_sha256": report["report_self_sha256"],
        "validation_manifest_path": report["validation_chunk_manifest_path"],
        "validation_manifest_self_sha256": manifest["manifest_sha256"],
        "validation_payload_sha256": payload_hash,
        "qualification_manifest_sha256": report["qualification_manifest_sha256"],
        "r1_evaluator": report["r1_evaluator_identity"],
    }


def run_check(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"B2 output is immutable; use --resume only for this run: {output_dir}")

    b_report = json.loads(B_REPORT_PATH.read_text(encoding="utf-8"))
    b_report_self_hash = _check_self_hash(b_report, "report_self_sha256", "Coverage-B report")
    if b_report.get("status") != "DISTRIBUTION_DIAGNOSTIC_COMPLETE" or b_report.get("v0_all_ten_pass") is not True:
        raise ValueError("B2 requires completed Coverage-B with the V0 gate passed")
    if len(b_report.get("later_results", {})) != 10:
        raise ValueError("Coverage-B report does not contain all ten Vall results")
    validation_manifest, chunks_by_document, validation_payload_hash = _load_validation_chunks(b_report)
    input_identity = _input_identity(b_report, validation_manifest, validation_payload_hash)

    if resume and (output_dir / "coverage_b2_progress.json").is_file():
        report = json.loads((output_dir / "coverage_b2_progress.json").read_text(encoding="utf-8"))
        if report.get("input_identity") != input_identity:
            raise ValueError("B2 input identity changed during resume")
    else:
        report = {
            "schema": "omega-train-data-coverage-b2-document-macro-check-v1",
            "unit": "OMEGA-TRAIN-DATA-COVERAGE-B2-DOCUMENT-MACRO-CHECK",
            "status": "B2_IN_PROGRESS",
            "input_identity": input_identity,
            "checkpoint_matrix": coverage_b._resolve_ten_checkpoints(),
            "evaluation_policy": {
                "purpose": "secondary predeclared document-macro metric for Coverage-B",
                "evaluator": "canonical r1.evaluate_validation",
                "precision": "FP32",
                "torch_intraop": 4,
                "torch_interop": 1,
                "documents": 60,
                "chunks": 452,
                "targets_per_chunk": 512,
                "state_reset_between_chunks": True,
                "within_chunk_state": "W0 to detached W1 state",
                "document_macro_definition": "arithmetic mean across documents of each document's NLL averaged over its own target tokens",
                "optimizer_updates": 0,
                "teacher_forward_used": False,
                "hidden_cache_used": False,
                "test_split_loaded": False,
            },
            "checkpoint_results": {},
            "scientific_updates": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        output_dir.mkdir(parents=True, exist_ok=False)

    if output_dir.exists() is False:
        output_dir.mkdir(parents=True, exist_ok=False)
    allowed_keys = {f"{condition}_seed_{seed}" for condition in ("old-513", "multichunk") for seed in coverage_b.SEEDS}
    if set(report.get("checkpoint_results", {})) - allowed_keys:
        raise ValueError("B2 resume contains results outside the frozen ten-checkpoint matrix")

    coverage_b.r1.configure_cpu_runtime()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("B2 requires CPU FP32 torch intraop/interop 4/1")

    routes = report["checkpoint_matrix"]
    if len(routes) != 10 or {f"{row['condition']}_seed_{int(row['seed'])}" for row in routes} != allowed_keys:
        raise ValueError("B2 checkpoint matrix differs from the frozen five-seed × two-condition matrix")
    for route in routes:
        key = f"{route['condition']}_seed_{int(route['seed'])}"
        if key in report["checkpoint_results"]:
            continue
        model, identity = coverage_b._load_checkpoint(
            backend="native",
            seed=int(route["seed"]),
            run_report_path=Path(route["run_report_path"]),
            expected_run_report_sha256=str(route["run_report_sha256"]),
            checkpoint_sha256=str(route["checkpoint_sha256"]),
            ce=coverage_b.ce,
        )
        before_hash = coverage_b.quality._value_hash(model.state_dict())
        per_document: list[dict[str, Any]] = []
        for document_order in range(60):
            rows = chunks_by_document[document_order]
            metric = coverage_b.r1.evaluate_validation(model, [{"tokens": row["tokens"]} for row in rows])
            expected_tokens = len(rows) * 512
            if not metric["finite"] or int(metric["tokens"]) != expected_tokens:
                raise ValueError(f"nonfinite or incomplete per-document validation for {key}, document {document_order}")
            per_document.append(
                {
                    "document_order_index": document_order,
                    "full_text_sha256": validation_manifest["documents"][document_order]["full_text_sha256"],
                    "chunk_count": len(rows),
                    "target_tokens": expected_tokens,
                    "nll": float(metric["nll"]),
                }
            )
        after_hash = coverage_b.quality._value_hash(model.state_dict())
        if before_hash != after_hash:
            raise RuntimeError(f"checkpoint weights changed during B2 evaluation for {key}")

        total_targets = sum(row["target_tokens"] for row in per_document)
        token_weighted_nll_from_docs = sum(row["nll"] * row["target_tokens"] for row in per_document) / total_targets
        stored_vall = float(b_report["later_results"][key]["Vall"]["nll"])
        report["checkpoint_results"][key] = {
            **identity,
            "condition": route["condition"],
            "seed": int(route["seed"]),
            "document_count": len(per_document),
            "target_tokens": total_targets,
            "document_macro_nll": sum(row["nll"] for row in per_document) / len(per_document),
            "reconstructed_token_weighted_nll": token_weighted_nll_from_docs,
            "coverage_b_stored_vall_nll": stored_vall,
            "reconstructed_minus_coverage_b_vall": token_weighted_nll_from_docs - stored_vall,
            "weights_unchanged": True,
            "finite": True,
            "per_document": per_document,
        }
        del model
        gc.collect()
        _write_json(output_dir / "coverage_b2_progress.json", report)

    if set(report["checkpoint_results"]) != allowed_keys:
        raise AssertionError("B2 did not finish all ten frozen checkpoints")
    paired: list[dict[str, Any]] = []
    descriptive_chunk_uniform: list[dict[str, Any]] = []
    for seed in coverage_b.SEEDS:
        old_key = f"old-513_seed_{seed}"
        multi_key = f"multichunk_seed_{seed}"
        old_macro = report["checkpoint_results"][old_key]["document_macro_nll"]
        multi_macro = report["checkpoint_results"][multi_key]["document_macro_nll"]
        old_weighted = report["checkpoint_results"][old_key]["reconstructed_token_weighted_nll"]
        multi_weighted = report["checkpoint_results"][multi_key]["reconstructed_token_weighted_nll"]
        stored_multi_weighted = float(b_report["later_results"][multi_key]["Vall"]["nll"])
        descriptive_chunk_uniform.append(
            {
                "seed": seed,
                "multichunk_document_macro_nll": multi_macro,
                "multichunk_chunk_uniform_training_stored_token_weighted_vall_nll": stored_multi_weighted,
                "document_macro_minus_stored_token_weighted_vall": multi_macro - stored_multi_weighted,
                "metric_comparison_note": "descriptive only: document-macro and token-weighted Vall use different evaluation weights",
            }
        )
        paired.append(
            {
                "seed": seed,
                "NLL_old_document_macro": old_macro,
                "NLL_multichunk_document_macro": multi_macro,
                "D_old_minus_multichunk_document_macro": old_macro - multi_macro,
                "old_token_weighted_vall_reconstructed": old_weighted,
                "multichunk_token_weighted_vall_reconstructed": multi_weighted,
                "multichunk_token_weighted_vall_stored": stored_multi_weighted,
            }
        )

    summary_rows = [
        {"seed": row["seed"], "D_old_minus_multichunk": row["D_old_minus_multichunk_document_macro"]}
        for row in paired
    ]
    report["paired_document_macro_deltas"] = paired
    report["document_macro_summary"] = coverage_b._paired_summary(summary_rows, "VALL_DOCUMENT_MACRO")
    report["descriptive_comparison_to_multichunk_chunk_uniform"] = {
        "per_seed": descriptive_chunk_uniform,
        "mean_document_macro_minus_stored_token_weighted_vall": sum(row["document_macro_minus_stored_token_weighted_vall"] for row in descriptive_chunk_uniform) / len(descriptive_chunk_uniform),
        "note": "descriptive only; the compared evaluation weightings differ",
    }
    report["vall_token_weighted_reproduction"] = {
        "absolute_differences_from_coverage_b_stored_vall": {
            key: result["reconstructed_minus_coverage_b_vall"] for key, result in report["checkpoint_results"].items()
        },
        "note": "same canonical scorer/data, regrouped by document; floating-point reduction order may cause small differences",
    }
    report["status"] = "B2_DOCUMENT_MACRO_COMPLETE"
    report["report_self_sha256"] = coverage_b.quality._canonical_hash(report)
    _write_json(output_dir / "coverage_b2_document_macro_report.json", report)
    _write_json(output_dir / "coverage_b2_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-coverage-b2-document-macro-check-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.confirm_coverage_b2_document_macro_check_go:
        parser.error("OMEGA-TRAIN-DATA-COVERAGE-B2 requires the explicit MD/291 GO")
    output_dir = args.output_dir.resolve()
    try:
        report = run_check(output_dir, resume=args.resume)
    except Exception as error:
        if output_dir.exists():
            _write_json(
                output_dir / "coverage_b2_failure.json",
                {
                    "status": "FAILED_RUNTIME",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "scientific_updates": 0,
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "document_macro_summary": report["document_macro_summary"],
                "descriptive_comparison_to_multichunk_chunk_uniform": report["descriptive_comparison_to_multichunk_chunk_uniform"],
                "scientific_updates": report["scientific_updates"],
                "report": str(output_dir / "coverage_b2_document_macro_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
