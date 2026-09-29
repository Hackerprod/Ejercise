"""Read-only Coverage-C V0/Vlater/Vall and document-macro evaluation."""

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
C_TRAINING_ROOT = HERE / "results" / "coverage_c_document_balanced_20260928_md291"
C_TRAINING_REPORT_PATH = C_TRAINING_ROOT / "coverage_c_training_report.json"
B_REPORT_PATH = HERE / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_distribution_report.json"
B2_REPORT_PATH = HERE / "results" / "coverage_b2_document_macro_20260928_md291" / "coverage_b2_document_macro_report.json"
DEFAULT_OUTPUT_DIR = HERE / "results" / "coverage_c_distribution_evaluation_20260928_md291"
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
K = 4
T_CRIT_90_DF4 = 2.131846786326383

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_coverage_b_distribution_diagnostic as coverage_b  # noqa: E402
import run_coverage_b2_document_macro_check as coverage_b2  # noqa: E402


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


def _paired_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"paired metric {metric} is missing one or more of the five frozen seeds")
    values = [float(by_seed[seed]["D_old_minus_document_balanced"]) for seed in SEEDS]
    mean = sum(values) / len(values)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half = T_CRIT_90_DF4 * sample_sd / math.sqrt(len(values))
    ci = [mean - half, mean + half]
    if ci[0] > 0:
        classification = "DOCUMENT_BALANCED_COVERAGE_IMPROVES"
    elif ci[1] < 0:
        classification = "DOCUMENT_BALANCED_COVERAGE_REGRESSES"
    else:
        classification = "NO_CLEAR_EFFECT"
    return {
        "metric": metric,
        "n": len(values),
        "per_seed_D_old_minus_document_balanced": {str(seed): by_seed[seed]["D_old_minus_document_balanced"] for seed in SEEDS},
        "mean": mean,
        "sample_sd": sample_sd,
        "t_critical_90_two_sided_df4": T_CRIT_90_DF4,
        "ci90": ci,
        "classification_by_ci_sign": classification,
    }


def _load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    training = json.loads(C_TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
    training_sig = _check_self_hash(training, "report_self_sha256", "Coverage-C training report")
    if training.get("status") != "TRAINING_COMPLETE_AWAITING_C_DISTRIBUTION_EVALUATION":
        raise PermissionError("Coverage-C distribution evaluation requires all five completed K4@2000 routes")
    if int(training.get("scientific_updates_completed", -1)) != 10000 or set(training.get("runs", {})) != {str(seed) for seed in SEEDS}:
        raise ValueError("Coverage-C training report is missing a seed or expected update count")
    if any(row.get("route_pass") is not True for row in training["runs"].values()):
        raise ValueError("Coverage-C has a failed or incomplete training route")

    b_report = json.loads(B_REPORT_PATH.read_text(encoding="utf-8"))
    b_sig = _check_self_hash(b_report, "report_self_sha256", "Coverage-B distribution report")
    if b_report.get("status") != "DISTRIBUTION_DIAGNOSTIC_COMPLETE" or len(b_report.get("later_results", {})) != 10:
        raise ValueError("Coverage-C paired comparison requires the completed Coverage-B report")
    b2_report = json.loads(B2_REPORT_PATH.read_text(encoding="utf-8"))
    b2_sig = _check_self_hash(b2_report, "report_self_sha256", "Coverage-B2 report")
    if b2_report.get("status") != "B2_DOCUMENT_MACRO_COMPLETE" or len(b2_report.get("checkpoint_results", {})) != 10:
        raise ValueError("Coverage-C document-macro comparison requires completed Coverage-B2")

    manifest, chunks_by_document, payload_hash = coverage_b2._load_validation_chunks(b_report)
    sources = {
        "coverage_c_training_report_path": str(C_TRAINING_REPORT_PATH),
        "coverage_c_training_report_file_sha256": _sha256_file(C_TRAINING_REPORT_PATH),
        "coverage_c_training_report_self_sha256": training_sig,
        "coverage_b_report_path": str(B_REPORT_PATH),
        "coverage_b_report_self_sha256": b_sig,
        "coverage_b2_report_path": str(B2_REPORT_PATH),
        "coverage_b2_report_self_sha256": b2_sig,
        "validation_manifest_path": b_report["validation_chunk_manifest_path"],
        "validation_manifest_self_sha256": manifest["manifest_sha256"],
        "validation_payload_sha256": payload_hash,
        "r1_evaluator": b_report["r1_evaluator_identity"],
    }
    return training, b_report, b2_report, manifest, chunks_by_document, sources


def _score_c_checkpoint(
    seed: int,
    route: dict[str, Any],
    manifest: dict[str, Any],
    chunks_by_document: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    run_report_path = Path(route["run_report_path"])
    checkpoint_path = Path(route["run_dir"]) / "checkpoint_02000.pt"
    checkpoint_sha = _sha256_file(checkpoint_path)
    sidecar = json.loads(checkpoint_path.with_suffix(checkpoint_path.suffix + ".identity.json").read_text(encoding="utf-8"))
    if checkpoint_sha != sidecar.get("sha256"):
        raise ValueError(f"Coverage-C seed{seed} endpoint checkpoint/sidecar SHA mismatch")
    if route.get("run_report_sha256") != _sha256_file(run_report_path):
        raise ValueError(f"Coverage-C seed{seed} run-report file hash changed")
    model, identity = coverage_b._load_checkpoint(
        backend="native",
        seed=seed,
        run_report_path=run_report_path,
        expected_run_report_sha256=str(route["run_report_sha256"]),
        checkpoint_sha256=checkpoint_sha,
        ce=coverage_b.ce,
    )
    before_hash = coverage_b.quality._value_hash(model.state_dict())
    all_chunks = [row for doc_order in range(60) for row in chunks_by_document[doc_order]]
    v0_chunks = [row for row in all_chunks if int(row["chunk_index"]) == 0]
    if len(v0_chunks) != 60:
        raise ValueError("Coverage-C V0 evaluation does not contain exactly one chunk per document")
    v0_metric = coverage_b.r1.evaluate_validation(model, [{"tokens": row["tokens"]} for row in v0_chunks])

    later_document_rows: list[dict[str, Any]] = []
    vall_document_rows: list[dict[str, Any]] = []
    for doc_order in range(60):
        rows = chunks_by_document[doc_order]
        doc_meta = manifest["documents"][doc_order]
        later_rows = [row for row in rows if int(row["chunk_index"]) >= 1]
        if later_rows:
            later_metric = coverage_b.r1.evaluate_validation(model, [{"tokens": row["tokens"]} for row in later_rows])
            if not later_metric["finite"] or int(later_metric["tokens"]) != 512 * len(later_rows):
                raise ValueError(f"invalid Vlater score for C seed{seed}, document {doc_order}")
            later_document_rows.append(
                {
                    "document_order_index": doc_order,
                    "full_text_sha256": doc_meta["full_text_sha256"],
                    "chunk_count": len(later_rows),
                    "target_tokens": int(later_metric["tokens"]),
                    "nll": float(later_metric["nll"]),
                }
            )
        vall_metric = coverage_b.r1.evaluate_validation(model, [{"tokens": row["tokens"]} for row in rows])
        if not vall_metric["finite"] or int(vall_metric["tokens"]) != 512 * len(rows):
            raise ValueError(f"invalid Vall score for C seed{seed}, document {doc_order}")
        vall_document_rows.append(
            {
                "document_order_index": doc_order,
                "full_text_sha256": doc_meta["full_text_sha256"],
                "chunk_count": len(rows),
                "target_tokens": int(vall_metric["tokens"]),
                "nll": float(vall_metric["nll"]),
            }
        )

    after_hash = coverage_b.quality._value_hash(model.state_dict())
    if before_hash != after_hash:
        raise RuntimeError(f"Coverage-C seed{seed} weights changed during read-only evaluation")
    if _sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError(f"Coverage-C seed{seed} checkpoint file changed during read-only evaluation")
    later_tokens = sum(row["target_tokens"] for row in later_document_rows)
    vall_tokens = sum(row["target_tokens"] for row in vall_document_rows)
    later_nll = sum(row["nll"] * row["target_tokens"] for row in later_document_rows) / later_tokens
    vall_nll = sum(row["nll"] * row["target_tokens"] for row in vall_document_rows) / vall_tokens
    vall_macro = sum(row["nll"] for row in vall_document_rows) / len(vall_document_rows)
    stored_v0 = float(route["endpoint_nll_validation"]["nll"])
    v0_diff = float(v0_metric["nll"]) - stored_v0
    if not v0_metric["finite"] or int(v0_metric["tokens"]) != 30720 or abs(v0_diff) > coverage_b.GUARDRAIL:
        raise ValueError(f"Coverage-C seed{seed} post-training V0 portability mismatch: {v0_diff}")
    return {
        **identity,
        "condition": "document-balanced-multichunk",
        "seed": seed,
        "checkpoint_rounds": K,
        "inference_rounds": K,
        "checkpoint_sha256": checkpoint_sha,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "weights_unchanged": True,
        "checkpoint_modified": False,
        "V0": {"nll": float(v0_metric["nll"]), "tokens": int(v0_metric["tokens"]), "finite": bool(v0_metric["finite"]), "stored_training_endpoint_nll": stored_v0, "absolute_difference_from_training_endpoint": abs(v0_diff)},
        "Vlater": {"nll": later_nll, "tokens": later_tokens, "finite": True, "documents_with_later_chunks": len(later_document_rows), "per_document": later_document_rows},
        "Vall": {"nll": vall_nll, "tokens": vall_tokens, "finite": True},
        "Vall_document_macro": {"nll": vall_macro, "documents": len(vall_document_rows), "per_document": vall_document_rows},
    }


def run_evaluation(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Coverage-C evaluation output is immutable; resume explicitly: {output_dir}")
    training, b_report, b2_report, manifest, chunks_by_document, source_identity = _load_sources()
    if resume and (output_dir / "coverage_c_evaluation_progress.json").is_file():
        report = json.loads((output_dir / "coverage_c_evaluation_progress.json").read_text(encoding="utf-8"))
        if report.get("source_identity") != source_identity:
            raise ValueError("Coverage-C evaluation resume source identity mismatch")
        report.pop("report_self_sha256", None)
    else:
        report = {
            "schema": "omega-train-data-coverage-c-distribution-evaluation-v1",
            "unit": "OMEGA-TRAIN-DATA-COVERAGE-C-DOCUMENT-BALANCED-MULTICHUNK",
            "status": "EVALUATION_IN_PROGRESS",
            "source_identity": source_identity,
            "evaluation_policy": {
                "evaluator": "canonical r1.evaluate_validation",
                "precision": "FP32",
                "torch_intraop": 4,
                "torch_interop": 1,
                "validation_documents": 60,
                "validation_chunks": 452,
                "targets_per_chunk": 512,
                "reset_between_chunks": True,
                "within_chunk_state": "W0 to detached W1 state",
                "V0_source": "update-2000 run_report endpoint plus exact post-run canonical rescore",
                "Vlater_selection": "chunk_index >= 1",
                "Vall_selection": "all full chunks",
                "Vall_document_macro": "equal mean of per-document token-mean NLL across the 60 documents",
                "scientific_updates": 0,
                "teacher_forward_used": False,
                "hidden_cache_used": False,
                "test_split_loaded": False,
            },
            "checkpoint_results": {},
            "scientific_updates": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        output_dir.mkdir(parents=True, exist_ok=False)
    expected_keys = {str(seed) for seed in SEEDS}
    if set(report.get("checkpoint_results", {})) - expected_keys:
        raise ValueError("Coverage-C evaluation progress contains an unexpected seed")

    coverage_b.r1.configure_cpu_runtime()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Coverage-C evaluation requires CPU FP32 torch 4/1")

    for seed in SEEDS:
        if str(seed) in report["checkpoint_results"]:
            continue
        route = training["runs"][str(seed)]
        result = _score_c_checkpoint(seed, route, manifest, chunks_by_document)
        report["checkpoint_results"][str(seed)] = result
        _write_json(output_dir / "coverage_c_evaluation_progress.json", report)

    if set(report["checkpoint_results"]) != expected_keys:
        raise AssertionError("Coverage-C evaluation did not finish all five checkpoints")
    summary_fields = {
        "V0_CHUNK0": ("V0",),
        "VLATER_CHUNK_INDEX_GE_1": ("Vlater",),
        "VALL_ALL_FULL_CHUNKS": ("Vall",),
        "VALL_DOCUMENT_MACRO": ("Vall_document_macro",),
    }
    paired: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for metric_name, (field,) in summary_fields.items():
        rows = []
        for seed in SEEDS:
            old_key = f"old-513_seed_{seed}"
            old_multi_key = f"multichunk_seed_{seed}"
            c_result = report["checkpoint_results"][str(seed)]
            if metric_name == "V0_CHUNK0":
                old_nll = float(b_report["v0_checks"][old_key]["v0_current_nll"])
            elif metric_name == "VLATER_CHUNK_INDEX_GE_1":
                old_nll = float(b_report["later_results"][old_key]["Vlater"]["nll"])
            elif metric_name == "VALL_ALL_FULL_CHUNKS":
                old_nll = float(b_report["later_results"][old_key]["Vall"]["nll"])
            else:
                old_nll = float(b2_report["checkpoint_results"][old_key]["document_macro_nll"])
            c_nll = float(c_result[field]["nll"])
            rows.append({"seed": seed, "NLL_old": old_nll, "NLL_document_balanced": c_nll, "D_old_minus_document_balanced": old_nll - c_nll})
        paired[metric_name] = rows
        summaries[metric_name] = _paired_summary(rows, metric_name)

    comparison_to_chunk_uniform = []
    for seed in SEEDS:
        result = report["checkpoint_results"][str(seed)]
        multi_key = f"multichunk_seed_{seed}"
        multi_token = float(b_report["later_results"][multi_key]["Vall"]["nll"])
        multi_macro = float(b2_report["checkpoint_results"][multi_key]["document_macro_nll"])
        comparison_to_chunk_uniform.append(
            {
                "seed": seed,
                "C_document_balanced_Vall_token_weighted": result["Vall"]["nll"],
                "A_multichunk_chunk_uniform_Vall_token_weighted": multi_token,
                "D_chunk_uniform_minus_C_token_weighted": multi_token - result["Vall"]["nll"],
                "C_document_balanced_Vall_document_macro": result["Vall_document_macro"]["nll"],
                "A_multichunk_chunk_uniform_Vall_document_macro": multi_macro,
                "D_chunk_uniform_minus_C_document_macro": multi_macro - result["Vall_document_macro"]["nll"],
            }
        )

    report["paired_deltas_by_stratum"] = paired
    report["summary_by_stratum"] = summaries
    report["primary_metric"] = "VALL_ALL_FULL_CHUNKS token-weighted NLL"
    report["primary_classification"] = summaries["VALL_ALL_FULL_CHUNKS"]["classification_by_ci_sign"]
    report["secondary_metric"] = "VALL_DOCUMENT_MACRO"
    report["comparison_to_chunk_uniform_multichunk"] = {
        "per_seed": comparison_to_chunk_uniform,
        "mean_D_chunk_uniform_minus_C_token_weighted": sum(row["D_chunk_uniform_minus_C_token_weighted"] for row in comparison_to_chunk_uniform) / len(comparison_to_chunk_uniform),
        "mean_D_chunk_uniform_minus_C_document_macro": sum(row["D_chunk_uniform_minus_C_document_macro"] for row in comparison_to_chunk_uniform) / len(comparison_to_chunk_uniform),
        "note": "descriptive comparison to Coverage-A chunk-uniform training, reported separately for matching token-weighted and document-macro evaluation metrics",
    }
    report["status"] = "COVERAGE_C_COMPLETE"
    report["scientific_updates"] = 0
    report["checkpoint_modified"] = False
    report["teacher_forward_used"] = False
    report["hidden_cache_used"] = False
    report["test_split_loaded"] = False
    report["report_self_sha256"] = coverage_b.quality._canonical_hash(report)
    _write_json(output_dir / "coverage_c_distribution_evaluation_report.json", report)
    _write_json(output_dir / "coverage_c_evaluation_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-coverage-c-evaluation-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.confirm_coverage_c_evaluation_go:
        parser.error("Coverage-C evaluation requires the explicit MD/291/293 GO and a completed training report")
    output_dir = args.output_dir.resolve()
    try:
        report = run_evaluation(output_dir, resume=args.resume)
    except Exception as error:
        if output_dir.exists():
            _write_json(
                output_dir / "coverage_c_evaluation_failure.json",
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
                "primary_metric": report["primary_metric"],
                "primary_summary": report["summary_by_stratum"]["VALL_ALL_FULL_CHUNKS"],
                "secondary_summary": report["summary_by_stratum"]["VALL_DOCUMENT_MACRO"],
                "report": str(output_dir / "coverage_c_distribution_evaluation_report.json"),
                "scientific_updates": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
