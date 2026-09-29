"""Read-only V0/Vlater/Vall distribution diagnostic for OMEGA coverage A."""

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

import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
R1_SCOPE = HERE.parents[1] / "campaign" / "omega_core_lm_0_r1_scientific_scoping_a"
CE_DIR = HERE.parents[1] / "campaign" / "omega_ce_only_baseline"
EXPANDED_DIR = HERE.parents[1] / "campaign" / "omega_expanded_frozen_validation"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
INPUTS_DIR = CAMPAIGN / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
VALIDATION_MANIFEST_PATH = INPUTS_DIR / "validation_manifest_60.json"
QUALIFICATION_MANIFEST_PATH = INPUTS_DIR / "qualification_manifest.json"
TRAIN_MANIFEST_PATH = INPUTS_DIR / "train_manifest_1000_pairs.json"
CURVE_ROOT = HERE / "results" / "k_curve_primary_20260927_md286"
BLOCK_A_REPORT_PATH = CURVE_ROOT / "block_A_report.json"
BLOCK_B_REPORT_PATH = CURVE_ROOT / "block_B_report.json"
MULTICHUNK_ROOT = HERE / "results" / "train_data_coverage_stage0_20260927_md288_retry2" / "multichunk_training"
MULTICHUNK_REPORT_PATH = MULTICHUNK_ROOT / "multichunk_coverage_report.json"
VAL_CHUNK_TOKENS = 513
VAL_CHUNK_STRIDE = 512
EXPECTED_DOCS = 60
EXPECTED_V0_TOKENS = 30_720
GUARDRAIL = 1.0e-4
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
K = 4
T_CRIT_90_DF4 = 2.131846786326383

for _path in (CAMPAIGN, R1_SCOPE, CE_DIR, EXPANDED_DIR, SCRIPTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_backend_quality_qualification as quality  # noqa: E402
import run_scientific_scoping_a as r1  # noqa: E402
import run_omega_ce_only_baseline as ce  # noqa: E402
import run_omega_expanded_frozen_validation as expanded  # noqa: E402
import run_omega_core_lm_0_r1_training_technical_preflight as technical  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_selfhash(value: dict[str, Any], field: str, label: str) -> None:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != quality._canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")


def _collect_full_validation_documents(
    validation_dataset: Any,
    tokenizer: Any,
    expected_documents: list[dict[str, Any]],
    excluded_train_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    expected_keys = [
        (str(row["full_text_sha256"]), str(row["retained_513_token_sha256"]))
        for row in expected_documents
    ]
    expected_by_key = {key: row for key, row in zip(expected_keys, expected_documents)}
    recovered: dict[tuple[str, str], dict[str, Any]] = {}
    for document_index, source in enumerate(technical.reconstruct_documents(validation_dataset)):
        text = str(source["text"])
        tokens = list(tokenizer.encode(text, add_special_tokens=False))
        if len(tokens) < VAL_CHUNK_TOKENS:
            continue
        first = tokens[:VAL_CHUNK_TOKENS]
        key = (
            technical.sha256_text(text),
            technical.sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in first)),
        )
        if key in excluded_train_keys:
            raise ValueError("frozen validation candidate overlaps the historical train dedupe keys")
        if key not in expected_by_key or key in recovered:
            continue
        expected = expected_by_key[key]
        recovered[key] = {
            "document_index": int(expected["document_index"]),
            "row_range": list(expected["row_range"]),
            "header": str(expected["header"]),
            "token_count": len(tokens),
            "selected_token_count": VAL_CHUNK_TOKENS,
            "full_text_sha256": key[0],
            "retained_513_token_sha256": key[1],
            "source_dataset_document_index": document_index,
            "tokens": tokens,
        }
    if list(recovered) != expected_keys or len(recovered) != EXPECTED_DOCS:
        missing = [key for key in expected_keys if key not in recovered]
        raise ValueError(f"could not reconstruct the exact frozen 60 validation docs; missing={len(missing)}")
    docs = [recovered[key] for key in expected_keys]
    for index, (actual, expected) in enumerate(zip(docs, expected_documents)):
        for field in ("document_index", "row_range", "header", "token_count", "selected_token_count", "full_text_sha256", "retained_513_token_sha256"):
            if actual[field] != expected[field]:
                raise ValueError(f"validation document identity drift at index {index}, field={field}")
    return docs


def _make_validation_strata(docs: list[dict[str, Any]], output_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path]:
    chunks_by_depth: list[list[dict[str, Any]]] = []
    per_doc_counts: list[int] = []
    for document_order, doc in enumerate(docs):
        tokens = doc["tokens"]
        starts = list(range(0, len(tokens) - VAL_CHUNK_TOKENS + 1, VAL_CHUNK_STRIDE))
        per_doc_counts.append(len(starts))
        previous_target_end: int | None = None
        for depth, start in enumerate(starts):
            target_start, target_end = start + 1, start + VAL_CHUNK_TOKENS
            if previous_target_end is not None and previous_target_end != target_start:
                raise AssertionError("validation chunks duplicate or skip target positions")
            previous_target_end = target_end
            if depth >= len(chunks_by_depth):
                chunks_by_depth.append([])
            chunk_tokens = tokens[start : start + VAL_CHUNK_TOKENS]
            chunk_hash = technical.sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in chunk_tokens))
            chunks_by_depth[depth].append(
                {
                    "document_index": -1,
                    "validation_document_order_index": document_order,
                    "source_document_index": int(doc["document_index"]),
                    "full_text_sha256": doc["full_text_sha256"],
                    "source_first_chunk_sha256": doc["retained_513_token_sha256"],
                    "chunk_index": depth,
                    "chunk_token_sha256": chunk_hash,
                    "token_start": start,
                    "token_end_exclusive": start + VAL_CHUNK_TOKENS,
                    "target_start": target_start,
                    "target_end_exclusive": target_end,
                    "token_count": VAL_CHUNK_TOKENS,
                    "tokens": chunk_tokens,
                }
            )
    all_chunks: list[dict[str, Any]] = []
    for depth, level in enumerate(chunks_by_depth):
        level.sort(key=lambda chunk: int(chunk["validation_document_order_index"]))
        all_chunks.extend(level)
    for pool_index, chunk in enumerate(all_chunks):
        chunk["chunk_pool_index"] = pool_index

    strata = {
        "V0_CHUNK0": [chunk for chunk in all_chunks if int(chunk["chunk_index"]) == 0],
        "VLATER_CHUNK_INDEX_GE_1": [chunk for chunk in all_chunks if int(chunk["chunk_index"]) >= 1],
        "VALL_ALL_FULL_CHUNKS": list(all_chunks),
    }
    if len(strata["V0_CHUNK0"]) != EXPECTED_DOCS or not strata["VLATER_CHUNK_INDEX_GE_1"]:
        raise ValueError("validation strata are incomplete or later-chunk stratum is empty")

    payload_path = output_dir / "coverage_b_validation_chunks.jsonl"
    with payload_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in all_chunks:
            stream.write(json.dumps(chunk, sort_keys=True, separators=(",", ":")) + "\n")
    payload_sha = _sha256_file(payload_path)
    manifest = {
        "schema": "omega-train-data-coverage-b-validation-manifest-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-B-DISTRIBUTION-DIAGNOSTIC",
        "mode": "VALIDATION_ONLY_READ_ONLY",
        "dataset": {
            "id": technical.DATASET_ID,
            "config": technical.DATASET_CONFIG,
            "revision": technical.DATASET_REVISION,
            "split": "validation",
        },
        "source_validation_manifest_sha256": quality._canonical_hash({key: value for key, value in json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8")).items() if key != "manifest_sha256"}),
        "document_count": len(docs),
        "documents": [
            {key: value for key, value in doc.items() if key != "tokens"}
            for doc in docs
        ],
        "chunk_policy": {
            "chunk_tokens": VAL_CHUNK_TOKENS,
            "stride_tokens": VAL_CHUNK_STRIDE,
            "full_chunks_only": True,
            "context_overlap_tokens": 1,
            "depth_major_order": True,
            "reset_state_between_chunks": True,
            "within_chunk_windows": [
                {"window": "W0", "input_range": [0, 256], "target_range": [1, 257]},
                {"window": "W1", "input_range": [256, 512], "target_range": [257, 513], "state": "detached W0 state"},
            ],
            "chunk_counts_by_depth": {str(depth): len(rows) for depth, rows in enumerate(chunks_by_depth)},
            "chunk_count_all": len(all_chunks),
        },
        "strata": {
            name: {
                "chunk_count": len(rows),
                "target_tokens": len(rows) * 512,
                "document_count": len({int(row["validation_document_order_index"]) for row in rows}),
                "selection": "chunk_index == 0" if name == "V0_CHUNK0" else "chunk_index >= 1" if name == "VLATER_CHUNK_INDEX_GE_1" else "all full chunks",
            }
            for name, rows in strata.items()
        },
        "chunk_payload": {"path": str(payload_path), "sha256": payload_sha, "format": "JSONL token IDs, 513 tokens per complete chunk"},
        "leakage": {"train_validation_full_text_sha_intersection": 0, "train_validation_dedupe_key_intersection": 0, "test_split_loaded": False},
        "training_or_optimizer_updates": 0,
        "teacher_forward_or_hidden_cache_used": False,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = quality._canonical_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    manifest_path = output_dir / "coverage_b_validation_manifest.json"
    quality._write_json(manifest_path, manifest)
    return manifest, strata, manifest_path


def _load_checkpoint(
    *,
    backend: str,
    seed: int,
    run_report_path: Path,
    expected_run_report_sha256: str,
    checkpoint_sha256: str,
    ce: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if _sha256_file(run_report_path) != expected_run_report_sha256:
        raise ValueError(f"run report SHA mismatch: {run_report_path}")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    if (
        run_report.get("status") != "COMPLETE"
        or run_report.get("backend") != "native"
        or int(run_report.get("K", -1)) != K
        or int(run_report.get("seed", -1)) != seed
        or run_report.get("test_split_loaded") is not False
        or int(run_report.get("endpoint_nll_validation", {}).get("update", -1)) != 2000
        or int(run_report.get("endpoint_nll_validation", {}).get("tokens", -1)) != 30720
    ):
        raise ValueError(f"K4@2000 checkpoint/report identity or endpoint mismatch: {run_report_path}")
    checkpoint_path = Path(run_report["run_dir"]) / "checkpoint_02000.pt"
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".identity.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_sha = _sha256_file(checkpoint_path)
    if actual_sha != checkpoint_sha256 or sidecar.get("sha256") != actual_sha:
        raise ValueError(f"K4 checkpoint file/sidecar SHA mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    identity = payload.get("identity", {})
    if (
        int(payload.get("completed_updates", -1)) != 2000
        or int(payload.get("next_update", -1)) != 2000
        or int(identity.get("K", -1)) != K
        or int(identity.get("seed", -1)) != seed
        or identity.get("backend") != "native"
        or payload.get("model") is None
    ):
        raise ValueError(f"checkpoint payload update/seed/K/backend mismatch: {checkpoint_path}")
    model = ce.fresh_model(seed, K)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    model_hash_before = quality._value_hash(model.state_dict())
    del payload
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_sha,
        "run_report_path": str(run_report_path),
        "run_report_sha256": expected_run_report_sha256,
        "update": 2000,
        "seed": seed,
        "K": K,
        "model_state_sha256_before": model_hash_before,
        "stored_primary_validation_nll": float(run_report["endpoint_nll_validation"]["nll"]),
    }


def _resolve_ten_checkpoints() -> list[dict[str, Any]]:
    block_a = json.loads(BLOCK_A_REPORT_PATH.read_text(encoding="utf-8"))
    block_b = json.loads(BLOCK_B_REPORT_PATH.read_text(encoding="utf-8"))
    multi = json.loads(MULTICHUNK_REPORT_PATH.read_text(encoding="utf-8"))
    for label, report, self_field in (("Block A", block_a, "self_sha256"), ("Block B", block_b, "self_sha256"), ("multichunk", multi, "report_self_sha256")):
        unsigned = dict(report)
        signature = unsigned.pop(self_field, None)
        if not signature or signature != quality._canonical_hash(unsigned):
            raise ValueError(f"{label} report self-hash failed")
    if block_a.get("status") != "COMPLETE" or block_b.get("status") != "COMPLETE" or multi.get("status") != "COMPLETE":
        raise PermissionError("distribution diagnostic requires all ten completed K4@2000 checkpoints")
    if len(multi.get("runs", {})) != 5:
        raise ValueError("multichunk report does not contain five K4 seeds")

    routes: list[dict[str, Any]] = []
    for seed in SEEDS:
        old_block = block_a if seed in (20260913, 20260914) else block_b
        old_row = old_block["routes"][f"native_K4_seed_{seed}"]
        multi_row = multi["runs"][str(seed)]
        old_run_report = json.loads(Path(old_row["run_report_path"]).read_text(encoding="utf-8"))
        old_checkpoint_path = Path(old_run_report["run_dir"]) / "checkpoint_02000.pt"
        old_checkpoint_sidecar = json.loads(old_checkpoint_path.with_suffix(old_checkpoint_path.suffix + ".identity.json").read_text(encoding="utf-8"))
        old_checkpoint_sha = _sha256_file(old_checkpoint_path)
        if old_checkpoint_sha != old_checkpoint_sidecar.get("sha256"):
            raise ValueError(f"old-513 seed{seed} K4 checkpoint/sidecar SHA mismatch")
        routes.extend(
            [
                {
                    "condition": "old-513",
                    "seed": seed,
                    "run_report_path": old_row["run_report_path"],
                    "run_report_sha256": old_row["run_report_sha256"],
                    "checkpoint_sha256": old_checkpoint_sha,
                    "stored_validation_nll": float(old_row["endpoint_nll_validation"]["nll"]),
                },
                {
                    "condition": "multichunk",
                    "seed": seed,
                    "run_report_path": multi_row["run_report_path"],
                    "run_report_sha256": multi_row["run_report_sha256"],
                    "checkpoint_sha256": multi_row["checkpoint_02000_sha256"],
                    "stored_validation_nll": float(multi_row["endpoint_nll_validation"]["nll"]),
                },
            ]
        )
    return routes


def _paired_summary(rows: list[dict[str, Any]], stratum: str) -> dict[str, Any]:
    by_seed = {row["seed"]: row for row in rows}
    values = [by_seed[seed]["D_old_minus_multichunk"] for seed in SEEDS]
    mean = sum(values) / len(values)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half = T_CRIT_90_DF4 * sample_sd / math.sqrt(len(values))
    ci = [mean - half, mean + half]
    if ci[0] > 0:
        classification = "MULTICHUNK_BETTER"
    elif ci[1] < 0:
        classification = "MULTICHUNK_WORSE"
    else:
        classification = "EFFECT_NOT_ESTABLISHED"
    return {
        "stratum": stratum,
        "per_seed_D_old_minus_multichunk": {str(seed): by_seed[seed]["D_old_minus_multichunk"] for seed in SEEDS},
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "t_critical_90_two_sided_df4": T_CRIT_90_DF4,
        "ci90": ci,
        "classification": classification,
    }


def run_diagnostic(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"validation diagnostic output already exists; resume explicitly: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sealed validation inputs and source identities; no teacher/cache APIs are imported or called.
    base_manifest = json.loads(QUALIFICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    unsigned = dict(base_manifest)
    base_hash = unsigned.pop("manifest_sha256", None)
    if not base_hash or base_hash != quality._canonical_hash(unsigned):
        raise ValueError("qualification manifest self-hash mismatch")
    validation_frozen = json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    val_unsigned = dict(validation_frozen)
    val_sig = val_unsigned.pop("manifest_sha256", None)
    if not val_sig or val_sig != quality._canonical_hash(val_unsigned):
        raise ValueError("frozen 60-document validation manifest self-hash mismatch")
    train_frozen = json.loads(TRAIN_MANIFEST_PATH.read_text(encoding="utf-8"))
    train_unsigned = dict(train_frozen)
    train_sig = train_unsigned.pop("manifest_sha256", None)
    if not train_sig or train_sig != quality._canonical_hash(train_unsigned):
        raise ValueError("historical train manifest self-hash mismatch")

    r1.configure_cpu_runtime()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("distribution evaluation requires CPU FP32 torch 4/1")
    documents_v0, validation_manifest_loaded, tokenizer = expanded._load_real_validation()
    if len(documents_v0) != EXPECTED_DOCS or validation_manifest_loaded["manifest_sha256"] != base_manifest["validation_manifest"]["verification"]["manifest_sha256"]:
        raise ValueError("normal V0 loader did not reproduce the sealed 60-document validation manifest")

    from datasets import DownloadConfig, load_dataset  # noqa: PLC0415

    validation_dataset = load_dataset(
        technical.DATASET_ID,
        technical.DATASET_CONFIG,
        split="validation",
        revision=technical.DATASET_REVISION,
        download_config=DownloadConfig(local_files_only=True),
    )
    expected_doc_keys = {
        (str(row["full_text_sha256"]), str(row["retained_513_token_sha256"]))
        for row in validation_frozen["documents"]
    }
    train_keys = {(str(row["full_text_sha256"]), str(row["retained_513_token_sha256"])) for row in train_frozen["documents"]}
    if train_keys & expected_doc_keys:
        raise ValueError("validation manifest leaks historical train dedupe keys")
    full_validation_docs = _collect_full_validation_documents(
        validation_dataset,
        tokenizer,
        validation_frozen["documents"],
        train_keys,
    )
    for old_doc, full_doc in zip(documents_v0, full_validation_docs):
        if old_doc["tokens"] != full_doc["tokens"][:VAL_CHUNK_TOKENS]:
            raise ValueError("V0 chunk0 does not match the normal evaluator's frozen token prefix")

    val_manifest, strata, val_chunk_manifest_path = _make_validation_strata(full_validation_docs, output_dir)
    val_manifest_self_hash = val_manifest["manifest_sha256"]
    val_payload_hash = val_manifest["chunk_payload"]["sha256"]

    routes = _resolve_ten_checkpoints()
    expected_route_keys = {(row["condition"], int(row["seed"])) for row in routes}
    if expected_route_keys != {(condition, seed) for condition in ("old-513", "multichunk") for seed in SEEDS}:
        raise ValueError("distribution diagnostic checkpoint matrix is not five seeds × both K4 conditions")

    if resume and (output_dir / "coverage_b_progress.json").is_file():
        report = json.loads((output_dir / "coverage_b_progress.json").read_text(encoding="utf-8"))
        if report.get("validation_chunk_manifest_sha256") != val_manifest_self_hash:
            raise ValueError("validation strata manifest changed during diagnostic resume")
    else:
        report = {
            "schema": "omega-train-data-coverage-b-distribution-diagnostic-v1",
            "unit": "OMEGA-TRAIN-DATA-COVERAGE-B-DISTRIBUTION-DIAGNOSTIC",
            "status": "V0_IN_PROGRESS",
            "checkpoint_matrix": routes,
            "validation_chunk_manifest_path": str(val_chunk_manifest_path),
            "validation_chunk_manifest_sha256": val_manifest_self_hash,
            "validation_chunk_payload_sha256": val_payload_hash,
            "validation_manifest_sha256": validation_manifest_loaded["manifest_sha256"],
            "qualification_manifest_sha256": base_hash,
            "evaluation_policy": {
                "evaluator": "canonical r1.evaluate_validation",
                "fp32": True,
                "torch_intraop": 4,
                "torch_interop": 1,
                "test_split_loaded": False,
                "teacher_forward_used": False,
                "hidden_cache_used": False,
                "checkpoint_load_read_only": True,
                "reset_between_chunks": True,
                "within_chunk_state": "W0 to detached W1 state",
                "macro_document_nll": "not computed (optional secondary metric)",
                "v0_guardrail_nats_per_token": GUARDRAIL,
            },
            "strata": {
                name: {"chunks": len(rows), "target_tokens": len(rows) * 512} for name, rows in strata.items()
            },
            "v0_checks": {},
            "later_results": {},
            "vlater_started": False,
            "vall_started": False,
            "scientific_updates": 0,
            "output_dir": str(output_dir),
        }
        quality._write_json(output_dir / "coverage_b_progress.json", report)

    r1_identity = {"path": str(Path(r1.__file__).resolve()), "sha256": _sha256_file(Path(r1.__file__).resolve())}
    if resume:
        existing_done = set(report.get("v0_checks", {}))
    else:
        existing_done = set()
    models: dict[str, torch.nn.Module] = {}
    identities: dict[str, dict[str, Any]] = {}

    # V0 for every checkpoint is always completed before any later/all scoring.
    for route in routes:
        key = f"{route['condition']}_seed_{route['seed']}"
        if key in existing_done:
            continue
        model, identity = _load_checkpoint(
            backend="native",
            seed=int(route["seed"]),
            run_report_path=Path(route["run_report_path"]),
            expected_run_report_sha256=str(route["run_report_sha256"]),
            checkpoint_sha256=str(route["checkpoint_sha256"]),
            ce=ce,
        )
        before_hash = quality._value_hash(model.state_dict())
        metric = r1.evaluate_validation(model, documents_v0)
        after_hash = quality._value_hash(model.state_dict())
        diff = abs(float(metric["nll"]) - float(route["stored_validation_nll"]))
        passed = (
            bool(metric["finite"])
            and int(metric["tokens"]) == EXPECTED_V0_TOKENS
            and diff <= GUARDRAIL
            and before_hash == after_hash
        )
        identity.update(
            {
                "condition": route["condition"],
                "expected_stored_normal_nll": route["stored_validation_nll"],
                "v0_current_nll": float(metric["nll"]),
                "abs_difference": diff,
                "guardrail_nats_per_token": GUARDRAIL,
                "tokens": int(metric["tokens"]),
                "weights_unchanged": before_hash == after_hash,
                "finite": bool(metric["finite"]),
                "pass": passed,
            }
        )
        report["v0_checks"][key] = identity
        quality._write_json(output_dir / "coverage_b_progress.json", report)
        if not passed:
            report["status"] = "HOLD_V0_PORTABILITY"
            report["hold_reason"] = f"V0 does not reproduce its stored NLL within 1e-4 for {key}; later strata not started"
            report["vlater_started"] = False
            report["vall_started"] = False
            report["r1_evaluator_identity"] = r1_identity
            report["report_self_sha256"] = quality._canonical_hash(report)
            quality._write_json(output_dir / "coverage_b_distribution_report.json", report)
            quality._write_json(output_dir / "coverage_b_progress.json", report)
            return report
        models[key] = model
        identities[key] = identity

    if len(report["v0_checks"]) != 10 or any(row.get("pass") is not True for row in report["v0_checks"].values()):
        raise AssertionError("V0 guard did not pass all ten checkpoints; refusing Vlater/Vall")
    report["v0_all_ten_pass"] = True
    report["r1_evaluator_identity"] = r1_identity
    report["status"] = "V0_PASS_LATER_IN_PROGRESS"
    quality._write_json(output_dir / "coverage_b_progress.json", report)

    # Only after the V0 gate passes for all ten checkpoints do we score later/all.
    report["vlater_started"] = True
    report["vall_started"] = True
    quality._write_json(output_dir / "coverage_b_progress.json", report)
    for route in routes:
        key = f"{route['condition']}_seed_{route['seed']}"
        if key in report["later_results"]:
            continue
        model = models.get(key)
        if model is None:
            model, identities[key] = _load_checkpoint(
                backend="native",
                seed=int(route["seed"]),
                run_report_path=Path(route["run_report_path"]),
                expected_run_report_sha256=str(route["run_report_sha256"]),
                checkpoint_sha256=str(route["checkpoint_sha256"]),
                ce=ce,
            )
        before_hash = quality._value_hash(model.state_dict())
        later = r1.evaluate_validation(model, [{"tokens": chunk["tokens"]} for chunk in strata["VLATER_CHUNK_INDEX_GE_1"]])
        all_chunks = r1.evaluate_validation(model, [{"tokens": chunk["tokens"]} for chunk in strata["VALL_ALL_FULL_CHUNKS"]])
        after_hash = quality._value_hash(model.state_dict())
        if before_hash != after_hash:
            raise RuntimeError(f"model weights changed during distribution evaluation for {key}")
        report["later_results"][key] = {
            **identities[key],
            "condition": route["condition"],
            "Vlater": {"nll": float(later["nll"]), "tokens": int(later["tokens"]), "finite": bool(later["finite"])},
            "Vall": {"nll": float(all_chunks["nll"]), "tokens": int(all_chunks["tokens"]), "finite": bool(all_chunks["finite"])},
            "weights_unchanged": True,
        }
        models.pop(key, None)
        quality._write_json(output_dir / "coverage_b_progress.json", report)

    per_seed_strata: dict[str, list[dict[str, Any]]] = {name: [] for name in ("V0_CHUNK0", "VLATER_CHUNK_INDEX_GE_1", "VALL_ALL_FULL_CHUNKS")}
    for seed in SEEDS:
        for stratum in per_seed_strata:
            old_key = f"old-513_seed_{seed}"
            multi_key = f"multichunk_seed_{seed}"
            if stratum == "V0_CHUNK0":
                old_nll = report["v0_checks"][old_key]["v0_current_nll"]
                multi_nll = report["v0_checks"][multi_key]["v0_current_nll"]
            else:
                field = "Vlater" if stratum == "VLATER_CHUNK_INDEX_GE_1" else "Vall"
                old_nll = report["later_results"][old_key][field]["nll"]
                multi_nll = report["later_results"][multi_key][field]["nll"]
            per_seed_strata[stratum].append(
                {"seed": seed, "NLL_old": old_nll, "NLL_multichunk": multi_nll, "D_old_minus_multichunk": old_nll - multi_nll}
            )
    summaries = {name: _paired_summary(rows, name) for name, rows in per_seed_strata.items()}
    report["paired_deltas_by_stratum"] = per_seed_strata
    report["summary_by_stratum"] = summaries
    report["status"] = "DISTRIBUTION_DIAGNOSTIC_COMPLETE"
    report["scientific_updates"] = 0
    report["teacher_forward_used"] = False
    report["hidden_cache_used"] = False
    report["test_split_loaded"] = False
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_dir / "coverage_b_distribution_report.json", report)
    quality._write_json(output_dir / "coverage_b_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-coverage-b-distribution-diagnostic-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.confirm_coverage_b_distribution_diagnostic_go:
        parser.error("OMEGA-TRAIN-DATA-COVERAGE-B requires the explicit MD/290 GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"coverage_b_distribution_diagnostic_{stamp}"
    try:
        report = run_diagnostic(output_dir, resume=args.resume)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "distribution_diagnostic_failure.json",
                {"status": "FAILED_RUNTIME", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "training_updates": 0, "output_dir": str(output_dir)},
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "v0_all_ten_pass": report.get("v0_all_ten_pass"),
                "later_strata_started": report.get("vlater_started") or report.get("vall_started"),
                "summary_by_stratum": report.get("summary_by_stratum"),
                "training_updates": report.get("scientific_updates", 0),
                "report": str(output_dir / "coverage_b_distribution_report.json") if report.get("status") == "DISTRIBUTION_DIAGNOSTIC_COMPLETE" else str(output_dir / "coverage_b_progress.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("status") == "DISTRIBUTION_DIAGNOSTIC_COMPLETE" or report.get("status") == "HOLD_V0_PORTABILITY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
