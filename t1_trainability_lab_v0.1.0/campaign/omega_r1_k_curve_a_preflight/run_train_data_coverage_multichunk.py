"""Train the five authorized K4 multichunk runs after Stage-0 verification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
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
P2R0 = HERE.parent / "omega_native_runtime_p2r0"
P0_DIR = HERE.parent / "omega_native_runtime_p0"
R1_SCOPE = HERE.parents[1] / "campaign" / "omega_core_lm_0_r1_scientific_scoping_a"
TEACHER_CACHE_DIR = HERE.parents[1] / "campaign" / "omega_teacher_hidden_cache_probe"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
STAGE0_DIR = HERE / "results" / "train_data_coverage_stage0_20260927_md288_retry2"
STAGE0_REPORT = STAGE0_DIR / "stage0_report.json"
CHUNK_MANIFEST_PATH = STAGE0_DIR / "multichunk_train_manifest.json"
CHUNK_PAYLOAD_PATH = STAGE0_DIR / "multichunk_train_chunks.jsonl"
MULTICHUNK_CACHE_DIR = STAGE0_DIR / "hidden_cache"
MULTICHUNK_CACHE_MANIFEST = MULTICHUNK_CACHE_DIR / "cache_manifest.json"
MULTICHUNK_CACHE_FILE = MULTICHUNK_CACHE_DIR / "teacher_hidden.multichunk.fp32"
CURVE_ROOT = HERE / "results" / "k_curve_primary_20260927_md286"
BLOCK_A_REPORT_PATH = CURVE_ROOT / "block_A_report.json"
BLOCK_B_REPORT_PATH = CURVE_ROOT / "block_B_report.json"
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
K = 4
TARGET_UPDATES = 2000
CHECKPOINT_INTERVAL = 500
MIN_AVAILABLE_BYTES = 1 * 1024**3
T_CRIT_90_DF4 = 2.131846786326383

for _path in (CAMPAIGN, P2R0, P0_DIR, R1_SCOPE, TEACHER_CACHE_DIR, SCRIPTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_backend_quality_qualification as quality  # noqa: E402
import run_omega_teacher_hidden_cache_probe as hidden_cache_runner  # noqa: E402
import run_omega_native_runtime_r1_bridge as frozen_gates  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_model_factory(ce: Any, technical_model: Any):
    original = ce.fresh_model

    def fresh_model(seed: int, rounds: int, **kwargs: Any) -> torch.nn.Module:
        if rounds in getattr(ce, "KS", ()):
            return original(seed, rounds, **kwargs)
        ce.r1.configure_cpu_runtime()
        ce.set_seed(seed)
        reference = technical_model.OmegaCoreLM0R1Technical(
            vocab_size=kwargs.get("vocab_size", ce.TOKENIZER_VOCAB),
            dimension=kwargs.get("dimension", ce.DIMENSION),
            slots=kwargs.get("slots", ce.SLOTS),
            rounds=rounds,
            variant="shared",
        ).to(dtype=torch.float32)
        try:
            return ce.OmegaCoreLMFast.from_reference(reference).to(dtype=torch.float32)
        finally:
            del reference

    return fresh_model


def _stage0_and_cache(stage0_report_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage0 = json.loads(stage0_report_path.read_text(encoding="utf-8"))
    unsigned = dict(stage0)
    signature = unsigned.pop("report_self_sha256", None)
    if not signature or signature != quality._canonical_hash(unsigned):
        raise ValueError("Stage-0 report self-hash mismatch before multichunk training")
    if stage0.get("status") != "STAGE0_PASS_PREPARED_NO_TRAINING":
        raise PermissionError("multichunk training requires passed Stage 0")
    if int(stage0["ram_decision"]["external_concurrency_for_this_unit"]) != 1:
        raise ValueError("this run is authorized only with Stage-0 external concurrency=1")

    chunk_manifest = json.loads(CHUNK_MANIFEST_PATH.read_text(encoding="utf-8"))
    chunk_unsigned = dict(chunk_manifest)
    chunk_signature = chunk_unsigned.pop("manifest_sha256", None)
    if not chunk_signature or chunk_signature != quality._canonical_hash(chunk_unsigned):
        raise ValueError("multichunk train-manifest self-hash mismatch")
    if _sha256_file(CHUNK_PAYLOAD_PATH) != chunk_manifest["chunk_payload"]["sha256"]:
        raise ValueError("multichunk payload SHA mismatch")

    cache_manifest = json.loads(MULTICHUNK_CACHE_MANIFEST.read_text(encoding="utf-8"))
    if not hidden_cache_runner.verify_self_hash(cache_manifest, "manifest_self_hash"):
        raise ValueError("multichunk hidden-cache manifest self-hash mismatch")
    if cache_manifest.get("status") != "CACHE_SEALED" or cache_manifest.get("access") != "READ_ONLY":
        raise PermissionError("multichunk hidden cache is not sealed read-only")
    if cache_manifest["source"].get("stage0_manifest_sha256") != chunk_signature:
        raise ValueError("multichunk hidden cache belongs to another Stage-0 manifest")
    if cache_manifest["cache_file_sha256"] != _sha256_file(MULTICHUNK_CACHE_FILE):
        raise ValueError("multichunk hidden-cache file SHA mismatch")
    if cache_manifest["shape"] != [2, int(chunk_manifest["chunking"]["chunk_count"]), 256, 768]:
        raise ValueError("multichunk hidden-cache shape does not match Stage 0")
    return stage0, chunk_manifest, cache_manifest


def _build_execution_manifest(
    output_dir: Path,
    stage0: dict[str, Any],
    chunk_manifest: dict[str, Any],
    cache_manifest: dict[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    adapter_dir = output_dir / "input_manifests"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    path = adapter_dir / "multichunk_execution_train_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        saved = existing.pop("manifest_sha256", None)
        if not saved or saved != quality._canonical_hash(existing):
            raise ValueError("existing multichunk execution manifest self-hash mismatch")
        existing["manifest_sha256"] = saved
        return existing, path, _sha256_file(path)

    documents: list[dict[str, Any]] = []
    for line in CHUNK_PAYLOAD_PATH.read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        documents.append(
            {
                "document_index": int(chunk["document_index"]),
                "document_order_index": int(chunk["source_document_order_index"]),
                "source_document_index": int(chunk["source_document_index"]),
                "chunk_depth": int(chunk["chunk_depth"]),
                "token_start": int(chunk["token_start"]),
                "token_end_exclusive": int(chunk["token_end_exclusive"]),
                "row_range": chunk["source_row_range"],
                "header": chunk["header"],
                "full_text_sha256": chunk["full_text_sha256"],
                "retained_513_token_sha256": chunk["chunk_token_sha256"],
                "selected_token_count": 513,
                "token_count": 513,
                "source_token_count": int(chunk["source_token_count"]),
            }
        )
    if len(documents) != int(chunk_manifest["chunking"]["chunk_count"]):
        raise ValueError("execution-manifest chunk/document count mismatch")

    pair_manifest = chunk_manifest["schedule"]["pair_manifest"]
    adapter: dict[str, Any] = {
        "schema": "omega-train-data-coverage-a-execution-train-manifest-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
        "stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
        "stage0_report_sha256": _sha256_file(STAGE0_REPORT),
        "dataset": chunk_manifest["dataset"],
        "split": "train",
        "reconstruction": chunk_manifest["reconstruction"],
        "dedupe_key": chunk_manifest["dedupe_key"],
        "chunking": chunk_manifest["chunking"],
        "chunk_payload": chunk_manifest["chunk_payload"],
        "teacher_hidden_cache_manifest_sha256": cache_manifest["manifest_self_hash"],
        "teacher_hidden_cache_sha256": cache_manifest["cache_file_sha256"],
        "documents": documents,
        "document_count": len(documents),
        "pair_count": int(pair_manifest["pair_count"]),
        "pairs": pair_manifest["pairs"],
        "documents_consumed": int(pair_manifest["documents_consumed"]),
        "cycle_count": int(pair_manifest["cycle_count"]),
        "wraps": int(pair_manifest["wraps"]),
        "pair_order_semantics": "Stage-0 depth-major chunk pool; fixed cyclic batch-8 schedule",
        "manifest_sha256": "",
    }
    adapter["manifest_sha256"] = quality._canonical_hash({key: value for key, value in adapter.items() if key != "manifest_sha256"})
    quality._write_json(path, adapter)
    return adapter, path, _sha256_file(path)


def _load_baseline_reports(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    block_a = json.loads(BLOCK_A_REPORT_PATH.read_text(encoding="utf-8"))
    block_b = json.loads(BLOCK_B_REPORT_PATH.read_text(encoding="utf-8"))
    for label, report in (("A", block_a), ("B", block_b)):
        unsigned = dict(report)
        signature = unsigned.pop("self_sha256", None)
        if not signature or signature != quality._canonical_hash(unsigned) or report.get("status") != "COMPLETE":
            raise ValueError(f"historical old-513 Block {label} report is not complete/self-hashed")
        if report.get("qualification_manifest_sha256") != quality._load_qualification_manifest()["manifest_sha256"]:
            raise ValueError(f"old-513 Block {label} report manifest mismatch")
    return block_a, block_b


def _baseline_route(block_a: dict[str, Any], block_b: dict[str, Any], seed: int) -> dict[str, Any]:
    report = block_a if seed in (20260913, 20260914) else block_b
    key = f"native_K4_seed_{seed}"
    row = report["routes"][key]
    run_report_path = Path(row["run_report_path"])
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    checkpoint0 = Path(run_report["run_dir"]) / "checkpoint_00000.pt"
    sidecar_path = checkpoint0.with_suffix(checkpoint0.suffix + ".identity.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if _sha256_file(checkpoint0) != sidecar["sha256"]:
        raise ValueError(f"old-513 K4 seed{seed} update0 checkpoint sidecar SHA mismatch")
    return {
        "run_report_path": str(run_report_path),
        "run_report_sha256": _sha256_file(run_report_path),
        "run_report": run_report,
        "checkpoint_00000_path": str(checkpoint0),
        "checkpoint_00000_sha256": sidecar["sha256"],
        "checkpoint_00000_identity_sha256": sidecar["identity_sha256"],
    }


def _verify_update0_origin(seed: int, bundle_path: Path, bundle_sha: str, baseline: dict[str, Any]) -> dict[str, Any]:
    if _sha256_file(bundle_path) != bundle_sha:
        raise ValueError(f"new K4 seed{seed} common-init bundle hash mismatch")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if int(bundle.get("seed", -1)) != seed or int(bundle.get("K", -1)) != K:
        raise ValueError(f"new K4 seed{seed} common-init identity mismatch")
    historical_path = Path(baseline["checkpoint_00000_path"]).resolve()
    checkpoint = torch.load(historical_path, map_location="cpu", weights_only=False)
    identity = checkpoint.get("identity", {})
    if (
        int(checkpoint.get("completed_updates", -1)) != 0
        or int(checkpoint.get("next_update", -1)) != 0
        or int(identity.get("seed", -1)) != seed
        or int(identity.get("K", -1)) != K
    ):
        raise ValueError(f"historical K4 seed{seed} origin is not update0")
    bundle_state = bundle["model_state"]
    historical_state = checkpoint["model"]
    exact = (
        list(bundle_state) == list(historical_state)
        and quality._value_hash(bundle_state) == quality._value_hash(historical_state)
        and all(torch.equal(bundle_state[name], historical_state[name]) for name in bundle_state)
    )
    return {
        "seed": seed,
        "K": K,
        "new_common_init_path": str(bundle_path),
        "new_common_init_sha256": bundle_sha,
        "new_model_state_sha256": quality._value_hash(bundle_state),
        "historical_old_513_checkpoint_00000_path": str(historical_path),
        "historical_checkpoint_file_sha256": baseline["checkpoint_00000_sha256"],
        "historical_model_state_sha256": quality._value_hash(historical_state),
        "model_state_bitwise_equal": exact,
        "mismatch_action": None if exact else "HOLD_ROUTE_NO_TRAINING",
    }


def _child_run(args: argparse.Namespace) -> int:
    manifest = quality._load_qualification_manifest()
    r2, p0, ce, _bridge, _modules = quality._load_real_dependencies()
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("multichunk scientific run must remain PyTorch 4/1")
    if int(policy["physical_batch"]) != 8 or int(policy["effective_batch"]) != 8:
        raise RuntimeError("multichunk scientific run must remain batch 8/8")
    if _sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("multichunk run certified DLL SHA mismatch")

    adapter_path = Path(args.execution_manifest).resolve()
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_unsigned = dict(adapter)
    adapter_signature = adapter_unsigned.pop("manifest_sha256", None)
    if not adapter_signature or adapter_signature != quality._canonical_hash(adapter_unsigned):
        raise ValueError("multichunk execution-manifest self-hash mismatch")
    quality.TRAIN_MANIFEST = adapter_path
    chunk_payload_path = Path(adapter["chunk_payload"]["path"])
    if _sha256_file(chunk_payload_path) != adapter["chunk_payload"]["sha256"]:
        raise ValueError("multichunk payload file SHA mismatch")
    cache_manifest_path = Path(args.cache_manifest).resolve()
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if not hidden_cache_runner.verify_self_hash(cache_manifest, "manifest_self_hash"):
        raise ValueError("multichunk hidden-cache manifest self-hash mismatch in worker")
    cache_path = Path(cache_manifest["cache_file"])
    if cache_manifest["cache_file_sha256"] != args.cache_sha256 or cache_manifest["source"]["stage0_manifest_sha256"] != adapter["source_stage0_manifest_sha256"]:
        raise ValueError("multichunk hidden-cache/execution-manifest mismatch")
    if int(psutil.virtual_memory().available) < 1 * 1024**3:
        raise MemoryError("multichunk worker pre-load available RAM <1 GiB")

    quality.TRAIN_MANIFEST = adapter_path
    execution_manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
    if execution_manifest["training_manifest"]["manifest_sha256"] != adapter_signature:
        raise ValueError("run identity manifest does not point to this multichunk execution manifest")
    execution_manifest["hidden_cache"]["manifest"]["path"] = str(cache_manifest_path)
    execution_manifest["hidden_cache"]["manifest"]["sha256"] = cache_manifest["manifest_self_hash"]
    execution_manifest["hidden_cache"]["cache_file"]["path"] = str(cache_path)
    execution_manifest["hidden_cache"]["cache_file"]["sha256"] = args.cache_sha256

    original_loader = r2._load_inputs

    def load_multichunk_inputs(p0_module: Any, _manifest_path: Path, _cache_file: Path):
        if Path(_manifest_path).resolve() != cache_manifest_path:
            raise ValueError("quality runner passed an unexpected hidden-cache manifest path")
        if Path(_cache_file).resolve() != cache_path.resolve():
            raise ValueError("quality runner passed an unexpected hidden-cache data path")
        sealed, resolved_cache = p0_module.hidden.hidden_cache_load_manifest(cache_manifest_path)
        if not hidden_cache_runner.verify_self_hash(sealed, "manifest_self_hash"):
            raise ValueError("hidden-cache self-hash failed in custom multichunk loader")
        resolved_cache = Path(resolved_cache)
        if resolved_cache.resolve() != cache_path.resolve():
            raise ValueError("hidden-cache path differs from the multichunk cache identity")
        payload, _load_seconds, payload_hash = p0_module.hidden.preload_hidden_cache(resolved_cache, sealed)
        if payload_hash != args.cache_sha256:
            raise ValueError("preloaded multichunk teacher cache hash mismatch")
        head_weight, head_bias = p0_module.hidden.load_lm_head(cache_manifest_path.parent, sealed["lm_head"])
        chunks = [json.loads(line) for line in chunk_payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(chunks) != len(adapter["documents"]) or len(chunks) != int(sealed["shape"][1]):
            raise ValueError("multichunk token payload/cache/document counts do not agree")
        if any(len(item.get("tokens", [])) != 513 for item in chunks):
            raise ValueError("multichunk payload contains an incomplete or non-513 token chunk")
        actual_keys = [(item["full_text_sha256"], item["chunk_token_sha256"]) for item in chunks]
        expected_keys = [(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in adapter["documents"]]
        if actual_keys != expected_keys:
            raise ValueError("multichunk payload order differs from adapter manifest")
        return chunks, payload, head_weight, head_bias

    r2._load_inputs = load_multichunk_inputs
    try:
        report = quality._quality_run_child(
            block="COVERAGE_A",
            backend="native",
            rounds=K,
            seed=int(args.seed),
            init_path=Path(args.init_bundle).resolve(),
            init_sha=str(args.init_sha256),
            run_dir=Path(args.run_dir).resolve(),
            manifest=execution_manifest,
            resume_path=Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None,
            run_until_update=TARGET_UPDATES,
            target_updates=TARGET_UPDATES,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            technical_resume_test=False,
        )
    finally:
        r2._load_inputs = original_loader
    print(json.dumps({"status": report["status"], "run_id": report["run_id"], "seed": int(args.seed), "K": K, "updates": report["updates"], "endpoint": report["endpoint_nll_validation"]}, sort_keys=True))
    return 0


def _checkpoint_resume_path(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[-1]), reverse=True)
    for path in candidates:
        meta_path = path.with_suffix(path.suffix + ".identity.json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _sha256_file(path) == meta.get("sha256"):
            return path
    return None


def _build_adapter(stage0: dict[str, Any], cache_manifest: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], Path, str]:
    manifest_path = output_dir / "input_manifests" / "multichunk_execution_train_manifest.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = value.pop("manifest_sha256", None)
        if not signature or signature != quality._canonical_hash(value):
            raise ValueError("existing multichunk execution manifest self-hash mismatch")
        value["manifest_sha256"] = signature
        return value, manifest_path, _sha256_file(manifest_path)
    chunk_manifest = json.loads(Path(stage0["manifest"]["path"]).read_text(encoding="utf-8"))
    chunks_path = Path(stage0["chunk_payload"]["path"])
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = [
        {
            "document_index": int(chunk["document_index"]),
            "document_order_index": int(chunk["source_document_order_index"]),
            "source_document_index": int(chunk["source_document_index"]),
            "chunk_depth": int(chunk["chunk_depth"]),
            "token_start": int(chunk["token_start"]),
            "token_end_exclusive": int(chunk["token_end_exclusive"]),
            "row_range": chunk["source_row_range"],
            "header": chunk["header"],
            "full_text_sha256": chunk["full_text_sha256"],
            "retained_513_token_sha256": chunk["chunk_token_sha256"],
            "token_count": 513,
            "selected_token_count": 513,
            "source_token_count": int(chunk["source_token_count"]),
        }
        for chunk in chunks
    ]
    pair_manifest = chunk_manifest["schedule"]["pair_manifest"]
    adapter: dict[str, Any] = {
        "schema": "omega-train-data-coverage-a-execution-train-manifest-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
        "source_stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
        "stage0_report_sha256": _sha256_file(STAGE0_REPORT),
        "dataset": chunk_manifest["dataset"],
        "split": "train",
        "reconstruction": chunk_manifest["reconstruction"],
        "dedupe_key": chunk_manifest["dedupe_key"],
        "chunking": chunk_manifest["chunking"],
        "chunk_payload": chunk_manifest["chunk_payload"],
        "hidden_cache_manifest_sha256": cache_manifest["manifest_self_hash"],
        "hidden_cache_sha256": cache_manifest["cache_file_sha256"],
        "documents": documents,
        "document_count": len(documents),
        "pair_count": int(pair_manifest["pair_count"]),
        "pairs": pair_manifest["pairs"],
        "documents_consumed": int(pair_manifest["documents_consumed"]),
        "cycle_count": int(pair_manifest["cycle_count"]),
        "wraps": int(pair_manifest["wraps"]),
        "pair_order_semantics": "depth-major pool; fixed cyclic batch-8 pairs; W0/W1 consume the same eight chunks",
        "manifest_sha256": "",
    }
    adapter["manifest_sha256"] = quality._canonical_hash({key: value for key, value in adapter.items() if key != "manifest_sha256"})
    quality._write_json(manifest_path, adapter)
    return adapter, manifest_path, _sha256_file(manifest_path)


def _update0_match(seed: int, init_path: Path, init_sha: str, historical_path: Path) -> dict[str, Any]:
    if _sha256_file(init_path) != init_sha:
        raise ValueError(f"new seed{seed} K4 common-init bundle SHA mismatch")
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    historic_sidecar_path = historical_path.with_suffix(historical_path.suffix + ".identity.json")
    if not historic_sidecar_path.is_file():
        raise FileNotFoundError(f"historical K4 update0 sidecar missing: {historic_sidecar_path}")
    historic_sidecar = json.loads(historic_sidecar_path.read_text(encoding="utf-8"))
    if _sha256_file(historical_path) != historic_sidecar.get("sha256"):
        raise ValueError(f"historical K4 update0 checkpoint file hash mismatch for seed{seed}")
    historic = torch.load(historical_path, map_location="cpu", weights_only=False)
    identity = historic.get("identity", {})
    if (
        int(historic.get("completed_updates", -1)) != 0
        or int(historic.get("next_update", -1)) != 0
        or int(identity.get("seed", -1)) != seed
        or int(identity.get("K", -1)) != K
    ):
        raise ValueError(f"historical K4 seed{seed} checkpoint is not the correct update0 origin")
    common_state = init["model_state"]
    old_state = historic["model"]
    exact = (
        int(init.get("seed", -1)) == seed
        and int(init.get("K", -1)) == K
        and list(common_state) == list(old_state)
        and quality._value_hash(common_state) == quality._value_hash(old_state)
        and all(torch.equal(common_state[name], old_state[name]) for name in common_state)
    )
    return {
        "seed": seed,
        "K": K,
        "old_513_update0_checkpoint_path": str(historical_path),
        "old_513_update0_checkpoint_sha256": historic_sidecar["sha256"],
        "old_513_model_state_sha256": quality._value_hash(old_state),
        "new_multichunk_init_path": str(init_path),
        "new_multichunk_init_sha256": init_sha,
        "new_multichunk_model_state_sha256": quality._value_hash(common_state),
        "model_state_bitwise_identical": exact,
    }


def _historical_k4_route(seed: int, block_a: dict[str, Any], block_b: dict[str, Any]) -> dict[str, Any]:
    report = block_a if seed in (20260913, 20260914) else block_b
    row = report["routes"][f"native_K4_seed_{seed}"]
    run_report = json.loads(Path(row["run_report_path"]).read_text(encoding="utf-8"))
    return run_report


def _load_block_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    block_a = json.loads(BLOCK_A_REPORT_PATH.read_text(encoding="utf-8"))
    block_b = json.loads(BLOCK_B_REPORT_PATH.read_text(encoding="utf-8"))
    for label, report in (("A", block_a), ("B", block_b)):
        unsigned = dict(report)
        signature = unsigned.pop("self_sha256", None)
        if not signature or signature != quality._canonical_hash(unsigned) or report.get("status") != "COMPLETE":
            raise ValueError(f"old-513 block {label} report fails its self-hash/completion check")
        if any(route.get("route_pass") is not True for route in report["routes"].values()):
            raise ValueError(f"old-513 block {label} contains a failed route")
    return block_a, block_b


def _run_unit(output_dir: Path, *, resume: bool = False, prepare_only: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"multichunk training output exists; resume must be explicit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads(STAGE0_REPORT.read_text(encoding="utf-8"))
    stage0_unsigned = dict(stage0)
    stage0_selfhash = stage0_unsigned.pop("report_self_sha256", None)
    if not stage0_selfhash or stage0_selfhash != quality._canonical_hash(stage0_unsigned) or stage0.get("status") != "STAGE0_PASS_PREPARED_NO_TRAINING":
        raise PermissionError("multichunk training requires the verified Stage-0 PASS")
    if int(stage0["ram_decision"]["external_concurrency_for_this_unit"]) != 1:
        raise PermissionError("this run is fixed to Stage-0 external concurrency=1")

    original_manifest = quality._load_qualification_manifest()
    r2, p0, ce, _bridge, _modules = quality._load_real_dependencies()
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1 or int(policy["intraop_threads"]) != 4 or int(policy["interop_threads"]) != 1:
        raise RuntimeError("multichunk route threading changed from the frozen 4/1 policy")
    chunk_manifest = json.loads(CHUNK_MANIFEST_PATH.read_text(encoding="utf-8"))
    cache_manifest = json.loads(MULTICHUNK_CACHE_MANIFEST.read_text(encoding="utf-8"))
    if not hidden_cache_runner.verify_self_hash(cache_manifest, "manifest_self_hash"):
        raise ValueError("multichunk hidden-cache manifest self-hash mismatch before training")
    if (
        cache_manifest.get("status") != "CACHE_SEALED"
        or cache_manifest.get("access") != "READ_ONLY"
        or cache_manifest.get("shape") != [2, 4378, 256, 768]
        or len(cache_manifest.get("entries", {})) != 8756
        or cache_manifest.get("source", {}).get("stage0_manifest_sha256") != chunk_manifest.get("manifest_sha256")
    ):
        raise ValueError("multichunk cache shape/entries/source identity failed pre-training checks")
    if _sha256_file(MULTICHUNK_CACHE_FILE) != cache_manifest["cache_file_sha256"]:
        raise ValueError("multichunk hidden cache file SHA mismatch before training")
    adapter, adapter_path, adapter_sha = _build_adapter(stage0, cache_manifest, output_dir)
    if adapter["document_count"] != 4378 or adapter["pair_count"] != 1000 or len(adapter["pairs"]) != 1000:
        raise ValueError("multichunk execution adapter count/pair contract mismatch")

    block_a, block_b = _load_block_reports()
    baseline_reports = {seed: _historical_k4_route(seed, block_a, block_b) for seed in SEEDS}
    if resume and (output_dir / "multichunk_coverage_progress.json").is_file():
        report = json.loads((output_dir / "multichunk_coverage_progress.json").read_text(encoding="utf-8"))
        if report.get("stage0_manifest_sha256") != chunk_manifest["manifest_sha256"] or report.get("execution_manifest_sha256") != adapter["manifest_sha256"]:
            raise ValueError("multichunk resume manifest identity mismatch")
    else:
        report = {
            "schema": "omega-train-data-coverage-a-report-v1",
            "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
            "status": "IN_PROGRESS",
            "stage0_report_sha256": _sha256_file(STAGE0_REPORT),
            "stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
            "execution_manifest_path": str(adapter_path),
            "execution_manifest_sha256": adapter["manifest_sha256"],
            "hidden_cache_manifest_path": str(MULTICHUNK_CACHE_MANIFEST),
            "hidden_cache_manifest_self_hash": cache_manifest["manifest_self_hash"],
            "hidden_cache_path": str(MULTICHUNK_CACHE_FILE),
            "hidden_cache_sha256": cache_manifest["cache_file_sha256"],
            "qualification_manifest_sha256": original_manifest["manifest_sha256"],
            "old_513_block_A_report_sha256": _sha256_file(BLOCK_A_REPORT_PATH),
            "old_513_block_B_report_sha256": _sha256_file(BLOCK_B_REPORT_PATH),
            "loss_contract": quality.LOSS_CONTRACT,
            "loss_source_sha256": _sha256_file(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
            "native_dll_sha256": quality.BE376_SHA256,
            "K": K,
            "seeds": list(SEEDS),
            "backend": "native",
            "updates_per_run": TARGET_UPDATES,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "batch_physical_effective": [8, 8],
            "bptt_window_tokens": 256,
            "state_reset_after_each_chunk": True,
            "external_concurrency": 1,
            "native_workers": 4,
            "torch_threads": {"intraop": 4, "interop": 1},
            "test_split_loaded": False,
            "scientific_updates_completed": 0,
            "update0_checks": {},
            "runs": {},
            "historical_baseline_old_513": {},
            "output_dir": str(output_dir),
        }
        quality._write_json(output_dir / "multichunk_coverage_progress.json", report)

    adapter_signature = adapter["manifest_sha256"]
    run_manifest = copy.deepcopy(original_manifest)
    run_manifest["training_manifest"] = {
        **run_manifest["training_manifest"],
        "manifest_sha256": adapter_signature,
        "source_sha256": adapter_sha,
        "source_path": str(adapter_path),
    }
    run_manifest["hidden_cache"] = {
        **run_manifest["hidden_cache"],
        "manifest": {"path": str(MULTICHUNK_CACHE_MANIFEST), "sha256": cache_manifest["manifest_self_hash"]},
        "cache_file": {"path": str(MULTICHUNK_CACHE_FILE), "sha256": cache_manifest["cache_file_sha256"]},
        "dataset_revision": chunk_manifest["dataset"]["revision"],
        "teacher_revision": cache_manifest["teacher"]["revision"],
    }
    run_manifest["multichunk_coverage"] = {
        "stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
        "execution_manifest_sha256": adapter_signature,
        "chunk_count": adapter["document_count"],
        "depth_major_order": True,
        "state_reset_after_each_chunk": True,
    }
    run_manifest_path = output_dir / "multichunk_run_manifest.json"
    run_manifest["runtime_manifest_self_sha256"] = quality._canonical_hash(run_manifest)
    quality._write_json(run_manifest_path, run_manifest)

    if prepare_only:
        preflight_report: dict[str, Any] = {
            "schema": "omega-train-data-coverage-a-update0-origin-preflight-v1",
            "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
            "status": "IN_PROGRESS",
            "stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
            "execution_manifest_sha256": adapter_signature,
            "hidden_cache_manifest_self_hash": cache_manifest["manifest_self_hash"],
            "hidden_cache_sha256": cache_manifest["cache_file_sha256"],
            "K": K,
            "seeds": list(SEEDS),
            "historical_update0_checks": {},
            "scientific_updates": 0,
            "output_dir": str(output_dir),
        }
        init_dir = output_dir / "common_initializations"
        init_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            old = baseline_reports[seed]
            old_run_dir = Path(old["run_dir"])
            historical_checkpoint = old_run_dir / "checkpoint_00000.pt"
            init_path = init_dir / f"common_init_K4_seed_{seed}.pt"
            if not init_path.is_file():
                quality._prepare_common_initialization(ce, seed, K, init_dir)
            init_sha = _sha256_file(init_path)
            check = _update0_match(seed, init_path, init_sha, historical_checkpoint)
            preflight_report["historical_update0_checks"][str(seed)] = check
            quality._write_json(output_dir / "multichunk_update0_preflight_progress.json", preflight_report)
            if check["model_state_bitwise_identical"] is not True:
                preflight_report["status"] = "HOLD_UPDATE0_MISMATCH"
                preflight_report["mismatch_seed"] = seed
                preflight_report["report_self_sha256"] = quality._canonical_hash(preflight_report)
                quality._write_json(output_dir / "multichunk_update0_preflight_report.json", preflight_report)
                return preflight_report
        preflight_report["status"] = "UPDATE0_ORIGINS_EXACT_PASS"
        preflight_report["report_self_sha256"] = quality._canonical_hash(preflight_report)
        quality._write_json(output_dir / "multichunk_update0_preflight_report.json", preflight_report)
        return preflight_report

    route_report_paths = []
    for seed in SEEDS:
        key = f"K4_seed_{seed}"
        run_dir = output_dir / "runs" / key
        final_report_path = run_dir / "run_report.json"
        old = baseline_reports[seed]
        old_run_dir = Path(old["run_dir"])
        historical_checkpoint = old_run_dir / "checkpoint_00000.pt"
        init_path = output_dir / "common_initializations" / f"common_init_K4_seed_{seed}.pt"
        if not init_path.is_file():
            init_path.parent.mkdir(parents=True, exist_ok=True)
            quality._prepare_common_initialization(ce, seed, K, init_path.parent)
        init_sha = _sha256_file(init_path)
        identity_check = _update0_match(seed, init_path, init_sha, historical_checkpoint)
        if identity_check["model_state_bitwise_identical"] is not True:
            report["status"] = "HOLD_UPDATE0_MISMATCH"
            report["update0_checks"][str(seed)] = identity_check
            quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
            return report
        report["update0_checks"][str(seed)] = identity_check
        report["historical_baseline_old_513"][str(seed)] = {
            "run_report_path": old["run_dir"] + "\\run_report.json",
            "run_report_sha256": _sha256_file(old_run_dir / "run_report.json"),
            "endpoint_nll_validation": old["endpoint_nll_validation"],
            "validation_curve": old["validation_curve"],
            "checkpoint_00000_sha256": identity_check["old_513_update0_checkpoint_sha256"],
        }

        if final_report_path.is_file():
            run_report = json.loads(final_report_path.read_text(encoding="utf-8"))
            checkpoint2000 = run_dir / "checkpoint_02000.pt"
            route_row = {
                "status": run_report["status"],
                "run_dir": str(run_dir),
                "run_report_path": str(final_report_path),
                "run_report_sha256": _sha256_file(final_report_path),
                "endpoint_nll_validation": run_report["endpoint_nll_validation"],
                "validation_curve": run_report["validation_curve"],
                "checkpoint_02000_sha256": _sha256_file(checkpoint2000),
                "updates": run_report["updates"],
                "route_pass": run_report["status"] == "COMPLETE" and int(run_report["updates"]) == TARGET_UPDATES,
            }
            report["runs"][str(seed)] = route_row
            route_report_paths.append(final_report_path)
            continue

        resume_checkpoint = _checkpoint_resume_path(run_dir) if run_dir.exists() and any(run_dir.iterdir()) else None
        if run_dir.exists() and any(run_dir.iterdir()) and resume_checkpoint is None:
            raise FileNotFoundError(f"multichunk route has data but no valid recovery checkpoint: {run_dir}")
        report["runs"][str(seed)] = {
            "status": "RUNNING",
            "run_dir": str(run_dir),
            "update0_identity_check": identity_check,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
        quality._write_json(output_dir / "multichunk_coverage_progress.json", report)

        worker_stdout = output_dir / "logs" / f"K4_seed_{seed}.stdout.log"
        worker_stderr = output_dir / "logs" / f"K4_seed_{seed}.stderr.log"
        worker_stdout.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--output-dir",
            str(output_dir),
            "--K",
            str(K),
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
            "--init-bundle",
            str(init_path),
            "--init-sha256",
            init_sha,
            "--execution-manifest",
            str(adapter_path),
            "--run-manifest",
            str(run_manifest_path),
            "--cache-manifest",
            str(MULTICHUNK_CACHE_MANIFEST),
            "--cache-sha256",
            str(cache_manifest["cache_file_sha256"]),
        ]
        if resume_checkpoint is not None:
            command.extend(["--resume-checkpoint", str(resume_checkpoint)])
        with worker_stdout.open("ab") as stdout_stream, worker_stderr.open("ab") as stderr_stream:
            process = subprocess.Popen(command, cwd=str(LAB_ROOT.parent), stdout=stdout_stream, stderr=stderr_stream)
            anomaly: str | None = None
            while process.poll() is None:
                if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
                    anomaly = "available_memory_below_1GiB"
                    break
                # Persist the most recent verified checkpoint boundary.
                checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[-1])) if run_dir.exists() else []
                if checkpoints:
                    report["runs"][str(seed)]["latest_checkpoint"] = str(checkpoints[-1])
                    report["runs"][str(seed)]["latest_checkpoint_sha256"] = _sha256_file(checkpoints[-1])
                    report["runs"][str(seed)]["latest_completed_update"] = int(checkpoints[-1].stem.split("_")[-1])
                quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
                time.sleep(10.0)
            if anomaly and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if anomaly:
            report["status"] = "HOLD_MEMORY_GUARD"
            report["technical_anomaly"] = {"seed": seed, "reason": anomaly}
            quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
            return report
        if process.returncode != 0 or not final_report_path.is_file():
            report["status"] = "HOLD_TECHNICAL_ANOMALY"
            report["technical_anomaly"] = {"seed": seed, "return_code": process.returncode, "run_report_exists": final_report_path.is_file()}
            quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
            return report
        run_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        checkpoint2000 = run_dir / "checkpoint_02000.pt"
        if not checkpoint2000.is_file():
            raise FileNotFoundError(f"multichunk K4 seed{seed} completed without update2000 checkpoint")
        route_pass = (
            run_report.get("status") == "COMPLETE"
            and int(run_report.get("updates", -1)) == TARGET_UPDATES
            and int(run_report.get("endpoint_nll_validation", {}).get("update", -1)) == TARGET_UPDATES
            and int(run_report.get("endpoint_nll_validation", {}).get("tokens", -1)) == 30720
            and math.isfinite(float(run_report.get("endpoint_nll_validation", {}).get("nll", float("nan"))))
            and run_report.get("test_split_loaded") is False
            and run_report.get("scientific_quality_training") is True
        )
        report["runs"][str(seed)] = {
            "status": run_report["status"],
            "run_dir": str(run_dir),
            "run_report_path": str(final_report_path),
            "run_report_sha256": _sha256_file(final_report_path),
            "endpoint_nll_validation": run_report["endpoint_nll_validation"],
            "validation_curve": run_report["validation_curve"],
            "checkpoint_02000_sha256": _sha256_file(checkpoint2000),
            "updates": run_report["updates"],
            "route_pass": route_pass,
            "update0_identity_check": identity_check,
        }
        route_report_paths.append(final_report_path)
        report["scientific_updates_completed"] = sum(int(row.get("updates", 0)) for row in report["runs"].values())
        quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
        if not route_pass:
            report["status"] = "HOLD_TECHNICAL_ANOMALY"
            report["technical_anomaly"] = {"seed": seed, "reason": "route endpoint/validation/test/finite contract check failed"}
            quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
            return report

    route_passes = len(report["runs"]) == len(SEEDS) and all(row.get("route_pass") is True for row in report["runs"].values())
    if not route_passes:
        report["status"] = "IN_PROGRESS"
        quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
        return report

    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        old_curve = {int(point["update"]): float(point["nll"]) for point in report["historical_baseline_old_513"][str(seed)]["validation_curve"]}
        new_curve = {int(point["update"]): float(point["nll"]) for point in report["runs"][str(seed)]["validation_curve"]}
        d_coverage = old_curve[TARGET_UPDATES] - new_curve[TARGET_UPDATES]
        drift_old = old_curve[2000] - old_curve[1000]
        drift_new = new_curve[2000] - new_curve[1000]
        paired.append({
            "seed": seed,
            "nll_old_513_2000": old_curve[TARGET_UPDATES],
            "nll_multichunk_2000": new_curve[TARGET_UPDATES],
            "D_coverage_old_minus_multichunk": d_coverage,
            "L_old_2000_minus_1000": drift_old,
            "L_multichunk_2000_minus_1000": drift_new,
            "L_multichunk_minus_old": drift_new - drift_old,
        })
    deltas = [float(row["D_coverage_old_minus_multichunk"]) for row in paired]
    mean = sum(deltas) / len(deltas)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1))
    half_width = T_CRIT_90_DF4 * sample_sd / math.sqrt(len(deltas))
    ci90 = [mean - half_width, mean + half_width]
    if ci90[0] > 0:
        classification = "COVERAGE-IMPROVES"
    elif ci90[1] < 0:
        classification = "COVERAGE-REGRESSES"
    else:
        classification = "NO-CLEAR-EFFECT"
    report["paired_analysis"] = {
        "per_seed": paired,
        "D_coverage_old_minus_multichunk": {
            "values": deltas,
            "n": len(deltas),
            "mean": mean,
            "sample_sd": sample_sd,
            "t_critical_90_two_sided_df4": T_CRIT_90_DF4,
            "ci90": ci90,
        },
        "G1_to_4_mean_for_reference": None,
        "classification": classification,
        "no_posthoc_magnitude_gate": True,
    }
    report["scientific_updates_completed"] = len(SEEDS) * TARGET_UPDATES
    report["status"] = "COMPLETE"
    report["report_self_sha256"] = quality._canonical_hash(report)
    report_path = output_dir / "multichunk_coverage_report.json"
    quality._write_json(report_path, report)
    quality._write_json(output_dir / "multichunk_coverage_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-multichunk-coverage-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="verify all five K4 update0 origins; never train")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--K", type=int, choices=(4,), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-bundle", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--execution-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        required = (args.K, args.seed, args.run_dir, args.init_bundle, args.init_sha256, args.execution_manifest, args.run_manifest, args.cache_manifest, args.cache_sha256)
        if any(value is None for value in required):
            parser.error("multichunk worker arguments incomplete")
        return _child_run(args)
    if not args.confirm_multichunk_coverage_go:
        parser.error("multichunk coverage requires the explicit Stage-0-conditioned GO")
    try:
        report = _run_unit(args.output_dir.resolve(), resume=args.resume, prepare_only=args.prepare_only)
    except Exception as error:
        if args.output_dir.exists():
            quality._write_json(
                args.output_dir / "multichunk_training_failure.json",
                {"status": "FAILED_RUNTIME", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "scientific_training_stopped": True, "output_dir": str(args.output_dir)},
            )
        raise
    print(json.dumps({
        "status": report["status"],
        "routes_completed": len([row for row in report.get("runs", {}).values() if row.get("route_pass") is True]),
        "scientific_updates_completed": report.get("scientific_updates_completed", 0),
        "paired_analysis": report.get("paired_analysis"),
        "report": str(args.output_dir / "multichunk_update0_preflight_report.json") if args.prepare_only else str(args.output_dir / "multichunk_coverage_report.json") if report["status"] == "COMPLETE" else str(args.output_dir / "multichunk_coverage_progress.json"),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
