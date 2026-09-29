"""Run the authorized document-balanced multichunk Coverage-C experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
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
STAGE_C0_ROOT = HERE / "results" / "coverage_c_stage0_document_balanced_20260928_md291"
STAGE_C0_REPORT_PATH = STAGE_C0_ROOT / "coverage_c_stage0_report.json"
DEFAULT_OUTPUT_DIR = HERE / "results" / "coverage_c_document_balanced_20260928_md291"
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
K = 4
UPDATES = 2000
CHECKPOINT_INTERVAL = 500
PAIR_COUNT = 1000
PHYSICAL_BATCH = 8
DOCUMENT_COUNT = 602
MIN_AVAILABLE_BYTES = 1 * 1024**3

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_train_data_coverage_multichunk as coverage_a  # noqa: E402

quality = coverage_a.quality
hidden_cache_runner = coverage_a.hidden_cache_runner


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != quality._canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")
    return str(signature)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    quality._write_json(path, value)


def _coverage_c_stage0() -> tuple[dict[str, Any], str, str]:
    report = json.loads(STAGE_C0_REPORT_PATH.read_text(encoding="utf-8"))
    signature = _check_self_hash(report, "report_self_sha256", "Coverage-C Stage-0 report")
    if report.get("status") != "STAGE0_COMPLETE_NO_TRAINING":
        raise PermissionError("Coverage-C training requires the sealed Stage-0 reachability report")
    if report.get("unit") != "OMEGA-TRAIN-DATA-COVERAGE-C-DOCUMENT-BALANCED-MULTICHUNK":
        raise ValueError("Coverage-C Stage-0 report identity mismatch")
    policy = report.get("policy", {})
    if (
        policy.get("document_schedule") != "historical cyclic order d=((8p+i) mod 602), p=0..999, i=0..7"
        or policy.get("chunk_schedule") != "on zero-based document visit j, choose chunk_index=j mod n_d"
        or int(policy.get("updates", -1)) != UPDATES
        or int(policy.get("pair_positions", -1)) != PAIR_COUNT
        or int(policy.get("physical_batch", -1)) != PHYSICAL_BATCH
    ):
        raise ValueError("Coverage-C Stage-0 does not bind the authorized document-balanced schedule")
    flags = report.get("execution_flags", {})
    if flags.get("optimizer_updates") != 0 or any(
        value is not False for key, value in flags.items() if key != "optimizer_updates"
    ):
        raise ValueError("Coverage-C Stage-0 execution flags show work beyond read-only arithmetic")
    return report, signature, _sha256_file(STAGE_C0_REPORT_PATH)


def _stage_a_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage_a, chunk_manifest, cache_manifest = coverage_a._stage0_and_cache(coverage_a.STAGE0_REPORT)
    return stage_a, chunk_manifest, cache_manifest


def _build_document_balanced_adapter(
    output_dir: Path,
    c_stage0: dict[str, Any],
    c_stage0_self_hash: str,
    c_stage0_file_sha256: str,
    chunk_manifest: dict[str, Any],
    cache_manifest: dict[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    adapter_path = output_dir / "input_manifests" / "coverage_c_document_balanced_execution_manifest.json"
    if adapter_path.is_file():
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter_signature = _check_self_hash(adapter, "manifest_sha256", "Coverage-C execution manifest")
        if (
            adapter_signature != adapter.get("manifest_sha256")
            or adapter.get("coverage_c_stage0_self_sha256") != c_stage0_self_hash
            or adapter.get("coverage_a_stage0_manifest_sha256") != chunk_manifest["manifest_sha256"]
            or adapter.get("hidden_cache_manifest_self_sha256") != cache_manifest["manifest_self_hash"]
        ):
            raise ValueError("existing Coverage-C execution manifest is bound to different inputs")
        return adapter, adapter_path, _sha256_file(adapter_path)

    chunk_payload_path = Path(chunk_manifest["chunk_payload"]["path"])
    chunks = [json.loads(line) for line in chunk_payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(chunks) != int(chunk_manifest["chunking"]["chunk_count"]) or len(chunks) != 4378:
        raise ValueError("Coverage-A full-chunk payload count changed")
    if any(len(row.get("tokens", [])) != 513 for row in chunks):
        raise ValueError("Coverage-A payload contains an incomplete chunk")
    if [int(row["document_index"]) for row in chunks] != list(range(len(chunks))):
        raise ValueError("Coverage-A chunk pool indices are not contiguous and ordered")

    by_document_depth: dict[tuple[int, int], int] = {}
    chunks_by_document: dict[int, list[dict[str, Any]]] = {index: [] for index in range(DOCUMENT_COUNT)}
    for row in chunks:
        document_order = int(row["source_document_order_index"])
        depth = int(row["chunk_depth"])
        key = (document_order, depth)
        if key in by_document_depth:
            raise ValueError(f"duplicate source document/chunk depth in payload: {key}")
        by_document_depth[key] = int(row["document_index"])
        chunks_by_document[document_order].append(row)
    if len(chunks_by_document) != DOCUMENT_COUNT or any(not rows for rows in chunks_by_document.values()):
        raise ValueError("full-chunk pool does not cover each of the 602 source documents")
    for document_order, rows in chunks_by_document.items():
        rows.sort(key=lambda row: int(row["chunk_depth"]))
        if [int(row["chunk_depth"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"full-chunk depth sequence has a gap for document {document_order}")

    stage0_doc_rows = c_stage0["per_document_reachability"]
    if len(stage0_doc_rows) != DOCUMENT_COUNT:
        raise ValueError("Coverage-C Stage-0 per-document reachability list is incomplete")
    if sum(len(rows) for rows in chunks_by_document.values()) != 4378:
        raise AssertionError("chunk grouping lost payload records")

    visits = [0] * DOCUMENT_COUNT
    observed_depths: list[set[int]] = [set() for _ in range(DOCUMENT_COUNT)]
    pair_rows: list[dict[str, Any]] = []
    wraps_total = 0
    for pair_index in range(PAIR_COUNT):
        start = PHYSICAL_BATCH * pair_index
        document_orders = [(start + offset) % DOCUMENT_COUNT for offset in range(PHYSICAL_BATCH)]
        selected_chunks: list[int] = []
        selected_keys: list[list[str]] = []
        selected_depths: list[int] = []
        selected_ordinals: list[int] = []
        for document_order in document_orders:
            source_chunks = chunks_by_document[document_order]
            visit_ordinal = visits[document_order]
            chunk_depth = visit_ordinal % len(source_chunks)
            chunk_pool_index = by_document_depth[(document_order, chunk_depth)]
            chunk_record = chunks[chunk_pool_index]
            selected_chunks.append(chunk_pool_index)
            selected_keys.append([str(chunk_record["full_text_sha256"]), str(chunk_record["chunk_token_sha256"])])
            selected_depths.append(chunk_depth)
            selected_ordinals.append(visit_ordinal)
            observed_depths[document_order].add(chunk_depth)
            visits[document_order] += 1
        wraps = ((start + PHYSICAL_BATCH - 1) // DOCUMENT_COUNT) - (start // DOCUMENT_COUNT)
        wraps_total += wraps
        pair_rows.append(
            {
                "pair": pair_index,
                "start_offset": start % DOCUMENT_COUNT,
                "document_indices": selected_chunks,
                "document_keys": selected_keys,
                "window_0_and_window_1_share_documents": True,
                "wraps": wraps,
                "source_document_order_indices": document_orders,
                "document_visit_ordinals": selected_ordinals,
                "selected_chunk_depths": selected_depths,
            }
        )

    for document_order, row in enumerate(stage0_doc_rows):
        seen = sorted(observed_depths[document_order])
        expected_count = int(row["reached_chunk_count"])
        if (
            int(row["document_order_index"]) != document_order
            or visits[document_order] != int(row["historical_visits"])
            or len(seen) != expected_count
            or seen != list(range(expected_count))
            or expected_count != min(visits[document_order], len(chunks_by_document[document_order]))
        ):
            raise ValueError(f"execution schedule disagrees with sealed Stage-0 rotation for document {document_order}")
    if sum(visits) != PAIR_COUNT * PHYSICAL_BATCH or sum(len(pair["document_indices"]) for pair in pair_rows) != 8000:
        raise AssertionError("Coverage-C pair schedule does not consume the historical 8000 document positions")
    if pair_rows[0]["source_document_order_indices"] != list(range(PHYSICAL_BATCH)) or pair_rows[0]["selected_chunk_depths"] != [0] * PHYSICAL_BATCH:
        raise ValueError("Coverage-C first pair does not preserve the exact old-513 chunk0 prefix")

    adapter_documents = []
    for row in chunks:
        adapter_documents.append(
            {
                "document_index": int(row["document_index"]),
                "document_order_index": int(row["source_document_order_index"]),
                "source_document_index": int(row["source_document_index"]),
                "chunk_depth": int(row["chunk_depth"]),
                "token_start": int(row["token_start"]),
                "token_end_exclusive": int(row["token_end_exclusive"]),
                "row_range": row["source_row_range"],
                "header": row["header"],
                "full_text_sha256": row["full_text_sha256"],
                "retained_513_token_sha256": row["chunk_token_sha256"],
                "selected_token_count": 513,
                "token_count": 513,
                "source_token_count": int(row["source_token_count"]),
            }
        )
    adapter: dict[str, Any] = {
        "schema": "omega-train-data-coverage-c-document-balanced-execution-manifest-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-C-DOCUMENT-BALANCED-MULTICHUNK",
        "mode": "SCIENTIFIC_TRAINING_INPUT",
        "dataset": chunk_manifest["dataset"],
        "split": "train",
        "coverage_a_stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
        "coverage_a_stage0_report_sha256": _sha256_file(coverage_a.STAGE0_REPORT),
        "coverage_c_stage0_report_sha256": c_stage0_file_sha256,
        "coverage_c_stage0_self_sha256": c_stage0_self_hash,
        "hidden_cache_manifest_path": str(coverage_a.MULTICHUNK_CACHE_MANIFEST),
        "hidden_cache_manifest_self_sha256": cache_manifest["manifest_self_hash"],
        "hidden_cache_path": str(coverage_a.MULTICHUNK_CACHE_FILE),
        "hidden_cache_sha256": cache_manifest["cache_file_sha256"],
        "chunk_payload": chunk_manifest["chunk_payload"],
        "document_count": len(adapter_documents),
        "documents": adapter_documents,
        "pair_count": PAIR_COUNT,
        "documents_consumed": PAIR_COUNT * PHYSICAL_BATCH,
        "cycle_count": (PAIR_COUNT * PHYSICAL_BATCH) // DOCUMENT_COUNT,
        "wraps": wraps_total,
        "pairs": pair_rows,
        "schedule": {
            "document_formula": "doc_position=((8*p+i) mod 602), p=0..999, i=0..7",
            "document_order_identical_to_historical_train_manifest": True,
            "document_exposure_counts": {"13": visits.count(13), "14": visits.count(14)},
            "chunk_formula": "on zero-based document visit j, choose chunk_index=j mod n_d",
            "chunk0_is_the_first_visit_for_each_document": True,
            "pair_updates": UPDATES,
            "windows_0_and_1_share_selected_chunks": True,
            "external_concurrency": 1,
        },
        "pair_order_semantics": "historical document cyclic schedule; replace each document visit with its deterministic rotating full chunk",
        "manifest_sha256": "",
    }
    adapter["manifest_sha256"] = quality._canonical_hash({key: value for key, value in adapter.items() if key != "manifest_sha256"})
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(adapter_path, adapter)
    return adapter, adapter_path, _sha256_file(adapter_path)


def _build_run_manifest(
    original: dict[str, Any], adapter: dict[str, Any], adapter_path: Path, adapter_file_sha: str,
    cache_manifest: dict[str, Any], output_dir: Path,
) -> tuple[dict[str, Any], Path, str]:
    path = output_dir / "coverage_c_run_manifest.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        signature = value.pop("runtime_manifest_self_sha256", None)
        if not signature or signature != quality._canonical_hash(value):
            raise ValueError("existing Coverage-C runtime run-manifest self-hash mismatch")
        value["runtime_manifest_self_sha256"] = signature
        if value.get("coverage_c_document_balanced", {}).get("execution_manifest_sha256") != adapter["manifest_sha256"]:
            raise ValueError("existing Coverage-C runtime manifest is bound to another schedule")
        return value, path, _sha256_file(path)

    run_manifest = copy.deepcopy(original)
    run_manifest["training_manifest"] = {
        **run_manifest["training_manifest"],
        "manifest_sha256": adapter["manifest_sha256"],
        "source_sha256": adapter_file_sha,
        "source_path": str(adapter_path),
    }
    run_manifest["hidden_cache"] = {
        **run_manifest["hidden_cache"],
        "manifest": {"path": str(coverage_a.MULTICHUNK_CACHE_MANIFEST), "sha256": cache_manifest["manifest_self_hash"]},
        "cache_file": {"path": str(coverage_a.MULTICHUNK_CACHE_FILE), "sha256": cache_manifest["cache_file_sha256"]},
        "dataset_revision": adapter["dataset"]["revision"],
        "teacher_revision": cache_manifest["teacher"]["revision"],
    }
    run_manifest["coverage_c_document_balanced"] = {
        "stage0_report_path": str(STAGE_C0_REPORT_PATH),
        "stage0_report_self_sha256": adapter["coverage_c_stage0_self_sha256"],
        "coverage_a_stage0_manifest_sha256": adapter["coverage_a_stage0_manifest_sha256"],
        "execution_manifest_sha256": adapter["manifest_sha256"],
        "documents": DOCUMENT_COUNT,
        "full_chunks": 4378,
        "schedule": adapter["schedule"],
    }
    run_manifest["runtime_manifest_self_sha256"] = quality._canonical_hash(run_manifest)
    _write_json(path, run_manifest)
    return run_manifest, path, _sha256_file(path)


def _baseline_routes(block_a: dict[str, Any], block_b: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {seed: coverage_a._historical_k4_route(seed, block_a, block_b) for seed in SEEDS}


def _ensure_init_bundle(ce: Any, technical_model: Any, init_dir: Path, seed: int) -> tuple[Path, str]:
    init_dir.mkdir(parents=True, exist_ok=True)
    path = init_dir / f"common_init_K4_seed_{seed}.pt"
    if path.is_file():
        digest = _sha256_file(path)
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if int(bundle.get("seed", -1)) != seed or int(bundle.get("K", -1)) != K:
            raise ValueError(f"existing common-init identity mismatch: {path}")
        if quality._value_hash(bundle.get("model_state", {})) != bundle.get("model_state_sha256"):
            raise ValueError(f"existing common-init state hash mismatch: {path}")
        return path, digest
    ce.fresh_model = coverage_a._fresh_model_factory(ce, technical_model)
    path, digest = quality._prepare_common_initialization(ce, seed, K, init_dir)
    return path, digest


def _update0_checkpoint_equal(seed: int, new_checkpoint: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    sidecar_path = new_checkpoint.with_suffix(new_checkpoint.suffix + ".identity.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if _sha256_file(new_checkpoint) != sidecar.get("sha256"):
        raise ValueError(f"Coverage-C seed{seed} update0 checkpoint/sidecar SHA mismatch")
    new = torch.load(new_checkpoint, map_location="cpu", weights_only=False)
    old_checkpoint_path = Path(baseline["run_dir"]) / "checkpoint_00000.pt"
    old_sidecar_path = old_checkpoint_path.with_suffix(old_checkpoint_path.suffix + ".identity.json")
    old_sidecar = json.loads(old_sidecar_path.read_text(encoding="utf-8"))
    if _sha256_file(old_checkpoint_path) != old_sidecar.get("sha256"):
        raise ValueError(f"historical seed{seed} update0 checkpoint/sidecar SHA mismatch")
    old = torch.load(old_checkpoint_path, map_location="cpu", weights_only=False)
    identity = old.get("identity", {})
    new_identity = new.get("identity", {})
    if int(old.get("completed_updates", -1)) != 0 or int(old.get("next_update", -1)) != 0 or int(identity.get("seed", -1)) != seed or int(identity.get("K", -1)) != K:
        raise ValueError(f"historical K4 seed{seed} update0 checkpoint identity mismatch")
    new_state = new.get("model", {})
    old_state = old.get("model", {})
    exact = (
        int(new.get("completed_updates", -1)) == 0
        and int(new.get("next_update", -1)) == 0
        and int(new_identity.get("seed", -1)) == seed
        and int(new_identity.get("K", -1)) == K
        and list(new_state) == list(old_state)
        and quality._value_hash(new_state) == quality._value_hash(old_state)
        and all(torch.equal(new_state[name], old_state[name]) for name in new_state)
    )
    return {
        "seed": seed,
        "K": K,
        "coverage_c_update0_checkpoint_path": str(new_checkpoint),
        "coverage_c_update0_checkpoint_sha256": sidecar["sha256"],
        "coverage_c_update0_model_state_sha256": quality._value_hash(new_state),
        "historical_old_513_update0_checkpoint_path": str(old_checkpoint_path),
        "historical_old_513_update0_checkpoint_sha256": old_sidecar["sha256"],
        "historical_old_513_model_state_sha256": quality._value_hash(old_state),
        "model_state_bitwise_equal": exact,
    }


def _worker(args: argparse.Namespace) -> int:
    original = quality._load_qualification_manifest()
    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    r1 = modules[0]
    policy = ce.validate_policy()
    r1.configure_cpu_runtime()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Coverage-C worker requires PyTorch CPU threads 4/1")
    if int(policy["physical_batch"]) != 8 or int(policy["effective_batch"]) != 8:
        raise RuntimeError("Coverage-C worker requires physical/effective batch 8/8")
    if _sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("Coverage-C worker be37623 DLL identity mismatch")

    adapter_path = Path(args.execution_manifest).resolve()
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    adapter_signature = _check_self_hash(adapter, "manifest_sha256", "Coverage-C execution manifest")
    if _sha256_file(adapter_path) != args.execution_manifest_file_sha256 or adapter_signature != args.execution_manifest_sha256:
        raise ValueError("Coverage-C worker execution manifest file/self hash mismatch")
    run_manifest_path = Path(args.run_manifest).resolve()
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_sig = _check_self_hash(run_manifest, "runtime_manifest_self_sha256", "Coverage-C runtime manifest")
    if run_sig != args.run_manifest_self_sha256 or run_manifest["training_manifest"]["manifest_sha256"] != adapter_signature:
        raise ValueError("Coverage-C worker run-manifest identity mismatch")
    if int(adapter["document_count"]) != 4378 or len(adapter["pairs"]) != PAIR_COUNT or int(adapter["documents_consumed"]) != 8000:
        raise ValueError("Coverage-C worker pair/document schedule identity mismatch")

    cache_manifest_path = Path(args.cache_manifest).resolve()
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if not hidden_cache_runner.verify_self_hash(cache_manifest, "manifest_self_hash"):
        raise ValueError("Coverage-C worker hidden-cache manifest self-hash mismatch")
    cache_path = Path(cache_manifest["cache_file"])
    if (
        cache_manifest.get("access") != "READ_ONLY"
        or cache_manifest.get("status") != "CACHE_SEALED"
        or cache_manifest.get("manifest_self_hash") != args.cache_manifest_self_sha256
        or cache_manifest.get("cache_file_sha256") != args.cache_sha256
        or cache_manifest.get("source", {}).get("stage0_manifest_sha256") != adapter["coverage_a_stage0_manifest_sha256"]
    ):
        raise ValueError("Coverage-C worker hidden-cache identity/access mismatch")
    if _sha256_file(cache_path) != args.cache_sha256:
        raise ValueError("Coverage-C worker hidden-cache file SHA mismatch")
    if int(psutil.virtual_memory().available) < MIN_AVAILABLE_BYTES:
        raise MemoryError("Coverage-C worker pre-load available RAM below 1 GiB")

    chunk_payload_path = Path(adapter["chunk_payload"]["path"])
    if _sha256_file(chunk_payload_path) != adapter["chunk_payload"]["sha256"]:
        raise ValueError("Coverage-C chunk payload SHA mismatch")
    original_loader = r2._load_inputs

    def load_document_balanced_inputs(p0_module: Any, requested_manifest: Path, requested_cache: Path):
        if Path(requested_manifest).resolve() != cache_manifest_path or Path(requested_cache).resolve() != cache_path.resolve():
            raise ValueError("quality runner requested an unexpected Coverage-C cache path")
        sealed, resolved_cache = p0_module.hidden.hidden_cache_load_manifest(cache_manifest_path)
        if not hidden_cache_runner.verify_self_hash(sealed, "manifest_self_hash"):
            raise ValueError("Coverage-C hidden-cache manifest failed verification in loader")
        resolved_cache = Path(resolved_cache)
        if resolved_cache.resolve() != cache_path.resolve():
            raise ValueError("Coverage-C hidden-cache resolved path drift")
        payload, _load_seconds, payload_hash = p0_module.hidden.preload_hidden_cache(resolved_cache, sealed)
        if payload_hash != args.cache_sha256:
            raise ValueError("Coverage-C preloaded hidden-cache hash mismatch")
        head_weight, head_bias = p0_module.hidden.load_lm_head(cache_manifest_path.parent, sealed["lm_head"])
        docs = [json.loads(line) for line in chunk_payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(docs) != len(adapter["documents"]) or len(docs) != int(sealed["shape"][1]):
            raise ValueError("Coverage-C training chunks/cache/document counts disagree")
        if any(len(item.get("tokens", [])) != 513 for item in docs):
            raise ValueError("Coverage-C training payload contains an incomplete chunk")
        actual_keys = [(item["full_text_sha256"], item["chunk_token_sha256"]) for item in docs]
        expected_keys = [(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in adapter["documents"]]
        if actual_keys != expected_keys:
            raise ValueError("Coverage-C chunk payload order differs from execution adapter")
        return docs, payload, head_weight, head_bias

    r2._load_inputs = load_document_balanced_inputs
    quality.TRAIN_MANIFEST = adapter_path
    try:
        run_report = quality._quality_run_child(
            block="COVERAGE_C",
            backend="native",
            rounds=K,
            seed=int(args.seed),
            init_path=Path(args.init_bundle).resolve(),
            init_sha=str(args.init_sha256),
            run_dir=Path(args.run_dir).resolve(),
            manifest=run_manifest,
            resume_path=Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None,
            run_until_update=UPDATES,
            target_updates=UPDATES,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            technical_resume_test=False,
        )
    finally:
        r2._load_inputs = original_loader
    print(json.dumps({"status": run_report["status"], "block": run_report["block"], "seed": int(args.seed), "K": K, "updates": run_report["updates"], "endpoint": run_report["endpoint_nll_validation"]}, sort_keys=True))
    return 0


def _run_unit(output_dir: Path, *, resume: bool, prepare_only: bool) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Coverage-C output exists; resume explicitly: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    c_stage0, c_stage0_self_hash, c_stage0_file_sha = _coverage_c_stage0()
    stage_a, chunk_manifest, cache_manifest = _stage_a_inputs()
    if c_stage0["source_identity"]["coverage_a_multichunk_manifest_self_sha256"] != chunk_manifest["manifest_sha256"]:
        raise ValueError("Coverage-C Stage-0 report is not bound to the frozen Coverage-A chunk manifest")
    original_manifest = quality._load_qualification_manifest()
    adapter, adapter_path, adapter_file_sha = _build_document_balanced_adapter(
        output_dir, c_stage0, c_stage0_self_hash, c_stage0_file_sha, chunk_manifest, cache_manifest
    )
    run_manifest, run_manifest_path, run_manifest_file_sha = _build_run_manifest(
        original_manifest, adapter, adapter_path, adapter_file_sha, cache_manifest, output_dir
    )
    block_a, block_b = coverage_a._load_block_reports()
    baseline_routes = _baseline_routes(block_a, block_b)
    r2, p0, ce, _bridge, _modules = quality._load_real_dependencies()
    policy = ce.validate_policy()
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Coverage-C parent requires PyTorch CPU threads 4/1")
    if int(policy["physical_batch"]) != PHYSICAL_BATCH or int(policy["effective_batch"]) != PHYSICAL_BATCH:
        raise RuntimeError("Coverage-C parent requires physical/effective batch 8/8")
    technical_path = coverage_a.SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415
    init_dir = output_dir / "common_initializations"
    init_specs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        init_path, init_sha = _ensure_init_bundle(ce, technical_model, init_dir, seed)
        init_specs[seed] = {"path": str(init_path), "sha256": init_sha}

    progress_path = output_dir / "coverage_c_training_progress.json"
    source_identity = {
        "coverage_c_stage0_self_sha256": c_stage0_self_hash,
        "coverage_c_stage0_file_sha256": c_stage0_file_sha,
        "coverage_a_stage0_manifest_sha256": chunk_manifest["manifest_sha256"],
        "execution_manifest_sha256": adapter["manifest_sha256"],
        "execution_manifest_file_sha256": adapter_file_sha,
        "run_manifest_self_sha256": run_manifest["runtime_manifest_self_sha256"],
        "run_manifest_file_sha256": run_manifest_file_sha,
        "hidden_cache_manifest_self_sha256": cache_manifest["manifest_self_hash"],
        "hidden_cache_sha256": cache_manifest["cache_file_sha256"],
        "qualification_manifest_sha256": original_manifest["manifest_sha256"],
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source_sha256": _sha256_file(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
        "native_dll_sha256": quality.BE376_SHA256,
    }
    if resume and progress_path.is_file():
        report = json.loads(progress_path.read_text(encoding="utf-8"))
        if report.get("source_identity") != source_identity:
            raise ValueError("Coverage-C progress/source identity mismatch on resume")
        report.pop("report_self_sha256", None)
    else:
        report = {
            "schema": "omega-train-data-coverage-c-document-balanced-training-v1",
            "unit": "OMEGA-TRAIN-DATA-COVERAGE-C-DOCUMENT-BALANCED-MULTICHUNK",
            "status": "UPDATE0_PREFLIGHT_IN_PROGRESS",
            "source_identity": source_identity,
            "K": K,
            "seeds": list(SEEDS),
            "updates_per_run": UPDATES,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "backend": "native",
            "dll_sha256": quality.BE376_SHA256,
            "loss_contract": quality.LOSS_CONTRACT,
            "batch_physical_effective": [PHYSICAL_BATCH, PHYSICAL_BATCH],
            "torch_threads": {"intraop": 4, "interop": 1},
            "external_concurrency": 1,
            "native_workers": quality.NATIVE_WORKERS,
            "document_schedule": adapter["schedule"]["document_formula"],
            "chunk_schedule": adapter["schedule"]["chunk_formula"],
            "test_split_loaded": False,
            "teacher_forward_used": False,
            "hidden_cache_used": True,
            "hidden_cache_access": "READ_ONLY",
            "scientific_updates_completed": 0,
            "update0_checks": {},
            "runs": {},
            "initializations": {str(seed): init_specs[seed] for seed in SEEDS},
            "output_dir": str(output_dir),
        }
        _write_json(progress_path, report)

    if prepare_only:
        for seed in SEEDS:
            baseline_run = baseline_routes[seed]
            historical_checkpoint = Path(baseline_run["run_dir"]) / "checkpoint_00000.pt"
            init = init_specs[seed]
            check = coverage_a._update0_match(seed, Path(init["path"]), str(init["sha256"]), historical_checkpoint)
            report["update0_checks"][str(seed)] = check
            _write_json(progress_path, report)
            if check["model_state_bitwise_identical"] is not True:
                report["status"] = "HOLD_UPDATE0_MISMATCH"
                report["mismatch_seed"] = seed
                report["report_self_sha256"] = quality._canonical_hash(report)
                _write_json(output_dir / "coverage_c_update0_preflight_report.json", report)
                return report
        report["status"] = "UPDATE0_ORIGINS_EXACT_PASS_NO_TRAINING"
        report["prepare_only"] = True
        report["scientific_updates_completed"] = 0
        report["report_self_sha256"] = quality._canonical_hash(report)
        _write_json(output_dir / "coverage_c_update0_preflight_report.json", report)
        _write_json(progress_path, report)
        return report

    run_report_paths: list[Path] = []
    for seed in SEEDS:
        key = f"K4_seed_{seed}"
        run_dir = output_dir / "runs" / key
        final_report_path = run_dir / "run_report.json"
        baseline_run = baseline_routes[seed]
        historical_checkpoint = Path(baseline_run["run_dir"]) / "checkpoint_00000.pt"
        init = init_specs[seed]

        # Per-user GO: gate each scientific route against its paired K4 update0 immediately before it starts.
        update0_check = coverage_a._update0_match(seed, Path(init["path"]), str(init["sha256"]), historical_checkpoint)
        report["update0_checks"][str(seed)] = update0_check
        if update0_check["model_state_bitwise_identical"] is not True:
            report["status"] = "HOLD_UPDATE0_MISMATCH"
            report["mismatch_seed"] = seed
            _write_json(progress_path, report)
            return report

        if final_report_path.is_file():
            run_report = json.loads(final_report_path.read_text(encoding="utf-8"))
            if run_report.get("status") != "COMPLETE" or int(run_report.get("updates", -1)) != UPDATES:
                raise ValueError(f"existing Coverage-C route is not complete: {final_report_path}")
            checkpoint0_check = _update0_checkpoint_equal(seed, run_dir / "checkpoint_00000.pt", baseline_run)
            if not checkpoint0_check["model_state_bitwise_equal"]:
                raise ValueError(f"existing Coverage-C route checkpoint0 differs from historical seed{seed}")
            report["runs"][str(seed)] = {
                "status": "COMPLETE",
                "run_dir": str(run_dir),
                "run_report_path": str(final_report_path),
                "run_report_sha256": _sha256_file(final_report_path),
                "endpoint_nll_validation": run_report["endpoint_nll_validation"],
                "updates": run_report["updates"],
                "route_pass": True,
                "update0_checkpoint_check": checkpoint0_check,
                "reused_on_resume": True,
            }
            run_report_paths.append(final_report_path)
            _write_json(progress_path, report)
            continue

        resume_checkpoint = coverage_a._checkpoint_resume_path(run_dir) if run_dir.exists() and any(run_dir.iterdir()) else None
        if run_dir.exists() and any(run_dir.iterdir()) and resume_checkpoint is None:
            raise FileNotFoundError(f"nonempty Coverage-C route has no valid checkpoint: {run_dir}")
        if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
            report["status"] = "HOLD_MEMORY_GUARD"
            report["technical_anomaly"] = {"seed": seed, "reason": "available_memory_below_1GiB_before_worker"}
            _write_json(progress_path, report)
            return report

        worker_stdout = output_dir / "logs" / f"{key}.stdout.log"
        worker_stderr = output_dir / "logs" / f"{key}.stderr.log"
        worker_stdout.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
            "--seed", str(seed), "--run-dir", str(run_dir),
            "--init-bundle", init["path"], "--init-sha256", init["sha256"],
            "--execution-manifest", str(adapter_path), "--execution-manifest-sha256", adapter["manifest_sha256"],
            "--execution-manifest-file-sha256", adapter_file_sha,
            "--run-manifest", str(run_manifest_path), "--run-manifest-self-sha256", run_manifest["runtime_manifest_self_sha256"],
            "--cache-manifest", str(coverage_a.MULTICHUNK_CACHE_MANIFEST),
            "--cache-manifest-self-sha256", cache_manifest["manifest_self_hash"],
            "--cache-sha256", cache_manifest["cache_file_sha256"],
        ]
        if resume_checkpoint is not None:
            command.extend(["--resume-checkpoint", str(resume_checkpoint)])
        report["runs"][str(seed)] = {
            "status": "RUNNING",
            "run_dir": str(run_dir),
            "historical_update0_check": update0_check,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
        report["status"] = "TRAINING_IN_PROGRESS"
        _write_json(progress_path, report)
        with worker_stdout.open("ab") as stdout_stream, worker_stderr.open("ab") as stderr_stream:
            process = subprocess.Popen(command, cwd=str(coverage_a.LAB_ROOT.parent), stdout=stdout_stream, stderr=stderr_stream)
            anomaly: str | None = None
            while process.poll() is None:
                if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
                    anomaly = "available_memory_below_1GiB"
                    break
                checkpoints = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[-1])) if run_dir.exists() else []
                if checkpoints:
                    latest = checkpoints[-1]
                    report["runs"][str(seed)]["latest_checkpoint"] = str(latest)
                    report["runs"][str(seed)]["latest_checkpoint_sha256"] = _sha256_file(latest)
                    report["runs"][str(seed)]["latest_completed_update"] = int(latest.stem.split("_")[-1])
                _write_json(progress_path, report)
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
            _write_json(progress_path, report)
            return report
        if process.returncode != 0 or not final_report_path.is_file():
            report["status"] = "HOLD_TECHNICAL_ANOMALY"
            report["technical_anomaly"] = {"seed": seed, "return_code": process.returncode, "run_report_exists": final_report_path.is_file()}
            _write_json(progress_path, report)
            return report

        run_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        checkpoint0_check = _update0_checkpoint_equal(seed, run_dir / "checkpoint_00000.pt", baseline_run)
        old_curve = {int(row["update"]): float(row["nll"]) for row in baseline_run["validation_curve"]}
        new_curve = {int(row["update"]): float(row["nll"]) for row in run_report["validation_curve"]}
        v0_nll_equal = new_curve.get(0) == old_curve.get(0)
        endpoint = run_report.get("endpoint_nll_validation", {})
        route_pass = (
            run_report.get("status") == "COMPLETE"
            and run_report.get("block") == "COVERAGE_C"
            and run_report.get("backend") == "native"
            and int(run_report.get("K", -1)) == K
            and int(run_report.get("seed", -1)) == seed
            and int(run_report.get("updates", -1)) == UPDATES
            and int(endpoint.get("update", -1)) == UPDATES
            and int(endpoint.get("tokens", -1)) == 30720
            and math.isfinite(float(endpoint.get("nll", float("nan"))))
            and run_report.get("test_split_loaded") is False
            and run_report.get("scientific_quality_training") is True
            and run_report.get("dll_sha256") == quality.BE376_SHA256
            and checkpoint0_check["model_state_bitwise_equal"] is True
            and v0_nll_equal
        )
        report["runs"][str(seed)] = {
            "status": run_report["status"],
            "run_dir": str(run_dir),
            "run_report_path": str(final_report_path),
            "run_report_sha256": _sha256_file(final_report_path),
            "endpoint_nll_validation": endpoint,
            "updates": run_report.get("updates"),
            "validation_curve": run_report.get("validation_curve"),
            "historical_v0_nll": old_curve.get(0),
            "coverage_c_v0_nll": new_curve.get(0),
            "v0_nll_exact_match": v0_nll_equal,
            "update0_checkpoint_check": checkpoint0_check,
            "route_pass": route_pass,
        }
        report["scientific_updates_completed"] = sum(int(row.get("updates", 0)) for row in report["runs"].values())
        _write_json(progress_path, report)
        if not route_pass:
            report["status"] = "HOLD_TECHNICAL_ANOMALY"
            report["technical_anomaly"] = {"seed": seed, "reason": "route identity/endpoint/update0/V0 verification failed"}
            _write_json(progress_path, report)
            return report
        run_report_paths.append(final_report_path)

    if len(report["runs"]) != len(SEEDS) or any(row.get("route_pass") is not True for row in report["runs"].values()):
        report["status"] = "TRAINING_IN_PROGRESS"
        _write_json(progress_path, report)
        return report
    report["scientific_updates_completed"] = len(SEEDS) * UPDATES
    report["status"] = "TRAINING_COMPLETE_AWAITING_C_DISTRIBUTION_EVALUATION"
    report["report_self_sha256"] = quality._canonical_hash(report)
    _write_json(output_dir / "coverage_c_training_report.json", report)
    _write_json(progress_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-coverage-c-document-balanced-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-bundle", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--init-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--execution-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--execution-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--execution-manifest-file-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--run-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-manifest-self-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--cache-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache-manifest-self-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--cache-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        required = (
            args.seed, args.run_dir, args.init_bundle, args.init_sha256,
            args.execution_manifest, args.execution_manifest_sha256, args.execution_manifest_file_sha256,
            args.run_manifest, args.run_manifest_self_sha256, args.cache_manifest,
            args.cache_manifest_self_sha256, args.cache_sha256,
        )
        if any(value is None for value in required):
            parser.error("Coverage-C worker arguments are incomplete")
        return _worker(args)
    if not args.confirm_coverage_c_document_balanced_go:
        parser.error("OMEGA-TRAIN-DATA-COVERAGE-C requires the explicit MD/291 GO")
    output_dir = args.output_dir.resolve()
    try:
        report = _run_unit(output_dir, resume=args.resume, prepare_only=args.prepare_only)
    except Exception as error:
        if output_dir.exists():
            _write_json(
                output_dir / "coverage_c_training_failure.json",
                {
                    "status": "FAILED_RUNTIME",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "scientific_training_stopped": True,
                    "output_dir": str(output_dir),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "update0_checks": {seed: row.get("model_state_bitwise_identical") for seed, row in report.get("update0_checks", {}).items()},
                "routes_completed": len([row for row in report.get("runs", {}).values() if row.get("route_pass") is True]),
                "scientific_updates_completed": report.get("scientific_updates_completed", 0),
                "report": str(output_dir / "coverage_c_update0_preflight_report.json") if args.prepare_only else str(output_dir / "coverage_c_training_report.json") if report.get("status") == "TRAINING_COMPLETE_AWAITING_C_DISTRIBUTION_EVALUATION" else str(output_dir / "coverage_c_training_progress.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("status") in ("UPDATE0_ORIGINS_EXACT_PASS_NO_TRAINING", "TRAINING_COMPLETE_AWAITING_C_DISTRIBUTION_EVALUATION") else 2


if __name__ == "__main__":
    raise SystemExit(main())
