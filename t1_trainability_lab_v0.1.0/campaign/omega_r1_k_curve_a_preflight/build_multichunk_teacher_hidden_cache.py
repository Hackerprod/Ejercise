"""Build the full-precision multichunk teacher hidden-cache for the authorized unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
TEACHER_CACHE_DIR = HERE.parents[1] / "campaign" / "omega_teacher_hidden_cache_probe"
TEACHER_LOGIT_DIR = HERE.parents[1] / "campaign" / "omega_teacher_logit_cache"
P2R0 = HERE.parents[1] / "campaign" / "omega_native_runtime_p2r0"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
STAGE0_DIR = HERE / "results" / "train_data_coverage_stage0_20260927_md288_retry2"
OLD_CACHE_PATH = Path(r"C:\omega_cache\teacher_hidden.fp32")
BATCH_SIZE = 8
WINDOWS = (0, 1)
CHUNK_TOKENS = 513
CHUNK_STRIDE = 512

for _path in (CAMPAIGN, TEACHER_CACHE_DIR, TEACHER_LOGIT_DIR, P2R0, SCRIPTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_backend_quality_qualification as quality  # noqa: E402
import run_omega_teacher_hidden_cache_probe as hidden_cache_runner  # noqa: E402
import run_omega_teacher_logit_cache as teacher_base  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _cache_entry(
    *,
    chunk: dict[str, Any],
    window: int,
    chunk_index: int,
    chunk_count: int,
    hidden_sha256: str,
    teacher_parameter_sha256: str,
    tokenizer_sha256: str,
    prefix_reused: bool,
) -> dict[str, Any]:
    fields = hidden_cache_runner.hidden_key_fields(
        chunk,
        window,
        teacher_parameter_sha256=teacher_parameter_sha256,
        tokenizer_hash=tokenizer_sha256,
    )
    fields["chunk_depth"] = int(chunk["chunk_depth"])
    fields["token_start"] = int(chunk["token_start"])
    offset = ((window * chunk_count + chunk_index) * 256 * hidden_cache_runner.HIDDEN_SIZE * 4)
    return {
        "document_index": chunk_index,
        "source_document_index": int(chunk["source_document_index"]),
        "source_document_order_index": int(chunk["source_document_order_index"]),
        "chunk_depth": int(chunk["chunk_depth"]),
        "chunk_token_start": int(chunk["token_start"]),
        "window": window,
        "offset_bytes": offset,
        "shape": [256, hidden_cache_runner.HIDDEN_SIZE],
        "sha256": hidden_sha256,
        "prefix_reused_from_historical_cache": prefix_reused,
        "cache_key": {"fields": fields, "digest": teacher_base.canonical_hash(fields)},
    }


def _load_chunks(payload_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_cache(stage0_report_path: Path, output_dir: Path, *, resume: bool = False, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    stage0_report = json.loads(stage0_report_path.read_text(encoding="utf-8"))
    stage0_unsigned = dict(stage0_report)
    stage0_signature = stage0_unsigned.pop("report_self_sha256", None)
    if not stage0_signature or stage0_signature != quality._canonical_hash(stage0_unsigned):
        raise ValueError("Stage-0 report self-hash mismatch")
    if stage0_report.get("status") != "STAGE0_PASS_PREPARED_NO_TRAINING":
        raise PermissionError("teacher hidden-cache build requires a passed Stage-0 report")
    stage0_dir = Path(stage0_report["output_dir"])
    chunk_manifest_path = Path(stage0_report["manifest"]["path"])
    chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    manifest_unsigned = dict(chunk_manifest)
    manifest_signature = manifest_unsigned.pop("manifest_sha256", None)
    if not manifest_signature or manifest_signature != quality._canonical_hash(manifest_unsigned):
        raise ValueError("multichunk train manifest self-hash mismatch")
    payload_path = Path(stage0_report["chunk_payload"]["path"])
    if _sha256_file(payload_path) != stage0_report["chunk_payload"]["sha256"]:
        raise ValueError("multichunk payload SHA mismatch")
    chunks = _load_chunks(payload_path)
    chunk_count = int(chunk_manifest["chunking"]["chunk_count"])
    if len(chunks) != chunk_count or chunk_count != 4378:
        raise ValueError(f"multichunk payload count mismatch: got {len(chunks)}, expected {chunk_count}")

    old_cache_manifest_path = TEACHER_CACHE_DIR / "results" / "cache_manifest.json"
    old_cache_manifest, old_cache_resolved = hidden_cache_runner.hidden_cache_load_manifest(old_cache_manifest_path)
    old_cache_resolved = Path(old_cache_resolved)
    if old_cache_resolved.resolve() != OLD_CACHE_PATH.resolve():
        raise ValueError("old frozen cache path differs from the sealed manifest")
    if _sha256_file(old_cache_resolved) != old_cache_manifest["cache_file_sha256"]:
        raise ValueError("old frozen hidden-cache hash mismatch")
    sealed_quality_manifest = quality._load_qualification_manifest()
    if _sha256_file(old_cache_manifest_path) != sealed_quality_manifest["hidden_cache"]["manifest"]["sha256"]:
        raise ValueError("old hidden-cache identity differs from the quality qualification manifest")

    planned_shape = [2, chunk_count, 256, hidden_cache_runner.HIDDEN_SIZE]
    expected_bytes = math.prod(planned_shape) * 4
    if expected_bytes != stage0_report["hidden_cache_ram_assessment"]["planned_cache_bytes"]:
        raise ValueError("new hidden-cache size differs from the Stage-0 memory calculation")
    if int(stage0_report["ram_decision"]["external_concurrency_for_this_unit"]) != 1:
        raise ValueError("Stage-0 RAM plan did not select the one-process topology required for cache build")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "teacher_hidden.multichunk.fp32"
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache_manifest_path = output_dir / "cache_manifest.json"
    progress_path = output_dir / "cache_build_progress.json"
    if cache_manifest_path.is_file() and cache_path.is_file():
        final_manifest, final_path = hidden_cache_runner.hidden_cache_load_manifest(cache_manifest_path)
        if _sha256_file(Path(final_path)) != final_manifest["cache_file_sha256"]:
            raise ValueError("existing multichunk cache is not sealed correctly")
        return {
            "status": "CACHE_SEALED",
            "cache_path": str(final_path),
            "cache_sha256": final_manifest["cache_file_sha256"],
            "manifest_path": str(cache_manifest_path),
            "manifest_self_hash": final_manifest["manifest_self_hash"],
            "chunks": chunk_count,
            "already_present": True,
        }
    final_payload_ready = False
    if cache_path.exists() and not cache_manifest_path.exists():
        if not resume or not progress_path.is_file():
            raise FileExistsError("unsealed final cache file exists; use --resume to verify and seal it")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("stage0_manifest_sha256") != manifest_signature
            or int(progress.get("next_chunk_index", -1)) != chunk_count
            or cache_path.stat().st_size != expected_bytes
        ):
            raise ValueError("unsealed final cache file is not a complete resumable payload")
        completed_chunks = chunk_count
        final_payload_ready = True
    elif temporary_cache.exists() and not resume:
        raise FileExistsError("partial hidden-cache payload exists; use --resume or inspect it")

    if final_payload_ready:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("stage0_manifest_sha256") != manifest_signature:
            raise ValueError("unsealed cache belongs to a different Stage-0 manifest")
        mmap_path = cache_path
    elif resume:
        if not progress_path.is_file() or not temporary_cache.is_file():
            raise FileNotFoundError("hidden-cache resume needs both progress metadata and the partial FP32 payload")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("stage0_manifest_sha256") != manifest_signature or progress.get("planned_cache_bytes") != expected_bytes:
            raise ValueError("partial cache belongs to a different Stage-0 manifest/shape")
        completed_chunks = int(progress["next_chunk_index"])
        if temporary_cache.stat().st_size != expected_bytes:
            raise ValueError("partial multichunk cache file has wrong size")
        mmap_path = temporary_cache
    else:
        completed_chunks = 0
        progress = {
            "schema": "omega-train-data-coverage-a-cache-build-progress-v1",
            "status": "IN_PROGRESS",
            "stage0_manifest_sha256": manifest_signature,
            "stage0_report_sha256": _sha256_file(stage0_report_path),
            "chunk_payload_sha256": _sha256_file(payload_path),
            "planned_cache_bytes": expected_bytes,
            "shape": planned_shape,
            "next_chunk_index": 0,
            "scientific_updates": 0,
        }
        quality._write_json(progress_path, progress)
        mmap_path = temporary_cache

    old_shape = tuple(int(value) for value in old_cache_manifest["shape"])
    if old_shape != (2, 602, 256, hidden_cache_runner.HIDDEN_SIZE):
        raise ValueError(f"historical cache shape mismatch: {old_shape}")
    old_mmap = np.memmap(old_cache_resolved, mode="r", dtype=np.float32, shape=old_shape, order="C")
    new_mmap = np.memmap(mmap_path, mode="r+" if resume or final_payload_ready else "w+", dtype=np.float32, shape=tuple(planned_shape), order="C")
    try:
        if not resume and not final_payload_ready:
            new_mmap[:, :602, :, :] = old_mmap
            new_mmap.flush()
            completed_chunks = 602
            progress["historical_prefix_chunks_copied"] = 602
            progress["next_chunk_index"] = completed_chunks
            progress["prefix_hidden_is_bitwise_reused"] = True
            quality._write_json(progress_path, progress)

        # Load the exact pinned teacher from the user-provided HuggingFace cache
        # only; local_files_only prohibits hub downloads or fallback revisions.
        from transformers import AutoModelForCausalLM  # noqa: PLC0415

        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_base.MODEL_ID,
            revision=teacher_base.MODEL_REVISION,
            local_files_only=True,
            cache_dir=r"C:\omega_cache\huggingface\hub",
        )
        teacher.float().eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_hash = teacher_base.parameter_hash(teacher)
        if teacher_hash != old_cache_manifest["teacher"]["parameter_sha256"]:
            raise ValueError("locally loaded teacher weights differ from the sealed historical cache teacher")
        tokenizer_hash = str(old_cache_manifest["tokenizer"]["sha256"])
        if teacher_base.MODEL_REVISION != old_cache_manifest["tokenizer"]["revision"]:
            raise ValueError("tokenizer revision differs from the historical hidden-cache tokenizer")

        # The first 602 chunk0 hidden states are copied from the historical cache
        # rather than recomputed, preserving the old recipe's prefix targets.
        entries: dict[str, Any] = {}
        for chunk_index in range(completed_chunks):
            chunk = chunks[chunk_index]
            for window in WINDOWS:
                if chunk_index < 602:
                    old_entry = old_cache_manifest["entries"][f"{window}:{chunk_index}"]
                    value_sha = old_entry["sha256"]
                    observed_sha = _payload_hash(np.asarray(new_mmap[window, chunk_index]))
                    if observed_sha != value_sha:
                        raise ValueError(f"copied historical hidden prefix differs at window={window}, chunk={chunk_index}")
                else:
                    value_sha = _payload_hash(np.asarray(new_mmap[window, chunk_index]))
                entries[f"{window}:{chunk_index}"] = _cache_entry(
                    chunk=chunk,
                    window=window,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    hidden_sha256=value_sha,
                    teacher_parameter_sha256=teacher_hash,
                    tokenizer_sha256=tokenizer_hash,
                    prefix_reused=chunk_index < 602,
                )

        build_started = time.perf_counter()
        for batch_start in range(completed_chunks, chunk_count, batch_size):
            batch_end = min(batch_start + batch_size, chunk_count)
            batch = chunks[batch_start:batch_end]
            source = torch.tensor([row["tokens"] for row in batch], dtype=torch.long)
            for window in WINDOWS:
                hidden_states = hidden_cache_runner.hidden_window_states(teacher, source, window)
                if tuple(hidden_states.shape) != (len(batch), 256, hidden_cache_runner.HIDDEN_SIZE):
                    raise ValueError(f"hidden batch shape mismatch for window={window}, batch={batch_start}:{batch_end}")
                for batch_offset, chunk in enumerate(batch):
                    chunk_index = batch_start + batch_offset
                    value = hidden_states[batch_offset].detach().cpu().contiguous().float()
                    new_mmap[window, chunk_index] = value.numpy()
                    fields_entry = _cache_entry(
                        chunk=chunk,
                        window=window,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        hidden_sha256=hidden_cache_runner.tensor_hash(value),
                        teacher_parameter_sha256=teacher_hash,
                        tokenizer_sha256=tokenizer_hash,
                        prefix_reused=False,
                    )
                    entries[f"{window}:{chunk_index}"] = fields_entry
            new_mmap.flush()
            completed_chunks = batch_end
            progress.update(
                {
                    "status": "IN_PROGRESS",
                    "next_chunk_index": completed_chunks,
                    "completed_chunk_count": completed_chunks,
                    "new_chunk_count_computed": completed_chunks - 602,
                    "last_batch_range": [batch_start, batch_end],
                    "elapsed_seconds": time.perf_counter() - build_started,
                    "scientific_updates": 0,
                }
            )
            quality._write_json(progress_path, progress)
        new_mmap.flush()
    finally:
        old_mmap._mmap.close()
        new_mmap._mmap.close()

    if completed_chunks != chunk_count or len(entries) != len(WINDOWS) * chunk_count:
        raise AssertionError("new hidden cache coverage is incomplete")
    payload_path_to_seal = cache_path if final_payload_ready else temporary_cache
    if payload_path_to_seal.stat().st_size != expected_bytes:
        raise ValueError("new raw FP32 cache byte size differs from planned shape")
    if not final_payload_ready:
        if cache_path.exists():
            raise FileExistsError(cache_path)
        os.replace(temporary_cache, cache_path)
    cache_file_sha = _sha256_file(cache_path)

    lm_head_meta = dict(old_cache_manifest["lm_head"])
    for filename_key, sha_key in (("weight_file", "weight_sha256"), ("bias_file", "bias_sha256")):
        filename = lm_head_meta.get(filename_key)
        if not filename:
            continue
        source_path = old_cache_manifest_path.parent / str(filename)
        destination_path = output_dir / str(filename)
        if not destination_path.is_file():
            shutil.copy2(source_path, destination_path)
        if _sha256_file(destination_path) != lm_head_meta[sha_key]:
            raise ValueError(f"copied teacher LM-head artifact hash mismatch: {filename}")

    teacher_metadata = dict(old_cache_manifest["teacher"])
    tokenizer_metadata = dict(old_cache_manifest["tokenizer"])
    result = {
        "schema": "omega-teacher-hidden-cache-multichunk-v1",
        "campaign_id": "OMEGA-TRAIN-DATA-COVERAGE-A",
        "status": "CACHE_SEALED",
        "access": "READ_ONLY",
        "cache_file": cache_path.resolve().as_posix(),
        "shape": planned_shape,
        "order": ["window", "chunk", "token", "hidden"],
        "representation": "raw_fp32_hidden",
        "dtype": "float32",
        "entry_bytes": int(math.prod([256, hidden_cache_runner.HIDDEN_SIZE]) * 4),
        "logical_bytes": expected_bytes,
        "cache_file_sha256": cache_file_sha,
        "source": {
            "stage0_manifest_sha256": manifest_signature,
            "stage0_report_self_sha256": stage0_report["report_self_sha256"],
            "chunk_payload_sha256": _sha256_file(payload_path),
            "dataset_id": old_cache_manifest["source"]["dataset_id"],
            "dataset_revision": old_cache_manifest["source"]["dataset_revision"],
            "retained_tokens": CHUNK_TOKENS,
            "chunk_count": chunk_count,
            "stride_tokens": CHUNK_STRIDE,
            "first_602_chunk0_hidden_reused_bitwise": True,
            "historical_cache_sha256": old_cache_manifest["cache_file_sha256"],
        },
        "teacher": teacher_metadata,
        "tokenizer": tokenizer_metadata,
        "lm_head": lm_head_meta,
        "build_provenance": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "external_concurrency": 1,
            "batch_size": batch_size,
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "fp16_or_compression": False,
            "teacher_model_forward_for_chunks": chunk_count - 602,
            "historical_chunk0_rows_recomputed": 0,
            "scientific_updates": 0,
        },
        "entries": entries,
    }
    manifest_path = output_dir / "cache_manifest.json"
    hidden_cache_runner.write_self_hashed(manifest_path, result, field="manifest_self_hash")
    sealed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not hidden_cache_runner.verify_self_hash(sealed_manifest, "manifest_self_hash"):
        raise ValueError("new multichunk hidden-cache manifest self-hash failed")
    if _sha256_file(cache_path) != sealed_manifest["cache_file_sha256"]:
        raise ValueError("new multichunk hidden-cache final file SHA mismatch")
    build_report = {
        "schema": "omega-train-data-coverage-a-hidden-cache-build-v1",
        "unit": "OMEGA-TRAIN-DATA-COVERAGE-A",
        "status": "CACHE_SEALED",
        "stage0_manifest_sha256": manifest_signature,
        "cache_manifest_path": str(manifest_path),
        "cache_manifest_self_hash": sealed_manifest["manifest_self_hash"],
        "cache_file_path": str(cache_path),
        "cache_file_sha256": cache_file_sha,
        "shape": planned_shape,
        "logical_bytes": expected_bytes,
        "chunk_count": chunk_count,
        "historical_chunk0_reused": 602,
        "new_chunk_teacher_forwards": chunk_count - 602,
        "teacher_parameter_sha256": teacher_hash,
        "tokenizer_sha256": tokenizer_hash,
        "scientific_updates": 0,
    }
    report_path = output_dir / "cache_build_report.json"
    build_report["report_self_sha256"] = quality._canonical_hash(build_report)
    quality._write_json(report_path, build_report)
    progress.update({"status": "CACHE_SEALED", "next_chunk_index": chunk_count, "cache_file_sha256": cache_file_sha, "cache_manifest_self_hash": sealed_manifest["manifest_self_hash"]})
    quality._write_json(progress_path, progress)
    return build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-multichunk-cache-build-go", action="store_true")
    parser.add_argument("--stage0-report", type=Path, default=STAGE0_DIR / "stage0_report.json")
    parser.add_argument("--output-dir", type=Path, default=STAGE0_DIR / "hidden_cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if not args.confirm_multichunk_cache_build_go:
        parser.error("multichunk teacher hidden-cache build requires explicit coverage-unit GO")
    try:
        result = build_cache(args.stage0_report.resolve(), args.output_dir.resolve(), resume=args.resume, batch_size=args.batch_size)
    except Exception as error:
        if args.output_dir.exists():
            quality._write_json(
                args.output_dir / "cache_build_failure.json",
                {"status": "FAILED_RUNTIME", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "scientific_updates": 0, "output_dir": str(args.output_dir)},
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "CACHE_SEALED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
