"""OMEGA R1 Step 3 cache extension.

This file is isolated from Step 1/Step 2 artifacts and never invokes SCOPE-A.
The parent process performs metadata-only traversal simulation. Each measured
route runs in its own fresh child process and constructs its own teacher/model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable

import psutil
import torch
from torch import Tensor, nn


UNIT_DIR = Path(__file__).resolve().parent
LAB_ROOT = UNIT_DIR.parents[1]
SCRIPTS_DIR = LAB_ROOT / "scripts"
SCOPE_DIR = LAB_ROOT / "campaign" / "omega_core_lm_0_r1_scientific_scoping_a"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(UNIT_DIR))
sys.path.insert(0, str(SCOPE_DIR))

from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_VOCAB,
    OmegaCoreLM0R1Technical,
    distillation_loss,
    is_level_one_header,
)
from run_scientific_scoping_a import (  # noqa: E402
    build_pair_manifest,
    is_eligible_document,
)
from omega_fast_candidate import (  # noqa: E402
    OmegaCoreLMFast,
    distillation_loss_lse,
    teacher_targets,
)


THREADS = int(os.environ.get("OMEGA_STEP3_TORCH_THREADS", "4"))
INTEROP_THREADS = int(os.environ.get("OMEGA_STEP3_TORCH_INTEROP_THREADS", "1"))
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(INTEROP_THREADS)

DEVICE = torch.device("cpu")
DTYPE = torch.float32
PAIR_COUNT = 1000
PHYSICAL_BATCH = 8
TOKENS_PER_WINDOW = 256
RETAINED_TOKENS = 513
DIMENSION = 128
SLOTS = 8
TEMPERATURE = 2.0
BASE_LR = 3e-4
CLIP_NORM = 1.0
VARIANTS = {"shared_K1": {"rounds": 1}, "shared_K4": {"rounds": 4}}
ROUTES = ("F", "C+L")
MAX_CACHE_BYTES = 4 * 1024**3
MEMORY_RESERVE_BYTES = 1 * 1024**3
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"
FORMULA_TEXT = "T_SCOPE-A = T_preparación + Σ(rutas/variantes)(N_aciertos×t_acierto + N_fallos×t_fallo) + T_validación/checkpoints/E-S"


class MemoryReserveError(RuntimeError):
    def __init__(self, stage: str, available_bytes: int) -> None:
        super().__init__(f"memory reserve unavailable at {stage}: {available_bytes} bytes available")
        self.stage = stage
        self.available_bytes = available_bytes


class CacheCapacityError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def memory_snapshot() -> dict[str, int | None]:
    info = psutil.Process().memory_info()
    peak_wset = getattr(info, "peak_wset", None)
    return {
        "rss_bytes": int(info.rss),
        "available_system_bytes": int(psutil.virtual_memory().available),
        "os_peak_working_set_bytes": int(peak_wset) if peak_wset is not None else None,
    }


def reserve_satisfied(available_bytes: int, reserve_bytes: int = MEMORY_RESERVE_BYTES) -> bool:
    return int(available_bytes) >= int(reserve_bytes)


def reserve_guard(stage: str) -> dict[str, int | None]:
    snapshot = memory_snapshot()
    available = int(snapshot["available_system_bytes"])
    if not reserve_satisfied(available):
        raise MemoryReserveError(stage, available)
    return snapshot


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def check_disk_before_write(path: Path, expected_bytes: int, stage: str) -> int:
    free = disk_free_bytes(path)
    if free < expected_bytes:
        raise CacheCapacityError(stage, f"disk free {free} below required {expected_bytes}")
    return free


def _iter_reconstructed_documents(dataset: Any):
    current_start: int | None = None
    current_lines: list[str] = []
    current_header = ""
    for row_index, row in enumerate(dataset):
        text = str(row["text"])
        if is_level_one_header(text):
            if current_start is not None:
                yield {"row_range": [current_start, row_index], "header": current_header, "text": "\n".join(current_lines)}
            current_start = row_index
            current_header = text.strip()
            current_lines = [text]
        elif current_start is not None:
            current_lines.append(text)
    if current_start is not None:
        yield {"row_range": [current_start, len(dataset)], "header": current_header, "text": "\n".join(current_lines)}


def load_train_metadata(needed_positions: set[int] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct and deduplicate train documents without retaining full corpus tokens."""
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True, local_files_only=True)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise RuntimeError(f"tokenizer vocab mismatch: expected {TOKENIZER_VOCAB}, got {len(tokenizer)}")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION, download_config=DownloadConfig(local_files_only=True))
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    scanned = 0
    for document_index, document in enumerate(_iter_reconstructed_documents(dataset)):
        scanned += 1
        candidate = is_eligible_document({**document, "document_index": document_index}, tokenizer)
        if candidate is None:
            continue
        key = (candidate["full_text_sha256"], candidate["retained_513_token_sha256"])
        if key in seen:
            continue
        seen.add(key)
        eligible_position = len(result)
        metadata = {key: value for key, value in candidate.items() if key != "tokens"}
        metadata["eligible_position"] = eligible_position
        if needed_positions is not None and eligible_position in needed_positions:
            metadata["tokens"] = candidate["tokens"]
        result.append(metadata)
    del dataset, tokenizer
    return result, {"documents_reconstructed": scanned, "eligible_document_count": len(result), "deduplication": "first occurrence by full_text_sha256 + retained_513_token_sha256"}


def document_key_fields(document: dict[str, Any], window: int) -> dict[str, Any]:
    context_range = [0, 256] if window == 0 else [0, 512]
    return {
        "document_identity": {"document_index": int(document["document_index"]), "full_text_sha256": document["full_text_sha256"]},
        "retained_token_hash": document["retained_513_token_sha256"],
        "teacher_model": MODEL_ID,
        "teacher_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "window": int(window),
        "teacher_context_range": context_range,
        "representation": "fp32_teacher_probs_plus_negative_entropy",
        "dtype": "float32",
        "temperature": TEMPERATURE,
    }


def canonical_cache_key(document: dict[str, Any], window: int) -> dict[str, str | dict[str, Any]]:
    fields = document_key_fields(document, window)
    return {"fields": fields, "digest": canonical_hash(fields)}


def entry_payload_bytes() -> int:
    return TOKENS_PER_WINDOW * TOKENIZER_VOCAB * 4 + TOKENS_PER_WINDOW * 4


def cache_capacity() -> dict[str, int]:
    payload = entry_payload_bytes()
    entries = MAX_CACHE_BYTES // payload
    return {"max_bytes": MAX_CACHE_BYTES, "entry_payload_bytes": payload, "max_entries": entries, "max_complete_documents": entries // 2}


def pair_documents(documents: list[dict[str, Any]], pair_index: int) -> list[dict[str, Any]]:
    if not documents:
        raise ValueError("cannot schedule pair without eligible documents")
    start = pair_index * PHYSICAL_BATCH
    return [documents[(start + offset) % len(documents)] for offset in range(PHYSICAL_BATCH)]


def simulate_full_traversal(documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not documents:
        raise RuntimeError("no eligible train documents")
    manifest = build_pair_manifest(documents, PAIR_COUNT)
    pair_rows = manifest["pairs"]
    first_repeat: int | None = None
    pair0 = tuple(pair_rows[0]["document_indices"])
    for row in pair_rows[1:]:
        if tuple(row["document_indices"]) == pair0:
            first_repeat = int(row["pair"])
            break
    if first_repeat is None:
        raise RuntimeError("no repeated pair found in authorized traversal")

    capacity = cache_capacity()
    cache: OrderedDict[str, int] = OrderedDict()
    last_access: dict[str, int] = {}
    pair_stats: list[dict[str, Any]] = []
    by_window = {"window_0": {"hits": 0, "misses": 0, "evictions": 0, "accesses": 0}, "window_1": {"hits": 0, "misses": 0, "evictions": 0, "accesses": 0}}
    reuse_distances: list[int] = []
    access_index = 0
    document_positions: dict[str, list[int]] = defaultdict(list)
    document_reuse_counts: dict[str, int] = defaultdict(int)
    for row in pair_rows:
        pair_index = int(row["pair"])
        pair_hit = pair_miss = pair_evictions = 0
        pair_windows: dict[str, dict[str, int]] = {}
        for window in (0, 1):
            window_name = f"window_{window}"
            hits = misses = evictions = 0
            for document_index in row["document_indices"]:
                # build_pair_manifest indexes eligible documents, not source rows.
                document = documents[int(document_index)]
                key = str(canonical_cache_key(document, window)["digest"])
                document_positions[str(document["eligible_position"])].append(pair_index)
                by_window[window_name]["accesses"] += 1
                if key in cache:
                    hits += 1
                    by_window[window_name]["hits"] += 1
                    reuse_distances.append(access_index - last_access[key] - 1)
                    cache.move_to_end(key)
                else:
                    misses += 1
                    by_window[window_name]["misses"] += 1
                    if len(cache) >= capacity["max_entries"]:
                        cache.popitem(last=False)
                        evictions += 1
                        by_window[window_name]["evictions"] += 1
                    cache[key] = access_index
                    document_reuse_counts[str(document["eligible_position"])] += 1
                last_access[key] = access_index
                access_index += 1
            pair_hit += hits
            pair_miss += misses
            pair_evictions += evictions
            pair_windows[window_name] = {"hits": hits, "misses": misses, "evictions": evictions}
        pair_stats.append({"pair": pair_index, "hits": pair_hit, "misses": pair_miss, "evictions": pair_evictions, "by_window": pair_windows})

    per_document = []
    for document in documents:
        position = str(document["eligible_position"])
        positions = document_positions[position]
        per_document.append({
            "eligible_position": int(document["eligible_position"]),
            "document_index": int(document["document_index"]),
            "full_text_sha256": document["full_text_sha256"],
            "retained_513_token_sha256": document["retained_513_token_sha256"],
            "pair_positions": positions,
            "pair_occurrences": len(positions),
            "reuse_count": max(0, len(positions) - 1),
        })
    return {
        "mode": "metadata_only_exact_traversal_simulation",
        "estimate_scope": "full traversal simulation estimates frequency; it is not a block trial cost",
        "eligible_document_count": len(documents),
        "pair_count": PAIR_COUNT,
        "documents_consumed": PAIR_COUNT * PHYSICAL_BATCH,
        "unique_documents_touched": len(documents),
        "complete_cycle_length_pairs": len(documents) // math.gcd(len(documents), PHYSICAL_BATCH),
        "first_repeated_pair_index": first_repeat,
        "pair_schedule_definition": "pair_index*8 modulo eligible_document_count with wrap handling identical to build_pair_manifest",
        "per_document": per_document,
        "cache_policy": {
            "kind": "persistent_on_disk_lru",
            "key": "canonical hash of document identity + retained-token hash + teacher/revision + tokenizer revision + window + context range + representation + dtype + temperature",
            "separate_per_document_window_entries": True,
            "cache_window_key_forbidden": True,
            **capacity,
        },
        "cache_simulation": {
            "route": "C+L",
            "by_window": by_window,
            "total_hits": sum(item["hits"] for item in by_window.values()),
            "total_misses": sum(item["misses"] for item in by_window.values()),
            "total_evictions": sum(item["evictions"] for item in by_window.values()),
            "reuse_distance_count": len(reuse_distances),
            "reuse_distance_min": min(reuse_distances) if reuse_distances else None,
            "reuse_distance_max": max(reuse_distances) if reuse_distances else None,
            "reuse_distance_mean": sum(reuse_distances) / len(reuse_distances) if reuse_distances else None,
            "costs_paid": {
                "miss": ["real teacher target acquisition", "FP32 transform", "persistent cache write", "delivery"],
                "hit": ["persistent entry access", "deserialization", "copy", "delivery"],
                "eviction": ["LRU entry removal before bounded write"],
            },
            "pair_stats": pair_stats,
            "document_reuse_counts": dict(document_reuse_counts),
        },
    }


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.total_bytes = 0
        self.evictions = 0
        self.new_files = 0
        self.max_total_bytes = 0
        self.minimum_free_disk = disk_free_bytes(self.root)

    def lookup(self, key: dict[str, Any]) -> dict[str, Tensor] | None:
        digest = str(key["digest"])
        record = self.entries.get(digest)
        if record is None:
            return None
        path = Path(record["path"])
        if not path.is_file():
            self.entries.pop(digest, None)
            self.total_bytes -= int(record["bytes"])
            return None
        loaded = torch.load(path, map_location=DEVICE, weights_only=True)
        self.entries.move_to_end(digest)
        return loaded

    def put(self, key: dict[str, Any], payload: dict[str, Tensor], stage: str) -> int:
        digest = str(key["digest"])
        if digest in self.entries:
            return int(self.entries[digest]["bytes"])
        for tensor in payload.values():
            if tensor.dtype != DTYPE:
                raise RuntimeError("cache payload must be FP32")
        expected = sum(int(tensor.numel()) * tensor.element_size() for tensor in payload.values())
        if expected > MAX_CACHE_BYTES:
            raise CacheCapacityError(stage, "one cache entry exceeds 4 GiB limit")
        while self.total_bytes + expected > MAX_CACHE_BYTES and self.entries:
            old_digest, old_record = self.entries.popitem(last=False)
            old_path = Path(old_record["path"])
            if old_path.exists():
                old_path.unlink()
            self.total_bytes -= int(old_record["bytes"])
            self.evictions += 1
        if self.total_bytes + expected > MAX_CACHE_BYTES:
            raise CacheCapacityError(stage, "cache capacity cannot admit entry")
        free = check_disk_before_write(self.root, expected, stage)
        self.minimum_free_disk = min(self.minimum_free_disk, free)
        path = self.root / f"{digest}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        actual = path.stat().st_size
        self.entries[digest] = {"path": path, "bytes": actual}
        self.total_bytes += actual
        self.new_files += 1
        self.max_total_bytes = max(self.max_total_bytes, self.total_bytes)
        if self.total_bytes > MAX_CACHE_BYTES:
            raise CacheCapacityError(stage, "serialized cache files exceed 4 GiB limit")
        return actual


def public_document(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "tokens"}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n").encode("utf-8"))


def canonical_report_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_hashed_report(path: Path, report: dict[str, Any]) -> tuple[str, str]:
    snapshot = json_safe(report)
    if not isinstance(snapshot, dict):
        raise TypeError("report must be a dictionary")
    snapshot["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    unsigned = canonical_report_bytes(snapshot)
    digest = sha256_bytes(unsigned)
    snapshot["artifact_self_hash"] = digest
    encoded = canonical_report_bytes(snapshot)
    path.write_bytes(encoded)
    persisted = path.read_bytes()
    parsed = json.loads(persisted.decode("utf-8"))
    stored = parsed["artifact_self_hash"]
    parsed["artifact_self_hash"] = SELF_HASH_PLACEHOLDER
    if persisted != encoded or canonical_report_bytes(parsed) != unsigned or stored != digest or sha256_bytes(canonical_report_bytes(parsed)) != stored:
        raise RuntimeError("report self-hash verification failed")
    return digest, sha256_bytes(persisted)


def teacher_window_logits(teacher: nn.Module, source: Tensor, window: int) -> Tensor:
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    with torch.no_grad():
        output = teacher(input_ids=context)
        result = output.logits[:, start : start + 256].detach().clone().float()
    del output
    if result.dtype != DTYPE:
        raise RuntimeError(f"teacher route produced non-FP32 logits: {result.dtype}")
    return result


def load_teacher() -> nn.Module:
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    teacher.to(DEVICE).float().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def timed_phase(label: str, fn: Callable[[], Any], timings: dict[str, float], samples: list[dict[str, Any]]) -> Any:
    before = reserve_guard(f"{label}:before")
    samples.append({"phase": label, "point": "before", **before})
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    timings[label] = timings.get(label, 0.0) + elapsed
    after = reserve_guard(f"{label}:after")
    samples.append({"phase": label, "point": "after", **after})
    return value


def make_model(variant: str) -> tuple[OmegaCoreLMFast, torch.optim.AdamW]:
    config = VARIANTS[variant]
    torch.manual_seed(20260913 if variant == "shared_K1" else 20260914)
    reference = OmegaCoreLM0R1Technical(vocab_size=TOKENIZER_VOCAB, dimension=DIMENSION, slots=SLOTS, rounds=config["rounds"], variant="shared").to(DEVICE).float()
    model = OmegaCoreLMFast.from_reference(reference).to(DEVICE).float()
    del reference
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    return model, optimizer


def acquire_lse_payload(
    *,
    teacher: nn.Module,
    source: Tensor,
    documents: list[dict[str, Any]],
    window: int,
    cache: CacheStore,
    timings: dict[str, float],
    samples: list[dict[str, Any]],
) -> tuple[Tensor, Tensor, str, dict[str, Any]]:
    keys = [canonical_cache_key(document, window) for document in documents]
    hit_indexes: list[int] = []
    miss_indexes: list[int] = []
    for index, key in enumerate(keys):
        if str(key["digest"]) in cache.entries:
            hit_indexes.append(index)
        else:
            miss_indexes.append(index)
    mode = "hit" if not miss_indexes else "miss" if not hit_indexes else "mixed"
    teacher_logits: Tensor | None = None
    if miss_indexes:
        teacher_source = source[miss_indexes]
        teacher_logits = timed_phase("teacher_acquisition", lambda: teacher_window_logits(teacher, teacher_source, window), timings, samples)
        del teacher_source
    reserve_guard("delivery:before")
    delivery_started = time.perf_counter()
    target_probs = torch.empty(PHYSICAL_BATCH, TOKENS_PER_WINDOW, TOKENIZER_VOCAB, dtype=DTYPE, device=DEVICE)
    target_entropy = torch.empty(PHYSICAL_BATCH, TOKENS_PER_WINDOW, dtype=DTYPE, device=DEVICE)
    timings["delivery"] = timings.get("delivery", 0.0) + time.perf_counter() - delivery_started
    samples.append({"phase": "delivery", "point": "after_allocation", **memory_snapshot()})
    reserve_guard("delivery:after_allocation")
    if teacher_logits is not None:
        for local_index, batch_index in enumerate(miss_indexes):
            reserve_guard(f"transform:before:{batch_index}")
            started = time.perf_counter()
            probs, entropy = teacher_targets(teacher_logits[local_index].float(), temperature=TEMPERATURE)
            timings["transform"] = timings.get("transform", 0.0) + time.perf_counter() - started
            samples.append({"phase": "transform", "point": f"after:{batch_index}", **memory_snapshot()})
            reserve_guard(f"transform:after:{batch_index}")
            reserve_guard(f"cache_write:before:{batch_index}")
            started = time.perf_counter()
            cache.put(keys[batch_index], {"teacher_probs": probs.contiguous(), "teacher_neg_entropy": entropy.contiguous()}, f"cache_write:{window}:{batch_index}")
            timings["cache_write"] = timings.get("cache_write", 0.0) + time.perf_counter() - started
            samples.append({"phase": "cache_write", "point": f"after:{batch_index}", **memory_snapshot()})
            reserve_guard(f"cache_write:after:{batch_index}")
            reserve_guard(f"delivery:copy_miss:{batch_index}")
            started = time.perf_counter()
            target_probs[batch_index].copy_(probs)
            target_entropy[batch_index].copy_(entropy)
            timings["delivery"] = timings.get("delivery", 0.0) + time.perf_counter() - started
            reserve_guard(f"delivery:after_miss_copy:{batch_index}")
            del probs, entropy
        del teacher_logits
    for index in hit_indexes:
        reserve_guard(f"cache_read:before:{index}")
        started = time.perf_counter()
        loaded = cache.lookup(keys[index])
        if loaded is None:
            raise RuntimeError("cache index changed between hit classification and read")
        timings["cache_read"] = timings.get("cache_read", 0.0) + time.perf_counter() - started
        samples.append({"phase": "cache_read", "point": f"after:{index}", **memory_snapshot()})
        reserve_guard(f"delivery:copy_hit:{index}")
        started = time.perf_counter()
        target_probs[index].copy_(loaded["teacher_probs"])
        target_entropy[index].copy_(loaded["teacher_neg_entropy"])
        timings["delivery"] = timings.get("delivery", 0.0) + time.perf_counter() - started
        reserve_guard(f"delivery:after_hit_copy:{index}")
        del loaded
    reserve_guard("cache_delivery:after")
    samples.append({"phase": "cache_delivery", "point": "after", **memory_snapshot()})
    return target_probs, target_entropy, mode, {"hit_count": len(hit_indexes), "miss_count": len(miss_indexes), "keys": [str(key["digest"]) for key in keys]}


def negative_lookup_probe(cache: CacheStore, documents: list[dict[str, Any]], window: int) -> dict[str, Any]:
    if len(documents) < 2:
        raise RuntimeError("negative lookup requires two distinct documents")
    first_key = canonical_cache_key(documents[0], window)
    second_key = canonical_cache_key(documents[1], window)
    sentinel = cache.root / "negative_lookup_probe.pt"
    torch.save({"teacher_probs": torch.zeros(2, 3, dtype=DTYPE), "teacher_neg_entropy": torch.zeros(2, dtype=DTYPE)}, sentinel)
    cache.entries[str(first_key["digest"])] = {"path": sentinel, "bytes": sentinel.stat().st_size}
    result = cache.lookup(second_key)
    cache.entries.pop(str(first_key["digest"]), None)
    sentinel.unlink(missing_ok=True)
    if result is not None:
        raise RuntimeError("different-document cache lookup returned previous payload")
    return {"same_window": True, "first_digest": first_key["digest"], "different_document_digest": second_key["digest"], "different_document_is_miss": True}


def run_route(route: str, simulation: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started_route = time.perf_counter()
    result: dict[str, Any] = {
        "route": route,
        "status": "blocked",
        "configuration": {"device": "cpu", "dtype": "float32", "execution": "eager", "physical_batch": PHYSICAL_BATCH, "window": TOKENS_PER_WINDOW, "dimension": DIMENSION, "slots": SLOTS, "vocab_size": TOKENIZER_VOCAB, "temperature": TEMPERATURE, "optimizer": {"type": "AdamW", "lr": BASE_LR, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "clip_norm": CLIP_NORM}},
        "thread_configuration": {"intraop": THREADS, "interop": INTEROP_THREADS, "changed_inside_loops": False},
        "variants": {},
        "exact_updates_completed": 0,
        "errors": [],
    }
    selected_pair_indexes = [0, int(simulation["first_repeated_pair_index"])]
    needed_positions = {int(item["eligible_position"]) for item in simulation["per_document"] if any(pair in selected_pair_indexes for pair in item["pair_positions"])}
    documents, load_meta = load_train_metadata(needed_positions)
    by_position = {int(document["eligible_position"]): document for document in documents}
    simulation_docs = {int(item["eligible_position"]): item for item in simulation["per_document"]}
    for position in needed_positions:
        if position not in by_position or "tokens" not in by_position[position]:
            raise RuntimeError(f"selected document tokens missing for eligible position {position}")
        for field in ("full_text_sha256", "retained_513_token_sha256", "document_index"):
            if by_position[position][field] != simulation_docs[position][field]:
                raise RuntimeError(f"selected document metadata drift for {position}/{field}")
    result["source_reconstruction"] = load_meta
    teacher = None
    try:
        reserve_guard("teacher_setup:before")
        teacher = load_teacher()
        reserve_guard("teacher_setup:after")
        for variant in VARIANTS:
            reserve_guard(f"model_setup:{variant}:before")
            model, optimizer = make_model(variant)
            reserve_guard(f"model_setup:{variant}:after")
            variant_dir = output_dir / "cache" / variant
            cache = CacheStore(variant_dir) if route == "C+L" else None
            ledger_path = output_dir / f"{route.replace('+', '_')}_{variant}_ledger.jsonl"
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            updates: list[dict[str, Any]] = []
            negative_probe = None
            for pair_index in selected_pair_indexes:
                pair_docs = [by_position[int(item["eligible_position"])] for item in pair_documents(simulation_docs_as_documents(simulation), pair_index)]
                source = timed_phase("source_assignment", lambda docs=pair_docs: torch.tensor([doc["tokens"] for doc in docs], dtype=torch.long, device=DEVICE), {}, [])
                state: Tensor | None = None
                for window in (0, 1):
                    optimizer.zero_grad(set_to_none=True)
                    if window == 0:
                        reserve_guard(f"state_reset:before:{variant}:{pair_index}")
                        state = model.initial_state(PHYSICAL_BATCH, device=DEVICE)
                    if state is None:
                        raise RuntimeError("window 1 missing detached state")
                    input_ids = source[:, :256] if window == 0 else source[:, 256:512]
                    targets = source[:, 1:257] if window == 0 else source[:, 257:513]
                    timings: dict[str, float] = {}
                    samples: list[dict[str, Any]] = []
                    update_started = time.perf_counter()
                    cache_mode = "direct"
                    cache_info: dict[str, Any] = {}
                    teacher_logits: Tensor | None = None
                    teacher_probs: Tensor | None = None
                    teacher_entropy: Tensor | None = None
                    trace: dict[str, Tensor] | None = None
                    if route == "F":
                        teacher_logits = timed_phase("teacher_cache_retrieval", lambda: teacher_window_logits(teacher, source, window), timings, samples)
                    else:
                        assert cache is not None
                        if negative_probe is None:
                            negative_probe = negative_lookup_probe(cache, pair_docs, window)
                        teacher_probs, teacher_entropy, cache_mode, cache_info = acquire_lse_payload(teacher=teacher, source=source, documents=pair_docs, window=window, cache=cache, timings=timings, samples=samples)
                    if route == "F":
                        reserve_guard(f"recurrence_forward:before:{variant}:{pair_index}:{window}")
                        rec_started = time.perf_counter()
                        next_state, student_logits, trace = model.forward_window(input_ids, state)
                        timings["recurrence_forward"] = time.perf_counter() - rec_started
                        samples.append({"phase": "recurrence_forward", "point": "after", **memory_snapshot()})
                        reserve_guard(f"recurrence_forward:after:{variant}:{pair_index}:{window}")
                    else:
                        reserve_guard(f"recurrence_forward:before:{variant}:{pair_index}:{window}")
                        rec_started = time.perf_counter()
                        next_state, _, candidate_states, readout_states = model.recur_states(input_ids, state)
                        trace = {"candidate_states": candidate_states, "readout_states": readout_states}
                        timings["recurrence_forward"] = time.perf_counter() - rec_started
                        samples.append({"phase": "recurrence_forward", "point": "after", **memory_snapshot()})
                        reserve_guard(f"recurrence_forward:after:{variant}:{pair_index}:{window}")
                    valid_mask = torch.ones_like(targets, dtype=torch.bool)
                    reserve_guard(f"loss:before:{variant}:{pair_index}:{window}")
                    loss_started = time.perf_counter()
                    if route == "F":
                        assert teacher_logits is not None and student_logits is not None
                        losses = distillation_loss(student_logits, teacher_logits, targets, valid_mask)
                    else:
                        assert trace is not None and teacher_probs is not None and teacher_entropy is not None
                        losses = distillation_loss_lse(model, trace["readout_states"], teacher_probs, teacher_entropy, targets, valid_mask, temperature=TEMPERATURE)
                    timings["loss"] = time.perf_counter() - loss_started
                    samples.append({"phase": "loss", "point": "after", **memory_snapshot()})
                    reserve_guard(f"loss:after:{variant}:{pair_index}:{window}")
                    reserve_guard(f"backward:before:{variant}:{pair_index}:{window}")
                    backward_started = time.perf_counter()
                    losses["total"].backward()
                    timings["backward"] = time.perf_counter() - backward_started
                    samples.append({"phase": "backward", "point": "after", **memory_snapshot()})
                    reserve_guard(f"backward:after:{variant}:{pair_index}:{window}")
                    reserve_guard(f"clipping:before:{variant}:{pair_index}:{window}")
                    clipping_started = time.perf_counter()
                    pre_clip = grad_norm(model)
                    returned_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
                    post_clip = grad_norm(model)
                    timings["clipping"] = time.perf_counter() - clipping_started
                    samples.append({"phase": "clipping", "point": "after", **memory_snapshot()})
                    reserve_guard(f"clipping:after:{variant}:{pair_index}:{window}")
                    reserve_guard(f"adamw:before:{variant}:{pair_index}:{window}")
                    adam_started = time.perf_counter()
                    optimizer.step()
                    timings["adamw"] = time.perf_counter() - adam_started
                    samples.append({"phase": "adamw", "point": "after", **memory_snapshot()})
                    reserve_guard(f"adamw:after:{variant}:{pair_index}:{window}")
                    state = next_state.detach()
                    finite = all(bool(torch.isfinite(losses[key]).item()) for key in ("ce", "kl", "total")) and all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())
                    if not finite:
                        raise RuntimeError(f"non-finite values in {route}/{variant}/{pair_index}/{window}")
                    ledger_started = time.perf_counter()
                    reserve_guard(f"ledger:before:{variant}:{pair_index}:{window}")
                    record = {
                        "route": route,
                        "variant": variant,
                        "pair_index": pair_index,
                        "window": window,
                        "cache_mode": cache_mode,
                        "cache_info": cache_info,
                        "document_indices": [int(doc["document_index"]) for doc in pair_docs],
                        "eligible_positions": [int(doc["eligible_position"]) for doc in pair_docs],
                        "full_text_sha256": [doc["full_text_sha256"] for doc in pair_docs],
                        "retained_513_token_sha256": [doc["retained_513_token_sha256"] for doc in pair_docs],
                        "teacher_context_range": [0, 256] if window == 0 else [0, 512],
                        "input_range": [0, 256] if window == 0 else [256, 512],
                        "target_range": [1, 257] if window == 0 else [257, 513],
                        "valid_tokens": PHYSICAL_BATCH * TOKENS_PER_WINDOW,
                        "phase_seconds": {**timings, "ledger": 0.0},
                        "loss": {key: float(losses[key].detach().item()) for key in ("ce", "kl", "total")},
                        "pre_clip_grad_norm": pre_clip,
                        "clip_returned_norm": returned_clip,
                        "post_clip_grad_norm": post_clip,
                        "finite": finite,
                        "parameter_hash_after": parameter_hash(model),
                        "memory_samples": samples,
                    }
                    phase_seconds = record["phase_seconds"]
                    with ledger_path.open("ab") as ledger:
                        ledger.write((json.dumps(json_safe(record), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
                        ledger.flush()
                        os.fsync(ledger.fileno())
                    phase_seconds["ledger"] = time.perf_counter() - ledger_started
                    record["total_seconds"] = time.perf_counter() - update_started
                    samples.append({"phase": "ledger", "point": "after", **memory_snapshot()})
                    reserve_guard(f"ledger:after:{variant}:{pair_index}:{window}")
                    updates.append(record)
                    result["exact_updates_completed"] += 1
                    del losses, valid_mask, next_state, input_ids, targets
                    if teacher_logits is not None:
                        del teacher_logits
                    if teacher_probs is not None:
                        del teacher_probs, teacher_entropy
                    if route == "F":
                        del student_logits
                    del trace
                del source, state
            variant_result = {
                "updates": updates,
                "updates_completed": len(updates),
                "K1_total_seconds": sum(item["total_seconds"] for item in updates) if variant == "shared_K1" else None,
                "K4_total_seconds": sum(item["total_seconds"] for item in updates) if variant == "shared_K4" else None,
                "by_cache_mode": {mode: {"count": sum(item["cache_mode"] == mode for item in updates), "total_seconds": sum(item["total_seconds"] for item in updates if item["cache_mode"] == mode)} for mode in sorted({item["cache_mode"] for item in updates})},
                "parameter_hash_after": parameter_hash(model),
                "ledger": ledger_path,
                "negative_lookup_probe": negative_probe,
            }
            if cache is not None:
                variant_result["cache"] = {"root": variant_dir, "new_files": cache.new_files, "final_bytes": cache.total_bytes, "max_bytes_observed": cache.max_total_bytes, "evictions": cache.evictions, "minimum_free_disk_bytes": cache.minimum_free_disk}
            result["variants"][variant] = variant_result
            del model, optimizer, cache
        result["status"] = "completed"
    except MemoryReserveError as exc:
        result["status"] = "blocked_memory_reserve"
        result["failure_stage"] = exc.stage
        result["errors"].append({"type": type(exc).__name__, "message": str(exc), "stage": exc.stage, "available_bytes": exc.available_bytes})
    except CacheCapacityError as exc:
        result["status"] = "blocked_cache_capacity"
        result["failure_stage"] = exc.stage
        result["errors"].append({"type": type(exc).__name__, "message": str(exc), "stage": exc.stage})
    except Exception as exc:
        result["status"] = "failed"
        result["failure_stage"] = result.get("failure_stage", "route")
        result["errors"].append({"type": type(exc).__name__, "message": str(exc), "stage": result["failure_stage"]})
    finally:
        if teacher is not None:
            del teacher
        result["memory"] = {
            "minimum_available_system_bytes": min((sample["available_system_bytes"] for variant in result["variants"].values() for update in variant.get("updates", []) for sample in update.get("memory_samples", []) if sample.get("available_system_bytes") is not None), default=int(psutil.virtual_memory().available)),
            "observed_peak_rss_bytes": max((sample["rss_bytes"] for variant in result["variants"].values() for update in variant.get("updates", []) for sample in update.get("memory_samples", []) if sample.get("rss_bytes") is not None), default=psutil.Process().memory_info().rss),
            "reserve_bytes": MEMORY_RESERVE_BYTES,
        }
        result["elapsed_seconds"] = time.perf_counter() - started_route
    return result


def simulation_docs_as_documents(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in sorted(simulation["per_document"], key=lambda item: int(item["eligible_position"]))]


def source_hashes() -> dict[str, str]:
    return {
        "own_runner_sha256": sha256_file(Path(__file__)),
        "own_test_sha256": sha256_file(UNIT_DIR / "test_step3_cache_extension.py"),
        "own_readme_sha256": sha256_file(UNIT_DIR / "README_step3.md"),
        "current_r1_source_sha256": sha256_file(SCRIPTS_DIR / "run_omega_core_lm_0_r1_training_technical_preflight.py"),
        "local_fast_candidate_sha256": sha256_file(UNIT_DIR / "omega_fast_candidate.py"),
        "scope_schedule_source_sha256": sha256_file(SCOPE_DIR / "run_scientific_scoping_a.py"),
    }


def launch_child(route: str, simulation_path: Path, output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / f"{route.replace('+', '_')}_child_result.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--child-route", route, "--simulation", str(simulation_path), "--child-output", str(result_path)]
    process = subprocess.Popen(command, cwd=str(UNIT_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["process_returncode"] = process.returncode
        result["child_stdout"] = stdout[-2000:]
        result["child_stderr"] = stderr[-2000:]
        return result
    return {"route": route, "status": "failed", "failure_stage": "child_process", "errors": [{"type": "ChildProcessError", "message": f"child exited {process.returncode} without result", "stdout": stdout[-2000:], "stderr": stderr[-2000:]}], "exact_updates_completed": 0}


def combine_block_costs(children: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"measured_definition": "block trial measures costs; it does not estimate full traversal frequency", "routes": {}}
    for route, child in children.items():
        updates = [update for variant in child.get("variants", {}).values() for update in variant.get("updates", [])]
        by_mode: dict[str, dict[str, Any]] = {}
        for mode in sorted({str(update["cache_mode"]) for update in updates}):
            matching = [update for update in updates if update["cache_mode"] == mode]
            by_mode[mode] = {"updates": len(matching), "total_seconds": sum(float(item["total_seconds"]) for item in matching), "mean_seconds": sum(float(item["total_seconds"]) for item in matching) / len(matching) if matching else None, "phase_seconds": {phase: sum(float(item["phase_seconds"].get(phase, 0.0)) for item in matching) for phase in sorted({phase for item in matching for phase in item["phase_seconds"]})}}
        result["routes"][route] = {"status": child.get("status"), "miss_hit_block_costs": by_mode, "K1_total_seconds": sum(float(item["total_seconds"]) for item in updates if item["variant"] == "shared_K1"), "K4_total_seconds": sum(float(item["total_seconds"]) for item in updates if item["variant"] == "shared_K4"), "total_seconds": sum(float(item["total_seconds"]) for item in updates), "exact_updates": len(updates)}
    return result


def run_parent(output_dir: Path) -> tuple[dict[str, Any], Path]:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    documents, reconstruction = load_train_metadata()
    simulation = simulate_full_traversal(documents)
    simulation["reconstruction"] = reconstruction
    simulation_path = run_dir / "full_traversal_simulation.json"
    write_json(simulation_path, simulation)
    del documents
    children: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        children[route] = launch_child(route, simulation_path, run_dir)
    all_completed = simulation["pair_count"] == PAIR_COUNT and all(child.get("status") == "completed" and child.get("exact_updates_completed") == 8 for child in children.values())
    report: dict[str, Any] = {
        "schema": "omega-core-lm-0-r1-cpu-fastpath-validation-step3-cache-extension-v1",
        "step": 3,
        "step_name": "metadata-only exact SCOPE-A cycle simulation plus targeted cache miss/hit trial",
        "run_id": run_id,
        "status": "completed" if all_completed else "blocked",
        "failure_stage": None if all_completed else next((f"{route}:{child.get('failure_stage', child.get('status'))}" for route, child in children.items() if child.get("status") != "completed"), "update_contract"),
        "scope_a_relaunched": False,
        "projection_calculated": False,
        "test_split_loaded": False,
        "step2_selector_preserved": True,
        "default_base_route": "F",
        "automatic_selector_l_documented_as_historical_benchmark_decision": True,
        "exact_update_contract": {"routes": list(ROUTES), "variants": list(VARIANTS), "selected_pair_positions": [0, int(simulation["first_repeated_pair_index"])], "windows_per_pair": [0, 1], "updates_per_route": 8, "total_updates": 16},
        "block_trial_costs": combine_block_costs(children),
        "full_traversal_simulation": simulation,
        "children": children,
        "formula_uncomputed": FORMULA_TEXT,
        "limitations": ["The block trial is a targeted miss/hit path probe, not full-traversal frequency evidence.", "Full traversal values are metadata-only frequency estimates; measured block costs are not blended into them.", "No validation/test split, SCOPE-A relaunch, projection, or quality interpretation was performed."],
        "provenance": {"dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION, "split": "train"}, "teacher": {"id": MODEL_ID, "revision": MODEL_REVISION, "window_contexts": {"window_0": [0, 256], "window_1": [0, 512]}}, "source_hashes": source_hashes()},
        "cache_capacity": cache_capacity(),
        "report_path": (run_dir / "step3_report.json").relative_to(UNIT_DIR).as_posix(),
        "elapsed_seconds": None,
    }
    report["elapsed_seconds"] = sum(float(child.get("elapsed_seconds", 0.0)) for child in children.values())
    report_path = run_dir / "step3_report.json"
    digest, file_hash = write_hashed_report(report_path, report)
    report["report_path"] = report_path.relative_to(UNIT_DIR).as_posix()
    report["report_self_hash"] = digest
    report["report_file_sha256"] = file_hash
    return report, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-route", choices=ROUTES)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=UNIT_DIR / "results" / "step3_cache_extension")
    args = parser.parse_args(argv)
    if args.child_route:
        if args.simulation is None or args.child_output is None:
            parser.error("child route requires --simulation and --child-output")
        simulation = json.loads(args.simulation.read_text(encoding="utf-8"))
        child = run_route(args.child_route, simulation, args.child_output.parent)
        write_json(args.child_output, child)
        return 0 if child["status"] == "completed" else 2
    report, path = run_parent(args.output_root.resolve())
    print(json.dumps({"status": report["status"], "report": path.relative_to(UNIT_DIR).as_posix(), "artifact_self_hash": report["report_self_hash"], "exact_updates_completed": sum(int(child.get("exact_updates_completed", 0)) for child in report["children"].values())}, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
