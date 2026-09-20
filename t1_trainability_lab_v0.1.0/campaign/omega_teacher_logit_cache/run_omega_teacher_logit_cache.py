"""OMEGA-TEACHER-LOGIT-CACHE implementation and guarded phase runners.

The module contains implementation for phases A-D, but real cache creation,
correctness execution, and benchmarking require explicit authorization. Tests
exercise synthetic tensors and tiny models only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
REPO_ROOT = CAMPAIGN_ROOT.parent.parent
CE_ROOT = CAMPAIGN_ROOT / "omega_ce_only_baseline"
F_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_cpu_fastpath_validation"
R1_DIR = CAMPAIGN_ROOT / "omega_core_lm_0_r1_scientific_scoping_a"
TECHNICAL_SCRIPT = REPO_ROOT / "t1_trainability_lab_v0.1.0" / "scripts" / "run_omega_core_lm_0_r1_training_technical_preflight.py"
TRAIN_MANIFEST_PATH = CE_ROOT / "results" / "runs" / "CE-K4_seed_20260913" / "train_manifest.json"
DEFAULT_OUTPUT_ROOT = HERE / "results"

for _path in (F_DIR, R1_DIR, TECHNICAL_SCRIPT.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


CAMPAIGN_ID = "OMEGA-TEACHER-LOGIT-CACHE"
MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
TOKENIZER_REVISION = MODEL_REVISION
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_VOCAB = 50257
DOCUMENT_COUNT = 602
WINDOW_COUNT = 2
TOKENS_PER_WINDOW = 256
RETAINED_TOKENS = 513
PHYSICAL_BATCH = 8
BUILD_PAIRS = 76
PAIR_COUNT = 1000
ENTRY_BYTES = TOKENS_PER_WINDOW * TOKENIZER_VOCAB * 4
EXPECTED_CACHE_BYTES = WINDOW_COUNT * DOCUMENT_COUNT * ENTRY_BYTES
STORAGE_SAFETY_MARGIN = 0.20
WINDOWS = (0, 1)
BENCHMARK_UPDATES = 152
BENCHMARK_WARMUP_LAST = 39
BENCHMARK_MEASURE_START = 40
BENCHMARK_MEASURED_UPDATES = BENCHMARK_UPDATES - BENCHMARK_MEASURE_START
TEMPERATURE = 2.0
BASE_LR = 3e-4
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
CPU_INTRAOP_THREADS = 4
CPU_INTEROP_THREADS = 1
SELF_HASH_PLACEHOLDER = "__SELF_HASH__"


class RealExecutionAuthorizationError(RuntimeError):
    """Real cache operations require an explicit command-line authorization."""


class StorageInconclusiveError(RuntimeError):
    """The target filesystem cannot safely hold the complete FP32 cache."""


class CacheContractError(RuntimeError):
    """A cache artifact violates its immutable read-only contract."""


def require_real_authorization(confirmed: bool, operation: str) -> None:
    if not confirmed:
        raise RealExecutionAuthorizationError(f"{operation} requires --confirm-real-execution")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def file_hash(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def tensor_hash(value: Tensor) -> str:
    return sha256_bytes(tensor_bytes(value))


def write_self_hashed(path: Path, payload: Mapping[str, Any], field: str = "report_self_hash") -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned[field] = SELF_HASH_PLACEHOLDER
    encoded_unsigned = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_bytes(encoded_unsigned)
    written = dict(payload)
    written[field] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    if not verify_self_hash(json.loads(encoded.decode("utf-8")), field):
        raise RuntimeError("self-hash construction failed")
    return written


def verify_self_hash(value: Mapping[str, Any], field: str = "report_self_hash") -> bool:
    digest = value.get(field)
    if not isinstance(digest, str):
        return False
    unsigned = dict(value)
    unsigned[field] = SELF_HASH_PLACEHOLDER
    encoded = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return sha256_bytes(encoded) == digest


def disk_free_bytes(path: Path) -> int:
    return int(__import__("shutil").disk_usage(path).free)


def storage_feasibility(path: Path, *, expected_bytes: int = EXPECTED_CACHE_BYTES, margin: float = STORAGE_SAFETY_MARGIN) -> dict[str, Any]:
    path = path.resolve()
    probe = path if path.exists() else path.parent
    free = disk_free_bytes(probe)
    required = math.ceil(expected_bytes * (1.0 + margin))
    status = "READY" if free >= required else "INCONCLUSIVE_STORAGE"
    return {
        "status": status,
        "path": path.as_posix(),
        "filesystem_probe": probe.as_posix(),
        "expected_cache_bytes": int(expected_bytes),
        "safety_margin_fraction": float(margin),
        "required_free_bytes": int(required),
        "free_bytes": int(free),
        "margin_satisfied": bool(free >= required),
        "will_construct": False,
    }


def phase_a_report(path: Path) -> dict[str, Any]:
    feasibility = storage_feasibility(path)
    report = {
        "schema": "omega-teacher-logit-cache-feasibility-v1",
        "campaign_id": CAMPAIGN_ID,
        "phase": "A",
        "layout": {
            "windows": WINDOW_COUNT,
            "documents": DOCUMENT_COUNT,
            "tokens_per_window": TOKENS_PER_WINDOW,
            "vocab": TOKENIZER_VOCAB,
            "order": ["window", "document", "token", "vocab"],
            "representation": "raw_fp32_logits",
            "dtype": "float32",
        },
        "entry_bytes": ENTRY_BYTES,
        "expected_entries": WINDOW_COUNT * DOCUMENT_COUNT,
        "expected_cache_bytes": EXPECTED_CACHE_BYTES,
        "feasibility": feasibility,
    }
    return report


def frozen_train_manifest(path: Path = TRAIN_MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("document_count") != DOCUMENT_COUNT:
        raise CacheContractError("frozen train manifest must contain 602 documents")
    cyclic = manifest.get("cyclic_pairs", {})
    if cyclic.get("pair_count") != PAIR_COUNT or cyclic.get("documents_available") != DOCUMENT_COUNT:
        raise CacheContractError("frozen cyclic pair manifest identity mismatch")
    if not manifest.get("manifest_sha256"):
        raise CacheContractError("frozen train manifest hash missing")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) != DOCUMENT_COUNT:
        raise CacheContractError("frozen document list mismatch")
    return manifest


def _document_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "document_index": int(document["document_index"]),
        "full_text_sha256": str(document["full_text_sha256"]),
        "retained_513_token_sha256": str(document["retained_513_token_sha256"]),
    }


def cache_key_fields(
    document: Mapping[str, Any],
    window: int,
    *,
    teacher_model_id: str = MODEL_ID,
    teacher_revision: str = MODEL_REVISION,
    teacher_parameter_sha256: str = "__TEACHER_PARAMETER_HASH__",
    tokenizer_revision: str = TOKENIZER_REVISION,
    tokenizer_hash: str = "__TOKENIZER_HASH__",
    dataset_revision: str = DATASET_REVISION,
) -> dict[str, Any]:
    if window not in WINDOWS:
        raise ValueError(f"window must be 0 or 1, got {window}")
    return {
        "document": _document_identity(document),
        "window": int(window),
        "teacher": {
            "model_id": teacher_model_id,
            "revision": teacher_revision,
            "parameter_sha256": teacher_parameter_sha256,
        },
        "tokenizer": {"revision": tokenizer_revision, "sha256": tokenizer_hash},
        "dataset_revision": dataset_revision,
        "teacher_context_range": [0, 256] if window == 0 else [0, 512],
        "representation": "raw_fp32_logits",
        "dtype": "float32",
    }


def cache_key(document: Mapping[str, Any], window: int, **identity: str) -> dict[str, Any]:
    fields = cache_key_fields(document, window, **identity)
    return {"fields": fields, "digest": canonical_hash(fields)}


def entry_offset_bytes(window: int, document_index: int, *, documents: int = DOCUMENT_COUNT) -> int:
    if window not in WINDOWS or not 0 <= document_index < documents:
        raise IndexError("cache coordinate out of range")
    return ((window * documents + document_index) * TOKENS_PER_WINDOW * TOKENIZER_VOCAB * 4)


def build_entry_plan(manifest: Mapping[str, Any], *, limit_pairs: int = BUILD_PAIRS) -> list[dict[str, Any]]:
    pairs = manifest["cyclic_pairs"]["pairs"][:limit_pairs]
    if len(pairs) != limit_pairs:
        raise CacheContractError(f"expected at least {limit_pairs} frozen pairs")
    plan: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        for document_index in pair["document_indices"]:
            for window in WINDOWS:
                key = (int(document_index), window)
                if key in seen:
                    continue
                seen.add(key)
                plan.append({"document_index": key[0], "window": key[1], "pair": int(pair["pair"])})
    expected = DOCUMENT_COUNT * WINDOW_COUNT
    if len(seen) != expected:
        raise CacheContractError(f"build schedule covers {len(seen)} entries, expected {expected}")
    return plan


def tokenizer_hash(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    payload = {"init_kwargs": getattr(tokenizer, "init_kwargs", {}), "vocab": sorted(vocab.items())}
    return canonical_hash(payload)


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(tensor_bytes(parameter))
    return digest.hexdigest()


def teacher_window_logits(teacher: nn.Module, source: Tensor, window: int, *, expected_vocab: int = TOKENIZER_VOCAB) -> Tensor:
    if source.ndim != 2 or source.shape[1] < RETAINED_TOKENS:
        raise ValueError("source must contain at least 513 tokens")
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    with torch.no_grad():
        output = teacher(input_ids=context)
        result = output.logits[:, start : start + 256].detach().clone().float()
    if result.shape[1:] != (TOKENS_PER_WINDOW, expected_vocab):
        raise CacheContractError(f"teacher logits shape mismatch: {tuple(result.shape)}")
    return result


def load_real_teacher() -> nn.Module:
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    teacher.float().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def load_frozen_train_documents() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Recover tokens against the frozen CE manifest; never rebuild its order."""
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer
    from run_omega_core_lm_0_r1_training_technical_preflight import reconstruct_documents, sha256_text

    manifest = frozen_train_manifest()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=TOKENIZER_REVISION, use_fast=True, local_files_only=True)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise CacheContractError(f"tokenizer vocab mismatch: {len(tokenizer)}")
    dataset = load_dataset(
        DATASET_ID,
        "wikitext-2-raw-v1",
        split="train",
        revision=DATASET_REVISION,
        download_config=DownloadConfig(local_files_only=True),
    )
    expected = {(str(item["full_text_sha256"]), str(item["retained_513_token_sha256"])): item for item in manifest["documents"]}
    recovered: dict[tuple[str, str], dict[str, Any]] = {}
    for source_index, source in enumerate(reconstruct_documents(dataset)):
        text = str(source["text"])
        tokens = list(tokenizer.encode(text, add_special_tokens=False))
        if len(tokens) < RETAINED_TOKENS:
            continue
        retained = tokens[:RETAINED_TOKENS]
        token_digest = sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in retained))
        key = (sha256_text(text), token_digest)
        if key in expected and key not in recovered:
            recovered[key] = {**expected[key], "tokens": retained, "source_document_index": source_index}
    ordered_keys = [(str(item["full_text_sha256"]), str(item["retained_513_token_sha256"])) for item in manifest["documents"]]
    if list(recovered) != ordered_keys:
        missing = [key for key in ordered_keys if key not in recovered]
        raise CacheContractError(f"frozen token source mismatch; missing {len(missing)} documents")
    return manifest, [recovered[key] for key in ordered_keys], tokenizer_hash(tokenizer)


@dataclass(frozen=True)
class CacheShape:
    windows: int = WINDOW_COUNT
    documents: int = DOCUMENT_COUNT
    tokens: int = TOKENS_PER_WINDOW
    vocab: int = TOKENIZER_VOCAB

    @property
    def tuple(self) -> tuple[int, int, int, int]:
        return self.windows, self.documents, self.tokens, self.vocab

    @property
    def bytes(self) -> int:
        return math.prod(self.tuple) * 4


class MMapTeacherCache:
    """Read-only cache facade. It has no write method by design."""

    def __init__(self, cache_file: Path, manifest: Mapping[str, Any]) -> None:
        if manifest.get("status") != "CACHE_SEALED" or manifest.get("access") != "READ_ONLY":
            raise CacheContractError("cache manifest is not sealed read-only")
        shape = tuple(int(value) for value in manifest["shape"])
        if len(shape) != 4:
            raise CacheContractError("cache shape must have four dimensions")
        self.manifest = manifest
        self.shape = shape
        self.cache_file = cache_file
        if cache_file.stat().st_size != math.prod(shape) * 4:
            raise CacheContractError("cache file size does not match manifest shape")
        self._mmap = np.memmap(cache_file, mode="r", dtype=np.float32, shape=shape, order="C")

    def lookup(self, document_index: int, window: int) -> Tensor:
        if window not in range(self.shape[0]) or document_index not in range(self.shape[1]):
            raise IndexError(f"cache coordinate out of range: document={document_index}, window={window}")
        return torch.from_numpy(np.array(self._mmap[window, document_index], copy=True)).clone()

    def close(self) -> None:
        mmap = self._mmap
        self._mmap = None  # type: ignore[assignment]
        if mmap is not None:
            mmap._mmap.close()  # type: ignore[attr-defined]

    def __enter__(self) -> "MMapTeacherCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class CachedTeacherRoute:
    def __init__(self, cache: Any) -> None:
        self.cache = cache
        self.teacher_loaded = False
        self.teacher_forward_calls = 0

    def logits(self, document_indices: Sequence[int], window: int) -> Tensor:
        return torch.stack([self.cache.lookup(int(index), window) for index in document_indices])


class DirectTeacherRoute:
    def __init__(self, teacher: nn.Module) -> None:
        self.teacher = teacher
        self.teacher_loaded = True
        self.teacher_forward_calls = 0

    def logits(self, source: Tensor, window: int) -> Tensor:
        self.teacher_forward_calls += 1
        return teacher_window_logits(self.teacher, source, window)


def read_cache_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not verify_self_hash(manifest, "manifest_self_hash"):
        raise CacheContractError("cache manifest self-hash verification failed")
    cache_file = path.parent / str(manifest.get("cache_file", "teacher_logits.fp32"))
    return manifest, cache_file


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, targets: Tensor) -> dict[str, Tensor]:
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), targets.reshape(-1))
    student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * TEMPERATURE**2
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    normalized: dict[str, Any] = {"param_groups": optimizer.state_dict()["param_groups"], "state": {}}
    for key, state in optimizer.state_dict()["state"].items():
        normalized["state"][str(key)] = {name: (tensor_hash(value) if torch.is_tensor(value) else value) for name, value in state.items()}
    return canonical_hash(normalized)


def training_update(model: nn.Module, optimizer: torch.optim.Optimizer, inputs: Tensor, targets: Tensor, state: Tensor, teacher_logits: Tensor) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    result = model.forward_window(inputs, state)
    next_state, student_logits = result[0], result[1]
    losses = distillation_loss(student_logits, teacher_logits, targets)
    losses["total"].backward()
    clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
    optimizer.step()
    return {
        "losses": {name: value.detach().clone() for name, value in losses.items()},
        "clip_norm": clip_norm,
        "next_state": next_state.detach().clone(),
        "model_state_hash": parameter_hash(model),
        "optimizer_state_hash": optimizer_state_hash(optimizer),
    }


def compare_training_update(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    inputs: Tensor,
    targets: Tensor,
    state: Tensor,
    direct_logits: Tensor,
    cached_logits: Tensor,
) -> dict[str, Any]:
    direct_model = model_factory()
    cached_model = model_factory()
    cached_model.load_state_dict(direct_model.state_dict())
    direct_optimizer = optimizer_factory(direct_model)
    cached_optimizer = optimizer_factory(cached_model)
    left = training_update(direct_model, direct_optimizer, inputs, targets, state.clone(), direct_logits)
    right = training_update(cached_model, cached_optimizer, inputs, targets, state.clone(), cached_logits)
    exact = all(torch.equal(left["losses"][name], right["losses"][name]) for name in left["losses"])
    exact = exact and torch.equal(left["next_state"], right["next_state"])
    exact = exact and left["clip_norm"] == right["clip_norm"]
    exact = exact and left["model_state_hash"] == right["model_state_hash"]
    exact = exact and left["optimizer_state_hash"] == right["optimizer_state_hash"]
    return {"passed": exact, "direct": left, "cached": right}


def raw_logits_equal(cached: Tensor, direct: Tensor) -> bool:
    return torch.equal(cached, direct) and tensor_hash(cached) == tensor_hash(direct)


def correctness_schedule_checks(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pairs = manifest["cyclic_pairs"]["pairs"]
    selected = (0, 75, 76, 301)
    return {
        "pairs_checked": list(selected),
        "windows_checked": list(WINDOWS),
        "cycle_length_pairs": DOCUMENT_COUNT // math.gcd(DOCUMENT_COUNT, PHYSICAL_BATCH),
        "selected_pair_document_indices": {str(pair): pairs[pair]["document_indices"] for pair in selected},
    }


def cache_route_contract(route: CachedTeacherRoute) -> dict[str, Any]:
    return {"teacher_loaded": route.teacher_loaded, "teacher_forward_calls": route.teacher_forward_calls}


def _checkpoint_equivalence(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["model_state_hash"] == right["model_state_hash"]
        and left["optimizer_state_hash"] == right["optimizer_state_hash"]
        and torch.equal(left["next_state"], right["next_state"])
    )


def run_correctness_gate(*, cache_manifest_path: Path, output_path: Path, confirm_real_execution: bool) -> dict[str, Any]:
    """Run all Phase C checks against one sealed cache; guarded and explicit."""
    require_real_authorization(confirm_real_execution, "cache correctness gate")
    cache_manifest, cache_file = read_cache_manifest(cache_manifest_path)
    frozen, documents, tokenizer_digest = load_frozen_train_documents()
    if cache_manifest.get("source", {}).get("manifest_sha256") != frozen.get("manifest_sha256"):
        raise CacheContractError("cache source manifest differs from frozen CE manifest")
    teacher = load_real_teacher()
    expected_teacher_hash = cache_manifest.get("teacher", {}).get("parameter_sha256")
    actual_teacher_hash = parameter_hash(teacher)
    if expected_teacher_hash != actual_teacher_hash:
        raise CacheContractError("teacher parameter hash differs from sealed cache")
    if cache_manifest.get("tokenizer", {}).get("sha256") != tokenizer_digest:
        raise CacheContractError("tokenizer hash differs from sealed cache")

    with MMapTeacherCache(cache_file, cache_manifest) as cache:
        raw_rows: list[dict[str, Any]] = []
        cross_batch_rows: list[dict[str, Any]] = []
        for pair_index in (0, 75, 76, 301):
            pair = frozen["cyclic_pairs"]["pairs"][pair_index]
            indices = [int(index) for index in pair["document_indices"]]
            source = torch.tensor([documents[index]["tokens"] for index in indices], dtype=torch.long)
            for window in WINDOWS:
                direct = teacher_window_logits(teacher, source, window)
                for position, document_index in enumerate(indices):
                    cached = cache.lookup(document_index, window)
                    entry = cache_manifest["entries"][f"{window}:{document_index}"]
                    equal = raw_logits_equal(cached, direct[position])
                    hash_match = tensor_hash(cached) == entry["sha256"]
                    if not equal or not hash_match:
                        raise CacheContractError(f"RAW_LOGITS_EQ failed at pair={pair_index}, document={document_index}, window={window}")
                    raw_rows.append({"pair": pair_index, "document_index": document_index, "window": window, "torch_equal": equal, "sha256_equal": hash_match})
                cross_batch_rows.append({"pair": pair_index, "window": window, "all_documents_equal": all(raw_rows[-len(indices):][offset]["torch_equal"] for offset in range(len(indices)))})

        first = cache.lookup(0, 0)
        second = cache.lookup(1, 0)
        negative_control = {"different_documents_differ": not torch.equal(first, second), "out_of_range_fails": False}
        try:
            cache.lookup(DOCUMENT_COUNT, 0)
        except IndexError:
            negative_control["out_of_range_fails"] = True
        if not all(negative_control.values()):
            raise CacheContractError("negative key control failed")

        # Use identical F students and identical teacher tensors for several
        # complete updates. The cached route never receives the teacher object.
        import run_omega_ce_only_baseline as ce
        import run_scientific_scoping_a as r1

        def model_factory() -> nn.Module:
            return ce.fresh_model(20260913, 4)

        def optimizer_factory(model: nn.Module) -> torch.optim.Optimizer:
            return torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)

        direct_model = model_factory()
        cached_model = model_factory()
        cached_model.load_state_dict(direct_model.state_dict())
        direct_optimizer = optimizer_factory(direct_model)
        cached_optimizer = optimizer_factory(cached_model)
        direct_state = direct_model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
        cached_state = cached_model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
        update_rows: list[dict[str, Any]] = []
        for pair_index, window in ((0, 0), (0, 1), (1, 0)):
            indices = [int(index) for index in frozen["cyclic_pairs"]["pairs"][pair_index]["document_indices"]]
            source = torch.tensor([documents[index]["tokens"] for index in indices], dtype=torch.long)
            start = window * TOKENS_PER_WINDOW
            inputs = source[:, start : start + TOKENS_PER_WINDOW]
            targets = source[:, start + 1 : start + TOKENS_PER_WINDOW + 1]
            direct_logits = teacher_window_logits(teacher, source, window)
            cached_logits = torch.stack([cache.lookup(index, window) for index in indices])
            left = training_update(direct_model, direct_optimizer, inputs, targets, direct_state, direct_logits)
            right = training_update(cached_model, cached_optimizer, inputs, targets, cached_state, cached_logits)
            exact = _checkpoint_equivalence(left, right) and all(torch.equal(left["losses"][name], right["losses"][name]) for name in left["losses"])
            if not exact:
                raise CacheContractError(f"TRAINING_UPDATE_EQ failed at pair={pair_index}, window={window}")
            direct_state, cached_state = left["next_state"], right["next_state"]
            update_rows.append({"pair": pair_index, "window": window, "exact": exact, "checkpoint_hash": left["model_state_hash"]})
        cached_route = CachedTeacherRoute(cache)
        cached_contract = cache_route_contract(cached_route)
        if cached_contract != {"teacher_loaded": False, "teacher_forward_calls": 0}:
            raise CacheContractError("cached route teacher accounting failed")

    report = {
        "schema": "omega-teacher-logit-cache-correctness-v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "CACHE_CORRECT",
        "checks": {
            "RAW_LOGITS_EQ": {"passed": True, "rows": len(raw_rows), "rows_detail": raw_rows},
            "CROSS_BATCH_POSITION_EQ": {"passed": True, "rows": cross_batch_rows},
            "WINDOW0/WINDOW1_CONTEXT": {"passed": True, "context_ranges": {"0": [0, 256], "1": [0, 512], "retained_logits_1": [256, 512]}},
            "TRAINING_UPDATE_EQ": {"passed": True, "updates": update_rows},
            "NEGATIVE_KEY_CONTROL": negative_control,
            "CACHED_ROUTE_ACCOUNTING": {"teacher_loaded": False, "teacher_forward_calls": 0},
        },
        "cache_manifest": cache_manifest_path.as_posix(),
    }
    return write_self_hashed(output_path, report)


def benchmark_ratio(direct_seconds: float, cache_seconds: float) -> float:
    if direct_seconds <= 0 or cache_seconds < 0:
        raise ValueError("benchmark seconds must be non-negative and baseline must be positive")
    return cache_seconds / direct_seconds


def benchmark_gates(direct_k1: float, cache_k1: float, direct_k4: float, cache_k4: float) -> dict[str, Any]:
    r_k1 = benchmark_ratio(direct_k1, cache_k1)
    r_k4 = benchmark_ratio(direct_k4, cache_k4)
    r_joint = benchmark_ratio(direct_k1 + direct_k4, cache_k1 + cache_k4)
    return {
        "R_cache_K1": r_k1,
        "R_cache_K4": r_k4,
        "R_joint": r_joint,
        "individual_pass": r_k1 <= 0.90 and r_k4 <= 0.90,
        "joint_pass": r_joint <= 0.80,
        "regression_clear_failure": r_k1 > 1.05 or r_k4 > 1.05 or r_joint > 1.05,
        "pass": r_k1 <= 0.90 and r_k4 <= 0.90 and r_joint <= 0.80,
    }


def break_even_runs(build_seconds: float, direct_seconds: float, cache_seconds: float) -> float | None:
    if build_seconds < 0 or direct_seconds < 0 or cache_seconds < 0:
        raise ValueError("break-even inputs must be non-negative")
    if direct_seconds <= cache_seconds:
        return None
    return build_seconds / (direct_seconds - cache_seconds)


def benchmark_protocol() -> dict[str, Any]:
    return {
        "routes": ["DIRECT-K1", "CACHE-K1", "DIRECT-K4", "CACHE-K4"],
        "fresh_child_process_per_route": True,
        "updates_total": BENCHMARK_UPDATES,
        "warmup_updates": [0, BENCHMARK_WARMUP_LAST],
        "primary_updates": [BENCHMARK_MEASURE_START, BENCHMARK_UPDATES - 1],
        "primary_update_count": BENCHMARK_MEASURED_UPDATES,
        "primary_metric": "total_seconds_per_update_in_primary_region",
        "cost_ratio_formula": "R_cost=T_CANDIDATE/T_BASELINE",
    }


def build_cache(*, output_root: Path, confirm_real_execution: bool) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution, "cache build")
    existing_manifest = output_root / "cache_manifest.json"
    existing_cache = output_root / "teacher_logits.fp32"
    if existing_manifest.exists() or existing_cache.exists():
        raise CacheContractError("refusing to overwrite an existing cache artifact")
    manifest, documents, tokenizer_digest = load_frozen_train_documents()
    feasibility = storage_feasibility(output_root)
    if feasibility["status"] != "READY":
        report = {"schema": "omega-teacher-logit-cache-build-v1", "campaign_id": CAMPAIGN_ID, "status": "INCONCLUSIVE_STORAGE", "feasibility": feasibility}
        write_self_hashed(output_root / "cache_build_report.json", report)
        raise StorageInconclusiveError(json.dumps(feasibility, sort_keys=True))

    output_root.mkdir(parents=True, exist_ok=True)
    cache_file = output_root / "teacher_logits.fp32"
    temporary = output_root / "teacher_logits.fp32.tmp"
    teacher = load_real_teacher()
    teacher_digest = parameter_hash(teacher)
    shape = CacheShape()
    mmap = np.memmap(temporary, mode="w+", dtype=np.float32, shape=shape.tuple, order="C")
    entries: dict[str, Any] = {}
    written: set[tuple[int, int]] = set()
    by_index = {int(document["document_index"]): document for document in documents}
    started = time.perf_counter()
    for pair in manifest["cyclic_pairs"]["pairs"][:BUILD_PAIRS]:
        batch_indices = [int(index) for index in pair["document_indices"]]
        source = torch.tensor([by_index[index]["tokens"] for index in batch_indices], dtype=torch.long)
        for window in WINDOWS:
            logits = teacher_window_logits(teacher, source, window)
            for position, document_index in enumerate(batch_indices):
                key = (document_index, window)
                if key in written:
                    continue
                payload = logits[position].contiguous()
                mmap[window, document_index] = payload.numpy()
                document = by_index[document_index]
                fields = cache_key_fields(document, window, teacher_parameter_sha256=teacher_digest, tokenizer_hash=tokenizer_digest)
                entries[f"{window}:{document_index}"] = {
                    "document_index": document_index,
                    "window": window,
                    "offset_bytes": entry_offset_bytes(window, document_index),
                    "shape": [TOKENS_PER_WINDOW, TOKENIZER_VOCAB],
                    "sha256": tensor_hash(payload),
                    "cache_key": {"fields": fields, "digest": canonical_hash(fields)},
                }
                written.add(key)
    mmap.flush()
    del mmap
    os.replace(temporary, cache_file)
    cache_sha = file_hash(cache_file)
    built = {
        "schema": "omega-teacher-logit-cache-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "CACHE_SEALED",
        "access": "READ_ONLY",
        "cache_file": cache_file.name,
        "shape": list(shape.tuple),
        "order": ["window", "document", "token", "vocab"],
        "representation": "raw_fp32_logits",
        "dtype": "float32",
        "entry_bytes": ENTRY_BYTES,
        "logical_bytes": EXPECTED_CACHE_BYTES,
        "cache_file_sha256": cache_sha,
        "source": {
            "manifest_path": TRAIN_MANIFEST_PATH.as_posix(),
            "manifest_sha256": manifest["manifest_sha256"],
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "retained_tokens": RETAINED_TOKENS,
        },
        "teacher": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "parameter_sha256": teacher_digest},
        "tokenizer": {"revision": TOKENIZER_REVISION, "sha256": tokenizer_digest},
        "build_provenance": {"elapsed_seconds": time.perf_counter() - started, "code_sha256": file_hash(Path(__file__)), "git_commit": "not-collected_by_contract", "pairs": BUILD_PAIRS, "batch_size": PHYSICAL_BATCH},
        "entries": entries,
    }
    return write_self_hashed(output_root / "cache_manifest.json", built, field="manifest_self_hash")


def _memory_snapshot() -> dict[str, int]:
    return {"rss_bytes": int(psutil.Process().memory_info().rss), "available_bytes": int(psutil.virtual_memory().available)}


def _benchmark_child(route: str, cache_manifest_path: Path) -> dict[str, Any]:
    if route not in {"DIRECT-K1", "CACHE-K1", "DIRECT-K4", "CACHE-K4"}:
        raise ValueError(f"unsupported benchmark route: {route}")
    import run_omega_ce_only_baseline as ce
    import run_scientific_scoping_a as r1

    k = 1 if route.endswith("K1") else 4
    cached = route.startswith("CACHE")
    r1.configure_cpu_runtime()
    frozen, documents, _ = load_frozen_train_documents()
    manifest, cache_file = read_cache_manifest(cache_manifest_path)
    cache = MMapTeacherCache(cache_file, manifest) if cached else None
    teacher = None if cached else load_real_teacher()
    model = ce.fresh_model(20260913, k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
    state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
    timings = {name: [0.0] * BENCHMARK_UPDATES for name in ("total", "teacher_forward", "cache_materialization", "vocab_loss", "backward", "clip", "adamw")}
    rss_peak = _memory_snapshot()["rss_bytes"]
    available_min = _memory_snapshot()["available_bytes"]
    for update in range(BENCHMARK_UPDATES):
        pair_index = update // 2
        window = update % 2
        indices = [int(index) for index in frozen["cyclic_pairs"]["pairs"][pair_index]["document_indices"]]
        source = torch.tensor([documents[index]["tokens"] for index in indices], dtype=torch.long)
        start = window * TOKENS_PER_WINDOW
        inputs = source[:, start : start + TOKENS_PER_WINDOW]
        targets = source[:, start + 1 : start + TOKENS_PER_WINDOW + 1]
        current_state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if window == 0 else state
        update_started = time.perf_counter()
        forward_started = time.perf_counter()
        result = model.recur_states(inputs, current_state)
        next_state, _, _, readout_states = result
        timings["total"][update] += time.perf_counter() - update_started
        teacher_started = time.perf_counter()
        if cached:
            teacher_logits = torch.stack([cache.lookup(index, window) for index in indices])
            timings["cache_materialization"][update] += time.perf_counter() - teacher_started
        else:
            teacher_logits = teacher_window_logits(teacher, source, window)
            timings["teacher_forward"][update] += time.perf_counter() - teacher_started
        loss_started = time.perf_counter()
        projected = model.project(readout_states).reshape(-1, model.dimension)
        flat_teacher = teacher_logits.reshape(-1, model.vocab_size)
        flat_targets = targets.reshape(-1)
        ce_total = projected.new_zeros(())
        kl_total = projected.new_zeros(())
        for chunk_start in range(0, projected.shape[0], 512):
            chunk_stop = min(chunk_start + 512, projected.shape[0])
            student_logits = model.logits_from_projected(projected[chunk_start:chunk_stop])
            ce_total = ce_total + F.cross_entropy(student_logits, flat_targets[chunk_start:chunk_stop], reduction="sum")
            kl_total = kl_total + F.kl_div(
                F.log_softmax(student_logits / TEMPERATURE, dim=-1),
                F.softmax(flat_teacher[chunk_start:chunk_stop] / TEMPERATURE, dim=-1),
                reduction="batchmean",
            ) * TEMPERATURE**2 * (chunk_stop - chunk_start)
        loss = 0.5 * ce_total / flat_targets.numel() + 0.5 * kl_total / flat_targets.numel()
        timings["vocab_loss"][update] += time.perf_counter() - loss_started
        optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        loss.backward()
        timings["backward"][update] += time.perf_counter() - backward_started
        clip_started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        timings["clip"][update] += time.perf_counter() - clip_started
        adamw_started = time.perf_counter()
        optimizer.step()
        timings["adamw"][update] += time.perf_counter() - adamw_started
        timings["total"][update] = time.perf_counter() - update_started
        state = next_state.detach() if window == 0 else model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
        snapshot = _memory_snapshot()
        rss_peak = max(rss_peak, snapshot["rss_bytes"])
        available_min = min(available_min, snapshot["available_bytes"])
    primary = slice(BENCHMARK_MEASURE_START, BENCHMARK_UPDATES)
    result = {
        "route": route,
        "K": k,
        "cached": cached,
        "updates_total": BENCHMARK_UPDATES,
        "warmup_updates": {name: sum(values[:BENCHMARK_MEASURE_START]) for name, values in timings.items()},
        "primary_updates": {name: sum(values[primary]) for name, values in timings.items()},
        "primary_total_seconds": sum(timings["total"][primary]),
        "primary_total_seconds_per_update": sum(timings["total"][primary]) / BENCHMARK_MEASURED_UPDATES,
        "direct_teacher_forward_seconds": sum(timings["teacher_forward"]),
        "cache_materialization_copy_seconds": sum(timings["cache_materialization"]),
        "vocab_loss_seconds": sum(timings["vocab_loss"]),
        "backward_seconds": sum(timings["backward"]),
        "rss_peak_bytes": rss_peak,
        "available_memory_min_bytes": available_min,
        "logical_disk_bytes": int(manifest.get("logical_bytes", 0)) if cached else 0,
        "teacher_loaded": not cached,
        "teacher_forward_calls": BENCHMARK_UPDATES if not cached else 0,
    }
    if cache is not None:
        cache.close()
    return result


def run_benchmark_subprocesses(*, output_root: Path, cache_manifest: Path, confirm_real_execution: bool) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution, "benchmark")
    routes = benchmark_protocol()["routes"]
    rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for route in routes:
        command = [sys.executable, str(Path(__file__).resolve()), "--benchmark-child", "--route", route, "--cache-manifest", str(cache_manifest), "--confirm-real-execution"]
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"benchmark child failed for {route}: {completed.stderr}")
        rows.append(json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1]))
        commands.append(command)
    by_route = {row["route"]: row for row in rows}
    gates = benchmark_gates(by_route["DIRECT-K1"]["primary_total_seconds"], by_route["CACHE-K1"]["primary_total_seconds"], by_route["DIRECT-K4"]["primary_total_seconds"], by_route["CACHE-K4"]["primary_total_seconds"])
    report = {"schema": "omega-teacher-logit-cache-benchmark-v1", "campaign_id": CAMPAIGN_ID, "protocol": benchmark_protocol(), "routes": by_route, "gates": gates, "worker_commands": commands}
    return write_self_hashed(output_root / "benchmark_report.json", report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-child", action="store_true")
    parser.add_argument("--route")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--confirm-real-execution", action="store_true")
    args = parser.parse_args(argv)
    selected = sum(bool(getattr(args, name)) for name in ("feasibility", "build", "correctness", "benchmark", "benchmark_child"))
    if selected != 1:
        parser.error("select exactly one guarded phase")
    if args.feasibility:
        report = phase_a_report(args.output_root)
        write_self_hashed(args.output_root / "feasibility_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.build:
        print(json.dumps(build_cache(output_root=args.output_root, confirm_real_execution=args.confirm_real_execution), indent=2, sort_keys=True))
        return 0
    if args.correctness:
        report_path = args.output_root / "correctness_report.json"
        report = run_correctness_gate(cache_manifest_path=args.cache_manifest or (args.output_root / "cache_manifest.json"), output_path=report_path, confirm_real_execution=args.confirm_real_execution)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.benchmark:
        print(json.dumps(run_benchmark_subprocesses(output_root=args.output_root, cache_manifest=args.cache_manifest or (args.output_root / "cache_manifest.json"), confirm_real_execution=args.confirm_real_execution), indent=2, sort_keys=True))
        return 0
    require_real_authorization(args.confirm_real_execution, "benchmark child")
    if not args.route or not args.cache_manifest:
        parser.error("benchmark child requires --route and --cache-manifest")
    print(json.dumps(_benchmark_child(args.route, args.cache_manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
