"""OMEGA hidden-state cache probe, phases A-D, guarded real execution.

This unit reuses frozen-document, student, loss, self-hash, and schedule
helpers from the completed teacher-logit-cache unit. Only cached representation
and benchmark route differ: hidden states are preloaded into RAM on NVMe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
OLD_DIR = CAMPAIGN_ROOT / "omega_teacher_logit_cache"
if str(OLD_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_DIR))
import run_omega_teacher_logit_cache as base  # noqa: E402


CAMPAIGN_ID = "OMEGA-TEACHER-HIDDEN-CACHE-PROBE"
DEFAULT_OUTPUT_ROOT = HERE / "results"
DEFAULT_BACKING_STORE = Path(r"C:\omega_cache\teacher_hidden.fp32")
HIDDEN_SIZE = 768
ENTRY_BYTES = base.TOKENS_PER_WINDOW * HIDDEN_SIZE * 4
EXPECTED_CACHE_BYTES = base.WINDOW_COUNT * base.DOCUMENT_COUNT * ENTRY_BYTES
WINDOWS = base.WINDOWS
BUILD_PAIRS = base.BUILD_PAIRS
BENCHMARK_UPDATES = base.BENCHMARK_UPDATES
BENCHMARK_MEASURE_START = base.BENCHMARK_MEASURE_START
BENCHMARK_MEASURED_UPDATES = base.BENCHMARK_MEASURED_UPDATES
TEMPERATURE = base.TEMPERATURE
BASE_LR = base.BASE_LR
ADAMW_BETAS = base.ADAMW_BETAS
ADAMW_EPS = base.ADAMW_EPS
WEIGHT_DECAY = base.WEIGHT_DECAY
CLIP_NORM = base.CLIP_NORM
PHYSICAL_BATCH = base.PHYSICAL_BATCH
SELF_HASH_PLACEHOLDER = base.SELF_HASH_PLACEHOLDER


class HiddenCacheContractError(RuntimeError):
    pass


class RealExecutionAuthorizationError(RuntimeError):
    pass


def require_real_authorization(confirmed: bool, operation: str) -> None:
    if not confirmed:
        raise RealExecutionAuthorizationError(f"{operation} requires --confirm-real-execution")


def tensor_bytes(value: Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def tensor_hash(value: Tensor) -> str:
    return base.sha256_bytes(tensor_bytes(value))


def write_self_hashed(path: Path, payload: Mapping[str, Any], field: str = "report_self_hash") -> dict[str, Any]:
    return base.write_self_hashed(path, payload, field)


def verify_self_hash(value: Mapping[str, Any], field: str = "report_self_hash") -> bool:
    return base.verify_self_hash(value, field)


def storage_feasibility(path: Path, *, expected_bytes: int = EXPECTED_CACHE_BYTES, margin: float = base.STORAGE_SAFETY_MARGIN) -> dict[str, Any]:
    return base.storage_feasibility(path, expected_bytes=expected_bytes, margin=margin)


def phase_a_report(path: Path) -> dict[str, Any]:
    feasibility = storage_feasibility(path)
    return {
        "schema": "omega-teacher-hidden-cache-feasibility-v1",
        "campaign_id": CAMPAIGN_ID,
        "phase": "A",
        "layout": {"windows": 2, "documents": 602, "tokens_per_window": 256, "hidden_size": HIDDEN_SIZE, "order": ["window", "document", "token", "hidden"], "representation": "raw_fp32_hidden"},
        "entry_bytes": ENTRY_BYTES,
        "expected_entries": 1204,
        "expected_cache_bytes": EXPECTED_CACHE_BYTES,
        "backing_store": DEFAULT_BACKING_STORE.as_posix(),
        "feasibility": feasibility,
    }


def hidden_key_fields(document: Mapping[str, Any], window: int, **identity: str) -> dict[str, Any]:
    fields = base.cache_key_fields(document, window, **identity)
    fields["representation"] = "raw_fp32_hidden"
    fields["hidden_size"] = HIDDEN_SIZE
    return fields


def hidden_cache_key(document: Mapping[str, Any], window: int, **identity: str) -> dict[str, Any]:
    fields = hidden_key_fields(document, window, **identity)
    return {"fields": fields, "digest": base.canonical_hash(fields)}


def hidden_offset_bytes(window: int, document_position: int) -> int:
    if window not in WINDOWS or not 0 <= document_position < base.DOCUMENT_COUNT:
        raise IndexError("hidden cache coordinate out of range")
    return ((window * base.DOCUMENT_COUNT + document_position) * base.TOKENS_PER_WINDOW * HIDDEN_SIZE * 4)


def build_entry_plan(manifest: Mapping[str, Any], *, limit_pairs: int = BUILD_PAIRS) -> list[dict[str, Any]]:
    return base.build_entry_plan(manifest, limit_pairs=limit_pairs)


def hidden_window_states(teacher: nn.Module, source: Tensor, window: int, *, expected_hidden: int = HIDDEN_SIZE) -> Tensor:
    if source.ndim != 2 or source.shape[1] < base.RETAINED_TOKENS:
        raise ValueError("source must contain at least 513 tokens")
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    transformer = getattr(teacher, "transformer", None)
    if transformer is None:
        raise HiddenCacheContractError("teacher has no transformer pre-lm_head route")
    with torch.no_grad():
        output = transformer(input_ids=context)
        hidden = output.last_hidden_state[:, start : start + 256].detach().clone().float()
    if hidden.shape[1:] != (base.TOKENS_PER_WINDOW, expected_hidden):
        raise HiddenCacheContractError(f"hidden shape mismatch: {tuple(hidden.shape)}")
    return hidden


def teacher_direct_logits(teacher: nn.Module, source: Tensor, window: int) -> Tensor:
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    with torch.no_grad():
        output = teacher(input_ids=context)
        return output.logits[:, start : start + 256].detach().clone().float()


def lm_head_parameters(teacher: nn.Module) -> tuple[Tensor, Tensor | None]:
    head = getattr(teacher, "lm_head", None)
    if head is None or not hasattr(head, "weight"):
        raise HiddenCacheContractError("teacher has no lm_head weight")
    weight = head.weight.detach().cpu().contiguous().float().clone()
    bias = getattr(head, "bias", None)
    return weight, None if bias is None else bias.detach().cpu().contiguous().float().clone()


def hidden_to_logits(hidden: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return F.linear(hidden, weight, bias)


def save_lm_head(root: Path, weight: Tensor, bias: Tensor | None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "lm_head.weight.fp32"
    weight_path.write_bytes(tensor_bytes(weight))
    result: dict[str, Any] = {"weight_file": weight_path.name, "weight_shape": list(weight.shape), "weight_sha256": base.file_hash(weight_path), "dtype": "float32"}
    if bias is not None:
        bias_path = root / "lm_head.bias.fp32"
        bias_path.write_bytes(tensor_bytes(bias))
        result.update({"bias_file": bias_path.name, "bias_shape": list(bias.shape), "bias_sha256": base.file_hash(bias_path)})
    return result


def load_lm_head(root: Path, metadata: Mapping[str, Any]) -> tuple[Tensor, Tensor | None]:
    weight_path = root / str(metadata["weight_file"])
    if base.file_hash(weight_path) != metadata["weight_sha256"]:
        raise HiddenCacheContractError("lm_head weight hash mismatch")
    weight = torch.from_numpy(np.fromfile(weight_path, dtype=np.float32).reshape(tuple(metadata["weight_shape"]))).clone()
    bias = None
    if metadata.get("bias_file"):
        bias_path = root / str(metadata["bias_file"])
        if base.file_hash(bias_path) != metadata["bias_sha256"]:
            raise HiddenCacheContractError("lm_head bias hash mismatch")
        bias = torch.from_numpy(np.fromfile(bias_path, dtype=np.float32).reshape(tuple(metadata["bias_shape"]))).clone()
    return weight, bias


class HiddenMMapCache:
    def __init__(self, cache_file: Path, manifest: Mapping[str, Any]) -> None:
        if manifest.get("status") != "CACHE_SEALED" or manifest.get("access") != "READ_ONLY":
            raise HiddenCacheContractError("hidden cache is not sealed read-only")
        self.shape = tuple(int(value) for value in manifest["shape"])
        if self.shape != (2, 602, 256, HIDDEN_SIZE) or cache_file.stat().st_size != math.prod(self.shape) * 4:
            raise HiddenCacheContractError("hidden mmap shape/size mismatch")
        self._mmap = np.memmap(cache_file, mode="r", dtype=np.float32, shape=self.shape, order="C")

    def lookup(self, document_position: int, window: int) -> Tensor:
        if window not in range(self.shape[0]) or document_position not in range(self.shape[1]):
            raise IndexError("hidden cache coordinate out of range")
        return torch.from_numpy(np.array(self._mmap[window, document_position], copy=True)).clone()

    def close(self) -> None:
        mmap = self._mmap
        self._mmap = None  # type: ignore[assignment]
        if mmap is not None:
            mmap._mmap.close()  # type: ignore[attr-defined]

    def __enter__(self) -> "HiddenMMapCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def preload_hidden_cache(cache_file: Path, manifest: Mapping[str, Any]) -> tuple[np.ndarray, float, str]:
    started = time.perf_counter()
    payload = np.fromfile(cache_file, dtype=np.float32)
    expected = math.prod(tuple(int(value) for value in manifest["shape"]))
    if payload.size != expected:
        raise HiddenCacheContractError("preloaded hidden cache size mismatch")
    digest = base.sha256_bytes(payload.tobytes())
    if digest != manifest["cache_file_sha256"]:
        raise HiddenCacheContractError("preloaded hidden cache hash mismatch")
    return payload.reshape(tuple(int(value) for value in manifest["shape"])), time.perf_counter() - started, digest


def hidden_cache_load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not verify_self_hash(manifest, "manifest_self_hash"):
        raise HiddenCacheContractError("hidden manifest self-hash failed")
    return manifest, Path(str(manifest["cache_file"]))


def benchmark_ratio(direct_seconds: float, hidden_seconds: float) -> float:
    if direct_seconds <= 0 or hidden_seconds < 0:
        raise ValueError("invalid benchmark timing")
    return hidden_seconds / direct_seconds


def benchmark_gates(direct_k1: float, hidden_k1: float, direct_k4: float, hidden_k4: float) -> dict[str, Any]:
    r1 = benchmark_ratio(direct_k1, hidden_k1)
    r4 = benchmark_ratio(direct_k4, hidden_k4)
    joint = benchmark_ratio(direct_k1 + direct_k4, hidden_k1 + hidden_k4)
    return {"R_hidden_K1": r1, "R_hidden_K4": r4, "R_joint": joint, "individual_pass": r1 <= 0.90 and r4 <= 0.90, "joint_pass": joint <= 0.80, "pass": r1 <= 0.90 and r4 <= 0.90 and joint <= 0.80}


def benchmark_protocol() -> dict[str, Any]:
    return {"routes": ["DIRECT-K1", "DIRECT-K4", "HIDDEN-RAM-K1", "HIDDEN-RAM-K4"], "fresh_child_process_per_route": True, "updates_total": 152, "warmup_updates": [0, 39], "primary_updates": [40, 151], "primary_update_count": 112, "primary_metric": "total_seconds_per_update_in_primary_region", "cost_ratio_formula": "R_hidden=T_HIDDEN_RAM/T_DIRECT"}


def build_hidden_cache(*, output_root: Path, backing_store: Path, confirm_real_execution: bool) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution, "hidden cache build")
    manifest_path = output_root / "cache_manifest.json"
    if manifest_path.exists() or backing_store.exists():
        raise HiddenCacheContractError("refusing to overwrite existing hidden cache artifact")
    frozen, documents, tokenizer_digest = base.load_frozen_train_documents()
    feasibility = storage_feasibility(backing_store.parent)
    if feasibility["status"] != "READY":
        report = {"schema": "omega-teacher-hidden-cache-build-v1", "campaign_id": CAMPAIGN_ID, "status": "INCONCLUSIVE_STORAGE", "feasibility": feasibility}
        write_self_hashed(output_root / "cache_build_report.json", report)
        raise HiddenCacheContractError("INCONCLUSIVE_STORAGE")
    output_root.mkdir(parents=True, exist_ok=True)
    backing_store.parent.mkdir(parents=True, exist_ok=True)
    temporary = backing_store.with_suffix(backing_store.suffix + ".tmp")
    teacher = base.load_real_teacher()
    teacher_hash = base.parameter_hash(teacher)
    weight, bias = lm_head_parameters(teacher)
    lm_head_metadata = save_lm_head(output_root, weight, bias)
    mmap = np.memmap(temporary, mode="w+", dtype=np.float32, shape=(2, 602, 256, HIDDEN_SIZE), order="C")
    entries: dict[str, Any] = {}
    written: set[tuple[int, int]] = set()
    started = time.perf_counter()
    for pair in frozen["cyclic_pairs"]["pairs"][:BUILD_PAIRS]:
        positions = [int(index) for index in pair["document_indices"]]
        batch_documents = base.scheduled_documents(documents, pair)
        source = torch.tensor([document["tokens"] for document in batch_documents], dtype=torch.long)
        for window in WINDOWS:
            hidden = hidden_window_states(teacher, source, window)
            for batch_position, document_position in enumerate(positions):
                if (document_position, window) in written:
                    continue
                payload = hidden[batch_position].contiguous()
                mmap[window, document_position] = payload.numpy()
                document = batch_documents[batch_position]
                fields = hidden_key_fields(document, window, teacher_parameter_sha256=teacher_hash, tokenizer_hash=tokenizer_digest)
                entries[f"{window}:{document_position}"] = {"document_index": document_position, "source_document_index": int(document["document_index"]), "window": window, "offset_bytes": hidden_offset_bytes(window, document_position), "shape": [256, HIDDEN_SIZE], "sha256": tensor_hash(payload), "cache_key": {"fields": fields, "digest": base.canonical_hash(fields)}}
                written.add((document_position, window))
    mmap.flush()
    del mmap
    os.replace(temporary, backing_store)
    cache_hash = base.file_hash(backing_store)
    result = {"schema": "omega-teacher-hidden-cache-manifest-v1", "campaign_id": CAMPAIGN_ID, "status": "CACHE_SEALED", "access": "READ_ONLY", "cache_file": backing_store.resolve().as_posix(), "shape": [2, 602, 256, HIDDEN_SIZE], "order": ["window", "document", "token", "hidden"], "representation": "raw_fp32_hidden", "dtype": "float32", "entry_bytes": ENTRY_BYTES, "logical_bytes": EXPECTED_CACHE_BYTES, "cache_file_sha256": cache_hash, "source": {"manifest_sha256": frozen["manifest_sha256"], "dataset_id": base.DATASET_ID, "dataset_revision": base.DATASET_REVISION, "retained_tokens": base.RETAINED_TOKENS}, "teacher": {"model_id": base.MODEL_ID, "revision": base.MODEL_REVISION, "parameter_sha256": teacher_hash}, "tokenizer": {"revision": base.TOKENIZER_REVISION, "sha256": tokenizer_digest}, "lm_head": lm_head_metadata, "build_provenance": {"elapsed_seconds": time.perf_counter() - started, "pairs": BUILD_PAIRS, "batch_size": PHYSICAL_BATCH, "code_sha256": base.file_hash(Path(__file__)), "git_commit": "not-collected_by_contract"}, "entries": entries}
    return write_self_hashed(output_root / "cache_manifest.json", result, field="manifest_self_hash")


def run_correctness_gate(*, cache_manifest_path: Path, output_path: Path, confirm_real_execution: bool) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution, "hidden cache correctness gate")
    manifest, cache_file = hidden_cache_load_manifest(cache_manifest_path)
    frozen, documents, tokenizer_digest = base.load_frozen_train_documents()
    teacher = base.load_real_teacher()
    weight, bias = load_lm_head(cache_manifest_path.parent, manifest["lm_head"])
    if manifest["source"]["manifest_sha256"] != frozen["manifest_sha256"] or manifest["tokenizer"]["sha256"] != tokenizer_digest or manifest["teacher"]["parameter_sha256"] != base.parameter_hash(teacher):
        raise HiddenCacheContractError("sealed identity mismatch")
    with HiddenMMapCache(cache_file, manifest) as cache:
        raw_rows = []
        cross_rows = []
        for pair_index in (0, 75, 76, 301):
            pair = frozen["cyclic_pairs"]["pairs"][pair_index]
            positions = [int(index) for index in pair["document_indices"]]
            source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
            for window in WINDOWS:
                direct = teacher_direct_logits(teacher, source, window)
                equal_rows = []
                for batch_position, document_position in enumerate(positions):
                    hidden = cache.lookup(document_position, window)
                    reconstructed = hidden_to_logits(hidden, weight, bias)
                    exact = torch.equal(reconstructed, direct[batch_position]) and tensor_hash(reconstructed) == tensor_hash(direct[batch_position])
                    entry = manifest["entries"][f"{window}:{document_position}"]
                    hidden_hash_ok = tensor_hash(hidden) == entry["sha256"]
                    if not exact or not hidden_hash_ok:
                        raise HiddenCacheContractError(f"RAW_LOGITS_EQ failed pair={pair_index}, position={document_position}, window={window}")
                    equal_rows.append(exact)
                    raw_rows.append({"pair": pair_index, "document_index": document_position, "window": window, "torch_equal": exact, "hidden_sha256_equal": hidden_hash_ok})
                cross_rows.append({"pair": pair_index, "window": window, "all_documents_equal": all(equal_rows)})
        negative = {"different_documents_differ": not torch.equal(cache.lookup(0, 0), cache.lookup(1, 0)), "out_of_range_fails": False}
        try:
            cache.lookup(602, 0)
        except IndexError:
            negative["out_of_range_fails"] = True
        if not all(negative.values()):
            raise HiddenCacheContractError("negative key control failed")

        import run_omega_ce_only_baseline as ce
        def model_factory() -> nn.Module:
            return ce.fresh_model(20260913, 4)
        def optimizer_factory(model: nn.Module) -> torch.optim.Optimizer:
            return torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
        left_model, right_model = model_factory(), model_factory()
        right_model.load_state_dict(left_model.state_dict())
        left_optimizer, right_optimizer = optimizer_factory(left_model), optimizer_factory(right_model)
        left_state = left_model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
        right_state = right_model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
        updates = []
        for pair_index, window in ((0, 0), (0, 1), (1, 0)):
            positions = [int(index) for index in frozen["cyclic_pairs"]["pairs"][pair_index]["document_indices"]]
            source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
            start = window * 256
            inputs, targets = source[:, start : start + 256], source[:, start + 1 : start + 257]
            direct = teacher_direct_logits(teacher, source, window)
            cached = torch.stack([hidden_to_logits(cache.lookup(position, window), weight, bias) for position in positions])
            a = base.training_update(left_model, left_optimizer, inputs, targets, left_state, direct)
            b = base.training_update(right_model, right_optimizer, inputs, targets, right_state, cached)
            exact = all(torch.equal(a["losses"][key], b["losses"][key]) for key in a["losses"]) and torch.equal(a["next_state"], b["next_state"]) and a["clip_norm"] == b["clip_norm"] and a["model_state_hash"] == b["model_state_hash"] and a["optimizer_state_hash"] == b["optimizer_state_hash"]
            if not exact:
                raise HiddenCacheContractError("TRAINING_UPDATE_EQ failed")
            left_state, right_state = a["next_state"], b["next_state"]
            updates.append({"pair": pair_index, "window": window, "exact": exact, "checkpoint_hash": a["model_state_hash"]})
    report = {"schema": "omega-teacher-hidden-cache-correctness-v1", "campaign_id": CAMPAIGN_ID, "status": "CACHE_CORRECT", "checks": {"RAW_LOGITS_EQ": {"passed": True, "rows": len(raw_rows), "rows_detail": raw_rows}, "CROSS_BATCH_POSITION_EQ": {"passed": True, "rows": cross_rows}, "WINDOW0/WINDOW1_CONTEXT": {"passed": True, "context_ranges": {"0": [0, 256], "1": [0, 512], "retained_hidden_1": [256, 512]}}, "TRAINING_UPDATE_EQ": {"passed": True, "updates": updates}, "NEGATIVE_KEY_CONTROL": negative}, "cache_manifest": cache_manifest_path.as_posix()}
    return write_self_hashed(output_path, report)


def _memory_snapshot() -> dict[str, int]:
    return {"rss_bytes": int(psutil.Process().memory_info().rss), "available_bytes": int(psutil.virtual_memory().available)}


def _benchmark_child(route: str, manifest_path: Path) -> dict[str, Any]:
    import run_omega_ce_only_baseline as ce
    frozen, documents, _ = base.load_frozen_train_documents()
    manifest, cache_file = hidden_cache_load_manifest(manifest_path)
    hidden_route = route.startswith("HIDDEN-RAM")
    k = 1 if route.endswith("K1") else 4
    teacher = None if hidden_route else base.load_real_teacher()
    weight, bias = load_lm_head(manifest_path.parent, manifest["lm_head"])
    preloaded = None
    cache_load_seconds = 0.0
    if hidden_route:
        preloaded, cache_load_seconds, _ = preload_hidden_cache(cache_file, manifest)
    model = ce.fresh_model(20260913, k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY)
    state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
    times = {name: [0.0] * BENCHMARK_UPDATES for name in ("total", "student_forward", "teacher_transformer_forward", "teacher_lm_head", "hidden_ram_read", "vocab_loss", "backward", "clip", "adamw")}
    rss_peak = _memory_snapshot()["rss_bytes"]
    available_min = _memory_snapshot()["available_bytes"]
    for update in range(BENCHMARK_UPDATES):
        pair = frozen["cyclic_pairs"]["pairs"][update // 2]
        positions = [int(index) for index in pair["document_indices"]]
        source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
        window = update % 2
        start = window * 256
        inputs, targets = source[:, start : start + 256], source[:, start + 1 : start + 257]
        current_state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if window == 0 else state
        total_started = time.perf_counter()
        student_started = time.perf_counter()
        result = model.recur_states(inputs, current_state)
        next_state, _, _, readout = result
        times["student_forward"][update] = time.perf_counter() - student_started
        if hidden_route:
            read_started = time.perf_counter()
            hidden = torch.from_numpy(np.array(preloaded[window, positions], copy=True)).float()  # type: ignore[index]
            times["hidden_ram_read"][update] = time.perf_counter() - read_started
            head_started = time.perf_counter()
            teacher_logits = hidden_to_logits(hidden, weight, bias)
            times["teacher_lm_head"][update] = time.perf_counter() - head_started
        else:
            context = source[:, :256] if window == 0 else source[:, :512]
            transformer_started = time.perf_counter()
            with torch.no_grad():
                hidden_all = teacher.transformer(input_ids=context).last_hidden_state
            times["teacher_transformer_forward"][update] = time.perf_counter() - transformer_started
            head_started = time.perf_counter()
            teacher_logits = teacher.lm_head(hidden_all[:, 0 if window == 0 else 256 : (256 if window == 0 else 512)])
            times["teacher_lm_head"][update] = time.perf_counter() - head_started
        loss_started = time.perf_counter()
        projected = model.project(readout).reshape(-1, model.dimension)
        flat_teacher, flat_targets = teacher_logits.reshape(-1, model.vocab_size), targets.reshape(-1)
        ce_total = projected.new_zeros(()); kl_total = projected.new_zeros(())
        for chunk_start in range(0, projected.shape[0], 512):
            stop = min(chunk_start + 512, projected.shape[0])
            student_logits = model.logits_from_projected(projected[chunk_start:stop])
            ce_total = ce_total + F.cross_entropy(student_logits, flat_targets[chunk_start:stop], reduction="sum")
            kl_total = kl_total + F.kl_div(F.log_softmax(student_logits / TEMPERATURE, dim=-1), F.softmax(flat_teacher[chunk_start:stop] / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2 * (stop - chunk_start)
        loss = 0.5 * ce_total / flat_targets.numel() + 0.5 * kl_total / flat_targets.numel()
        times["vocab_loss"][update] = time.perf_counter() - loss_started
        optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter(); loss.backward(); times["backward"][update] = time.perf_counter() - backward_started
        clip_started = time.perf_counter(); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM); times["clip"][update] = time.perf_counter() - clip_started
        adam_started = time.perf_counter(); optimizer.step(); times["adamw"][update] = time.perf_counter() - adam_started
        times["total"][update] = time.perf_counter() - total_started
        state = next_state.detach() if window == 0 else model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
        snapshot = _memory_snapshot(); rss_peak = max(rss_peak, snapshot["rss_bytes"]); available_min = min(available_min, snapshot["available_bytes"])
    primary = slice(BENCHMARK_MEASURE_START, BENCHMARK_UPDATES)
    return {"route": route, "K": k, "cached": hidden_route, "updates_total": BENCHMARK_UPDATES, "cache_load_seconds": cache_load_seconds, "warmup_updates": {name: sum(values[:BENCHMARK_MEASURE_START]) for name, values in times.items()}, "primary_updates": {name: sum(values[primary]) for name, values in times.items()}, "primary_total_seconds": sum(times["total"][primary]), "primary_total_seconds_per_update": sum(times["total"][primary]) / BENCHMARK_MEASURED_UPDATES, "rss_peak_bytes": rss_peak, "available_memory_min_bytes": available_min, "logical_disk_bytes": int(manifest["logical_bytes"]) if hidden_route else 0, "teacher_loaded": not hidden_route, "teacher_forward_calls": BENCHMARK_UPDATES if not hidden_route else 0}


def run_benchmark(*, output_root: Path, manifest_path: Path, confirm_real_execution: bool) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution, "hidden cache benchmark")
    manifest_path = manifest_path.resolve()
    routes = benchmark_protocol()["routes"]
    rows = []
    commands = []
    for route in routes:
        command = [sys.executable, str(Path(__file__).resolve()), "--benchmark-child", "--route", route, "--cache-manifest", str(manifest_path), "--confirm-real-execution"]
        completed = subprocess.run(command, cwd=base.REPO_ROOT, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"benchmark child failed for {route}: {completed.stderr}")
        rows.append(json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])); commands.append(command)
    by_route = {row["route"]: row for row in rows}
    gates = benchmark_gates(by_route["DIRECT-K1"]["primary_total_seconds"], by_route["HIDDEN-RAM-K1"]["primary_total_seconds"], by_route["DIRECT-K4"]["primary_total_seconds"], by_route["HIDDEN-RAM-K4"]["primary_total_seconds"])
    report = {"schema": "omega-teacher-hidden-cache-benchmark-v1", "campaign_id": CAMPAIGN_ID, "protocol": benchmark_protocol(), "routes": by_route, "gates": gates, "worker_commands": commands}
    return write_self_hashed(output_root / "benchmark_report.json", report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for flag in ("feasibility", "build", "correctness", "benchmark", "benchmark_child"):
        parser.add_argument(f"--{flag.replace('_', '-')}", action="store_true")
    parser.add_argument("--backing-store", type=Path, default=DEFAULT_BACKING_STORE)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--route")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--confirm-real-execution", action="store_true")
    args = parser.parse_args(argv)
    selected = sum(bool(getattr(args, name)) for name in ("feasibility", "build", "correctness", "benchmark", "benchmark_child"))
    if selected != 1:
        parser.error("select exactly one phase")
    if args.feasibility:
        report = phase_a_report(args.backing_store.parent); print(json.dumps(write_self_hashed(args.output_root / "feasibility_report.json", report), indent=2, sort_keys=True)); return 0
    if args.build:
        print(json.dumps(build_hidden_cache(output_root=args.output_root, backing_store=args.backing_store, confirm_real_execution=args.confirm_real_execution), indent=2, sort_keys=True)); return 0
    if args.correctness:
        path = args.cache_manifest or args.output_root / "cache_manifest.json"; print(json.dumps(run_correctness_gate(cache_manifest_path=path, output_path=args.output_root / "correctness_report.json", confirm_real_execution=args.confirm_real_execution), indent=2, sort_keys=True)); return 0
    if args.benchmark:
        path = args.cache_manifest or args.output_root / "cache_manifest.json"; print(json.dumps(run_benchmark(output_root=args.output_root, manifest_path=path, confirm_real_execution=args.confirm_real_execution), indent=2, sort_keys=True)); return 0
    require_real_authorization(args.confirm_real_execution, "benchmark child")
    if not args.route or not args.cache_manifest:
        parser.error("benchmark child requires --route and --cache-manifest")
    print(json.dumps(_benchmark_child(args.route, args.cache_manifest), sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
