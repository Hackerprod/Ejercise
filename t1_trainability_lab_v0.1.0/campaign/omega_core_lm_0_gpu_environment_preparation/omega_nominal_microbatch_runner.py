"""Prepared OMEGA nominal runner with recoverable microbatch updates.

This module is not invoked by environment preparation. It is ready for a
future explicitly authorized pod run and has no provider lifecycle authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import argparse
from pathlib import Path
from typing import Any, Callable

import torch

LOCAL_ROOT = Path(__file__).resolve().parents[2]
ROOT = LOCAL_ROOT if (LOCAL_ROOT / "scripts").is_dir() else Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
import sys

sys.path.insert(0, str(SCRIPTS))
from run_omega_core_lm_0_r1_training_technical_preflight import (  # noqa: E402
    BASE_LR,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    OmegaCoreLM0R1Technical,
    TOKENIZER_VOCAB,
    distillation_loss,
    select_documents,
    sha256_bytes,
)


PHASES = ("update_started", "student_forward_completed", "teacher_forward_completed", "loss_completed", "memory_guard_failed", "backward_completed", "optimizer_step_started", "optimizer_step_completed", "update_completed")
WINDOWS = (
    {"window": 0, "input_range": [0, 256], "target_range": [1, 257], "teacher_context_range": [0, 256]},
    {"window": 1, "input_range": [256, 512], "target_range": [257, 513], "teacher_context_range": [0, 512]},
)
TECHNICAL_VARIANT_SEEDS = {"shared_K1": 20260913, "shared_K4": 20260914, "untied_K4": 20260915}
APPROVED_MANIFEST_PATH = "campaign/omega_core_lm_0_r1_training_technical_preflight/training_technical_preflight_v2.json"
APPROVED_MANIFEST_SHA256 = "0afa54c708cc746ca6d44503946a6b3c715bec44b7bc97cacbd84ea346d777ac"
APPROVED_SELECTION_MANIFEST = {
    "provenance": {
        "path": APPROVED_MANIFEST_PATH,
        "artifact_sha256": APPROVED_MANIFEST_SHA256,
    },
    "selected_documents": [
        {"document_index": 0, "header": "= Valkyria Chronicles III =", "row_range": [1, 51], "selected_token_count": 513, "text_sha256": "f3a9ae3686a2c64c70212770b06836d08e81dbb5bf81d7158459008c041935fb", "token_count": 4486, "token_sha256": "d199468db926ad1db93f42f156cc21a14782db0c844b8540e357b26000e8b6ba"},
        {"document_index": 1, "header": "= Tower Building of the Little Rock Arsenal =", "row_range": [51, 133], "selected_token_count": 513, "text_sha256": "65f085f7dfb80e45298333c99c54cb232bfe5d8fae3d7b064ac1f81207eecac3", "token_count": 4512, "token_sha256": "0ca3fb1716f4e21178b371c3dca9bd76308602ae5e9dad67933ba0115b4b4ce8"},
        {"document_index": 2, "header": "= Cicely Mary Barker =", "row_range": [133, 271], "selected_token_count": 513, "text_sha256": "630d2265952215dc960026317255ce6cd6edeb00cb387f103c1b4ccdd6e81754", "token_count": 3801, "token_sha256": "ce7997d9ec69aa58fb77610a33782250383b52bedc6ffc7c1bf62e5e0d43cc49"},
        {"document_index": 3, "header": "= Gambia women 's national football team =", "row_range": [271, 287], "selected_token_count": 513, "text_sha256": "230f909aedcfc18015ff6874e35f89d672927ff1c8d96a1536afb1397661cbd2", "token_count": 795, "token_sha256": "483874a11b2cb40386e21d6261bd2c75bdc65873c14a448230f407be8b0a5db5"},
        {"document_index": 4, "header": "= Plain maskray =", "row_range": [287, 316], "selected_token_count": 513, "text_sha256": "dbc801d057ace38f37739023db048819e0e49b324f704a3586e0cbe346c7f8db", "token_count": 1607, "token_sha256": "c1cda6663d4dd7737d9688f21e13126249cd81733843df3b48f0ac460b461a5c"},
        {"document_index": 5, "header": "= 2011 \u2013 12 Columbus Blue Jackets season =", "row_range": [316, 378], "selected_token_count": 513, "text_sha256": "3db398fa1f9a8600fb4e65fcae3e81d2b4d0bb4961a6fd1ae6355e2c51d71dd8", "token_count": 3759, "token_sha256": "5514e0addb8669980c2e20b9cec3e55ca521d16048b979b923c20a0223372158"},
        {"document_index": 12, "header": "= Saves ; Sv % =", "row_range": [394, 406], "selected_token_count": 513, "text_sha256": "37867c3498e05b00263e74a0243fe5e7de707d177fd608e4e169cfd369a6f6df", "token_count": 568, "token_sha256": "d6be108736ba25a863ed3f273bc0ea9a3a415383fd8802a946b4d1016e2e41ec"},
        {"document_index": 13, "header": "= Gregorian Tower =", "row_range": [406, 433], "selected_token_count": 513, "text_sha256": "98e75a186668b2e33348d01579dcf29eb9d313e0685c9e11c44cc2bd5093ace2", "token_count": 1862, "token_sha256": "ef4f8b1201ad14392a787485cfb6e7725fd8971485d60fe6737430fd73747c8f"},
    ],
    "windows": [
        {"window": 0, "input_range": [0, 256], "target_range": [1, 257], "teacher_context_range": [0, 256]},
        {"window": 1, "input_range": [256, 512], "target_range": [257, 513], "teacher_context_range": [0, 512]},
    ],
}


class RunLedger:
    """Append-only run ledger; never deletes or silently reuses an old run."""

    def __init__(self, run_dir: Path, run_id: str, *, resume: bool = False) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.events_path = run_dir / "events.jsonl"
        if self.events_path.exists() and not resume:
            raise FileExistsError(f"run ledger already exists; choose new run_id or explicit resume: {self.events_path}")
        if resume and not self.events_path.exists():
            raise FileNotFoundError(f"cannot resume missing run ledger: {self.events_path}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        required = {
            "run_id", "variant", "update", "microbatch", "phase", "status", "elapsed_seconds", "memory",
            "document_id", "input_range", "target_range", "teacher_context_range", "window",
            "state_reset", "state_source_update", "valid_tokens",
        }
        missing = required.difference(event)
        if missing:
            raise ValueError(f"ledger event missing fields: {sorted(missing)}")
        if event["phase"] not in PHASES:
            raise ValueError(f"unknown phase: {event['phase']}")
        encoded = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with self.events_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ExecutionTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        synchronize(device)
        self.host_started = time.perf_counter()
        self.cuda_started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        self.cuda_finished = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if self.cuda_started is not None:
            self.cuda_started.record()

    def elapsed(self) -> tuple[float, float | None]:
        synchronize(self.device)
        cuda_elapsed = None
        if self.cuda_started is not None and self.cuda_finished is not None:
            self.cuda_finished.record()
            self.cuda_finished.synchronize()
            cuda_elapsed = self.cuda_started.elapsed_time(self.cuda_finished) / 1000.0
        return time.perf_counter() - self.host_started, cuda_elapsed


def memory_observed(device: torch.device) -> dict[str, int | None]:
    import psutil

    synchronize(device)
    process = psutil.Process()
    info = process.memory_info()
    observed: dict[str, int | None] = {"rss_bytes": info.rss, "available_system_bytes": psutil.virtual_memory().available, "peak_working_set_bytes": getattr(info, "peak_wset", None), "gpu_allocated_bytes": None, "gpu_reserved_bytes": None, "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    if device.type == "cuda":
        observed.update({"gpu_allocated_bytes": torch.cuda.memory_allocated(device), "gpu_reserved_bytes": torch.cuda.memory_reserved(device), "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    return observed


def compact_teacher_logits(teacher: torch.nn.Module, source: torch.Tensor, window: int) -> torch.Tensor:
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    with torch.no_grad():
        output = teacher(input_ids=context)
        compact = output.logits[:, start : start + 256].detach().clone()
    del output
    return compact


def window_for_update(update: int) -> int:
    if update < 0:
        raise ValueError("update must be non-negative")
    return update % 2


def update_window_schedule(updates: int) -> list[int]:
    if updates < 0:
        raise ValueError("updates must be non-negative")
    return [window_for_update(update) for update in range(updates)]


def state_source_update_for_update(update: int) -> int | None:
    window = window_for_update(update)
    return update - 1 if window == 1 else None


def validate_window_schedule(schedule: list[int]) -> None:
    expected = update_window_schedule(len(schedule))
    if schedule != expected:
        raise ValueError(f"invalid causal window schedule: expected {expected}, got {schedule}")


def construct_adamw(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def set_technical_seed(seed: int, *, configure_determinism: bool = True) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if configure_determinism:
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
    return backend_options(seed)


def backend_options(seed: int | None = None) -> dict[str, Any]:
    cudnn = getattr(torch.backends, "cudnn", None)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "seed": seed,
        "backend_flags": {
            "cudnn_deterministic": getattr(cudnn, "deterministic", None),
            "cudnn_benchmark": getattr(cudnn, "benchmark", None),
            "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        },
    }


def _window_metadata(manifest: dict[str, Any] | None, window: int) -> dict[str, list[int]]:
    if window not in (0, 1):
        raise ValueError(f"window must be 0 or 1, got {window}")
    if manifest is None:
        return WINDOWS[window]
    windows = manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != 2:
        raise ValueError("source manifest must contain exactly two windows")
    metadata = next((item for item in windows if item.get("window") == window), None)
    if metadata is None:
        raise ValueError(f"source manifest missing window {window}")
    return {key: list(metadata[key]) for key in ("input_range", "target_range", "teacher_context_range")}


def _event(*, run_id: str, variant: str, update: int, microbatch: int | str | None, phase: str, timer: ExecutionTimer, input_shape: list[int], student_shape: list[int] | None, teacher_shape: list[int] | None, valid_tokens: int, document_ids: list[str], window: int, state_source_update: int | None, manifest: dict[str, Any] | None = None, status: str = "completed", metrics: dict[str, float] | None = None) -> dict[str, Any]:
    elapsed_seconds, cuda_elapsed_seconds = timer.elapsed()
    ranges = _window_metadata(manifest, window)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "variant": variant,
        "update": update,
        "microbatch": microbatch,
        "phase": phase,
        "status": status,
        "input_shape": input_shape,
        "student_logits_shape": student_shape,
        "teacher_logits_shape": teacher_shape,
        "document_id": document_ids,
        "input_range": ranges["input_range"],
        "target_range": ranges["target_range"],
        "teacher_context_range": ranges["teacher_context_range"],
        "window": window,
        "state_reset": window == 0,
        "state_source_update": state_source_update,
        "valid_tokens": valid_tokens,
        "elapsed_seconds": elapsed_seconds,
        "cuda_elapsed_seconds": cuda_elapsed_seconds,
        "memory": memory_observed(timer.device),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def read_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ledger_summary(path: Path) -> dict[str, Any]:
    events = read_ledger(path)
    applied = sorted({event["update"] for event in events if event["phase"] == "optimizer_step_completed" and event["status"] == "APPLIED"})
    uncertain = sorted({event["update"] for event in events if event["phase"] == "optimizer_step_started"} - set(applied))
    completed = sorted({event["update"] for event in events if event["phase"] == "update_completed"})
    return {"events": len(events), "applied_updates": applied, "uncertain_updates": uncertain, "completed_updates": completed, "phases": [event["phase"] for event in events]}


def microbatch_update(*, ledger: RunLedger, run_id: str, variant: str, update: int, student: OmegaCoreLM0R1Technical, teacher: torch.nn.Module, source: torch.Tensor, valid_mask: torch.Tensor, optimizer: torch.optim.Optimizer, window: int, persistent_state: torch.Tensor | None, input_valid_mask: torch.Tensor | None = None, memory_limit_rss_bytes: int | None = None, clip_max_norm: float = 1.0, fault: str | None = None, physical_batch: int = 2, document_ids: list[str] | None = None, source_manifest: dict[str, Any] | None = None, state_source_update: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    if physical_batch != 2:
        raise ValueError("OMEGA nominal runner requires physical_batch=2")
    if clip_max_norm <= 0:
        raise ValueError("clip_max_norm must be positive")
    if window not in (0, 1):
        raise ValueError(f"window must be 0 or 1, got {window}")
    if source.shape[0] != 8 or source.shape[1] != 513:
        raise ValueError(f"source must be [8,513], got {tuple(source.shape)}")
    if valid_mask.shape != (8, 256):
        raise ValueError(f"valid_mask must be [8,256], got {tuple(valid_mask.shape)}")
    if input_valid_mask is None:
        input_valid_mask = torch.ones_like(valid_mask)
    if input_valid_mask.shape != (8, 256):
        raise ValueError(f"input_valid_mask must be [8,256], got {tuple(input_valid_mask.shape)}")
    if valid_mask.dtype != torch.bool:
        raise TypeError("target valid_mask must be bool")
    if input_valid_mask.dtype != torch.bool:
        raise TypeError("input_valid_mask must be bool")
    total_valid = int(valid_mask.sum().item())
    if total_valid <= 0:
        raise ValueError("at least one valid target required")
    if window == 1 and persistent_state is None:
        raise ValueError("window 1 requires detached per-sequence persistent state")
    if document_ids is None:
        document_ids = [f"sequence-{index}" for index in range(8)]
    if len(document_ids) != 8:
        raise ValueError("document_ids must contain one ID per source sequence")
    if state_source_update is None:
        state_source_update = None if window == 0 else update - 1
    if window == 0 and state_source_update is not None:
        raise ValueError("window 0 cannot consume persistent state")
    if window == 1 and state_source_update != update - 1:
        raise ValueError("window 1 state must come from immediately preceding window 0 update")
    optimizer.zero_grad(set_to_none=True)
    update_timer = ExecutionTimer(source.device)
    if memory_limit_rss_bytes is not None:
        observed = memory_observed(source.device)
        if observed["rss_bytes"] is not None and observed["rss_bytes"] >= memory_limit_rss_bytes:
            ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="memory_guard_failed", timer=update_timer, input_shape=[8, 256], student_shape=None, teacher_shape=None, valid_tokens=total_valid, document_ids=document_ids, window=window, state_source_update=state_source_update, manifest=source_manifest, status="ABORTED", metrics={"rss_bytes": float(observed["rss_bytes"]), "memory_limit_rss_bytes": float(memory_limit_rss_bytes)}))
            raise MemoryError("RSS memory guard exceeded before update allocation")
    next_state = torch.zeros(8, student.slots, student.dimension, device=source.device)
    micro_metrics: list[dict[str, Any]] = []
    for microbatch, start in enumerate(range(0, 8, physical_batch)):
        stop = start + physical_batch
        if memory_limit_rss_bytes is not None:
            observed = memory_observed(source.device)
            if observed["rss_bytes"] is not None and observed["rss_bytes"] >= memory_limit_rss_bytes:
                ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="memory_guard_failed", timer=update_timer, input_shape=[physical_batch, 256], student_shape=None, teacher_shape=None, valid_tokens=int(valid_mask[start:stop].sum().item()), document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest, status="ABORTED", metrics={"rss_bytes": float(observed["rss_bytes"]), "memory_limit_rss_bytes": float(memory_limit_rss_bytes)}))
                raise MemoryError("RSS memory guard exceeded before microbatch forward")
        input_ids = source[start:stop, :256] if window == 0 else source[start:stop, 256:512]
        targets = source[start:stop, 1:257] if window == 0 else source[start:stop, 257:513]
        target_mask = valid_mask[start:stop]
        input_mask = input_valid_mask[start:stop]
        valid_count = int(target_mask.sum().item())
        previous = torch.zeros(physical_batch, student.slots, student.dimension, device=source.device) if window == 0 else persistent_state[start:stop]
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="update_started", timer=update_timer, input_shape=list(input_ids.shape), student_shape=None, teacher_shape=None, valid_tokens=valid_count, document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest))
        next_micro_state, student_logits = student.forward_window(input_ids, previous, input_mask)
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="student_forward_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=None, valid_tokens=valid_count, document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest))
        teacher_source = source[start:stop]
        teacher_logits = compact_teacher_logits(teacher, teacher_source, window)
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="teacher_forward_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest))
        losses = distillation_loss(student_logits, teacher_logits, targets, target_mask)
        scale = valid_count / total_valid
        scalar_metrics = {"ce": float(losses["ce"].detach().item()), "kl": float(losses["kl"].detach().item()), "total_loss_mean": float(losses["total"].detach().item()), "weight_from_real_mask": scale, "valid_tokens_total": total_valid}
        loss_event = _event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="loss_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest, metrics=scalar_metrics)
        ledger.append(loss_event)
        if memory_limit_rss_bytes is not None:
            observed = loss_event["memory"]
            if observed["rss_bytes"] is not None and observed["rss_bytes"] >= memory_limit_rss_bytes:
                ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="memory_guard_failed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest, status="ABORTED", metrics={"rss_bytes": float(observed["rss_bytes"]), "memory_limit_rss_bytes": float(memory_limit_rss_bytes)}))
                raise MemoryError("RSS memory guard exceeded after loss; backward aborted")
        if fault == "after_loss":
            raise RuntimeError("controlled runner fault after loss")
        (losses["total"] * scale).backward()
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="backward_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, document_ids=document_ids[start:stop], window=window, state_source_update=state_source_update, manifest=source_manifest))
        if fault == "after_backward":
            raise RuntimeError("controlled runner fault after backward")
        next_state[start:stop] = next_micro_state.detach()
        del next_micro_state, student_logits, teacher_logits, losses
        micro_metrics.append({"microbatch": microbatch, "valid_tokens": valid_count, "weight": scale})
    pre_clip_grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), clip_max_norm).item())
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_started", timer=update_timer, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, document_ids=document_ids, window=window, state_source_update=state_source_update, manifest=source_manifest, status="started", metrics={"clip_max_norm": clip_max_norm, "pre_clip_grad_norm": pre_clip_grad_norm}))
    try:
        optimizer.step()
    except Exception:
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_started", timer=update_timer, input_shape=[8, 256], student_shape=None, teacher_shape=None, valid_tokens=total_valid, document_ids=document_ids, window=window, state_source_update=state_source_update, manifest=source_manifest, status="UNCERTAIN"))
        raise
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_completed", timer=update_timer, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, document_ids=document_ids, window=window, state_source_update=state_source_update, manifest=source_manifest, status="APPLIED"))
    optimizer.zero_grad(set_to_none=True)
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="update_completed", timer=update_timer, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, document_ids=document_ids, window=window, state_source_update=state_source_update, manifest=source_manifest))
    elapsed_seconds, cuda_elapsed_seconds = update_timer.elapsed()
    return next_state.detach(), {"update": update, "window": window, "effective_batch": 8, "physical_batch": physical_batch, "microbatches": 4, "valid_tokens": total_valid, "microbatch_metrics": micro_metrics, "pre_clip_grad_norm": pre_clip_grad_norm, "clip_max_norm": clip_max_norm, "elapsed_seconds": elapsed_seconds, "cuda_elapsed_seconds": cuda_elapsed_seconds}


def load_fixed_dataset_and_tokenizer(cache_root: Path) -> tuple[Any, Any]:
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise ValueError("fixed tokenizer vocabulary mismatch")
    return dataset, tokenizer


def load_fixed_teacher(cache_root: Path) -> torch.nn.Module:
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def load_fixed_real_runtime(cache_root: Path) -> tuple[Any, Any, Any]:
    """Compatibility loader; validates source before loading teacher."""
    dataset, tokenizer = load_fixed_dataset_and_tokenizer(cache_root)
    _preflight_source(dataset, tokenizer, torch.device("cpu"))
    return dataset, tokenizer, load_fixed_teacher(cache_root)


VARIANT_CONFIG = {
    "shared_K1": {"variant": "shared", "rounds": 1},
    "shared_K4": {"variant": "shared", "rounds": 4},
    "untied_K4": {"variant": "untied", "rounds": 4},
}


def validate_selected_documents(selected: list[dict[str, Any]], manifest: dict[str, Any], approved_manifest: dict[str, Any] | None = None) -> list[str]:
    manifest_documents = manifest.get("selected_documents")
    if not isinstance(manifest_documents, list) or len(manifest_documents) != 8 or len(selected) != 8:
        raise ValueError("source must contain exactly eight selected documents")
    if approved_manifest is not None:
        approved_documents = approved_manifest.get("selected_documents")
        approved_windows = approved_manifest.get("windows")
        if manifest_documents != approved_documents:
            raise ValueError("selected documents do not match approved selection manifest")
        if manifest.get("windows") != approved_windows:
            raise ValueError("causal windows do not match approved selection manifest")
    by_index = {item.get("document_index"): item for item in manifest_documents}
    if len(by_index) != 8:
        raise ValueError("selected document manifest IDs are not unique")
    document_ids: list[str] = []
    for item in selected:
        if len(item.get("tokens", [])) != 513 or item.get("selected_token_count") != 513:
            raise ValueError("selected document must retain exactly 513 tokens")
        expected_hash = sha256_bytes(b"".join(int(token).to_bytes(4, "little") for token in item["tokens"]))
        if item.get("token_sha256") != expected_hash:
            raise ValueError(f"selected document token hash mismatch: {item.get('document_index')}")
        manifest_item = by_index.get(item.get("document_index"))
        if manifest_item is None:
            raise ValueError("selected document is absent from manifest")
        for key in ("row_range", "header", "token_count", "selected_token_count", "text_sha256", "token_sha256"):
            if manifest_item.get(key) != item.get(key):
                raise ValueError(f"selected document manifest mismatch for {key}")
        document_id = f"wikitext-train-document-{item['document_index']}"
        item["document_id"] = document_id
        manifest_item["document_id"] = document_id
        document_ids.append(document_id)
    if len(set(document_ids)) != 8:
        raise ValueError("selected document IDs are not unique")
    return document_ids


def _preflight_source(dataset: Any, tokenizer: Any, device: torch.device, approved_manifest: dict[str, Any] | None = None) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    selected, manifest = select_documents(dataset, tokenizer)
    pinned_manifest = APPROVED_SELECTION_MANIFEST if approved_manifest is None else approved_manifest
    document_ids = validate_selected_documents(selected, manifest, pinned_manifest)
    source = torch.tensor([item["tokens"] for item in selected], dtype=torch.long, device=device)
    if tuple(source.shape) != (8, 513):
        raise ValueError(f"selected source must be [8,513], got {tuple(source.shape)}")
    input_masks = [torch.ones((8, 256), dtype=torch.bool, device=device) for _ in WINDOWS]
    target_masks = [torch.ones((8, 256), dtype=torch.bool, device=device) for _ in WINDOWS]
    if any(int(mask.sum().item()) != 2048 for mask in target_masks):
        raise ValueError("each causal window must contain exactly 2048 valid target tokens")
    manifest["source_shape"] = [8, 513]
    manifest["selected_document_ids"] = document_ids
    manifest["valid_target_tokens_per_window"] = [int(mask.sum().item()) for mask in target_masks]
    manifest["approved_manifest_provenance"] = pinned_manifest.get("provenance", {"source": "injected"})
    return source, input_masks, target_masks, manifest


def run_preflight(*, cache_root: Path, output_root: Path, device_name: str, updates: int, memory_limit_rss_bytes: int | None = None) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    schedule = update_window_schedule(updates)
    validate_window_schedule(schedule)
    dataset, tokenizer = load_fixed_dataset_and_tokenizer(cache_root)
    source, input_masks, target_masks, manifest = _preflight_source(dataset, tokenizer, device)
    document_ids = list(manifest["selected_document_ids"])
    teacher = load_fixed_teacher(cache_root)
    teacher.to(device)
    results: dict[str, Any] = {
        "device": str(device),
        "updates_requested": updates,
        "window_schedule": schedule,
        "selection_manifest": manifest,
        "backend": {"before_variants": backend_options(), "variant_seeds": {}},
        "variants": {},
    }
    for name, config in VARIANT_CONFIG.items():
        variant_seed = TECHNICAL_VARIANT_SEEDS[name]
        results["backend"]["variant_seeds"][name] = set_technical_seed(variant_seed)
        variant_dir = output_root / name
        ledger = RunLedger(variant_dir, f"preflight-{name}")
        student = OmegaCoreLM0R1Technical(vocab_size=len(tokenizer), rounds=config["rounds"], variant=config["variant"]).to(device)
        optimizer = construct_adamw(student)
        state: torch.Tensor | None = None
        updates_completed = 0
        for update, window in enumerate(schedule):
            state_source_update = state_source_update_for_update(update)
            if window == 0:
                state = None
            state, metrics = microbatch_update(ledger=ledger, run_id=f"preflight-{name}", variant=name, update=update, student=student, teacher=teacher, source=source, valid_mask=target_masks[window], input_valid_mask=input_masks[window], optimizer=optimizer, window=window, persistent_state=state, document_ids=document_ids, source_manifest=manifest, state_source_update=state_source_update, memory_limit_rss_bytes=memory_limit_rss_bytes)
            updates_completed += 1
        results["variants"][name] = {"rounds": config["rounds"], "variant": config["variant"], "seed": variant_seed, "updates_completed": updates_completed, "last_metrics": metrics, "ledger": str(ledger.events_path)}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="explicitly run fixed-revision runtime preflight")
    preflight.add_argument("--cache-root", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--device", default="cpu")
    preflight.add_argument("--updates", type=int, default=1)
    preflight.add_argument("--memory-limit-rss-bytes", type=int, default=None)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_preflight(cache_root=args.cache_root, output_root=args.output_root, device_name=args.device, updates=args.updates, memory_limit_rss_bytes=args.memory_limit_rss_bytes), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
