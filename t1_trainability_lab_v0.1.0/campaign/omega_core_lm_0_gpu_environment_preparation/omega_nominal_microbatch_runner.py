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
)


PHASES = ("update_started", "student_forward_completed", "teacher_forward_completed", "loss_completed", "memory_guard_failed", "backward_completed", "optimizer_step_started", "optimizer_step_completed", "update_completed")


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
        required = {"run_id", "variant", "update", "microbatch", "phase", "status", "elapsed_seconds", "memory"}
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


def _event(*, run_id: str, variant: str, update: int, microbatch: int | str | None, phase: str, timer: ExecutionTimer, input_shape: list[int], student_shape: list[int] | None, teacher_shape: list[int] | None, valid_tokens: int, status: str = "completed", metrics: dict[str, float] | None = None) -> dict[str, Any]:
    elapsed_seconds, cuda_elapsed_seconds = timer.elapsed()
    payload: dict[str, Any] = {"run_id": run_id, "variant": variant, "update": update, "microbatch": microbatch, "phase": phase, "status": status, "input_shape": input_shape, "student_logits_shape": student_shape, "teacher_logits_shape": teacher_shape, "valid_tokens": valid_tokens, "elapsed_seconds": elapsed_seconds, "cuda_elapsed_seconds": cuda_elapsed_seconds, "memory": memory_observed(timer.device)}
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


def microbatch_update(*, ledger: RunLedger, run_id: str, variant: str, update: int, student: OmegaCoreLM0R1Technical, teacher: torch.nn.Module, source: torch.Tensor, valid_mask: torch.Tensor, optimizer: torch.optim.Optimizer, window: int, persistent_state: torch.Tensor | None, input_valid_mask: torch.Tensor | None = None, memory_limit_rss_bytes: int | None = None, clip_max_norm: float = 1.0, fault: str | None = None, physical_batch: int = 2) -> tuple[torch.Tensor, dict[str, Any]]:
    if physical_batch != 2:
        raise ValueError("OMEGA nominal runner requires physical_batch=2")
    if clip_max_norm <= 0:
        raise ValueError("clip_max_norm must be positive")
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
    optimizer.zero_grad(set_to_none=True)
    update_timer = ExecutionTimer(source.device)
    if memory_limit_rss_bytes is not None:
        observed = memory_observed(source.device)
        if observed["rss_bytes"] is not None and observed["rss_bytes"] >= memory_limit_rss_bytes:
            ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="memory_guard_failed", timer=update_timer, input_shape=[8, 256], student_shape=None, teacher_shape=None, valid_tokens=total_valid, status="ABORTED", metrics={"rss_bytes": float(observed["rss_bytes"]), "memory_limit_rss_bytes": float(memory_limit_rss_bytes)}))
            raise MemoryError("RSS memory guard exceeded before update allocation")
    next_state = torch.zeros(8, student.slots, student.dimension, device=source.device)
    micro_metrics: list[dict[str, Any]] = []
    for microbatch, start in enumerate(range(0, 8, physical_batch)):
        stop = start + physical_batch
        if memory_limit_rss_bytes is not None:
            observed = memory_observed(source.device)
            if observed["rss_bytes"] is not None and observed["rss_bytes"] >= memory_limit_rss_bytes:
                ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="memory_guard_failed", timer=update_timer, input_shape=[physical_batch, 256], student_shape=None, teacher_shape=None, valid_tokens=int(valid_mask[start:stop].sum().item()), status="ABORTED", metrics={"rss_bytes": float(observed["rss_bytes"]), "memory_limit_rss_bytes": float(memory_limit_rss_bytes)}))
                raise MemoryError("RSS memory guard exceeded before microbatch forward")
        input_ids = source[start:stop, :256] if window == 0 else source[start:stop, 256:512]
        targets = source[start:stop, 1:257] if window == 0 else source[start:stop, 257:513]
        target_mask = valid_mask[start:stop]
        input_mask = input_valid_mask[start:stop]
        valid_count = int(target_mask.sum().item())
        previous = torch.zeros(physical_batch, student.slots, student.dimension, device=source.device) if window == 0 else persistent_state[start:stop]
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="update_started", timer=update_timer, input_shape=list(input_ids.shape), student_shape=None, teacher_shape=None, valid_tokens=valid_count))
        next_micro_state, student_logits = student.forward_window(input_ids, previous, input_mask)
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="student_forward_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=None, valid_tokens=valid_count))
        teacher_source = source[start:stop]
        teacher_logits = compact_teacher_logits(teacher, teacher_source, window)
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="teacher_forward_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count))
        losses = distillation_loss(student_logits, teacher_logits, targets, target_mask)
        scale = valid_count / total_valid
        scalar_metrics = {"ce": float(losses["ce"].detach().item()), "kl": float(losses["kl"].detach().item()), "total_loss_mean": float(losses["total"].detach().item()), "weight_from_real_mask": scale, "valid_tokens_total": total_valid}
        loss_event = _event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="loss_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, metrics=scalar_metrics)
        ledger.append(loss_event)
        if memory_limit_rss_bytes is not None:
            observed = loss_event["memory"]
            if observed["rss_bytes"] is not None and observed["rss_bytes"] >= memory_limit_rss_bytes:
                ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="memory_guard_failed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, status="ABORTED", metrics={"rss_bytes": float(observed["rss_bytes"]), "memory_limit_rss_bytes": float(memory_limit_rss_bytes)}))
                raise MemoryError("RSS memory guard exceeded after loss; backward aborted")
        if fault == "after_loss":
            raise RuntimeError("controlled runner fault after loss")
        (losses["total"] * scale).backward()
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="backward_completed", timer=update_timer, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count))
        if fault == "after_backward":
            raise RuntimeError("controlled runner fault after backward")
        next_state[start:stop] = next_micro_state.detach()
        del next_micro_state, student_logits, teacher_logits, losses
        micro_metrics.append({"microbatch": microbatch, "valid_tokens": valid_count, "weight": scale})
    pre_clip_grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), clip_max_norm).item())
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_started", timer=update_timer, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, status="started", metrics={"clip_max_norm": clip_max_norm, "pre_clip_grad_norm": pre_clip_grad_norm}))
    try:
        optimizer.step()
    except Exception:
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_started", timer=update_timer, input_shape=[8, 256], student_shape=None, teacher_shape=None, valid_tokens=total_valid, status="UNCERTAIN"))
        raise
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_completed", timer=update_timer, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, status="APPLIED"))
    optimizer.zero_grad(set_to_none=True)
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="update_completed", timer=update_timer, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid))
    elapsed_seconds, cuda_elapsed_seconds = update_timer.elapsed()
    return next_state.detach(), {"update": update, "window": window, "effective_batch": 8, "physical_batch": physical_batch, "microbatches": 4, "valid_tokens": total_valid, "microbatch_metrics": micro_metrics, "pre_clip_grad_norm": pre_clip_grad_norm, "clip_max_norm": clip_max_norm, "elapsed_seconds": elapsed_seconds, "cuda_elapsed_seconds": cuda_elapsed_seconds}


def load_fixed_real_runtime(cache_root: Path) -> tuple[Any, Any, Any]:
    """Future-pod loader; deliberately not called by this preparation unit."""
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if len(tokenizer) != TOKENIZER_VOCAB:
        raise ValueError("fixed tokenizer vocabulary mismatch")
    return dataset, tokenizer, teacher


VARIANT_CONFIG = {
    "shared_K1": {"variant": "shared", "rounds": 1},
    "shared_K4": {"variant": "shared", "rounds": 4},
    "untied_K4": {"variant": "untied", "rounds": 4},
}


def _preflight_source(dataset: Any, tokenizer: Any, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    text = str(dataset[0].get("text", "")).strip()
    if not text:
        raise ValueError("fixed dataset first record has no usable text")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer([text] * 8, padding="max_length", truncation=True, max_length=513, return_tensors="pt")
    source = encoded["input_ids"].to(device)
    attention = encoded.get("attention_mask", torch.ones_like(source)).to(device).bool()
    input_masks = [attention[:, :256], attention[:, 256:512]]
    target_masks = [attention[:, 1:257], attention[:, 257:513]]
    return source, input_masks, target_masks


def run_preflight(*, cache_root: Path, output_root: Path, device_name: str, updates: int, memory_limit_rss_bytes: int | None = None) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    dataset, tokenizer, teacher = load_fixed_real_runtime(cache_root)
    teacher.to(device)
    source, input_masks, target_masks = _preflight_source(dataset, tokenizer, device)
    results: dict[str, Any] = {"device": str(device), "updates_requested": updates, "variants": {}}
    for name, config in VARIANT_CONFIG.items():
        variant_dir = output_root / name
        ledger = RunLedger(variant_dir, f"preflight-{name}")
        student = OmegaCoreLM0R1Technical(vocab_size=len(tokenizer), rounds=config["rounds"], variant=config["variant"]).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=BASE_LR)
        state: torch.Tensor | None = None
        updates_completed = 0
        for update in range(updates):
            window = 0 if update == 0 else 1
            state, metrics = microbatch_update(ledger=ledger, run_id=f"preflight-{name}", variant=name, update=update, student=student, teacher=teacher, source=source, valid_mask=target_masks[window], input_valid_mask=input_masks[window], optimizer=optimizer, window=window, persistent_state=state, memory_limit_rss_bytes=memory_limit_rss_bytes)
            updates_completed += 1
        results["variants"][name] = {"rounds": config["rounds"], "variant": config["variant"], "updates_completed": updates_completed, "last_metrics": metrics, "ledger": str(ledger.events_path)}
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
