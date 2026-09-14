"""Prepared OMEGA nominal runner with recoverable microbatch updates.

This module is not invoked by environment preparation. It is ready for a
future explicitly authorized pod run and has no provider lifecycle authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
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


PHASES = ("update_started", "student_forward_completed", "teacher_forward_completed", "loss_completed", "backward_completed", "optimizer_step_started", "optimizer_step_completed", "update_completed")


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


def memory_observed() -> dict[str, int | None]:
    import psutil

    process = psutil.Process()
    info = process.memory_info()
    return {"rss_bytes": info.rss, "available_system_bytes": psutil.virtual_memory().available, "peak_working_set_bytes": getattr(info, "peak_wset", None)}


def compact_teacher_logits(teacher: torch.nn.Module, source: torch.Tensor, window: int) -> torch.Tensor:
    context = source[:, :256] if window == 0 else source[:, :512]
    start = 0 if window == 0 else 256
    with torch.no_grad():
        output = teacher(input_ids=context)
        compact = output.logits[:, start : start + 256].detach().clone()
    del output
    return compact


def _event(*, run_id: str, variant: str, update: int, microbatch: int | str | None, phase: str, started: float, input_shape: list[int], student_shape: list[int] | None, teacher_shape: list[int] | None, valid_tokens: int, status: str = "completed", metrics: dict[str, float] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"run_id": run_id, "variant": variant, "update": update, "microbatch": microbatch, "phase": phase, "status": status, "input_shape": input_shape, "student_logits_shape": student_shape, "teacher_logits_shape": teacher_shape, "valid_tokens": valid_tokens, "elapsed_seconds": time.perf_counter() - started, "memory": memory_observed()}
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


def microbatch_update(*, ledger: RunLedger, run_id: str, variant: str, update: int, student: OmegaCoreLM0R1Technical, teacher: torch.nn.Module, source: torch.Tensor, valid_mask: torch.Tensor, optimizer: torch.optim.Optimizer, window: int, persistent_state: torch.Tensor | None, fault: str | None = None, physical_batch: int = 2) -> tuple[torch.Tensor, dict[str, Any]]:
    if source.shape[0] != 8 or source.shape[1] != 513:
        raise ValueError(f"source must be [8,513], got {tuple(source.shape)}")
    if valid_mask.shape != (8, 256):
        raise ValueError(f"valid_mask must be [8,256], got {tuple(valid_mask.shape)}")
    total_valid = int(valid_mask.sum().item())
    if total_valid <= 0:
        raise ValueError("at least one valid target required")
    if window == 1 and persistent_state is None:
        raise ValueError("window 1 requires detached per-sequence persistent state")
    optimizer.zero_grad(set_to_none=True)
    update_started = time.perf_counter()
    next_state = torch.zeros(8, student.slots, student.dimension, device=source.device)
    micro_metrics: list[dict[str, Any]] = []
    for microbatch, start in enumerate(range(0, 8, physical_batch)):
        stop = start + physical_batch
        input_ids = source[start:stop, :256] if window == 0 else source[start:stop, 256:512]
        targets = source[start:stop, 1:257] if window == 0 else source[start:stop, 257:513]
        mask = valid_mask[start:stop]
        valid_count = int(mask.sum().item())
        previous = torch.zeros(physical_batch, student.slots, student.dimension, device=source.device) if window == 0 else persistent_state[start:stop]
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="update_started", started=update_started, input_shape=list(input_ids.shape), student_shape=None, teacher_shape=None, valid_tokens=valid_count))
        next_micro_state, student_logits = student.forward_window(input_ids, previous)
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="student_forward_completed", started=update_started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=None, valid_tokens=valid_count))
        teacher_source = source[start:stop]
        teacher_logits = compact_teacher_logits(teacher, teacher_source, window)
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="teacher_forward_completed", started=update_started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count))
        losses = distillation_loss(student_logits, teacher_logits, targets, mask)
        scale = valid_count / total_valid
        scalar_metrics = {"ce": float(losses["ce"].detach().item()), "kl": float(losses["kl"].detach().item()), "total_loss_mean": float(losses["total"].detach().item()), "weight_from_real_mask": scale, "valid_tokens_total": total_valid}
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="loss_completed", started=update_started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count, metrics=scalar_metrics))
        if fault == "after_loss":
            raise RuntimeError("controlled runner fault after loss")
        (losses["total"] * scale).backward()
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch=microbatch, phase="backward_completed", started=update_started, input_shape=list(input_ids.shape), student_shape=list(student_logits.shape), teacher_shape=list(teacher_logits.shape), valid_tokens=valid_count))
        if fault == "after_backward":
            raise RuntimeError("controlled runner fault after backward")
        next_state[start:stop] = next_micro_state.detach()
        del next_micro_state, student_logits, teacher_logits, losses
        micro_metrics.append({"microbatch": microbatch, "valid_tokens": valid_count, "weight": scale})
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_started", started=update_started, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, status="started"))
    try:
        optimizer.step()
    except Exception:
        ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_started", started=update_started, input_shape=[8, 256], student_shape=None, teacher_shape=None, valid_tokens=total_valid, status="UNCERTAIN"))
        raise
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="optimizer_step_completed", started=update_started, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid, status="APPLIED"))
    optimizer.zero_grad(set_to_none=True)
    ledger.append(_event(run_id=run_id, variant=variant, update=update, microbatch="all", phase="update_completed", started=update_started, input_shape=[8, 256], student_shape=[8, 256, student.vocab_size], teacher_shape=[8, 256, student.vocab_size], valid_tokens=total_valid))
    return next_state.detach(), {"update": update, "window": window, "effective_batch": 8, "physical_batch": physical_batch, "microbatches": 4, "valid_tokens": total_valid, "microbatch_metrics": micro_metrics, "elapsed_seconds": time.perf_counter() - update_started}


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


if __name__ == "__main__":
    raise SystemExit("This is a prepared library, not an executable training command; use an explicitly authorized future-pod runner.")
