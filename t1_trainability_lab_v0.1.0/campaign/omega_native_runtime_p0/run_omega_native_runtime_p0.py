"""OMEGA native-runtime P0 profiler.

Profiles the current Python/PyTorch hidden-cache production route only.  This
unit deliberately contains no C++, CUDA, AVX, or cache-building code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
INTEGRATION_DIR = CAMPAIGN_ROOT / "omega_hidden_cache_production_integration"
CE_DIR = CAMPAIGN_ROOT / "omega_ce_only_baseline"
PROBE_DIR = CAMPAIGN_ROOT / "omega_teacher_hidden_cache_probe"
for path in (INTEGRATION_DIR, CE_DIR, PROBE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_omega_hidden_cache_production_integration as integration  # noqa: E402
import run_omega_ce_only_baseline as ce  # noqa: E402


hidden = integration.hidden

CAMPAIGN_ID = "OMEGA-NATIVE-RUNTIME-P0"
DEFAULT_OUTPUT_ROOT = HERE / "results"
SEALED_MANIFEST = integration.SEALED_MANIFEST
DEFAULT_CACHE_FILE = integration.DEFAULT_CACHE_FILE
PHYSICAL_BATCH = integration.PHYSICAL_BATCH
WINDOW_TOKENS = 256
DEFAULT_TOTAL_UPDATES = 20
DEFAULT_WARMUP_UPDATES = 4
K_VALUES = (1, 4)
STAGE_NAMES = (
    "student_embedding_prelude",
    "recurrent_forward",
    "student_readout_projection",
    "teacher_hidden_fetch",
    "teacher_lm_head",
    "ce_kl_forward",
    "student_vocab_projection",
    "backward_total",
    "backward_loss_logits",
    "backward_vocab_readout",
    "backward_recurrent",
    "backward_embedding_prelude",
    "clip",
    "adamw",
)


class ProfileContractError(RuntimeError):
    """Raised when instrumentation cannot represent the production route."""


class RealExecutionAuthorizationError(RuntimeError):
    pass


def require_real_authorization(confirmed: bool) -> None:
    if not confirmed:
        raise RealExecutionAuthorizationError("P0 profiling requires --confirm-real-execution")


def _sync(value: Tensor | None = None) -> None:
    if value is not None and value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


@contextmanager
def timed(timings: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - started


def _student_forward_timed(
    model: nn.Module,
    tokens: Tensor,
    previous_state: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor], dict[str, float]]:
    """Mirror production ``OmegaCoreLMFast.recur_states`` with stage timers.

    The returned boundary tensors are the only tensors used for backward
    segmentation.  The forward equations intentionally match the production
    implementation line-for-line.
    """
    required = ("embedding", "prelude", "blocks", "depth_embedding", "gate_logits", "output_projection")
    if any(not hasattr(model, name) for name in required):
        raise ProfileContractError("model does not expose the production OmegaCoreLMFast stages")
    batch, sequence = tokens.shape
    slots = int(model.slots)
    dimension = int(model.dimension)
    state_dimension = slots * dimension
    timings: dict[str, float] = {}

    with timed(timings, "student_embedding_prelude"):
        prelude_weight = model.prelude.weight
        embedded = model.embedding(tokens)
        token_part = F.linear(embedded, prelude_weight[:, :dimension], model.prelude.bias)
    _sync(token_part)

    with timed(timings, "recurrent_forward"):
        state_part_weight = prelude_weight[:, dimension:]
        depth_biases, gates = model._round_constants()
        blocks = [model.blocks[0] if model.variant == "shared" else model.blocks[index] for index in range(model.rounds)]
        state = previous_state
        continuation_states: list[Tensor] = []
        readout_states: list[Tensor] = []
        for position in range(sequence):
            write = token_part[:, position] + F.linear(state.mean(dim=1), state_part_weight)
            anchor = F.rms_norm(
                state.flatten(start_dim=1) + write,
                (state_dimension,),
                model.prelude_norm_weight,
                1e-6,
            ).view(batch, slots, dimension)
            candidate = anchor
            for round_index, block in enumerate(blocks):
                candidate = block(candidate, anchor, depth_biases[round_index], gates[round_index])
            readout_state = candidate
            if valid_mask is None:
                continuation = candidate
            else:
                active = valid_mask[:, position].view(-1, 1, 1)
                continuation = torch.where(active, candidate, state)
            continuation_states.append(continuation)
            readout_states.append(readout_state)
            state = continuation
        continuation_stack = torch.stack(continuation_states, dim=1)
        readout_stack = torch.stack(readout_states, dim=1)
    _sync(readout_stack)

    with timed(timings, "student_readout_projection"):
        projected = model.project(readout_stack)
    _sync(projected)
    with timed(timings, "student_vocab_projection"):
        student_logits = model.logits_from_projected(projected)
    _sync(student_logits)
    return state, student_logits, {
        "embedding_prelude_boundary": token_part,
        "recurrent_readout_boundary": readout_stack,
        "student_vocab_boundary": student_logits,
        "continuation_states": continuation_stack,
    }, timings


def _backward_timed(loss: Tensor, boundaries: Mapping[str, Tensor]) -> dict[str, Any]:
    """Measure one real backward using hooks, never three backward calls."""
    events: dict[str, float] = {}
    handles = []

    def record(name: str):
        def hook(_: Tensor) -> None:
            events.setdefault(name, time.perf_counter())

        return hook

    for key in ("student_vocab_boundary", "recurrent_readout_boundary", "embedding_prelude_boundary"):
        tensor = boundaries[key]
        if not tensor.requires_grad:
            return {"method": "unsegmented", "status": "boundary_without_grad", "backward_total": None}
        handles.append(tensor.register_hook(record(key)))
    started = time.perf_counter()
    try:
        loss.backward()
    finally:
        ended = time.perf_counter()
        for handle in handles:
            handle.remove()
    total = ended - started
    ordered = ("student_vocab_boundary", "recurrent_readout_boundary", "embedding_prelude_boundary")
    if not all(key in events for key in ordered):
        return {"method": "unsegmented", "status": "boundary_hook_missing", "backward_total": total}
    vocab_hook, readout_hook, prelude_hook = (events[key] for key in ordered)
    if not (vocab_hook <= readout_hook <= prelude_hook <= ended):
        return {"method": "unsegmented", "status": "boundary_hook_order_invalid", "backward_total": total}
    segments = {
        "backward_loss_logits": vocab_hook - started,
        "backward_vocab_readout": readout_hook - vocab_hook,
        "backward_recurrent": prelude_hook - readout_hook,
        "backward_embedding_prelude": ended - prelude_hook,
    }
    reconstructed = sum(segments.values())
    if not math.isclose(total, reconstructed, rel_tol=1e-9, abs_tol=1e-9):
        raise ProfileContractError(
            "single-backward boundary timing does not conserve elapsed time: "
            f"total={total}, reconstructed={reconstructed}"
        )
    return {
        "method": "boundary_hooks_single_backward",
        "status": "segmented",
        "backward_total": total,
        **segments,
    }


def _teacher_hidden_timed(payload: np.ndarray, weight: Tensor, bias: Tensor | None, positions: Sequence[int], window: int) -> tuple[Tensor, dict[str, float]]:
    timings: dict[str, float] = {}
    with timed(timings, "teacher_hidden_fetch"):
        hidden_states = torch.from_numpy(np.array(payload[window, list(positions)], copy=True)).float()
    _sync(hidden_states)
    with timed(timings, "teacher_lm_head"):
        teacher_logits = hidden.hidden_to_logits(hidden_states, weight, bias)
    _sync(teacher_logits)
    return teacher_logits, timings


def profile_update(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    state: Tensor,
    teacher_payload: np.ndarray,
    teacher_weight: Tensor,
    teacher_bias: Tensor | None,
    document_positions: Sequence[int],
    window: int,
) -> tuple[dict[str, Any], Tensor]:
    timings: dict[str, float] = {}
    update_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    teacher_logits, teacher_timings = _teacher_hidden_timed(teacher_payload, teacher_weight, teacher_bias, document_positions, window)
    timings.update(teacher_timings)
    next_state, student_logits, boundaries, student_timings = _student_forward_timed(model, inputs, state)
    timings.update(student_timings)
    with timed(timings, "ce_kl_forward"):
        losses = hidden.base.distillation_loss(student_logits, teacher_logits, targets)
    _sync(losses["total"])
    backward = _backward_timed(losses["total"], boundaries)
    timings.update({key: value for key, value in backward.items() if key.startswith("backward_") and value is not None})
    with timed(timings, "clip"):
        clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
    with timed(timings, "adamw"):
        optimizer.step()
    _sync()
    timings["total"] = time.perf_counter() - update_started
    timings["backward_method"] = backward["method"]  # type: ignore[assignment]
    timings["backward_status"] = backward["status"]  # type: ignore[assignment]
    return {
        "timings": timings,
        "losses": {name: float(value.detach().item()) for name, value in losses.items()},
        "clip_norm": clip_norm,
        "next_state": next_state.detach().clone(),
    }, next_state.detach().clone()


def _aggregate(records: Sequence[Mapping[str, Any]], warmup_updates: int) -> dict[str, Any]:
    measured = list(records[warmup_updates:])
    if not measured:
        raise ValueError("warmup_updates leaves no measured updates")
    values: dict[str, float] = {}
    for name in STAGE_NAMES:
        values[name] = float(sum(float(record["timings"].get(name, 0.0)) for record in measured) / len(measured))
    values["total"] = float(sum(float(record["timings"].get("total", 0.0)) for record in measured) / len(measured))
    total = values["total"]
    percentages = {name: (100.0 * value / total if total else 0.0) for name, value in values.items() if name != "total"}
    statuses = sorted({str(record["timings"].get("backward_status", "unknown")) for record in measured})
    methods = sorted({str(record["timings"].get("backward_method", "unknown")) for record in measured})
    return {
        "updates_total": len(records),
        "warmup_updates": warmup_updates,
        "measured_updates": len(measured),
        "mean_seconds": values,
        "percent_of_total": percentages,
        "backward_statuses": statuses,
        "backward_methods": methods,
    }


def derive_backward_gap_report(report_path: Path) -> dict[str, Any]:
    """Add loss-to-logits backward bucket to an existing profile report.

    This does not execute a model or touch the cache.  It derives the missing
    bucket from timestamps already represented by the existing three segments
    and the backward envelope in the self-hashed report.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not hidden.verify_self_hash(report):
        raise ProfileContractError("cannot derive backward bucket from invalid report self-hash")
    for combination in report["combinations"].values():
        records = combination["records"]
        for record in records:
            timings = record["timings"]
            if timings.get("backward_status") != "segmented":
                continue
            total = float(timings["backward_total"])
            vocab_readout = float(timings["backward_vocab_readout"])
            recurrent = float(timings["backward_recurrent"])
            embedding = float(timings["backward_embedding_prelude"])
            loss_logits = total - vocab_readout - recurrent - embedding
            reconstructed = loss_logits + vocab_readout + recurrent + embedding
            if loss_logits < -1e-9 or not math.isclose(total, reconstructed, rel_tol=1e-9, abs_tol=1e-9):
                raise ProfileContractError("derived backward loss/logits bucket violates conservation invariant")
            timings["backward_loss_logits"] = loss_logits
        combination["profile"] = _aggregate(records, int(combination["profile"]["warmup_updates"]))
    report["backward_derivation"] = {
        "status": "DERIVED_FROM_EXISTING_REPORT",
        "formula": "backward_total - backward_vocab_readout - backward_recurrent - backward_embedding_prelude",
        "real_profile_rerun": False,
    }
    return hidden.write_self_hashed(report_path, report)


def run_profile(
    *,
    manifest_path: Path,
    cache_file: Path,
    output_root: Path,
    confirm_real_execution: bool,
    total_updates: int = DEFAULT_TOTAL_UPDATES,
    warmup_updates: int = DEFAULT_WARMUP_UPDATES,
) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution)
    if total_updates <= 0 or total_updates % 2 or warmup_updates < 0 or warmup_updates >= total_updates:
        raise ValueError("total_updates must be positive/even and warmup_updates must leave measured updates")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_file.with_suffix(cache_file.suffix + ".p0.lock")
    with integration.CacheLock(lock_path):
        memory = integration.memory_safety_gate()
        provenance = integration.verify_provenance(manifest_path, cache_file=cache_file)
        manifest, resolved_cache = hidden.hidden_cache_load_manifest(manifest_path)
        frozen, documents, _ = hidden.base.load_frozen_train_documents()
        payload, cache_load_seconds, _ = hidden.preload_hidden_cache(resolved_cache, manifest)
        teacher_weight, teacher_bias = hidden.load_lm_head(manifest_path.parent, manifest["lm_head"])
        combinations: dict[str, Any] = {}
        for k in K_VALUES:
            model = ce.fresh_model(20260913, k)
            optimizer = torch.optim.AdamW(model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS, eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY)
            state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
            records: list[dict[str, Any]] = []
            for update in range(total_updates):
                pair_index = update // 2
                window = update % 2
                positions = [int(index) for index in frozen["cyclic_pairs"]["pairs"][pair_index]["document_indices"]]
                source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
                offset = window * WINDOW_TOKENS
                inputs = source[:, offset : offset + WINDOW_TOKENS]
                targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
                current_state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if window == 0 else state
                result, state = profile_update(model=model, optimizer=optimizer, inputs=inputs, targets=targets, state=current_state, teacher_payload=payload, teacher_weight=teacher_weight, teacher_bias=teacher_bias, document_positions=positions, window=window)
                records.append({"update": update, "pair": pair_index, "window": window, **result})
            combinations[f"K{k}"] = {"K": k, "profile": _aggregate(records, warmup_updates), "records": [{key: value for key, value in record.items() if key != "next_state"} for record in records]}
    report = {
        "schema": "omega-native-runtime-p0-v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "PROFILE_PASS",
        "route": {"model": "R1/F", "loss": "CE+KL", "teacher": "RAM hidden-cache + online exact lm_head", "bptt_tokens": WINDOW_TOKENS, "dtype": "float32", "batch": PHYSICAL_BATCH, "optimizer": "AdamW", "cpp_or_cuda": False},
        "provenance": provenance,
        "memory": memory,
        "cache_load_seconds": cache_load_seconds,
        "schedule": {"total_updates": total_updates, "warmup_updates": warmup_updates, "measured_updates": total_updates - warmup_updates, "window_pattern": "update%2", "pair_pattern": "update//2"},
        "combinations": combinations,
    }
    return hidden.write_self_hashed(output_root / "profile_report.json", report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile", action="store_true")
    mode.add_argument("--derive-backward-gap", action="store_true")
    parser.add_argument("--manifest", type=Path, default=SEALED_MANIFEST)
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_ROOT / "profile_report.json")
    parser.add_argument("--total-updates", type=int, default=DEFAULT_TOTAL_UPDATES)
    parser.add_argument("--warmup-updates", type=int, default=DEFAULT_WARMUP_UPDATES)
    parser.add_argument("--confirm-real-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.derive_backward_gap:
        print(json.dumps(derive_backward_gap_report(args.report), indent=2, sort_keys=True))
        return 0
    report = run_profile(manifest_path=args.manifest, cache_file=args.cache_file, output_root=args.output_root, confirm_real_execution=args.confirm_real_execution, total_updates=args.total_updates, warmup_updates=args.warmup_updates)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
