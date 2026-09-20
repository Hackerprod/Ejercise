"""OMEGA TBPTT-16 directional screening probe.

This unit is intentionally isolated from historical campaign runners.  It
implements parameter-gradient accumulation across detached 16-token chunks,
one clipping operation, and one optimizer step per 256-token update.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import psutil
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
REPO_ROOT = CAMPAIGN_ROOT.parent.parent
BASELINE_DIR = CAMPAIGN_ROOT / "omega_ce_only_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import run_omega_ce_only_baseline as base  # noqa: E402


CAMPAIGN_ID = "OMEGA-TBPTT-DIRECTIONAL-PROBE"
ADDENDUM = 225
HORIZON = 16
K = 4
SEED = 20260913
TOTAL_UPDATES = 2000
BOUNDARIES = (0, 500, 1000, 1500, 2000)
PHYSICAL_BATCH = 8
WINDOW_TOKENS = 256
SEGMENTS_PER_UPDATE = WINDOW_TOKENS // HORIZON
WARMUP_UPDATES = 2
MEASURED_UPDATES = 8
PREFLIGHT_UPDATES = WARMUP_UPDATES + MEASURED_UPDATES
CLIP_NORM = 1.0
ATOL = 1e-5
RTOL = 1e-5
FULL_COMBINATION = "FULL-BPTT256-K4-seed13"
TBPTT_COMBINATION = "TBPTT16-K4-seed13"
CPU_INTRAOP_THREADS = 4
CPU_INTEROP_THREADS = 1
RATIO_TECH_POSITIVE = 0.90
RATIO_TECH_NEUTRAL = 1.05
QUALITY_SAFE = 0.05
QUALITY_TRADEOFF = 0.10


class RealExecutionAuthorizationError(RuntimeError):
    pass


class CorrectnessGateError(AssertionError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def self_hash(value: Mapping[str, Any], field: str = "report_self_hash") -> str:
    unsigned = dict(value)
    unsigned[field] = "__SELF_HASH__"
    return hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def with_self_hash(value: Mapping[str, Any], field: str = "report_self_hash") -> dict[str, Any]:
    result = dict(value)
    result[field] = self_hash(result, field)
    return result


def verify_self_hash(value: Mapping[str, Any], field: str = "report_self_hash") -> bool:
    digest = value.get(field)
    return isinstance(digest, str) and self_hash(value, field) == digest


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def configure_policy() -> dict[str, Any]:
    base.r1.configure_cpu_runtime()
    if torch.get_num_threads() != CPU_INTRAOP_THREADS:
        raise AssertionError("TBPTT intraop policy drift")
    return {
        "device": "cpu",
        "dtype": "float32",
        "execution": "eager",
        "intraop_threads": CPU_INTRAOP_THREADS,
        "interop_threads": CPU_INTEROP_THREADS,
        "physical_batch": PHYSICAL_BATCH,
        "window_tokens": WINDOW_TOKENS,
        "horizon": HORIZON,
        "segments_per_update": SEGMENTS_PER_UPDATE,
        "optimizer": {"type": "AdamW", "lr": base.BASE_LR, "betas": list(base.ADAMW_BETAS), "eps": base.ADAMW_EPS, "weight_decay": base.WEIGHT_DECAY, "clip_norm": CLIP_NORM},
    }


def fresh_model(*, vocab_size: int = base.TOKENIZER_VOCAB, dimension: int = base.DIMENSION, slots: int = base.SLOTS) -> base.OmegaCoreLMFast:
    return base.fresh_model(SEED, K, vocab_size=vocab_size, dimension=dimension, slots=slots)


def _split_inputs(source: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = source[:, start : start + WINDOW_TOKENS]
    targets = source[:, start + 1 : start + WINDOW_TOKENS + 1]
    return inputs, targets


def _weighted_chunk_loss(
    model: base.OmegaCoreLMFast,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    state: torch.Tensor,
    valid_mask: torch.Tensor,
    total_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    next_state, _, losses = base.ce_only_forward_loss(model, input_ids, targets, state, valid_mask)
    chunk_valid = valid_mask.to(dtype=next_state.dtype).sum()
    weighted = losses["ce"] * (chunk_valid / total_valid.clamp_min(1.0))
    return next_state, weighted, {"ce": losses["ce"], "valid_tokens": int(chunk_valid.item())}


def _tbptt_update(
    model: base.OmegaCoreLMFast,
    optimizer: torch.optim.Optimizer,
    source: torch.Tensor,
    state: torch.Tensor,
    *,
    backward_mode: str,
    measure: bool = False,
) -> dict[str, Any]:
    if backward_mode not in {"incremental", "reference"}:
        raise ValueError(backward_mode)
    optimizer.zero_grad(set_to_none=True)
    total_valid = torch.tensor(float(source.shape[0] * WINDOW_TOKENS), dtype=torch.float32)
    state = state.detach()
    losses: list[torch.Tensor] = []
    forward_seconds = 0.0
    ce_seconds = 0.0
    backward_seconds = 0.0
    for chunk_index in range(SEGMENTS_PER_UPDATE):
        start = chunk_index * HORIZON
        inputs = source[:, start : start + HORIZON]
        targets = source[:, start + 1 : start + HORIZON + 1]
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        started = time.perf_counter()
        next_state, weighted_loss, _ = _weighted_chunk_loss(model, inputs, targets, state, valid_mask, total_valid)
        elapsed = time.perf_counter() - started
        forward_seconds += elapsed
        ce_started = time.perf_counter()
        loss_value = weighted_loss.detach()
        ce_seconds += time.perf_counter() - ce_started
        if backward_mode == "incremental":
            backward_started = time.perf_counter()
            weighted_loss.backward()
            backward_seconds += time.perf_counter() - backward_started
        else:
            losses.append(weighted_loss)
        state = next_state.detach()
    if backward_mode == "reference":
        backward_started = time.perf_counter()
        torch.stack(losses).sum().backward()
        backward_seconds = time.perf_counter() - backward_started
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    clip_started = time.perf_counter()
    clip_value = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
    clip_seconds = time.perf_counter() - clip_started
    adam_started = time.perf_counter()
    optimizer.step()
    adamw_seconds = time.perf_counter() - adam_started
    return {
        "loss": float(loss_value.item()),
        "gradients": gradients,
        "state": state,
        "clip_norm": clip_value,
        "clip_seconds": clip_seconds,
        "adamw_seconds": adamw_seconds,
        "forward_seconds": forward_seconds,
        "ce_seconds": ce_seconds,
        "backward_seconds": backward_seconds,
    }


def _nested_allclose(left: Any, right: Any, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.is_tensor(left) and torch.is_tensor(right) and torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_nested_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, type(right)) and len(left) == len(right) and all(_nested_allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right))
    return left == right


def tbptt_correctness_gate(*, model_factory: Callable[[], base.OmegaCoreLMFast] | None = None) -> dict[str, Any]:
    """Compare incremental and summed-loss TBPTT with identical detach boundaries."""
    configure_policy()
    factory = model_factory or fresh_model
    model_incremental = factory()
    model_reference = copy.deepcopy(model_incremental)
    optimizer_incremental = torch.optim.AdamW(model_incremental.parameters(), lr=base.BASE_LR, betas=base.ADAMW_BETAS, eps=base.ADAMW_EPS, weight_decay=base.WEIGHT_DECAY)
    optimizer_reference = torch.optim.AdamW(model_reference.parameters(), lr=base.BASE_LR, betas=base.ADAMW_BETAS, eps=base.ADAMW_EPS, weight_decay=base.WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    source = torch.randint(0, model_incremental.vocab_size, (PHYSICAL_BATCH, WINDOW_TOKENS + 1), generator=generator)
    initial = model_incremental.initial_state(PHYSICAL_BATCH, device=torch.device("cpu"))
    left = _tbptt_update(model_incremental, optimizer_incremental, source, initial, backward_mode="incremental")
    right = _tbptt_update(model_reference, optimizer_reference, source, initial.clone(), backward_mode="reference")
    checks = {
        "loss": math.isclose(left["loss"], right["loss"], rel_tol=RTOL, abs_tol=ATOL),
        "gradients": _nested_allclose(left["gradients"], right["gradients"]),
        "state": _nested_allclose(left["state"], right["state"]),
        "clip_norm": math.isclose(left["clip_norm"], right["clip_norm"], rel_tol=RTOL, abs_tol=ATOL),
        "parameters_after_step": _nested_allclose(model_incremental.state_dict(), model_reference.state_dict()),
        "adamw_moments": _nested_allclose(optimizer_incremental.state_dict(), optimizer_reference.state_dict()),
    }
    if not all(checks.values()):
        raise CorrectnessGateError(f"TBPTT16 incremental/reference mismatch: {checks}")
    return {"passed": True, "atol": ATOL, "rtol": RTOL, "checks": checks, "horizon": HORIZON, "segments_per_update": SEGMENTS_PER_UPDATE}


def _rss() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _phase0_docs(smoke: bool) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if smoke:
        docs = [{"tokens": [((index + 3) * 7 + pos) % 31 for pos in range(base.RETAINED_TOKENS)]} for index in range(PHYSICAL_BATCH)]
        manifest = {"cyclic_pairs": {"pairs": [{"document_indices": list(range(PHYSICAL_BATCH))}]}}
        return docs, manifest
    manifests, train_documents, _, _ = base.load_real_frozen_documents()
    return train_documents, manifests["train"]


def _preflight_worker(combination: str, *, smoke: bool) -> dict[str, Any]:
    if combination not in {FULL_COMBINATION, TBPTT_COMBINATION}:
        raise ValueError(combination)
    configure_policy()
    docs, manifest = _phase0_docs(smoke)
    model = fresh_model(vocab_size=31, dimension=4, slots=1) if smoke else fresh_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.BASE_LR, betas=base.ADAMW_BETAS, eps=base.ADAMW_EPS, weight_decay=base.WEIGHT_DECAY)
    timings = {key: 0.0 for key in ("total_seconds", "backward_seconds", "student_forward_seconds", "vocab_ce_seconds", "clip_seconds", "adamw_seconds")}
    peak_rss = _rss()
    state: torch.Tensor | None = None
    for update in range(PREFLIGHT_UPDATES):
        pair = manifest["cyclic_pairs"]["pairs"][0]
        batch = [docs[index] for index in pair["document_indices"]]
        source = torch.tensor([doc["tokens"] for doc in batch], dtype=torch.long)
        source = source[:, : WINDOW_TOKENS + 1]
        state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if state is None else state.detach()
        started = time.perf_counter()
        if combination == FULL_COMBINATION:
            optimizer.zero_grad(set_to_none=True)
            forward_started = time.perf_counter()
            recurrent_result = model.recur_states(source[:, :WINDOW_TOKENS], state)
            forward_elapsed = time.perf_counter() - forward_started
            next_state, _, _, readout_states = recurrent_result
            ce_started = time.perf_counter()
            losses = base.ce_only_loss(model, readout_states, source[:, 1:WINDOW_TOKENS + 1])
            ce_elapsed = time.perf_counter() - ce_started
            backward_started = time.perf_counter()
            losses["total"].backward()
            backward_elapsed = time.perf_counter() - backward_started
            clip_started = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            clip_elapsed = time.perf_counter() - clip_started
            adam_started = time.perf_counter()
            optimizer.step()
            adam_elapsed = time.perf_counter() - adam_started
            state = next_state.detach()
        else:
            result = _tbptt_update(model, optimizer, source, state, backward_mode="incremental")
            forward_elapsed = result["forward_seconds"]
            ce_elapsed = result["ce_seconds"]
            backward_elapsed = result["backward_seconds"]
            clip_elapsed = result["clip_seconds"]
            adam_elapsed = result["adamw_seconds"]
            state = result["state"]
        peak_rss = max(peak_rss, _rss())
        if update >= WARMUP_UPDATES:
            timings["total_seconds"] += time.perf_counter() - started
            timings["backward_seconds"] += backward_elapsed
            timings["student_forward_seconds"] += forward_elapsed
            timings["vocab_ce_seconds"] += ce_elapsed
            timings["clip_seconds"] += clip_elapsed
            timings["adamw_seconds"] += adam_elapsed
    return {"combination": combination, "fresh_process": True, "updates_total": PREFLIGHT_UPDATES, "warmup_updates": WARMUP_UPDATES, "measured_updates": MEASURED_UPDATES, "worker_pid": os.getpid(), "peak_rss_bytes": peak_rss, **timings}


def require_real_authorization(confirm: bool, action: str) -> None:
    if not confirm:
        raise RealExecutionAuthorizationError(f"{action} requires --confirm-real-execution")


def build_cost_report(measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {str(row["combination"]): dict(row) for row in measurements}
    if set(rows) != {FULL_COMBINATION, TBPTT_COMBINATION}:
        raise ValueError("cost preflight requires exactly full and TBPTT rows")
    full = rows[FULL_COMBINATION]
    tbptt = rows[TBPTT_COMBINATION]
    rt = tbptt["total_seconds"] / full["total_seconds"]
    report = {
        "schema": "omega-tbptt-directional-probe-cost-report-v1",
        "campaign_id": CAMPAIGN_ID,
        "addendum": ADDENDUM,
        "quality_gate": None,
        "protocol": {"fresh_process_per_combination": True, "total_updates": 20, "warmup_updates": WARMUP_UPDATES, "measured_updates": MEASURED_UPDATES, "horizon": HORIZON, "combinations": [FULL_COMBINATION, TBPTT_COMBINATION]},
        "measurements": {FULL_COMBINATION: full, TBPTT_COMBINATION: tbptt},
        "R_t": rt,
        "formula": "R_t=T_TBPTT16/T_FULL256",
    }
    return with_self_hash(report)


def run_cost_preflight(output_dir: Path, *, confirm_real_execution: bool, smoke: bool = False) -> dict[str, Any]:
    if not smoke:
        require_real_authorization(confirm_real_execution, "cost preflight real execution")
    rows: list[dict[str, Any]] = []
    for combination in (FULL_COMBINATION, TBPTT_COMBINATION):
        command = [sys.executable, str(Path(__file__).resolve()), "--cost-worker", "--combination", combination]
        command.append("--smoke-worker" if smoke else "--confirm-real-execution")
        completed = __import__("subprocess").run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"cost worker failed for {combination}: {completed.stderr}")
        rows.append(json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1]))
    report = build_cost_report(rows)
    report["worker_commands"] = [[sys.executable, str(Path(__file__).resolve()), "--cost-worker", "--combination", row["combination"]] for row in rows]
    report = with_self_hash(report)
    write_json(output_dir / "cost_preflight_report.json", report)
    return report


def classify_screening(*, rt: float, delta_2000: float, rss_reduction: float) -> dict[str, Any]:
    tech = "TECH_POSITIVE" if rt <= RATIO_TECH_POSITIVE else "TECH_NEUTRAL" if rt <= RATIO_TECH_NEUTRAL else "TECH_NEGATIVE"
    quality = "QUALITY_SAFE" if delta_2000 <= QUALITY_SAFE else "QUALITY_TRADEOFF" if delta_2000 <= QUALITY_TRADEOFF else "QUALITY_BAD"
    if quality == "QUALITY_BAD":
        final = "DIRECTIONAL-STOP"
    elif quality == "QUALITY_SAFE":
        if rt <= RATIO_TECH_POSITIVE or rss_reduction >= 0.20:
            final = "DIRECTIONAL-GO"
        elif rt <= RATIO_TECH_NEUTRAL:
            final = "DIRECTIONAL-NEUTRAL"
        else:
            final = "DIRECTIONAL-STOP"
    elif rt <= 0.80:
        final = "DIRECTIONAL-PROMISING-TRADEOFF"
    elif rt <= RATIO_TECH_POSITIVE:
        final = "DIRECTIONAL-WEAK-TRADEOFF"
    elif rt <= RATIO_TECH_NEUTRAL or rss_reduction >= 0.20:
        final = "UNCLASSIFIED"
    else:
        final = "DIRECTIONAL-STOP"
    return {"technical": tech, "quality": quality, "final": final}


def load_frozen_full_curve() -> list[dict[str, Any]]:
    path = BASELINE_DIR / "results" / "runs" / "CE-K4_seed_20260913" / "validation_curve.json"
    curve = json.loads(path.read_text(encoding="utf-8"))
    return curve["curve"] if isinstance(curve, Mapping) else curve


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, update: int, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"update": update, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": dict(config)}, path)


def run_tbptt_training(output_dir: Path, *, confirm_real_execution: bool, cost_report_path: Path) -> dict[str, Any]:
    require_real_authorization(confirm_real_execution, "TBPTT real training")
    cost_report = json.loads(cost_report_path.read_text(encoding="utf-8"))
    if not verify_self_hash(cost_report):
        raise ValueError("cost preflight report self-hash verification failed")
    gate = tbptt_correctness_gate()
    manifests, train_documents, validation_documents, _ = base.load_real_frozen_documents()
    model = fresh_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.BASE_LR, betas=base.ADAMW_BETAS, eps=base.ADAMW_EPS, weight_decay=base.WEIGHT_DECAY)
    config = {"campaign_id": CAMPAIGN_ID, "seed": SEED, "K": K, "horizon": HORIZON, "window_tokens": WINDOW_TOKENS, "loss": "full_cross_entropy_only", "detach_boundary": HORIZON, "optimizer": configure_policy(), "teacher_loaded": False, "frozen_full_curve": str(BASELINE_DIR / "results" / "runs" / "CE-K4_seed_20260913" / "validation_curve.json")}
    run_dir = output_dir / "runs" / "TBPTT16-K4_seed_20260913"
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "train_manifest.json", manifests["train"])
    write_json(run_dir / "validation_manifest.json", manifests["validation"])
    curve: list[dict[str, Any]] = [{"update": 0, **base.evaluate_ce_only(model, validation_documents)}]
    _save_checkpoint(run_dir / "checkpoint_00000.pt", model, optimizer, 0, config)
    state: torch.Tensor | None = None
    ledger: list[dict[str, Any]] = []
    for update in range(TOTAL_UPDATES):
        pair = manifests["train"]["cyclic_pairs"]["pairs"][update % len(manifests["train"]["cyclic_pairs"]["pairs"])]
        documents = [train_documents[int(index)] for index in pair["document_indices"]]
        source = torch.tensor([document["tokens"] for document in documents], dtype=torch.long)[:, : WINDOW_TOKENS + 1]
        state = model.initial_state(PHYSICAL_BATCH, device=torch.device("cpu")) if state is None else state.detach()
        result = _tbptt_update(model, optimizer, source, state, backward_mode="incremental")
        state = result["state"]
        completed = update + 1
        ledger.append({"update": completed, "segments": SEGMENTS_PER_UPDATE, "backward_calls": SEGMENTS_PER_UPDATE, "optimizer_steps": 1, "teacher_loaded": False, "loss": result["loss"]})
        if completed in BOUNDARIES:
            model.eval()
            curve.append({"update": completed, **base.evaluate_ce_only(model, validation_documents)})
            model.train()
            _save_checkpoint(run_dir / f"checkpoint_{completed:05d}.pt", model, optimizer, completed, config)
    write_json(run_dir / "validation_curve.json", {"curve": curve})
    write_json(run_dir / "ledger.json", {"updates": ledger, "teacher_loaded": False, "segments_per_update": SEGMENTS_PER_UPDATE})
    full_curve = load_frozen_full_curve()
    tb_curve = {int(point["update"]): float(point["nll"]) for point in curve}
    full_by_update = {int(point["update"]): float(point["nll"]) for point in full_curve}
    delta_2000 = tb_curve[TOTAL_UPDATES] - full_by_update[TOTAL_UPDATES]
    full_rss = cost_report["measurements"][FULL_COMBINATION]["peak_rss_bytes"]
    tb_rss = cost_report["measurements"][TBPTT_COMBINATION]["peak_rss_bytes"]
    rss_reduction = 1.0 - (tb_rss / full_rss)
    report = {"schema": "omega-tbptt-directional-probe-final-report-v1", "campaign_id": CAMPAIGN_ID, "training_performed": True, "cost_report": cost_report_path.as_posix(), "correctness_gate": gate, "run_dir": run_dir.as_posix(), "tbptt_validation_curve": curve, "frozen_full_bptt_curve": str(BASELINE_DIR / "results" / "runs" / "CE-K4_seed_20260913" / "validation_curve.json"), "delta_16": [{"update": update, "delta_tbptt_minus_full": tb_curve[update] - full_by_update[update]} for update in BOUNDARIES], "delta_16_at_2000": delta_2000, "R_t": cost_report["R_t"], "rss_reduction": rss_reduction, "classification": classify_screening(rt=cost_report["R_t"], delta_2000=delta_2000, rss_reduction=rss_reduction)}
    report = with_self_hash(report)
    write_json(output_dir / "tbptt_directional_probe_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-preflight", action="store_true")
    parser.add_argument("--cost-worker", action="store_true")
    parser.add_argument("--tbptt-training", action="store_true")
    parser.add_argument("--combination", choices=[FULL_COMBINATION, TBPTT_COMBINATION])
    parser.add_argument("--smoke-worker", action="store_true")
    parser.add_argument("--confirm-real-execution", action="store_true")
    parser.add_argument("--cost-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.cost_worker:
        if args.combination is None:
            parser.error("--cost-worker requires --combination")
        if not args.smoke_worker:
            require_real_authorization(args.confirm_real_execution, "cost worker real execution")
        print(json.dumps(_preflight_worker(args.combination, smoke=args.smoke_worker), sort_keys=True))
        return 0
    if sum(bool(value) for value in (args.cost_preflight, args.tbptt_training)) != 1:
        parser.error("select exactly one of --cost-preflight or --tbptt-training")
    if args.cost_preflight:
        report = run_cost_preflight(args.output_dir, confirm_real_execution=args.confirm_real_execution)
    else:
        if args.cost_report is None:
            parser.error("--tbptt-training requires --cost-report")
        report = run_tbptt_training(args.output_dir, confirm_real_execution=args.confirm_real_execution, cost_report_path=args.cost_report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
