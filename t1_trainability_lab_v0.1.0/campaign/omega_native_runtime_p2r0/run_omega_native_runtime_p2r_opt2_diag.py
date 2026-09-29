"""Fresh-process P2-R-OPT2 diagnostic runner.

Part A uses T=1 and external total_update wall time for one residual mask at a
time. Part B uses T=4 and captures independent native runtime timing snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
DEFAULT_BUILD_ROOT = HERE / "native"
R1_SEED = 20260913
WINDOW_TOKENS = 256
WARMUP_UPDATES = 2
MEASURED_UPDATES = 6

VARIANTS: dict[str, int] = {
    "baseline": 0,
    "attention_scores_softmax_mixing": 1 << 0,
    "depth_embedding_pairwise": 1 << 1,
    "rmsnorm_gates": 1 << 2,
    "state_prelude": 1 << 3,
    "history_buffer_bookkeeping": 1 << 4,
    "depth_embedding_carry": 1 << 5,
}
MASK_TO_NAME = {mask: name for name, mask in VARIANTS.items()}


def _load_modules() -> tuple[Any, Any, Any, Any]:
    for path in (CAMPAIGN_ROOT / "omega_native_runtime_p0", CAMPAIGN_ROOT / "omega_ce_only_baseline", HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import omega_recurrent_diagnostic_bridge as diagnostic  # noqa: PLC0415
    import run_omega_ce_only_baseline as ce  # noqa: PLC0415
    import run_omega_native_runtime_p0 as p0  # noqa: PLC0415
    import run_omega_native_runtime_r2_benchmark as r2  # noqa: PLC0415

    return p0, ce, r2, diagnostic


def _dll_metadata(path: Path, *, mask: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "build_metadata": {
            "configuration": "Release",
            "target": "omega_recurrent_diagnostic_shared",
            "OMEGA_P2R_DIAGNOSTIC": True,
            "OMEGA_DIAGNOSTIC_ELIDE_MASK": mask,
            "OMEGA_PROFILE_INTERNAL": False,
            "clock": "std::chrono::steady_clock",
            "isa": "AVX2 shared diagnostic target",
        },
    }


def _measurement_setup(p0: Any, ce: Any, manifest: Path, cache: Path, rounds: int) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    documents, payload, teacher_weight, teacher_bias = _load_inputs(p0, manifest, cache)
    model = ce.fresh_model(R1_SEED, rounds)
    state_part_weight = _prepare_native_model(model, rounds)
    optimizer = __import__("torch").optim.AdamW(
        model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY,
    )
    physical_batch = int(p0.PHYSICAL_BATCH)
    state = model.initial_state(physical_batch, device=__import__("torch").device("cpu"))
    frozen, _, _ = p0.hidden.base.load_frozen_train_documents()
    return documents, payload, teacher_weight, teacher_bias, model, state_part_weight, optimizer, (physical_batch, frozen, state)


def _load_inputs(p0: Any, manifest: Path, cache: Path) -> tuple[list[dict[str, Any]], Any, Any, Any]:
    p0.integration.verify_provenance(manifest, cache_file=cache)
    manifest_data, resolved_cache = p0.hidden.hidden_cache_load_manifest(manifest)
    payload, _, _ = p0.hidden.preload_hidden_cache(resolved_cache, manifest_data)
    teacher_weight, teacher_bias = p0.hidden.load_lm_head(manifest.parent, manifest_data["lm_head"])
    _, documents, _ = p0.hidden.base.load_frozen_train_documents()
    return documents, payload, teacher_weight, teacher_bias


def _prepare_native_model(model: Any, rounds: int) -> Any:
    dimension = int(model.dimension)
    state_part_weight = model.prelude.weight[:, dimension:]
    expected_shape = (int(model.slots) * dimension, dimension)
    if tuple(state_part_weight.shape) != expected_shape:
        raise RuntimeError(f"diagnostic state view shape {tuple(state_part_weight.shape)} != {expected_shape}")
    if tuple(state_part_weight.stride()) != (2 * dimension, 1):
        raise RuntimeError("diagnostic state view is not fused prelude view")
    if int(model.rounds) != rounds:
        raise RuntimeError("diagnostic model rounds mismatch")
    return state_part_weight


def _teacher_logits(p0: Any, payload: Any, teacher_weight: Any, teacher_bias: Any, positions: Sequence[int], window: int) -> Any:
    torch = __import__("torch")
    hidden_states = torch.from_numpy(payload[window, list(positions)].copy()).float()
    return p0.hidden.hidden_to_logits(hidden_states, teacher_weight, teacher_bias)


def _run_update(
    *, model: Any, optimizer: Any, inputs: Any, targets: Any, previous_state: Any,
    teacher_logits: Any, ce: Any, p0: Any, diagnostic: Any, capture_native: bool,
) -> tuple[dict[str, Any], Any]:
    torch = __import__("torch")
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    next_state, student_logits, _ = diagnostic_apply(model, inputs, previous_state, diagnostic)
    forward_snapshot = diagnostic.snapshot() if capture_native else None
    losses = p0.hidden.base.distillation_loss(student_logits, teacher_logits, targets)
    losses["total"].backward()
    clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
    backward_snapshot = diagnostic.snapshot() if capture_native else None
    optimizer.step()
    return {
        "timing_seconds": {"total_update": time.perf_counter() - started},
        "loss": float(losses["total"].detach().item()),
        "clip_norm": clip_norm,
        "forward_snapshot": forward_snapshot,
        "backward_snapshot": backward_snapshot,
    }, next_state.detach()


def diagnostic_apply(model: Any, inputs: Any, previous_state: Any, diagnostic: Any) -> tuple[Any, Any, Any]:
    import torch.nn.functional as F

    dimension = int(model.dimension)
    token_part = F.linear(model.embedding(inputs), model.prelude.weight[:, :dimension], model.prelude.bias)
    block = model.blocks[0]
    next_state, readout_states = diagnostic.apply(
        token_part, previous_state, model.prelude.weight[:, dimension:], model.prelude_norm_weight,
        block.qkv.weight, block.qkv.bias, block.out.weight, block.out.bias,
        block.fc1.weight, block.fc1.bias, block.fc2.weight, block.fc2.bias,
        block.norm_weight, model.depth_embedding.weight, model.gate_logits, int(model.rounds),
    )
    student_logits = model.logits_from_projected(model.project(readout_states))
    return next_state, student_logits, readout_states


def _run_variant(*, variant: str, dll: Path, manifest: Path, cache: Path, threads: int, rounds: int) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown diagnostic variant: {variant}")
    p0, ce, _, diagnostic = _load_modules()
    diagnostic.configure_library(dll)
    diagnostic.configure_runtime(threads)
    try:
        documents, payload, teacher_weight, teacher_bias, model, state_part_weight, optimizer, setup = _measurement_setup(
            p0, ce, manifest, cache, rounds
        )
        physical_batch, frozen, state = setup
        warmups: list[dict[str, Any]] = []
        measured: list[dict[str, Any]] = []

        def run_items(items: Sequence[Mapping[str, int]], records: list[dict[str, Any]], capture_native: bool) -> None:
            nonlocal state
            for item in items:
                positions = [int(index) for index in frozen["cyclic_pairs"]["pairs"][int(item["pair"])] ["document_indices"]]
                source = __import__("torch").tensor([documents[position]["tokens"] for position in positions], dtype=__import__("torch").long)
                offset = int(item["window"]) * WINDOW_TOKENS
                inputs = source[:, offset : offset + WINDOW_TOKENS]
                targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
                if int(item["window"]) == 0:
                    state = model.initial_state(physical_batch, device=__import__("torch").device("cpu"))
                teacher = _teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, int(item["window"]))
                measurement, state = _run_update(
                    model=model, optimizer=optimizer, inputs=inputs, targets=targets, previous_state=state,
                    teacher_logits=teacher, ce=ce, p0=p0, diagnostic=diagnostic, capture_native=capture_native,
                )
                records.append({"update": int(item["update"]), **measurement})

        run_items(_schedule(WARMUP_UPDATES), warmups, False)
        state = model.initial_state(physical_batch, device=__import__("torch").device("cpu"))
        run_items(_schedule(MEASURED_UPDATES), measured, threads == 4)
        return {
            "variant": variant,
            "mask": VARIANTS[variant],
            "threads": threads,
            "rounds": rounds,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "dll_metadata": _dll_metadata(dll, mask=VARIANTS[variant]),
            "warmup": warmups,
            "updates": measured,
        }
    finally:
        diagnostic.shutdown_runtime()


def _schedule(count: int) -> list[dict[str, int]]:
    return [{"update": index, "window": index % 2, "pair": index // 2} for index in range(count)]


def _snapshot_metrics(snapshot: Mapping[str, Any], operation: str) -> dict[str, Any]:
    starts = [int(value) for value in snapshot["worker_start_ns"]]
    ends = [int(value) for value in snapshot["worker_end_ns"]]
    durations = [end - start for start, end in zip(starts, ends)]
    worker_max = max(durations)
    worker_min = min(durations)
    worker_mean = sum(durations) / len(durations)
    reduction_start = int(snapshot["final_gradient_reduction_start_ns"])
    reduction_end = int(snapshot["final_gradient_reduction_end_ns"])
    reduction = reduction_end - reduction_start if reduction_start else 0
    dispatch_join = int(snapshot["all_workers_done_ns"]) - int(snapshot["dispatch_start_ns"]) - worker_max
    return {
        "operation": operation,
        "worker_compute_max_ns": worker_max,
        "worker_compute_min_ns": worker_min,
        "worker_compute_mean_ns": worker_mean,
        "worker_imbalance_max_over_mean": worker_max / worker_mean if worker_mean else 0.0,
        "serial_final_reduction_ns": reduction,
        "dispatch_join_overhead_ns": dispatch_join,
        "dispatch_return_overhead_ns": int(snapshot["return_ns"]) - int(snapshot["dispatch_start_ns"]) - worker_max - reduction,
    }


def _runtime_aggregate(report: Mapping[str, Any]) -> dict[str, Any]:
    raw_calls: list[dict[str, Any]] = []
    for update in report["updates"]:
        for operation in ("forward", "backward"):
            snapshot = update[f"{operation}_snapshot"]
            raw_calls.append({"update": int(update["update"]), **_snapshot_metrics(snapshot, operation), "snapshot": snapshot})
    durations = [
        int(end) - int(start)
        for call in raw_calls
        for start, end in zip(call["snapshot"]["worker_start_ns"], call["snapshot"]["worker_end_ns"])
    ]
    reductions = [int(call["serial_final_reduction_ns"]) for call in raw_calls if int(call["serial_final_reduction_ns"]) > 0]
    joins = [int(call["dispatch_join_overhead_ns"]) for call in raw_calls]
    mean_worker = sum(durations) / len(durations)
    max_worker = max(durations)
    by_operation: dict[str, dict[str, Any]] = {}
    for operation in ("forward", "backward"):
        operation_calls = [call for call in raw_calls if call["operation"] == operation]
        operation_durations = [
            int(end) - int(start)
            for call in operation_calls
            for start, end in zip(call["snapshot"]["worker_start_ns"], call["snapshot"]["worker_end_ns"])
        ]
        operation_mean = sum(operation_durations) / len(operation_durations)
        by_operation[operation] = {
            "call_count": len(operation_calls),
            "worker_compute_max_ns": max(operation_durations),
            "worker_compute_min_ns": min(operation_durations),
            "worker_compute_mean_ns": operation_mean,
            "worker_imbalance_max_over_mean": max(operation_durations) / operation_mean if operation_mean else 0.0,
        }
    return {
        "call_count": len(raw_calls),
        "worker_compute_max_ns": max_worker,
        "worker_compute_min_ns": min(durations),
        "worker_compute_mean_ns": mean_worker,
        "worker_imbalance_max_over_mean": max_worker / mean_worker if mean_worker else 0.0,
        "serial_final_reduction_total_ns": sum(reductions),
        "serial_final_reduction_mean_ns": sum(reductions) / len(reductions) if reductions else 0.0,
        "dispatch_join_overhead_total_ns": sum(joins),
        "dispatch_join_overhead_mean_ns": sum(joins) / len(joins),
        "by_operation": by_operation,
        "external_total_update_mean_seconds": sum(float(update["timing_seconds"]["total_update"]) for update in report["updates"]) / len(report["updates"]),
        "raw_per_call_slots": raw_calls,
    }


def _default_dll(build_root: Path, mask: int) -> Path:
    return build_root / f"build-p2r-opt2-diag-mask{mask}" / "python" / "omega_recurrent_diagnostic.dll"


def _run_residual_aggregate(build_root: Path, manifest: Path, cache: Path, output: Path) -> dict[str, Any]:
    process_exit_codes: dict[str, int] = {}
    warnings: dict[str, str] = {}
    reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="omega-p2r-opt2-diag-") as temp_dir:
        for variant, mask in VARIANTS.items():
            dll = _default_dll(build_root, mask)
            child_output = Path(temp_dir) / f"{variant}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--variant", variant,
                "--dll", str(dll), "--manifest", str(manifest), "--cache-file", str(cache),
                "--output", str(child_output),
            ]
            completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True, check=False)
            process_exit_codes[variant] = int(completed.returncode)
            if completed.stderr.strip():
                warnings[variant] = completed.stderr[-4000:]
            if completed.returncode != 0:
                raise RuntimeError(f"diagnostic child {variant} failed with exit {completed.returncode}: {completed.stderr}")
            reports[variant] = json.loads(child_output.read_text(encoding="utf-8"))
    baseline_mean = sum(float(item["timing_seconds"]["total_update"]) for item in reports["baseline"]["updates"]) / MEASURED_UPDATES
    variants: dict[str, Any] = {}
    for variant, report in reports.items():
        mean = sum(float(item["timing_seconds"]["total_update"]) for item in report["updates"]) / MEASURED_UPDATES
        delta = mean - baseline_mean
        variants[variant] = {
            "mask": VARIANTS[variant],
            "dll_metadata": report["dll_metadata"],
            "measured_total_update_seconds": [float(item["timing_seconds"]["total_update"]) for item in report["updates"]],
            "mean_total_update_seconds": mean,
            "delta_seconds_vs_baseline": delta,
            "delta_percent_vs_baseline": (100.0 * delta / baseline_mean) if baseline_mean else 0.0,
            "speedup_percent_vs_baseline": (-100.0 * delta / baseline_mean) if baseline_mean else 0.0,
        }
    result = {
        "campaign_id": "P2-R-OPT2-DIAG",
        "part": "A",
        "measurement": "external total_update wall clock",
        "fresh_process_per_variant": True,
        "threads": 1,
        "rounds": 1,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "baseline_mean_total_update_seconds": baseline_mean,
        "requested_blocks": [name for name in VARIANTS if name != "baseline"],
        "other_non_avx2_residual_blocks": ["depth_embedding_carry"],
        "variants": variants,
        "process_exit_codes": process_exit_codes,
        "warnings": warnings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _run_runtime(build_root: Path, manifest: Path, cache: Path, output: Path) -> dict[str, Any]:
    dll = _default_dll(build_root, 0)
    p0, _, _, _ = _load_modules()
    report = _run_variant(variant="baseline", dll=dll, manifest=manifest, cache=cache, threads=4, rounds=1)
    result = {
        "campaign_id": "P2-R-OPT2-DIAG",
        "part": "B",
        "measurement": "independent native monotonic runtime snapshots plus external total_update wall clock",
        "fresh_process": True,
        "threads": 4,
        "rounds": 1,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "hidden_cache_route": "sealed manifest and production benchmark setup",
        "dll_metadata": report["dll_metadata"],
        "process_exit_codes": {"runtime_t4": 0},
        "warnings": {},
        "updates": report["updates"],
        "aggregate": _runtime_aggregate(report),
        "production_module_loaded": bool(p0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("A", "B", "all"), default="all")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-file", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "p2r_opt2_diag.json")
    args = parser.parse_args(argv)
    p0 = None
    if args.manifest is None or args.cache_file is None:
        p0, _, _, _ = _load_modules()
        args.manifest = args.manifest or p0.SEALED_MANIFEST
        args.cache_file = args.cache_file or p0.DEFAULT_CACHE_FILE
    if args.variant is not None:
        if args.dll is None:
            raise ValueError("--variant requires --dll")
        report = _run_variant(
            variant=args.variant, dll=args.dll, manifest=args.manifest, cache=args.cache_file, threads=1, rounds=1
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"variant": args.variant, "output": str(args.output)}, sort_keys=True))
        return 0
    if args.part in ("A", "all"):
        residual_output = args.output if args.part == "A" else args.output.with_name(args.output.stem + "_part_a.json")
        residual = _run_residual_aggregate(args.build_root, args.manifest, args.cache_file, residual_output)
        print(json.dumps({"part_a_output": str(residual_output), "variants": list(residual["variants"])}, sort_keys=True))
    if args.part in ("B", "all"):
        runtime_output = args.output if args.part == "B" else args.output.with_name(args.output.stem + "_part_b.json")
        runtime = _run_runtime(args.build_root, args.manifest, args.cache_file, runtime_output)
        print(json.dumps({"part_b_output": str(runtime_output), "call_count": runtime["aggregate"]["call_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
